"""Commute Buddy advice for the Hosting forecast page (``?advice=commute`` in main.py).

Runs inside forecast-api. ``main`` hands it the cached 5-hour curve (the same one the page's
``?curve=forecast&hours=5`` request returns); this module adds the latest Maps reading from the
public ``traffic-24h.json``, works out the facts in code, and asks OpenAI only to phrase them.

Facts come from code, not from the LLM:
- per direction, the best window to leave from now to 5 hours ahead (leaving now counts), the worst
  time, the comparison with the typical day (``baseline_min``) and the slot furthest from it;
- Singapore and Malaysia public holidays, eves, long weekends and school holidays from
  ``holidays_sg_my.csv``. School holidays are only stated for years the file covers.

The LLM writes one ``insight`` line (why traffic differs from usual, hedged) and at most two tips.
The page sends no input, and one answer per forecast origin is cached, so the OpenAI cost is about
one call per 10-minute origin per instance. The key is read from Secret Manager at run time, never
mounted at deploy, so a missing key or permission cannot fail a forecast-api deploy: the response
then carries the facts with ``advice: null``.
"""

import csv
import datetime
import json
import logging
import os
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

SGT = ZoneInfo("Asia/Singapore")
TRAFFIC_API = os.environ.get("TRAFFIC_API", "https://storage.googleapis.com/swiftborder-public/traffic-24h.json")
OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
FORECAST_HOURS = 5
OPENAI_TIMEOUT_S = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "40"))
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_SECONDS", "600"))
FAILED_RETRY_S = 60  # a failed OpenAI call is retried after this long, not on every visit
CALENDAR_PATH = Path(os.environ.get("CALENDAR_PATH", Path(__file__).parent / "holidays_sg_my.csv"))
SECRET_PROJECT = os.environ.get("BQ_PROJECT", "swiftborder")
SECRET_NAME = os.environ.get("OPENAI_SECRET", "openai-api-key")
KEY_TTL_S = 3600  # re-read the secret hourly, so a new version is picked up without a deploy
KEY_RETRY_S = 300  # after a failed read, wait this long before asking Secret Manager again
METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
LOOKAHEAD_DAYS = 21
GOOD_WINDOW_MIN = 2.0  # a slot within this many minutes of the best counts as "about as good"
NOW_STALE_MIN = 20  # an older Maps reading is not offered as "leave now"
# Decision tree (decide): minutes that make a difference worth acting on
SAVE_MIN = 5.0  # waiting must save at least this much against leaving now
WORSE_MIN = 5.0  # traffic "gets worse" once a slot is this much slower than now
REOPEN_MIN = 2.0  # after the peak, back to "about now" within this much
APPROACH_MIN = int(os.environ.get("APPROACH_MIN", "30"))  # drive from home to the start of the measured route

DIRECTIONS = (
    # key in forecast-api, key in traffic-24h.json, words
    {"api": "SG_TO_MY", "series": "mandai_to_shell_jb", "label": "To Johor (Singapore to Johor Bahru)"},
    {"api": "MY_TO_SG", "series": "jb_to_woodlands", "label": "To Singapore (Johor Bahru to Singapore)"},
)

_cache = {}
_lock = threading.Lock()
_calendar = None
_key = {"value": None, "at": -1e9, "error": None}


# --- inputs --------------------------------------------------------------------------------


def fetch_traffic():
    res = requests.get(TRAFFIC_API, timeout=20)
    res.raise_for_status()
    return res.json()


def load_calendar(path=None):
    """{date: row} with 0/1 flags as ints, plus the years whose school-holiday columns are filled."""
    rows = {}
    with open(path or CALENDAR_PATH, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            day = datetime.date.fromisoformat(raw["date"])
            row = {k: (int(v) if k not in ("date", "sg_names", "my_names") else v) for k, v in raw.items()}
            row["date"] = day
            rows[day] = row
    school_years = {
        country: sorted({d.year for d, r in rows.items() if r[f"{country}_school"]}) for country in ("sg", "my")
    }
    return {"rows": rows, "school_years": school_years}


def calendar():
    global _calendar
    if _calendar is None:
        _calendar = load_calendar()
    return _calendar


# --- facts ---------------------------------------------------------------------------------


def _parse(ts):
    return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _clock(dt):
    return dt.astimezone(SGT).strftime("%H:%M")


def latest_now(traffic, series, now=None):
    """Newest Maps travel time on the feed, or None when there is none or it is over NOW_STALE_MIN old."""
    points = [p for p in (traffic.get("series", {}).get(series, {}).get("points") or []) if p and p[1] is not None]
    if not points:
        return None
    ts, sec = points[-1][0], points[-1][1]
    at = _parse(ts)
    if at.tzinfo is None:
        at = at.replace(tzinfo=SGT)
    if now is not None and (now - at).total_seconds() > NOW_STALE_MIN * 60:
        return None
    return {"at": _clock(at), "at_iso": at.isoformat(), "minutes": round(sec / 60.0, 1)}


def decide(slots):
    """What to tell a driver, as one of four scenarios. ``slots`` are leave-home times; ``slots[0]`` is now.

    - ``wait``: a later window saves at least SAVE_MIN against now. Window: the run of slots within
      GOOD_WINDOW_MIN of the best one.
    - ``steady``: nothing later saves SAVE_MIN and nothing gets WORSE_MIN worse: leave any time.
    - ``leave_now``: nothing later saves SAVE_MIN, but traffic gets WORSE_MIN worse later. Window:
      now until the last slot before it does.
    - ``leave_now_or_later``: as ``leave_now``, and after the peak it comes back to within REOPEN_MIN
      of now before the 5 hours end (``reopen_iso``).
    """
    iso = lambda s: s["at"].isoformat()  # noqa: E731
    n = slots[0]["min"]
    best_i = min(range(len(slots)), key=lambda i: slots[i]["min"])
    best = slots[best_i]["min"]
    if n - best >= SAVE_MIN:
        lo = hi = best_i
        while lo > 1 and slots[lo - 1]["min"] <= best + GOOD_WINDOW_MIN:
            lo -= 1
        while hi < len(slots) - 1 and slots[hi + 1]["min"] <= best + GOOD_WINDOW_MIN:
            hi += 1
        return {"scenario": "wait", "window_start_iso": iso(slots[lo]), "window_end_iso": iso(slots[hi]),
                "crossing_min": round(best, 1), "saving_min": round(n - best, 1)}
    bad_i = next((i for i, s in enumerate(slots) if s["min"] >= n + WORSE_MIN), None)
    if bad_i is None:
        return {"scenario": "steady", "window_start_iso": iso(slots[0]), "window_end_iso": iso(slots[-1]),
                "crossing_min": round(n, 1), "range_min": [round(min(s["min"] for s in slots), 1),
                                                           round(max(s["min"] for s in slots), 1)]}
    peak_i = max(range(bad_i, len(slots)), key=lambda i: slots[i]["min"])
    out = {"scenario": "leave_now", "window_start_iso": iso(slots[0]), "window_end_iso": iso(slots[max(bad_i - 1, 0)]),
           "crossing_min": round(n, 1), "worse_from_iso": iso(slots[bad_i]),
           "peak": {"at_iso": iso(slots[peak_i]), "crossing_min": round(slots[peak_i]["min"], 1)}}
    reopen = next((s for s in slots[peak_i + 1:] if s["min"] <= n + REOPEN_MIN), None)
    if reopen is not None:
        out.update(scenario="leave_now_or_later", reopen_iso=iso(reopen), reopen_crossing_min=round(reopen["min"], 1))
    return out


def direction_facts(curve_dir, now_point):
    """When to leave home, from now to about 4.5 hours ahead, and the comparison with the typical day.

    A forecast slot is the Maps time for the measured route starting at ``forecast_for``. A driver
    needs about APPROACH_MIN to get there, so every time here is a leave-home time: the slot's start
    minus APPROACH_MIN. The first slot is therefore "leave now". The latest Maps reading (``now``) is
    kept for context only: leaving home now reaches the route after it.
    """
    approach = datetime.timedelta(minutes=APPROACH_MIN)
    pts = [
        {"at": _parse(p["forecast_for"]) - approach, "min": float(p["forecast_min"]),
         "typical": None if p.get("baseline_min") is None else float(p["baseline_min"])}
        for p in (curve_dir or {}).get("points", [])
        if p.get("forecast_min") is not None and p.get("model") != "profile"
    ]
    if not pts:
        return None
    best_i = min(range(len(pts)), key=lambda i: pts[i]["min"])
    worst_i = max(range(len(pts)), key=lambda i: pts[i]["min"])
    best = pts[best_i]["min"]
    lo = hi = best_i
    while lo > 0 and pts[lo - 1]["min"] <= best + GOOD_WINDOW_MIN:
        lo -= 1
    while hi < len(pts) - 1 and pts[hi + 1]["min"] <= best + GOOD_WINDOW_MIN:
        hi += 1
    diffs = [p["min"] - p["typical"] for p in pts if p["typical"] is not None]
    gaps = [p for p in pts if p["typical"] is not None]
    gap = max(gaps, key=lambda p: abs(p["min"] - p["typical"])) if gaps else None
    facts = {
        "times_are": f"when to leave home, assuming about {APPROACH_MIN} min to reach the Causeway route",
        "forecast": [{"leave_at": _clock(p["at"]), "at_iso": p["at"].isoformat(), "crossing_min": round(p["min"], 1),
                      "typical_min": None if p["typical"] is None else round(p["typical"], 1)} for p in pts],
        "best": {"leave_at": _clock(pts[best_i]["at"]), "crossing_min": round(best, 1),
                 "window": f"{_clock(pts[lo]['at'])}-{_clock(pts[hi]['at'])}",
                 "window_start_iso": pts[lo]["at"].isoformat(), "window_end_iso": pts[hi]["at"].isoformat(),
                 "leave_now": lo == 0},
        "worst": {"leave_at": _clock(pts[worst_i]["at"]), "crossing_min": round(pts[worst_i]["min"], 1)},
        "vs_typical_avg_min": round(sum(diffs) / len(diffs), 1) if diffs else None,
        # the slot furthest from usual, so the insight can say when the difference is largest
        "biggest_gap_vs_typical": None if gap is None else {
            "leave_at": _clock(gap["at"]), "crossing_min": round(gap["min"], 1),
            "typical_min": round(gap["typical"], 1), "diff_min": round(gap["min"] - gap["typical"], 1)},
        "now": now_point,  # the latest reading on the route itself, for context
        "decision": decide(pts),  # what the card says; see decide()
    }
    if now_point:
        facts["saving_vs_now_min"] = round(now_point["minutes"] - best, 1)
    return facts


def _names(row, country):
    return [n.strip() for n in (row.get(f"{country}_names") or "").split(";") if n.strip()]


def _school_run(rows, day, country):
    """(first, last) day of the school-holiday run that contains ``day``."""
    col = f"{country}_school"
    first = last = day
    while rows.get(first - datetime.timedelta(days=1), {}).get(col):
        first -= datetime.timedelta(days=1)
    while rows.get(last + datetime.timedelta(days=1), {}).get(col):
        last += datetime.timedelta(days=1)
    return first, last


def calendar_facts(today, cal=None):
    """What today is, and what is coming in the next LOOKAHEAD_DAYS, in Singapore and Malaysia."""
    cal = cal or calendar()
    rows, school_years = cal["rows"], cal["school_years"]
    row = rows.get(today)
    out = {"date": today.isoformat(), "weekday": today.strftime("%A"), "in_calendar": row is not None}
    if row is None:
        return out
    out["today"] = {
        "sg_public_holiday": _names(row, "sg") if row["sg_public"] else [],
        "my_public_holiday": _names(row, "my") if row["my_public"] else [],
        "eve_of_sg_public_holiday": bool(row["sg_public_eve"]),
        "eve_of_my_public_holiday": bool(row["my_public_eve"]),
        "sg_long_weekend": bool(row["sg_long_weekend"]),
        "last_day_of_sg_long_weekend": bool(row["sg_long_weekend_last_day"]),
    }
    for country, word in (("sg", "Singapore"), ("my", "Malaysia")):
        if today.year not in school_years[country]:
            out[f"{country}_school_holidays"] = "unknown (no school calendar for this year)"
        elif row[f"{country}_school"]:
            first, last = _school_run(rows, today, country)
            out[f"{country}_school_holidays"] = f"on now, {first.isoformat()} to {last.isoformat()}"
        else:
            nxt = next((today + datetime.timedelta(days=k) for k in range(1, LOOKAHEAD_DAYS + 1)
                        if rows.get(today + datetime.timedelta(days=k), {}).get(f"{country}_school")), None)
            out[f"{country}_school_holidays"] = (
                f"start {nxt.isoformat()} (in {(nxt - today).days} days)" if nxt else f"none in the next {LOOKAHEAD_DAYS} days"
            )
    upcoming = []
    for k in range(1, LOOKAHEAD_DAYS + 1):
        day = today + datetime.timedelta(days=k)
        r = rows.get(day)
        if r is None:
            break
        for country, word in (("sg", "Singapore"), ("my", "Malaysia")):
            if r[f"{country}_public"]:
                upcoming.append({"date": day.isoformat(), "in_days": k, "country": word, "names": _names(r, country)})
    out["upcoming_public_holidays"] = upcoming
    return out


def build_facts(curve, traffic, now):
    dirs = {}
    for d in DIRECTIONS:
        now_point = latest_now(traffic, d["series"], now) if traffic else None
        f = direction_facts((curve.get("directions") or {}).get(d["api"]), now_point)
        if f:
            dirs[d["api"]] = dict(f, label=d["label"])
    return {
        "generated_at_sgt": now.astimezone(SGT).strftime("%Y-%m-%d %H:%M"),
        "forecast_origin": {k: (curve.get("directions") or {}).get(k, {}).get("origin_ts") for k in dirs},
        "directions": dirs,
        "calendar": calendar_facts(now.astimezone(SGT).date()),
    }


# --- LLM ---------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a seasoned daily commuter across the Woodlands Causeway between Singapore and Johor Bahru. The app already shows, for each direction, when to leave and the expected trip time on a clock. Your job is the "why": explain in ONE line why traffic is different from usual today.

Facts (user JSON):
- Every "leave_at" and decision time is when to LEAVE HOME (about 30 min before reaching the Causeway route; "times_are"). "crossing_min": forecast Google Maps time for the Causeway route for that departure. "typical_min": the usual time. Do not mention the 30 minutes.
- "vs_typical_avg_min": average over the next 5 hours; > 0 means slower than usual. "biggest_gap_vs_typical": the slot furthest from usual.
- "decision": what the app already tells the driver (wait for a better window, leave now before it gets worse, leave any time). Never contradict it.
- "calendar": weekday, Singapore and Malaysia public holidays, eves, Singapore long weekends, school holidays, upcoming public holidays.

General Causeway patterns you may use as likely causes (they are well known, not facts about today):
- Friday evenings, eves of public holidays and the start of school holidays or long weekends: heavy towards Johor.
- Sunday evenings and the last day of a holiday or long weekend: heavy towards Singapore.
- Weekday mornings: Malaysian commuters heavy towards Singapore; weekday evenings: heavy back towards Johor.
- A Malaysian public holiday that is not a Singapore one often means lighter Singapore-bound commuter traffic.

How to write the insight:
- Pick the direction that differs most from usual (use vs_typical_avg_min and biggest_gap_vs_typical). If both are within about 2 minutes of usual, say traffic is about normal for this time.
- Say how much and when (numbers and times from the facts only), then the most likely cause from the calendar and the patterns above.
- Causes are guesses: use "likely" or "probably". Never invent events, accidents, weather, roadworks or checkpoint conditions. If nothing in the facts explains it, say the cause is unclear.
- If school holidays are "unknown", say nothing about school holidays.
- Times are SGT, written like 3:40 PM. Plain words, no greetings, no emojis.

Reply with JSON only:
{"insight": "ONE sentence, at most 22 words: which direction, how much vs usual, and the likely why",
 "tips": ["at most 2 tips, each at most 12 words"]}"""

ADVICE_KEYS = ("insight", "tips")


def _read_secret():
    """The latest version of the OpenAI secret, read with the service account's metadata token."""
    import base64

    tok = requests.get(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"}, timeout=5)
    tok.raise_for_status()
    url = (f"https://secretmanager.googleapis.com/v1/projects/{SECRET_PROJECT}/secrets/"
           f"{SECRET_NAME}/versions/latest:access")
    res = requests.get(url, headers={"Authorization": f"Bearer {tok.json()['access_token']}"}, timeout=10)
    if res.status_code != 200:
        raise RuntimeError(f"Secret Manager returned {res.status_code} for {SECRET_NAME}")
    return base64.b64decode(res.json()["payload"]["data"]).decode("utf-8").strip()


def openai_key():
    """OPENAI_API_KEY from the environment (local runs), else Secret Manager, cached in memory."""
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    age = time.monotonic() - _key["at"]
    if _key["value"] and age < KEY_TTL_S:
        return _key["value"]
    if _key["error"] and age < KEY_RETRY_S:
        raise RuntimeError(_key["error"])
    try:
        _key.update(value=_read_secret(), at=time.monotonic(), error=None)
    except Exception as exc:
        _key.update(at=time.monotonic(), error=f"OpenAI key unavailable: {exc}")
        if not _key["value"]:
            raise RuntimeError(_key["error"]) from exc
        logger.warning("keeping the cached OpenAI key: %s", exc)
    return _key["value"]


def ask_llm(facts):
    key = openai_key()
    res = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "temperature": 0.4,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
            ],
        },
        timeout=OPENAI_TIMEOUT_S,
    )
    if res.status_code != 200:
        raise RuntimeError(f"OpenAI returned {res.status_code}: {res.text[:200]}")
    return parse_advice(res.json()["choices"][0]["message"]["content"])


def parse_advice(text):
    data = json.loads(text)
    advice = {k: data.get(k) for k in ADVICE_KEYS}
    if not isinstance(advice["insight"], str) or not advice["insight"].strip():
        raise ValueError("advice insight missing")
    advice["insight"] = " ".join(advice["insight"].split())[:200]  # one line
    tips = advice["tips"] if isinstance(advice["tips"], list) else []
    advice["tips"] = [str(t).strip()[:140] for t in tips if str(t).strip()][:2]
    return advice


# --- entry point ----------------------------------------------------------------------------


def advise(curve, now=None):
    """The advice payload for this curve (``main.cached_curve`` output), cached per forecast origin."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    origins = tuple(sorted(((k, (v or {}).get("origin_ts")) for k, v in (curve.get("directions") or {}).items()),
                           key=lambda kv: kv[0]))
    key = (origins, now.astimezone(SGT).date().isoformat())
    with _lock:  # one OpenAI call per key, even when several visitors arrive together
        hit = _cache.get(key)
        if hit:
            ttl = CACHE_TTL_S if hit[1]["advice"] is not None else FAILED_RETRY_S
            if time.monotonic() - hit[0] < ttl:
                return hit[1]
        try:
            traffic = fetch_traffic()
        except Exception:  # "now" is optional; the forecast is not
            logger.exception("traffic-24h.json read failed")
            traffic = None
        facts = build_facts(curve, traffic, now)
        payload = {"kind": "commute advice", "facts": facts, "advice": None, "llm_model": OPENAI_MODEL,
                   "note": "Times and colours come from the SwiftBorder forecast. The insight and tips are "
                           "AI-written; causes are educated guesses. Estimates only."}
        if facts["directions"]:
            try:
                payload["advice"] = ask_llm(facts)
            except Exception as exc:
                logger.exception("LLM advice failed")
                payload["advice_error"] = ("The OpenAI key is not available." if "key unavailable" in str(exc)
                                           else "Advice is unavailable right now.")
        _cache.clear()  # only the newest origin is asked for again
        _cache[key] = (time.monotonic(), payload)  # a failure is retried after FAILED_RETRY_S, not per visitor
        return payload
