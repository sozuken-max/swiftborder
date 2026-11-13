"""commute-advisor: an LLM "commute buddy" on top of the forecast-api 5-hour curve.

A separate Cloud Run service. It never changes forecast-api: it reads the same public URLs the
Hosting page reads (``forecast-api ?curve=forecast&hours=5`` and ``traffic-24h.json``), works out
the facts in code, and asks OpenAI only to phrase them as advice.

Facts come from code, not from the LLM:
- per direction, the best and worst time to leave in the next 5 hours, and the crossing time then;
- how the next 5 hours compare with the typical day (the calendar profile, ``baseline_min``);
- Singapore and Malaysia public holidays, eves, long weekends and school holidays from
  ``data/holidays_sg_my.csv``. School holidays are only stated for years the file covers.

The page sends no input. Every visitor in the same forecast origin gets the same cached answer, so
the OpenAI cost is about one call per forecast origin (10 minutes) per instance. If OpenAI fails,
the response still carries the facts and ``advice`` is null.
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

import functions_framework
import requests

logger = logging.getLogger(__name__)

SGT = ZoneInfo("Asia/Singapore")
FORECAST_API = os.environ.get("FORECAST_API", "https://forecast-api-1095552466513.asia-southeast1.run.app/")
TRAFFIC_API = os.environ.get("TRAFFIC_API", "https://storage.googleapis.com/swiftborder-public/traffic-24h.json")
OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")  # GET only, no input, cached: previews may call it
FORECAST_HOURS = 5
FORECAST_TIMEOUT_S = float(os.environ.get("FORECAST_TIMEOUT_SECONDS", "150"))  # forecast-api may be cold
OPENAI_TIMEOUT_S = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "40"))
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_SECONDS", "600"))
RECHECK_S = 60  # how long an answer is reused without asking forecast-api for a newer origin
CALENDAR_PATH = Path(os.environ.get("CALENDAR_PATH", Path(__file__).parent / "data" / "holidays_sg_my.csv"))
LOOKAHEAD_DAYS = 21
GOOD_WINDOW_MIN = 2.0  # a slot within this many minutes of the best counts as "about as good"
NOW_STALE_MIN = 20  # an older Maps reading is not offered as "leave now"

DIRECTIONS = (
    # key in forecast-api, key in traffic-24h.json, words
    {"api": "SG_TO_MY", "series": "mandai_to_shell_jb", "label": "To Johor (Singapore to Johor Bahru)"},
    {"api": "MY_TO_SG", "series": "jb_to_woodlands", "label": "To Singapore (Johor Bahru to Singapore)"},
)

_cache = {}
_lock = threading.Lock()
_calendar = None


class UpstreamError(Exception):
    """forecast-api or the travel-time file could not be read."""


# --- inputs --------------------------------------------------------------------------------


def fetch_curve():
    res = requests.get(FORECAST_API, params={"curve": "forecast", "hours": FORECAST_HOURS}, timeout=FORECAST_TIMEOUT_S)
    if res.status_code != 200:
        raise UpstreamError(f"forecast-api returned {res.status_code}")
    return res.json()


def fetch_traffic():
    res = requests.get(TRAFFIC_API, timeout=20)
    if res.status_code != 200:
        raise UpstreamError(f"traffic-24h.json returned {res.status_code}")
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


def direction_facts(curve_dir, now_point):
    """Best and worst time to leave, from now to 5 hours ahead, and the comparison with the typical day.

    Leaving now (the latest Maps reading) is a candidate alongside the forecast slots, so the advice
    can be "leave now". The best window is the contiguous run of candidates within GOOD_WINDOW_MIN
    of the best one.
    """
    pts = [
        {"at": _parse(p["forecast_for"]), "min": float(p["forecast_min"]),
         "typical": None if p.get("baseline_min") is None else float(p["baseline_min"]), "now": False}
        for p in (curve_dir or {}).get("points", [])
        if p.get("forecast_min") is not None and p.get("model") != "profile"
    ]
    if not pts:
        return None
    slots = ([{"at": _parse(now_point["at_iso"]), "min": now_point["minutes"], "typical": None, "now": True}]
             if now_point else []) + pts
    best_i = min(range(len(slots)), key=lambda i: slots[i]["min"])
    worst_i = max(range(len(slots)), key=lambda i: slots[i]["min"])
    best = slots[best_i]["min"]
    lo = hi = best_i
    while lo > 0 and slots[lo - 1]["min"] <= best + GOOD_WINDOW_MIN:
        lo -= 1
    while hi < len(slots) - 1 and slots[hi + 1]["min"] <= best + GOOD_WINDOW_MIN:
        hi += 1
    diffs = [p["min"] - p["typical"] for p in pts if p["typical"] is not None]
    facts = {
        "forecast": [{"leave_at": _clock(p["at"]), "at_iso": p["at"].isoformat(), "crossing_min": round(p["min"], 1),
                      "typical_min": None if p["typical"] is None else round(p["typical"], 1)} for p in pts],
        "best": {"leave_at": _clock(slots[best_i]["at"]), "crossing_min": round(best, 1),
                 "window": f"{_clock(slots[lo]['at'])}-{_clock(slots[hi]['at'])}",
                 "window_start_iso": slots[lo]["at"].isoformat(), "window_end_iso": slots[hi]["at"].isoformat(),
                 "leave_now": bool(slots[lo]["now"])},
        "worst": {"leave_at": _clock(slots[worst_i]["at"]), "crossing_min": round(slots[worst_i]["min"], 1)},
        "vs_typical_avg_min": round(sum(diffs) / len(diffs), 1) if diffs else None,
        "now": now_point,
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

SYSTEM_PROMPT = """You are a seasoned daily commuter across the Woodlands Causeway between Singapore and Johor Bahru. The app already shows, for each direction, when to leave and the expected trip time on a clock. You add a short human touch.

Rules:
- Use ONLY the facts in the user's JSON. Never invent times, minutes, holidays, events, weather, accidents or checkpoint conditions.
- Every time and number you mention must appear in the facts. Times are SGT; write them like 3:40 PM.
- "crossing_min" is the forecast Google Maps driving time if you leave at "leave_at"; "typical_min" is the usual time then. "vs_typical_avg_min" > 0 means busier than usual over the next 5 hours.
- If school holidays are "unknown", say nothing about school holidays. Mention holidays, eves or long weekends only when they are in the facts.
- Plain, simple words. No hedging, no greetings, no emojis.

Reply with JSON only:
{"headline": "at most 10 words: the one thing to know right now",
 "tips": ["at most 2 tips, each at most 12 words"]}"""

ADVICE_KEYS = ("headline", "tips")


def ask_llm(facts):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
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
    if not isinstance(advice["headline"], str) or not advice["headline"].strip():
        raise ValueError("advice headline missing")
    advice["headline"] = advice["headline"].strip()[:120]
    tips = advice["tips"] if isinstance(advice["tips"], list) else []
    advice["tips"] = [str(t).strip()[:140] for t in tips if str(t).strip()][:2]
    return advice


# --- HTTP ----------------------------------------------------------------------------------


def _headers(max_age=0):
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Max-Age": "3600",
        "Content-Type": "application/json",
        "Cache-Control": f"public, max-age={max_age}" if max_age > 0 else "no-store",
    }


def _error(message, status):
    return (json.dumps({"error": message}), status, _headers())


def advise(now=None):
    """The advice payload for the current forecast origin, from the cache when it is fresh."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    with _lock:
        recent = _cache.get("recent")
        if recent and time.monotonic() - recent[0] < RECHECK_S:
            return recent[1]  # a visitor in the last minute: skip the forecast-api round trip
    curve = fetch_curve()
    try:
        traffic = fetch_traffic()
    except Exception:  # "now" is optional; the forecast is not
        logger.exception("traffic-24h.json read failed")
        traffic = None
    origins = tuple(sorted(((k, (v or {}).get("origin_ts")) for k, v in (curve.get("directions") or {}).items()),
                           key=lambda kv: kv[0]))
    key = (origins, now.astimezone(SGT).date().isoformat())
    with _lock:  # one OpenAI call per key, even when several visitors arrive together
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
            _cache["recent"] = (time.monotonic(), hit[1])
            return hit[1]
        facts = build_facts(curve, traffic, now)
        if not facts["directions"]:
            raise UpstreamError("forecast-api returned no forecast points")
        payload = {"kind": "commute advice", "facts": facts, "advice": None, "llm_model": OPENAI_MODEL,
                   "note": "AI-generated from the SwiftBorder forecast and a holiday calendar. The forecast is a model "
                           "estimate; check live conditions before you travel."}
        try:
            payload["advice"] = ask_llm(facts)
        except Exception as exc:
            logger.exception("LLM advice failed")
            payload["advice_error"] = "Advice is unavailable right now." if "OPENAI_API_KEY" not in str(exc) else str(exc)
        _cache.clear()  # only the newest origin is ever asked for again
        _cache["recent"] = (time.monotonic(), payload)  # a failed LLM call is retried after a minute, not per visitor
        if payload["advice"] is not None:
            _cache[key] = _cache["recent"]
        return payload


@functions_framework.http
def commute_advice(request):
    if request.method == "OPTIONS":
        return ("", 204, _headers())
    if request.method != "GET":
        return _error("method not allowed", 405)
    try:
        payload = advise()
    except UpstreamError as exc:
        return _error(str(exc), 503)
    except requests.RequestException:
        logger.exception("upstream request failed")
        return _error("forecast-api did not answer", 503)
    except Exception:
        logger.exception("advice failed")
        return _error("Advice failed on the server", 502)
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False), 200, _headers(60 if payload["advice"] else 0))
