"""HTTP read of the Woodlands 30-minute Maps-duration forecast, served from local models (ADR 0004).

Not the camera service. Cloud Build trigger 76bbca35 deploys camdetect/ only; forecastapi/cloudbuild.yaml
deploys Cloud Run forecast-api from main.

Callable ids:

- ``served`` (``production``): the per-direction selection in ``SERVED_SELECTION``.
- ``persistence`` and the six local models in ``local_models.LOCAL_MODELS`` (``local``). Each is fitted
  in this process from ``v_training_set`` rows, once per SGT day for the daily models, and scores the
  latest ``v_training_set`` row per direction. The fit uses the same rows, settings and seeds as the
  harness fold for that day, so a served forecast equals what ``eval/joined.py`` scored.

Other ids are listed from the eval harness and are not callable (``artifact`` / ``code-only``).

Exploratory horizons (``horizon_min`` 60 .. 1440, docs/horizon-study.md): ``served``, ``persistence``,
``xgb[maps]``, ``xgb[maps+prof]`` and the ``profile`` baseline take them; responses say
``status: exploratory`` and carry the study's MAE. Every forecast response also carries the calendar
profile as ``baseline`` (labelled: not a forecast), and ``?baseline=profile&hours=N`` returns its curve.

Manual selection: ``model`` (default ``served``) picks one id for every requested direction;
``model_sg_to_my`` / ``model_my_to_sg`` override it for one direction. A request with an override
answers ``model: custom`` and names the model used in each direction.
BigQuery ML (``lin_h30``, ``xgb_h30``, ``v_forecast_recent``) is no longer called.
"""

import datetime
import json
import logging
import os
import threading
import time

import functions_framework
import pandas as pd

import local_models as lm

logger = logging.getLogger("forecastapi")

BQ_PROJECT = os.environ.get("BQ_PROJECT", "swiftborder")
BQ_LOCATION = "US"
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
COMMIT_SHA = os.environ.get("COMMIT_SHA", "")
CACHE_TTL_SECONDS = 300
DIRECTIONS = ("SG_TO_MY", "MY_TO_SG")
SOURCES = frozenset({"local model", "bigquery view", "fixture"})
CALLABLE_STATES = frozenset({"production", "local", "exploratory", "baseline"})

# What ``served`` returns per direction. Interim value (until the frozen run, docs/roadmap.md): the
# 2026-09-12 registry choice rebuilt from local replicas (lin_h30 for SG_TO_MY, persistence for
# MY_TO_SG). After the frozen run it is set by the ADR 0004 selection rule.
SERVED_SELECTION = {"SG_TO_MY": "lin_bq[frozen]", "MY_TO_SG": "persistence"}
SELECTION_ID = "registry-2026-09-12-local-replica"

# Exploratory horizons (docs/horizon-study.md). ``served`` at a horizon other than 30 minutes is the
# lower-MAE model of eval/horizon_study.py on 13-30 Sep (both directions): xgb[maps] up to 1 h,
# xgb[maps+prof] from 1.5 h. Chosen on the study window, not confirmed on Run B: every such response
# says ``status: exploratory``. Beyond about 4 h no model beat the profile baseline in the study.
EXPLORATORY_SELECTION = {h: ("xgb[maps]" if h <= 60 else "xgb[maps+prof]") for h in (60, 90, 120, 180, 240, 360, 480, 720, 1080, 1440)}
EXPLORATORY_SELECTION_ID = "horizon-study-2026-10-04"
EXPLORATORY_MODELS = frozenset({"xgb[maps+prof]"})
BASELINE_LABEL = "Baseline: typical for this day and time (calendar profile), not a forecast"
CURVE_MAX_HOURS = 24
STUDY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horizon_study.json")


def _load_study():
    try:
        with open(STUDY_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


STUDY = _load_study()
PERSISTENCE_VERSION = "latest-bin"
OVERRIDE_PARAMS = {"SG_TO_MY": "model_sg_to_my", "MY_TO_SG": "model_my_to_sg"}

# Availability contract (docs/runbooks/forecast-api.md). A 10-minute bin [t, t+10) is an origin only
# once it has closed plus INGEST_GRACE (the harness treats a bin as known at t+10). The forecast is the
# mean Maps duration over [t+30, t+40). An origin is served only while that target bin has not started,
# so the newest closed bin, or the one before it after a missed fetch, can be served; anything older
# is unavailable (503), never a 200 with an expired forecast.
BIN = datetime.timedelta(minutes=10)
TARGET_SHIFT = datetime.timedelta(minutes=30)
INGEST_GRACE = datetime.timedelta(seconds=int(os.environ.get("INGEST_GRACE_SECONDS", "60")))
LATEST_WINDOW = datetime.timedelta(hours=3)  # enough bins for the 60-minute lags and the gap rule

BINS_VIEW = "`swiftborder.traffic_prediction.v_bins_10min`"
_BIN_COLUMNS = "route_id, direction, bin_ts, dur_min, congestion_ratio, speed_kmh"

# Features are built in Python with the harness's time-based lags (lm.features_from_bins), not read
# from v_training_set, whose positional LAG/LEAD shift after a skipped bin.
_LATEST_SQL = f"""
SELECT {_BIN_COLUMNS}
FROM {BINS_VIEW}
WHERE direction IN UNNEST(@directions)
  AND bin_ts >= TIMESTAMP_SUB(@now, INTERVAL {int(LATEST_WINDOW.total_seconds())} SECOND)
  AND TIMESTAMP_ADD(bin_ts, INTERVAL @closed_after_sec SECOND) <= @now
"""

_TRAINING_SQL = f"""
SELECT {_BIN_COLUMNS}
FROM {BINS_VIEW}
WHERE TIMESTAMP_ADD(bin_ts, INTERVAL 10 MINUTE) <= @labels_before
"""

REPORT_RUN_ID = "20261003T035807Z_offline-bqml-joined-deep-fuzzy-ensemble"
NOT_DEPLOYED_VERSION = "not-deployed"
BOTH = ("SG_TO_MY", "MY_TO_SG")
MY_ONLY = ("MY_TO_SG",)
_JOINED_SCORED = ("maps", "maps+weather", "maps+camfc", "maps+mpfc", "maps+mpfc+camfc")
_JOINED_CODE_ONLY = ("maps+weather+camera",)


def _spec(model_id, family, version, horizon_min, directions, deploy_state, eval_id=None, horizons=None):
    if horizon_min not in (30, 60):
        raise RuntimeError("horizon")
    return {
        "id": model_id,
        "family": family,
        "default_version": version,
        "horizon_min": horizon_min,
        "horizons": frozenset(horizons or {horizon_min}),
        "directions": directions,
        "deploy_state": deploy_state,
        "callable": deploy_state in CALLABLE_STATES,
        "eval_id": eval_id,
    }


def _build_models():
    all_h = lm.HORIZONS
    rows = [
        _spec("served", "selection", SELECTION_ID, 30, BOTH, "production", horizons=all_h),
        _spec("persistence", "baseline", PERSISTENCE_VERSION, 30, BOTH, "local", "persistence", horizons=all_h),
        _spec("profile", "baseline", None, 30, BOTH, "baseline", "horizon_study:profile", horizons=all_h),
    ]
    for model_id, info in lm.LOCAL_MODELS.items():
        family = "bqml-replica" if "_bq" in model_id else "sklearn"
        state = "exploratory" if model_id in EXPLORATORY_MODELS else "local"
        hs = all_h if model_id in lm.MULTI_HORIZON_MODELS else None
        rows.append(_spec(model_id, family, None, 30, BOTH, state, info["eval_id"], horizons=hs))
    local = set(lm.LOCAL_MODELS)
    for feature_set, state, version in (
        *((name, "artifact", REPORT_RUN_ID) for name in _JOINED_SCORED),
        *((name, "code-only", NOT_DEPLOYED_VERSION) for name in _JOINED_CODE_ONLY),
    ):
        for model_name, family in (("ridge", "sklearn"), ("xgb", "sklearn"), ("ensemble", "ensemble")):
            model_id = "%s[%s]" % (model_name, feature_set)
            if model_id not in local:
                rows.append(_spec(model_id, family, version, 30, BOTH, state))
    for model_id in ("ensemble_mean", "mean[models]", "stack", "select", "fuzzy stack"):
        rows.append(_spec(model_id, "ensemble", REPORT_RUN_ID, 30, BOTH, "artifact"))
    rows.append(_spec("XGB (sklearn)", "sklearn", REPORT_RUN_ID, 60, MY_ONLY, "artifact"))
    for model_id in ("lstm", "gru", "transformer", "transformer_raw"):
        rows.append(_spec(model_id, "deep", REPORT_RUN_ID, 60, MY_ONLY, "artifact"))
    rows.append(_spec("rules", "fuzzy", REPORT_RUN_ID, 60, BOTH, "artifact"))
    rows.append(_spec("xgb_to_fuzzy", "fuzzy", REPORT_RUN_ID, 60, BOTH, "artifact"))
    for model_id in ("mean[XGB+deep]", "mean[deep]", "stack[XGB+deep]"):
        rows.append(_spec(model_id, "ensemble", REPORT_RUN_ID, 60, MY_ONLY, "artifact"))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate model id")
    if set(SERVED_SELECTION) != set(DIRECTIONS) or not set(SERVED_SELECTION.values()) <= local | {"persistence"}:
        raise RuntimeError("SERVED_SELECTION must name a local model or persistence for each direction")
    if set(EXPLORATORY_SELECTION) != set(lm.HORIZONS) - {30} or not set(EXPLORATORY_SELECTION.values()) <= set(lm.MULTI_HORIZON_MODELS):
        raise RuntimeError("EXPLORATORY_SELECTION must name a multi-horizon model for every horizon but 30")
    return {row["id"]: row for row in rows}


MODELS = _build_models()

_cache = {}
_clock = time.monotonic
_now = lambda: datetime.datetime.now(datetime.timezone.utc)  # noqa: E731
_fit_lock = threading.Lock()
_fitted = {}
_training = {}


def current_version(model_id, horizon=30):
    spec = MODELS[model_id]
    if model_id == "served":
        return SELECTION_ID if horizon == 30 else EXPLORATORY_SELECTION_ID
    if model_id == "persistence":
        return PERSISTENCE_VERSION
    if model_id == "profile":
        return profile_version(_now())
    if model_id in lm.LOCAL_MODELS:
        return lm.version_for(model_id, _now())
    return spec["default_version"]


def profile_version(now):
    return "labels_before=" + lm.serving_day_start(now).tz_convert(lm.SGT).isoformat()


def status_for(model_id, horizon):
    """``evaluated``: a 30-minute model of the report harness; otherwise ``exploratory`` or ``baseline``."""
    if model_id == "profile":
        return "baseline"
    if horizon != 30 or model_id in EXPLORATORY_MODELS:
        return "exploratory"
    return "evaluated"


# --- BigQuery reads (replaced in tests) ----------------------------------------------------


def _bigquery():
    from google.cloud import bigquery

    return bigquery


def fetch_latest(directions, now):
    """Closed ``v_bins_10min`` rows of the last ``LATEST_WINDOW`` (the open bin is never read)."""
    bq = _bigquery()
    client = bq.Client(project=BQ_PROJECT)
    cfg = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter("directions", "STRING", list(directions)),
            bq.ScalarQueryParameter("now", "TIMESTAMP", now),
            bq.ScalarQueryParameter("closed_after_sec", "INT64", int((BIN + INGEST_GRACE).total_seconds())),
        ]
    )
    return client.query(_LATEST_SQL, job_config=cfg, location=BQ_LOCATION).to_dataframe()


def fetch_training(labels_before):
    bq = _bigquery()
    client = bq.Client(project=BQ_PROJECT)
    cfg = bq.QueryJobConfig(query_parameters=[bq.ScalarQueryParameter("labels_before", "TIMESTAMP", labels_before.to_pydatetime())])
    return client.query(_TRAINING_SQL, job_config=cfg, location=BQ_LOCATION).to_dataframe()


# --- fitting --------------------------------------------------------------------------------


def _training_frame(day_start):
    """All labelled rows observed before ``day_start``; one read per SGT day per process."""
    key = day_start.isoformat()
    if key not in _training:
        _training.clear()
        _training[key] = lm.prepare(lm.features_from_bins(fetch_training(day_start)))
    return _training[key]


def fitted_profile():
    """The calendar-profile baseline of the serving day (every closed bin before 00:00 SGT)."""
    version = profile_version(_now())
    key = ("profile", version, 0)
    with _fit_lock:
        hit = _fitted.get(key)
        if hit is not None:
            return hit
        frame = _training_frame(lm.serving_day_start(_now()))
        profile = lm.Profile().fit(frame)
        for k in [k for k in _fitted if k[0] == "profile"]:
            _fitted.pop(k)
        _fitted[key] = profile
        return profile


def fitted_model(model_id, horizon=30):
    version = lm.version_for(model_id, _now())
    key = (model_id, version, int(horizon))
    with _fit_lock:
        hit = _fitted.get(key)
    if hit is not None:
        return hit
    profile = fitted_profile() if lm.LOCAL_MODELS[model_id]["kind"] == "harness_xgb_prof" else None
    with _fit_lock:
        hit = _fitted.get(key)
        if hit is not None:
            return hit
        day_start = lm.serving_day_start(_now())
        rows = lm.LOCAL_MODELS[model_id]["rows"]
        frame = _training_frame(day_start)  # every closed bin before day_start, harness features
        train = lm.training_rows(frame, rows, day_start, horizon)
        if len(train) < 100:
            raise RuntimeError(f"only {len(train)} training rows for {model_id} at {horizon} min")
        if profile is not None:
            train = lm.with_profile(train, profile, horizon)
        model = lm.fit(model_id, train, horizon)
        for k in [k for k in _fitted if k[0] == model_id and k[2] == int(horizon)]:
            _fitted.pop(k)  # yesterday's fit
        _fitted[key] = model
        return model


def servable_origins(bins, now):
    """One origin row per direction that may be served at ``now``, or none for that direction.

    The origin is the newest bin that closed at least ``INGEST_GRACE`` ago. It is dropped when its
    target bin has already started (stale data) or when it lies within an hour of a gap of more than
    25 minutes (``after_gap``: the harness never scores such rows).
    """
    now = pd.Timestamp(now)
    if bins is None or len(bins) == 0:
        return lm.prepare(lm.features_from_bins(_empty_bins()))
    frame = bins.copy()
    frame["bin_ts"] = pd.to_datetime(frame["bin_ts"], utc=True)
    frame = frame[frame["bin_ts"] + BIN + INGEST_GRACE <= now]  # the fetch already does this; keep it local too
    feats = lm.prepare(lm.features_from_bins(frame, unknown_first_gap=True))
    latest = feats.sort_values("bin_ts").groupby("direction").tail(1)
    fresh = latest["bin_ts"] + TARGET_SHIFT > now
    complete = latest["after_gap"] == 0
    for _, row in latest[~(fresh & complete)].iterrows():
        logger.warning(
            "no servable origin for %s: latest closed bin %s (%s)",
            row["direction"], row["bin_ts"].isoformat(), "stale" if not fresh[row.name] else "after a gap",
        )
    return latest[fresh & complete].reset_index(drop=True)


def _empty_bins():
    return pd.DataFrame(columns=["route_id", "direction", "bin_ts", "dur_min", "congestion_ratio", "speed_kmh"])


def resolve(model_id, direction, horizon=30):
    """The concrete model behind an id in one direction (``served`` -> its selection)."""
    if model_id != "served":
        return model_id
    return SERVED_SELECTION[direction] if horizon == 30 else EXPLORATORY_SELECTION[horizon]


def query_forecast(model_id, directions, horizon=30):
    """Forecast rows for one callable id. Tests replace this function."""
    spec = MODELS[model_id]
    if not spec["callable"]:
        raise RuntimeError("model is not deployed")
    return query_selection({d: resolve(model_id, d, horizon) for d in directions}, horizon)


def query_selection(chosen, horizon=30):
    """Forecast rows for a direction -> concrete model mapping, plus the profile baseline.

    Tests replace this function.
    """
    directions = list(chosen)
    now = _now()
    origins = servable_origins(fetch_latest(directions, now), now)
    shift = datetime.timedelta(minutes=int(horizon))
    profile = fitted_profile()
    rows, meta, baseline = [], {}, {}
    for direction in directions:
        sub = origins[origins["direction"] == direction]
        if sub.empty:
            continue
        mid = chosen[direction]
        origin = sub["bin_ts"].iloc[0]
        target_value = float(profile.predict([direction], [origin + shift])[0])
        baseline[direction] = {"forecast_min": round(target_value, 1)}
        if mid == "persistence":
            value = float(sub["y_persistence"].iloc[0])
            meta[mid] = {"eval_id": "persistence", "version": PERSISTENCE_VERSION}
        elif mid == "profile":
            value = target_value
            meta[mid] = {"eval_id": "horizon_study:profile", "version": profile_version(now), "training_rows": profile.n_rows}
        else:
            model = fitted_model(mid, horizon)
            x = lm.with_profile(sub, profile, horizon) if lm.LOCAL_MODELS[mid]["kind"] == "harness_xgb_prof" else sub
            value = float(model.predict(x)[0])
            meta[mid] = {
                "eval_id": lm.LOCAL_MODELS[mid]["eval_id"],
                "version": lm.version_for(mid, _now()),
                "training_rows": model.n_rows,
                "seeds": model.seeds,
            }
        rows.append(
            {
                "direction": direction,
                "forecast_min": value,
                "forecast_for": _iso(origin + shift),
                "origin_ts": _iso(origin),
                "model": mid,
            }
        )
    return {
        "source": "local model",
        "rows": rows,
        "meta": meta,
        "baseline": {"model": "profile", "status": "baseline", "label": BASELINE_LABEL,
                     "version": profile_version(now), "directions": baseline},
    }


def profile_curve(directions, hours):
    """The profile baseline every 10 minutes for the next ``hours``, from the next bin start."""
    now = pd.Timestamp(_now())
    start = now.floor("10min") + pd.Timedelta(minutes=10)
    ts = pd.Series(pd.date_range(start, periods=int(hours) * 6, freq="10min"))
    profile = fitted_profile()
    out = {}
    for d in directions:
        vals = profile.predict([d] * len(ts), ts)
        out[d] = [{"bin_start": _iso(t), "forecast_min": round(float(v), 1)} for t, v in zip(ts, vals)]
    return {
        "model": "profile",
        "status": "baseline",
        "label": BASELINE_LABEL,
        "version": profile_version(now.to_pydatetime()),
        "generated_at": _iso(now),
        "hours": int(hours),
        "note": "Mean Maps duration of each 10-minute bin expected from the time of day and weekday/weekend only; "
                "it ignores current traffic. Use it to compare forecasts against, not as a forecast.",
        "directions": out,
        "study": STUDY and {"status": STUDY.get("status"), "profile_mae_min": {
            str(h["horizon_min"]): h["mae_min"]["profile"] for h in STUDY.get("horizons", [])}},
    }


# --- HTTP -----------------------------------------------------------------------------------


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "3600",
    }


class NoRecentForecast(Exception):
    """The query ran and did not return every requested direction."""


def _error(message, status):
    headers = {**_cors_headers(), "Content-Type": "application/json"}
    return (json.dumps({"error": message}), status, headers)


def _param(body, args, name):
    """Body wins over query string; a missing, null or empty-string body value falls back."""
    value = body.get(name) if isinstance(body, dict) else None
    if value is None or value == "":
        value = args.get(name)
    return value


def _iso(value):
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        if text.endswith("+00:00"):
            return text[:-6] + "Z"
        return text
    return str(value)


def _parse_ts(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("timestamp without zone")
    return ts.tz_convert("UTC").to_pydatetime()


def _with_timing(payload, now):
    """Copy of ``payload`` with ages measured at ``now``; None once any target bin has started."""
    out = dict(payload, directions={})
    for direction, row in payload["directions"].items():
        target = _parse_ts(row["forecast_for"])
        origin = _parse_ts(row["origin_ts"])
        if target <= now or origin + TARGET_SHIFT <= now:
            return None  # target started, or the origin is older than any servable origin
        closed = _parse_ts(row["origin_closed_at"])
        out["directions"][direction] = dict(
            row,
            observation_age_min=round((now - closed).total_seconds() / 60.0, 1),
            lead_min=round((target - now).total_seconds() / 60.0, 1),
        )
    return out


def study_for(horizon, models):
    """The horizon study's MAE at this horizon for the models used, the profile and persistence."""
    if not STUDY:
        return None
    entry = next((h for h in STUDY.get("horizons", []) if h["horizon_min"] == horizon), None)
    if entry is None:
        return None
    names = [m for m in models if m in entry["mae_min"]] + ["profile", "persistence"]
    return {
        "status": STUDY.get("status", "exploratory"),
        "window_sgt": STUDY.get("window_sgt"),
        "mae_min": {m: entry["mae_min"][m] for m in dict.fromkeys(names)},
        "vs_profile": {m: v for m, v in entry.get("vs_profile", {}).items() if m in models},
        "note": STUDY.get("note"),
    }


def list_models():
    """Catalog a caller can read before asking for a forecast. No BigQuery."""
    models = []
    for spec in MODELS.values():
        row = {
            "id": spec["id"],
            "family": spec["family"],
            "version": current_version(spec["id"]),
            "horizon_min": spec["horizon_min"],
            "directions": list(spec["directions"]),
            "deploy_state": spec["deploy_state"],
            "callable": spec["callable"],
        }
        row["horizons_min"] = sorted(spec["horizons"])
        if spec["id"] == "served":
            row["selection"] = dict(SERVED_SELECTION)
            row["selection_by_horizon"] = {str(h): m for h, m in EXPLORATORY_SELECTION.items()}
        models.append(row)
    return {
        "models": models,
        "parameters": {
            "model": "callable id for every requested direction (default served)",
            "model_sg_to_my": "callable id for SG_TO_MY only (overrides model)",
            "model_my_to_sg": "callable id for MY_TO_SG only (overrides model)",
            "direction": "SG_TO_MY, MY_TO_SG or both (default both)",
            "version": "optional; must equal the current version (not allowed with an override)",
            "horizon_min": "30 (evaluated) or %s (exploratory; see list=horizon-study)" % ", ".join(str(h) for h in lm.HORIZONS if h != 30),
            "baseline": "profile, with hours=1..%d: the calendar-profile baseline curve (not a forecast)" % CURVE_MAX_HOURS,
            "list": "models, or horizon-study for the exploratory study results",
        },
    }


def _cache_get(key):
    hit = _cache.get(key)
    if hit is None:
        return None
    expires_at, payload = hit
    if _clock() >= expires_at:
        _cache.pop(key, None)
        return None
    return payload


def _cache_put(key, payload):
    _cache[key] = (_clock() + CACHE_TTL_SECONDS, payload)


def _parse_horizon(raw, spec):
    if raw is None or str(raw).strip() == "":
        horizon = spec["horizon_min"]
    else:
        text = str(raw).strip()
        if not text.isdigit():
            return None
        horizon = int(text)
    if horizon not in spec["horizons"]:
        return None
    return horizon


def _horizon_error(spec):
    allowed = sorted(spec["horizons"])
    if len(allowed) == 1:
        return "horizon_min must be %s" % allowed[0]
    return "horizon_min must be %s" % " or ".join(str(item) for item in allowed)


def _parse_directions(raw):
    if raw is None or str(raw).strip() == "":
        return list(DIRECTIONS)
    token = str(raw).strip()
    if token == "both":
        return list(DIRECTIONS)
    if token in DIRECTIONS:
        return [token]
    return None


def _direction_payload(row):
    direction = row.get("direction")
    if direction not in DIRECTIONS:
        raise ValueError("unexpected direction")
    forecast_min = row.get("forecast_min")
    if isinstance(forecast_min, bool) or forecast_min is None:
        raise ValueError("forecast_min")
    try:
        forecast_min = float(forecast_min)
    except (TypeError, ValueError):
        raise ValueError("forecast_min") from None
    forecast_for = row.get("forecast_for")
    origin_ts = row.get("origin_ts")
    if not forecast_for or not origin_ts:
        raise ValueError("timestamp")
    target = _parse_ts(forecast_for)
    origin = _parse_ts(origin_ts)
    payload = {
        "forecast_min": round(float(forecast_min), 1),
        "forecast_for": _iso(target),
        "forecast_window_end": _iso(target + BIN),
        "origin_ts": _iso(origin),
        "origin_closed_at": _iso(origin + BIN),
    }
    if row.get("model"):
        payload["model"] = str(row["model"])
    return direction, payload


def _assemble(model_id, version_id, horizon, directions, result):
    source = result.get("source") if isinstance(result, dict) else None
    rows = result.get("rows") if isinstance(result, dict) else None
    if source not in SOURCES or not isinstance(rows, list):
        raise ValueError("shape")
    by_direction = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row")
        direction, payload = _direction_payload(row)
        if direction not in directions or direction in by_direction:
            raise ValueError("direction set")
        by_direction[direction] = payload
    if set(by_direction) != set(directions):
        raise NoRecentForecast("missing direction")
    ordered = {direction: by_direction[direction] for direction in directions}
    used = {payload.get("model") for payload in ordered.values()} - {None}
    statuses = {status_for(m, horizon) for m in used} or {status_for(model_id, horizon)}
    out = {
        "model": model_id,
        "version": version_id,
        "horizon_min": horizon,
        # evaluated (30-minute report harness) / exploratory (horizon study) / baseline (profile)
        "status": "exploratory" if "exploratory" in statuses else ("baseline" if statuses == {"baseline"} else "evaluated"),
        "generated_at": _iso(_now()),
        "source": source,
        "label": "maps_duration_in_traffic_min",
        # the label is the mean over the target bin, h to h+10 minutes after the origin bin starts
        "target_offset_min": [horizon, horizon + 10],
        "directions": ordered,
    }
    meta = result.get("meta") if isinstance(result, dict) else None
    if meta:
        out["model_meta"] = meta
    baseline = result.get("baseline") if isinstance(result, dict) else None
    if baseline:
        out["baseline"] = baseline
    study = study_for(horizon, used)
    if study:
        out["study"] = study
    if COMMIT_SHA:
        out["commit"] = COMMIT_SHA
    return out


@functions_framework.http
def forecast(request):
    if request.method == "OPTIONS":
        return ("", 204, _cors_headers())
    if request.method != "GET":
        return _error("method not allowed", 405)

    args = request.args or {}
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}

    headers = {**_cors_headers(), "Content-Type": "application/json", "Cache-Control": "private, max-age=300"}
    list_raw = _param(body, args, "list")
    if list_raw is not None and str(list_raw).strip() != "":
        what = str(list_raw).strip()
        if what == "models":
            return (json.dumps(list_models()), 200, headers)
        if what == "horizon-study":
            if not STUDY:
                return _error("horizon study results are not bundled", 503)
            return (json.dumps(STUDY), 200, headers)
        return _error("list must be models or horizon-study", 400)

    baseline_raw = _param(body, args, "baseline")
    if baseline_raw is not None and str(baseline_raw).strip() != "":
        if str(baseline_raw).strip() != "profile":
            return _error("baseline must be profile", 400)
        hours_raw = _param(body, args, "hours")
        text = "24" if hours_raw is None or str(hours_raw).strip() == "" else str(hours_raw).strip()
        if not text.isdigit() or not 1 <= int(text) <= CURVE_MAX_HOURS:
            return _error("hours must be 1 to %d" % CURVE_MAX_HOURS, 400)
        directions = _parse_directions(_param(body, args, "direction"))
        if directions is None:
            return _error("direction must be SG_TO_MY, MY_TO_SG, or both", 400)
        key = ("baseline-curve", profile_version(_now()), tuple(directions), int(text), pd.Timestamp(_now()).floor("10min").isoformat())
        payload = _cache_get(key)
        if payload is None:
            try:
                payload = profile_curve(directions, int(text))
            except Exception:
                logger.exception("baseline curve failed")
                return _error("Forecast query failed", 502)
            _cache_put(key, payload)
        return (json.dumps(payload), 200, headers)

    model_raw = _param(body, args, "model")
    model_id = "served" if model_raw is None or str(model_raw).strip() == "" else str(model_raw).strip()
    spec = MODELS.get(model_id)
    if spec is None:
        return _error("unknown model", 400)
    if not spec["callable"]:
        return _error("model is not deployed", 400)

    overrides = {}
    for direction, name in OVERRIDE_PARAMS.items():
        raw = _param(body, args, name)
        if raw is None or str(raw).strip() == "":
            continue
        mid = str(raw).strip()
        if mid not in MODELS:
            return _error("unknown model in %s" % name, 400)
        if not MODELS[mid]["callable"]:
            return _error("model in %s is not deployed" % name, 400)
        overrides[direction] = mid

    horizon = _parse_horizon(_param(body, args, "horizon_min"), spec)
    if horizon is None:
        return _error(_horizon_error(spec), 400)
    for direction, mid in overrides.items():
        if horizon not in MODELS[mid]["horizons"]:
            return _error("model in %s has no %d-minute horizon" % (OVERRIDE_PARAMS[direction], horizon), 400)

    version_raw = _param(body, args, "version")
    has_version = version_raw is not None and str(version_raw).strip() != ""

    directions = _parse_directions(_param(body, args, "direction"))
    if directions is None or any(direction not in spec["directions"] for direction in directions):
        return _error("direction must be SG_TO_MY, MY_TO_SG, or both", 400)
    if any(d not in directions for d in overrides):
        return _error("a per-direction model is set for a direction that was not requested", 400)

    if overrides:
        if has_version:
            return _error("version cannot be combined with a per-direction model", 400)
        chosen = {d: resolve(overrides.get(d, model_id), d, horizon) for d in directions}
        model_id = "custom"
        version_id = ",".join("%s=%s:%s" % (d, m, current_version(m, horizon)) for d, m in chosen.items())
    else:
        chosen = None
        version_id = current_version(model_id, horizon)
        if has_version and str(version_raw).strip() != version_id:
            return _error("unknown version", 400)

    cache_key = (model_id, version_id, tuple(directions), horizon)
    cached = _cache_get(cache_key)
    payload = _with_timing(cached, _now()) if cached is not None else None
    if payload is None:
        _cache.pop(cache_key, None)  # absent, or a cached target bin has started
        try:
            result = query_selection(chosen, horizon) if chosen else query_forecast(model_id, directions, horizon)
            fresh = _assemble(model_id, version_id, horizon, directions, result)
        except NoRecentForecast:
            return _error("No recent forecast for the requested direction", 503)
        except Exception:
            logger.exception("forecast query failed")
            return _error("Forecast query failed", 502)
        payload = _with_timing(fresh, _now())
        if payload is None:
            logger.warning("forecast target already started; refusing a stale forecast")
            return _error("No recent forecast for the requested direction", 503)
        _cache_put(cache_key, fresh)

    headers = {**_cors_headers(), "Content-Type": "application/json", "Cache-Control": "private, max-age=300"}
    return (json.dumps(payload), 200, headers)
