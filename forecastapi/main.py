"""HTTP read of the Woodlands 30-minute Maps-duration forecast.

Not the camera service. Cloud Build trigger 76bbca35 deploys camdetect/ only.
forecastapi/cloudbuild.yaml deploys Cloud Run forecast-api on main. Until that
build runs, this module is not deployed.

Model ids and version strings are an allow-list. Served BigQuery rows are
refreshed from the read-only queries in docs/runbooks/forecast-api.md
(2026-10-03). Other ids are the eval harness names. A forecast is returned
only for a callable deploy_state: ``production`` (model id ``served``, the
registry mix that v_forecast_recent publishes) or ``api`` (a single model
queried directly; not what the live system serves). BigQuery has no separate model-version
objects for lin_h30 and xgb_h30. Their version string is the model resource
creationTime. persistence uses the registry decided_on date.
The SQL below does not add a VERSION clause: the live view calls the model
with none, and no other version was verified.
"""

import datetime
import json
import logging
import os
import time

import functions_framework
from google.cloud import bigquery

logger = logging.getLogger("forecastapi")

BQ_PROJECT = os.environ.get("BQ_PROJECT", "swiftborder")
BQ_LOCATION = "US"
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
CACHE_TTL_SECONDS = 300
DIRECTIONS = ("SG_TO_MY", "MY_TO_SG")
SOURCES = frozenset({"bigquery view", "ML.PREDICT", "fixture"})

# creationTime from `bq show --model` on 2026-10-03. One training run each.
LIN_H30_VERSION = "2026-09-12T06:15:33.163Z"
XGB_H30_VERSION = "2026-09-12T06:19:18.783Z"
# model_registry.decided_on. persistence is not a BigQuery ML model.
PERSISTENCE_VERSION = "2026-09-12"
# model_registry.decided_on for the per-direction mix that v_forecast_recent serves.
REGISTRY_VERSION = "2026-09-12"
# deploy_state values that return a forecast. ``production`` is what the live view serves;
# ``api`` is a direct query of one model, which the registry may not serve in that direction.
CALLABLE_STATES = frozenset({"production", "api"})

_BASE_FILTER = """
  lag_60 IS NOT NULL
  AND bin_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
  AND direction IN UNNEST(@directions)
"""

_LIN_SQL = f"""
WITH base AS (
  SELECT *
  FROM `swiftborder.traffic_prediction.v_training_set`
  WHERE {_BASE_FILTER}
)
SELECT
  direction,
  bin_ts AS origin_ts,
  TIMESTAMP_ADD(bin_ts, INTERVAL 30 MINUTE) AS forecast_for_ts,
  ROUND(predicted_y_30, 1) AS forecast_min
FROM ML.PREDICT(MODEL `swiftborder.traffic_prediction.lin_h30`, TABLE base)
QUALIFY ROW_NUMBER() OVER (PARTITION BY direction ORDER BY bin_ts DESC) = 1
"""

_XGB_SQL = f"""
WITH base AS (
  SELECT *
  FROM `swiftborder.traffic_prediction.v_training_set`
  WHERE {_BASE_FILTER}
)
SELECT
  direction,
  bin_ts AS origin_ts,
  TIMESTAMP_ADD(bin_ts, INTERVAL 30 MINUTE) AS forecast_for_ts,
  ROUND(predicted_y_30, 1) AS forecast_min
FROM ML.PREDICT(MODEL `swiftborder.traffic_prediction.xgb_h30`, TABLE base)
QUALIFY ROW_NUMBER() OVER (PARTITION BY direction ORDER BY bin_ts DESC) = 1
"""

_PERSISTENCE_SQL = f"""
SELECT
  direction,
  bin_ts AS origin_ts,
  TIMESTAMP_ADD(bin_ts, INTERVAL 30 MINUTE) AS forecast_for_ts,
  ROUND(y_persistence, 1) AS forecast_min
FROM `swiftborder.traffic_prediction.v_training_set`
WHERE {_BASE_FILTER}
QUALIFY ROW_NUMBER() OVER (PARTITION BY direction ORDER BY bin_ts DESC) = 1
"""

_SERVED_SQL = """
SELECT
  direction,
  bin_ts AS origin_ts,
  forecast_for_ts,
  forecast_30min_min AS forecast_min
FROM `swiftborder.traffic_prediction.v_forecast_recent`
WHERE direction IN UNNEST(@directions)
QUALIFY ROW_NUMBER() OVER (PARTITION BY direction ORDER BY bin_ts DESC) = 1
"""

# Promoted harness snapshot. Version label for models that run scored and saved
# nothing a Cloud Run service can load. Not a semver.
REPORT_RUN_ID = "20260930T202959Z_offline-bqml-joined-deep-fuzzy-ensemble"
NOT_DEPLOYED_VERSION = "not-deployed"
BOTH = ("SG_TO_MY", "MY_TO_SG")
MY_ONLY = ("MY_TO_SG",)

# joined.py FEATURE_SETS that the report run scored as ``{model}[{set}]``.
_JOINED_SCORED = ("maps", "maps+weather", "maps+camfc", "maps+mpfc", "maps+mpfc+camfc")
# In FEATURE_SETS, and not a candidate row in that run.
_JOINED_CODE_ONLY = ("maps+weather+camera",)


def _spec(
    model_id,
    family,
    version,
    horizon_min,
    directions,
    deploy_state,
    callable,
    source=None,
    sql=None,
):
    if callable != (deploy_state in CALLABLE_STATES):
        raise RuntimeError("callable must match a callable deploy_state")
    if callable and not sql:
        raise RuntimeError("a callable model needs SQL")
    if horizon_min not in (30, 60):
        raise RuntimeError("horizon")
    return {
        "id": model_id,
        "family": family,
        "default_version": version,
        "versions": frozenset({version}),
        "horizon_min": horizon_min,
        "horizons": frozenset({horizon_min}),
        "directions": directions,
        "deploy_state": deploy_state,
        "callable": callable,
        "source": source,
        "sql": sql,
    }


def _build_models():
    """Allow-list. ``served`` mirrors v_forecast_recent; the ``api`` rows are the 2026-10-03 BigQuery read. The rest are
    ids from eval code (joined, ensemble, fuzzy_traffic, timeseries_xgb,
    deep_forecast). The camera Fourier profile is not an entry.
    """
    rows = [
        _spec("served", "registry", REGISTRY_VERSION, 30, BOTH, "production", True, "bigquery view", _SERVED_SQL),
        _spec("lin_h30", "bqml", LIN_H30_VERSION, 30, BOTH, "api", True, "ML.PREDICT", _LIN_SQL),
        _spec("xgb_h30", "bqml", XGB_H30_VERSION, 30, BOTH, "api", True, "ML.PREDICT", _XGB_SQL),
        _spec(
            "persistence",
            "baseline",
            PERSISTENCE_VERSION,
            30,
            BOTH,
            "api",
            True,
            "bigquery view",
            _PERSISTENCE_SQL,
        ),
    ]
    for feature_set, state, version in (
        *((name, "artifact", REPORT_RUN_ID) for name in _JOINED_SCORED),
        *((name, "code-only", NOT_DEPLOYED_VERSION) for name in _JOINED_CODE_ONLY),
    ):
        for model_name, family in (("ridge", "sklearn"), ("xgb", "sklearn"), ("ensemble", "ensemble")):
            rows.append(
                _spec(
                    "%s[%s]" % (model_name, feature_set),
                    family,
                    version,
                    30,
                    BOTH,
                    state,
                    False,
                )
            )
    for model_id in ("ensemble_mean", "mean[models]", "stack", "select", "fuzzy stack"):
        rows.append(_spec(model_id, "ensemble", REPORT_RUN_ID, 30, BOTH, "artifact", False))
    rows.append(_spec("XGB (sklearn)", "sklearn", REPORT_RUN_ID, 60, MY_ONLY, "artifact", False))
    for model_id in ("lstm", "gru", "transformer", "transformer_raw"):
        rows.append(_spec(model_id, "deep", REPORT_RUN_ID, 60, MY_ONLY, "artifact", False))
    # fuzzy_traffic.CANDIDATES keys ``rules`` and ``xgb`` (level, not minutes).
    rows.append(_spec("rules", "fuzzy", REPORT_RUN_ID, 60, BOTH, "artifact", False))
    rows.append(_spec("xgb_to_fuzzy", "fuzzy", REPORT_RUN_ID, 60, BOTH, "artifact", False))
    for model_id in ("mean[XGB+deep]", "mean[deep]", "stack[XGB+deep]"):
        rows.append(_spec(model_id, "ensemble", REPORT_RUN_ID, 60, MY_ONLY, "artifact", False))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate model id")
    return {row["id"]: row for row in rows}


MODELS = _build_models()

_cache = {}
_clock = time.monotonic


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "3600",
    }


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
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        if text.endswith("+00:00"):
            return text[:-6] + "Z"
        return text
    return str(value)


def list_models():
    """Catalog a caller can read before asking for a forecast. No BigQuery."""
    models = []
    for spec in MODELS.values():
        models.append(
            {
                "id": spec["id"],
                "family": spec["family"],
                "version": spec["default_version"],
                "horizon_min": spec["horizon_min"],
                "directions": list(spec["directions"]),
                "deploy_state": spec["deploy_state"],
                "callable": spec["callable"],
            }
        )
    return {"models": models}


def query_forecast(model_id, directions):
    """Run the allow-listed SQL for one served model. Tests replace this function."""
    spec = MODELS[model_id]
    if not spec["callable"]:
        raise RuntimeError("model is not deployed")
    client = bigquery.Client(project=BQ_PROJECT)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("directions", "STRING", list(directions)),
        ]
    )
    job = client.query(spec["sql"], job_config=job_config, location=BQ_LOCATION)
    rows = []
    for row in job.result():
        rows.append(
            {
                "direction": row["direction"],
                "forecast_min": row["forecast_min"],
                "forecast_for": _iso(row["forecast_for_ts"]),
                "origin_ts": _iso(row["origin_ts"]),
            }
        )
    return {"source": spec["source"], "rows": rows}


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
    return direction, {
        "forecast_min": round(float(forecast_min), 1),
        "forecast_for": str(forecast_for),
        "origin_ts": str(origin_ts),
    }


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
        raise ValueError("missing direction")
    ordered = {direction: by_direction[direction] for direction in directions}
    return {
        "model": model_id,
        "version": version_id,
        "horizon_min": horizon,
        "generated_at": _iso(datetime.datetime.now(datetime.timezone.utc)),
        "source": source,
        "label": "maps_duration_in_traffic_min",
        "directions": ordered,
    }


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

    list_raw = _param(body, args, "list")
    if list_raw is not None and str(list_raw).strip() != "":
        if str(list_raw).strip() != "models":
            return _error("list must be models", 400)
        headers = {
            **_cors_headers(),
            "Content-Type": "application/json",
            "Cache-Control": "private, max-age=300",
        }
        return (json.dumps(list_models()), 200, headers)

    model_raw = _param(body, args, "model")
    if model_raw is None or str(model_raw).strip() == "":
        return _error("model is required", 400)
    model_id = str(model_raw).strip()
    spec = MODELS.get(model_id)
    if spec is None:
        return _error("unknown model", 400)
    if not spec["callable"]:
        return _error("model is not deployed", 400)

    version_raw = _param(body, args, "version")
    if version_raw is None or str(version_raw).strip() == "":
        version_id = spec["default_version"]
    else:
        version_id = str(version_raw).strip()
        if version_id not in spec["versions"]:
            return _error("unknown version", 400)

    directions = _parse_directions(_param(body, args, "direction"))
    if directions is None or any(direction not in spec["directions"] for direction in directions):
        return _error("direction must be SG_TO_MY, MY_TO_SG, or both", 400)

    horizon = _parse_horizon(_param(body, args, "horizon_min"), spec)
    if horizon is None:
        return _error(_horizon_error(spec), 400)

    cache_key = (model_id, version_id, tuple(directions), horizon)
    cached = _cache_get(cache_key)
    if cached is None:
        try:
            result = query_forecast(model_id, directions)
            payload = _assemble(model_id, version_id, horizon, directions, result)
        except Exception:
            logger.exception("forecast query failed")
            return _error("Forecast query failed", 502)
        _cache_put(cache_key, payload)
    else:
        payload = cached

    headers = {
        **_cors_headers(),
        "Content-Type": "application/json",
        "Cache-Control": "private, max-age=300",
    }
    return (json.dumps(payload), 200, headers)
