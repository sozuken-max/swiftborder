#!/usr/bin/env python3
"""SwiftBorder Layer B: TimesFM 2.5 travel-time forecasts fused with Layer A vehicle counts.

Layer B forecasts the Google Maps travel time in traffic on the Woodlands Causeway, per direction,
10 to 60 minutes ahead. This module uses Google's TimesFM 2.5 time-series foundation model through
BigQuery ``AI.FORECAST`` and adds the Layer A (vision) signal on top of it.

Why the design looks like this
------------------------------
In BigQuery, TimesFM 2.5 is univariate: ``AI.FORECAST`` accepts the target series only, so the
Layer A vehicle counts cannot be passed into the model. They are therefore fused *around* it, the
way the TimesFM library itself handles covariates (in-context regression on the residuals):

    B0  timesfm             zero-shot TimesFM 2.5 forecast of the travel-time series
    B0c timesfm_calibrated  B0 + ridge on its residual using non-vision features only
    B1  timesfm_layer_a     B0 + ridge on its residual using the same features plus Layer A counts

B1 differs from B0c only by the vision features, so B1 - B0c measures what the camera adds
("identical model specification except for vision features", proposal slide 15).

Decision rules (proposal slides 11, 14 and 15), fixed in code and applied on validation days only,
per direction and per horizon step:

1. Calibration is adopted if B0c beats B0 (95% day-block bootstrap interval of the paired MAE
   difference entirely below zero).
2. A vision feature advances to Layer B only if, on rows where Layer A was available, B1 beats
   both B0c (the identical model without vision) and the policy from step 1, and Layer A covered
   at least half of the validation rows. When a live frame is missing, the step-1 policy is served.
3. The resulting policy is served only if it beats persistence; otherwise persistence is served.

Layer A data contract
---------------------
Layer A is the ``swiftbackend`` Cloud Run service (``camdetect/`` in this repository). It is called
with ``format=json`` and returns per-direction vehicle counts and a congestion level. Its labels
``SG-MY`` / ``MY-SG`` map to ``SG_TO_MY`` / ``MY_TO_SG``. Counts measure density (vehicles in view),
not flow, so they are used as features, never as the target. A frame with zero vehicles while
Google reports congestion (slide 4: haze or heavy rain) is rejected.

The historical detections live in ``asia-southeast1`` while the travel times live in the BigQuery
``US`` location, and BigQuery cannot join across locations. This module therefore logs Layer A
observations into ``<dataset>.layer_a_counts`` in the same location as the travel times.

Time conventions
----------------
Everything runs on the 10-minute bins of ``traffic_prediction.v_bins_10min`` (bin start in UTC).
The forecast origin is the last complete bin ``t``. Step ``k`` targets bin ``t + 10k`` minutes,
which matches ``y_30`` / ``y_60`` in ``v_training_set`` (steps 3 and 6). Persistence is the value of
bin ``t``. Calendar dates on the command line are Singapore dates (UTC+8, no daylight saving).

Commands
--------
    python layer_b_timesfm.py ensure-tables
    python layer_b_timesfm.py run-cycle                       # Cloud Run Job, every 10 min
    python layer_b_timesfm.py ingest-layer-a                  # one live Layer A observation
    python layer_b_timesfm.py backfill-layer-a --start 2026-09-13 --end 2026-09-30 --max-calls 200
    python layer_b_timesfm.py backtest --start 2026-09-13 --end 2026-09-30
    python layer_b_timesfm.py train --start 2026-09-13 --end 2026-09-30 --validation-days 5 --register
    python layer_b_timesfm.py show-sql

``backfill-layer-a`` is a dry run unless ``--execute`` is given; every call costs Roboflow credits.
``backtest`` and ``train`` refuse the team's protected confirmation window (default 1-19 Oct 2026)
unless ``--allow-protected-window`` is given, so October labels are not scored before the
pre-registered frozen run.

Deployment (Cloud Run Job triggered by Cloud Scheduler)
-------------------------------------------------------
Two companion lines are needed for a source deploy of ``Causeway/`` with Google Cloud buildpacks.
``Causeway/requirements.txt`` must contain::

    google-cloud-bigquery==3.45.2
    requests==2.34.2

and ``Causeway/Procfile`` must name the start command (buildpacks read it; overriding
``--command`` would bypass the buildpack launcher)::

    web: python3 layer_b_timesfm.py run-cycle

Environment variables::

    SWIFTBORDER_PROJECT   GCP project (default: swiftborder)
    BQ_LOCATION           location of the travel-time datasets (default: US)
    LAYER_B_DATASET       dataset for the Layer B tables (default: traffic_prediction)
    BINS_VIEW             10-minute bins view (default: <project>.traffic_prediction.v_bins_10min)
    LAYER_A_URL           swiftbackend URL; Layer A ingestion is skipped when unset
    LAYER_A_CAMERA_ID     default 2701 (the only camera with a dividing line)
    LAYER_A_CONFIDENCE    minimum detection confidence sent to swiftbackend (default 0.25)
    TIMESFM_CONTEXT_WINDOW  one of 64..15360 accepted by TimesFM 2.5 (default 1024, ~7 days)
    PROTECTED_WINDOW      SGT dates START..END that backtest/train refuse (default 2026-10-01..2026-10-19)

Example::

    gcloud run jobs deploy layer-b-timesfm --source Causeway --region asia-southeast1 \\
        --max-retries 1 --task-timeout 600s \\
        --service-account layer-b@swiftborder.iam.gserviceaccount.com \\
        --set-env-vars SWIFTBORDER_PROJECT=swiftborder,LAYER_A_URL=https://<swiftbackend-url>
    gcloud scheduler jobs create http layer-b-timesfm-10min --location asia-southeast1 \\
        --schedule "*/10 * * * *" --time-zone Asia/Singapore --http-method POST \\
        --uri https://run.googleapis.com/v2/projects/swiftborder/locations/asia-southeast1/jobs/layer-b-timesfm:run \\
        --oauth-service-account-email layer-b@swiftborder.iam.gserviceaccount.com

Live Layer A frames are captured at the start of each cycle and used from the next cycle on, so
the frame paired with origin ``t`` was taken inside bin ``t`` both live and in the backfill
(which requests ``t + 5 min``). The Firebase front end can read the latest forecasts with::

    SELECT * FROM `swiftborder.traffic_prediction.layer_b_forecasts`
    WHERE origin_ts = (SELECT MAX(origin_ts) FROM `swiftborder.traffic_prediction.layer_b_forecasts`)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY direction, step ORDER BY created_at DESC) = 1

``run-cycle`` creates the four Layer B tables on first use (``ensure-tables`` does the same on
demand). The job's service account needs ``roles/bigquery.jobUser`` on the project,
``roles/bigquery.dataEditor`` on the Layer B dataset and ``roles/bigquery.dataViewer`` on
``causeway``; the scheduler's account needs ``roles/run.invoker`` on the job.

Exit codes: 0 success, 1 unexpected error, 2 invalid configuration or arguments,
3 no usable forecast (stale or insufficient data).

References
----------
A. Das, W. Kong, R. Sen, Y. Zhou, "A decoder-only foundation model for time-series forecasting",
ICML 2024. Google Cloud, "The AI.FORECAST function" (BigQuery documentation).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import random
import re
import shlex
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------

UTC = dt.timezone.utc
SGT = dt.timezone(dt.timedelta(hours=8), "SGT")  # Singapore has no daylight saving time
BIN = dt.timedelta(minutes=10)
BIN_MINUTES = 10

DIRECTIONS: Tuple[str, str] = ("SG_TO_MY", "MY_TO_SG")
LAYER_A_KEYS: Mapping[str, str] = {"SG_TO_MY": "sg_my", "MY_TO_SG": "my_sg"}
CONGESTION_ORDINAL: Mapping[str, int] = {"Free Flow": 0, "Quarter Way": 1, "Half Way": 2, "Back to Back": 3}

TIMESFM_MODEL = "TimesFM 2.5"
TIMESFM_CONTEXT_WINDOWS = frozenset({64, 128, 256, 512, 1024, 2048, 4096, 8192, 15360})
MAX_HORIZON_STEPS = 36  # 6 hours of 10-minute bins; the proposal targets 30-60 minutes

MORNING_PEAK_HOURS = range(6, 11)   # 06:00-10:59 SGT
EVENING_PEAK_HOURS = range(16, 22)  # 16:00-21:59 SGT
WITHIN_MINUTES = 15.0               # slide 15: share of forecasts within +/-15 minutes

MODELS = ("persistence", "seasonal_d1", "seasonal_d7", "seasonal_tod7", "timesfm", "timesfm_calibrated", "timesfm_layer_a")
SERVABLE = ("persistence", "timesfm", "timesfm_calibrated", "timesfm_layer_a")
BASE_FEATURES: Tuple[str, ...] = ("timesfm_minus_persistence",)
VISION_FEATURES: Tuple[str, ...] = ("vehicle_count", "other_direction_count", "congestion_ordinal")

MIN_TRAIN_ROWS = 100
MIN_VALIDATION_DAYS = 3
MIN_VISION_COVERAGE = 0.5

TABLE_LAYER_A = "layer_a_counts"
TABLE_FORECASTS = "layer_b_forecasts"
TABLE_BACKTEST = "layer_b_backtest"
TABLE_REGISTRY = "layer_b_registry"

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_NO_FORECAST = 0, 1, 2, 3

_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_DATASET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,1023}$")
_TABLE_REF_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """Invalid configuration or command-line arguments."""


class LayerAError(RuntimeError):
    """The Layer A service failed or returned an unusable payload."""


class LayerANoFrame(LayerAError):
    """No camera frame exists for the requested time (HTTP 404 from swiftbackend)."""


class NoForecastError(RuntimeError):
    """No direction had fresh enough data to forecast."""


def log(severity: str, message: str, **fields: Any) -> None:
    """Structured log line; Cloud Logging reads ``severity`` and ``message`` from JSON on stdout."""
    record = {"severity": severity, "message": message, "component": "layer_b_timesfm"}
    record.update({k: _jsonable(v) for k, v in fields.items()})
    print(json.dumps(record, sort_keys=True), flush=True)


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Config:
    project: str = "swiftborder"
    bq_location: str = "US"
    dataset: str = "traffic_prediction"
    bins_view: str = ""
    layer_a_url: str = ""
    camera_id: str = "2701"
    layer_a_confidence: float = 0.25
    context_window: int = 1024
    horizon_steps: int = 6
    confidence_level: float = 0.8
    min_history_bins: int = 144          # one day of 10-minute bins (capped at context_window)
    max_fill_bins: int = 3               # gaps up to 30 minutes are carried forward
    congested_ratio: float = 1.5         # Google congestion ratio above which a zero count is rejected
    layer_a_max_age_bins: int = 2        # a Layer A frame up to 20 minutes before the origin is usable
    ridge_lambda: float = 10.0
    bootstrap_reps: int = 2000
    seed: int = 20261004
    http_timeout_sec: float = 150.0      # swiftbackend can take ~120 s (frame, download, Roboflow)
    protected_window: Tuple[str, str] = ("2026-10-01", "2026-10-19")

    @property
    def min_bins(self) -> int:
        return min(self.min_history_bins, self.context_window)

    @property
    def view(self) -> str:
        return self.bins_view or f"{self.project}.traffic_prediction.v_bins_10min"

    def table(self, name: str) -> str:
        return f"{self.project}.{self.dataset}.{name}"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Config:
        env = os.environ if env is None else env
        project = env.get("SWIFTBORDER_PROJECT") or env.get("GOOGLE_CLOUD_PROJECT") or cls.project
        window = env.get("PROTECTED_WINDOW", "2026-10-01..2026-10-19")
        parts = tuple(p.strip() for p in window.split("..")) if window else ("", "")
        if len(parts) != 2:
            raise ConfigError(f"PROTECTED_WINDOW must look like START..END, got {window!r}")
        try:
            cfg = cls(
                project=project,
                bq_location=env.get("BQ_LOCATION", cls.bq_location),
                dataset=env.get("LAYER_B_DATASET", cls.dataset),
                bins_view=env.get("BINS_VIEW", ""),
                layer_a_url=env.get("LAYER_A_URL", "").rstrip("/"),
                camera_id=env.get("LAYER_A_CAMERA_ID", cls.camera_id),
                layer_a_confidence=float(env.get("LAYER_A_CONFIDENCE", cls.layer_a_confidence)),
                context_window=int(env.get("TIMESFM_CONTEXT_WINDOW", cls.context_window)),
                protected_window=(parts[0], parts[1]),
            )
        except ValueError as exc:
            raise ConfigError(f"invalid numeric environment variable: {exc}") from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not _PROJECT_RE.match(self.project):
            raise ConfigError(f"invalid project id {self.project!r}")
        if not _DATASET_RE.match(self.dataset):
            raise ConfigError(f"invalid dataset name {self.dataset!r}")
        if not _TABLE_REF_RE.match(self.view):
            raise ConfigError(f"BINS_VIEW must be project.dataset.table, got {self.view!r}")
        if not re.match(r"^[A-Za-z0-9-]{1,40}$", self.bq_location):
            raise ConfigError(f"invalid BigQuery location {self.bq_location!r}")
        if not re.match(r"^[0-9]{1,6}$", self.camera_id):
            raise ConfigError(f"invalid camera id {self.camera_id!r}")
        if self.layer_a_url and not self.layer_a_url.startswith("https://"):
            raise ConfigError("LAYER_A_URL must be an https:// URL")
        if self.context_window not in TIMESFM_CONTEXT_WINDOWS:
            raise ConfigError(f"context_window must be one of {sorted(TIMESFM_CONTEXT_WINDOWS)} for {TIMESFM_MODEL}")
        if not 1 <= self.horizon_steps <= MAX_HORIZON_STEPS:
            raise ConfigError(f"horizon_steps must be between 1 and {MAX_HORIZON_STEPS}")
        if not 0.0 < self.confidence_level < 1.0:
            raise ConfigError("confidence_level must be strictly between 0 and 1")
        if not 0.0 <= self.layer_a_confidence <= 1.0:
            raise ConfigError("LAYER_A_CONFIDENCE must be between 0 and 1")
        if self.min_history_bins < 1:
            raise ConfigError("min_history_bins must be at least 1")
        if not 0 <= self.max_fill_bins <= 12:
            raise ConfigError("max_fill_bins must be between 0 and 12")
        if self.ridge_lambda < 0 or self.bootstrap_reps < 100:
            raise ConfigError("ridge_lambda must be >= 0 and bootstrap_reps >= 100")
        if any(self.protected_window):
            start, end = (parse_sgt_date(d) for d in self.protected_window)
            if end < start:
                raise ConfigError("PROTECTED_WINDOW end is before its start")


# ----------------------------------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------------------------------


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def as_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        raise ValueError(f"naive datetime {ts!r}; timestamps must carry a time zone")
    return ts.astimezone(UTC)


def floor_bin(ts: dt.datetime) -> dt.datetime:
    """Start of the 10-minute bin containing ``ts`` (UTC), as ``v_bins_10min`` computes it."""
    ts = as_utc(ts)
    seconds = int(ts.timestamp()) // (BIN_MINUTES * 60) * (BIN_MINUTES * 60)
    return dt.datetime.fromtimestamp(seconds, UTC)


def last_complete_bin(now: dt.datetime) -> dt.datetime:
    return floor_bin(now) - BIN


def parse_sgt_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"expected a date YYYY-MM-DD, got {value!r}") from exc


def sgt_day_bounds(start: dt.date, end: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    """First and last bin start (UTC) of the SGT calendar days ``start``..``end`` inclusive."""
    if end < start:
        raise ConfigError(f"end date {end} is before start date {start}")
    first = dt.datetime.combine(start, dt.time(0, 0), SGT).astimezone(UTC)
    last = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0, 0), SGT).astimezone(UTC) - BIN
    return first, last


def sgt_date(ts: dt.datetime) -> dt.date:
    return as_utc(ts).astimezone(SGT).date()


def parse_layer_a_time(value: Any) -> dt.datetime:
    """swiftbackend echoes ``date_time`` as Singapore local time ``YYYY-MM-DDTHH:MM:SS``."""
    if not isinstance(value, str) or not value:
        raise LayerAError("payload has no date_time")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LayerAError(f"unparseable date_time {value!r}") from exc
    return as_utc(parsed if parsed.tzinfo else parsed.replace(tzinfo=SGT))


def period_of(ts: dt.datetime) -> str:
    hour = as_utc(ts).astimezone(SGT).hour
    if hour in MORNING_PEAK_HOURS:
        return "morning_peak"
    if hour in EVENING_PEAK_HOURS:
        return "evening_peak"
    return "off_peak"


def day_type_of(ts: dt.datetime) -> str:
    return "weekend" if as_utc(ts).astimezone(SGT).weekday() >= 5 else "weekday"


def check_protected_window(cfg: Config, first: dt.datetime, last: dt.datetime, allow: bool) -> None:
    """Refuse to score labels inside the team's pre-registered confirmation window."""
    if allow or not all(cfg.protected_window):
        return
    p_first, p_last = sgt_day_bounds(*(parse_sgt_date(d) for d in cfg.protected_window))
    # labels are capped at the window's last bin (forecast_sql: @label_cutoff), so origins and
    # labels both stay inside [first, last]
    if first <= p_last and last >= p_first:
        raise ConfigError(
            f"window overlaps the protected confirmation window {cfg.protected_window[0]}..{cfg.protected_window[1]} "
            "(SGT); pass --allow-protected-window only for the pre-registered frozen run"
        )


# ----------------------------------------------------------------------------------------------
# JSON and BigQuery plumbing
# ----------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return as_utc(value).isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


SCHEMAS: Dict[str, List[Tuple[str, str, str]]] = {
    TABLE_LAYER_A: [
        ("camera_id", "STRING", "REQUIRED"),
        ("obs_ts", "TIMESTAMP", "REQUIRED"),
        ("bin_ts", "TIMESTAMP", "REQUIRED"),
        ("direction", "STRING", "REQUIRED"),
        ("vehicle_count", "INT64", "REQUIRED"),
        ("other_direction_count", "INT64", "REQUIRED"),
        ("unknown_count", "INT64", "REQUIRED"),
        ("total_count", "INT64", "REQUIRED"),
        ("congestion_level", "STRING", "NULLABLE"),
        ("congestion_ordinal", "INT64", "NULLABLE"),
        ("min_confidence", "FLOAT64", "NULLABLE"),
        ("source_image", "STRING", "NULLABLE"),
        ("source", "STRING", "REQUIRED"),
        ("ingested_at", "TIMESTAMP", "REQUIRED"),
    ],
    TABLE_FORECASTS: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("origin_ts", "TIMESTAMP", "REQUIRED"),
        ("direction", "STRING", "REQUIRED"),
        ("step", "INT64", "REQUIRED"),
        ("horizon_min", "INT64", "REQUIRED"),
        ("target_ts", "TIMESTAMP", "REQUIRED"),
        ("persistence", "FLOAT64", "NULLABLE"),
        ("timesfm", "FLOAT64", "NULLABLE"),
        ("timesfm_lower", "FLOAT64", "NULLABLE"),
        ("timesfm_upper", "FLOAT64", "NULLABLE"),
        ("confidence_level", "FLOAT64", "NULLABLE"),
        ("timesfm_calibrated", "FLOAT64", "NULLABLE"),
        ("timesfm_layer_a", "FLOAT64", "NULLABLE"),
        ("served_model", "STRING", "REQUIRED"),
        ("served_value", "FLOAT64", "NULLABLE"),
        ("layer_a_status", "STRING", "REQUIRED"),
        ("registry_run_id", "STRING", "NULLABLE"),
        ("model", "STRING", "REQUIRED"),
        ("context_window", "INT64", "REQUIRED"),
    ],
    TABLE_BACKTEST: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("origin_ts", "TIMESTAMP", "REQUIRED"),
        ("direction", "STRING", "REQUIRED"),
        ("step", "INT64", "REQUIRED"),
        ("target_ts", "TIMESTAMP", "REQUIRED"),
        ("actual", "FLOAT64", "NULLABLE"),
        ("persistence", "FLOAT64", "NULLABLE"),
        ("seasonal_d1", "FLOAT64", "NULLABLE"),
        ("seasonal_d7", "FLOAT64", "NULLABLE"),
        ("seasonal_tod7", "FLOAT64", "NULLABLE"),
        ("timesfm", "FLOAT64", "NULLABLE"),
        ("timesfm_lower", "FLOAT64", "NULLABLE"),
        ("timesfm_upper", "FLOAT64", "NULLABLE"),
        ("n_context_bins", "INT64", "NULLABLE"),
        ("model", "STRING", "REQUIRED"),
        ("context_window", "INT64", "REQUIRED"),
    ],
    TABLE_REGISTRY: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("model", "STRING", "REQUIRED"),
        ("context_window", "INT64", "REQUIRED"),
        ("train_start", "DATE", "REQUIRED"),
        ("train_end", "DATE", "REQUIRED"),
        ("validation_start", "DATE", "REQUIRED"),
        ("validation_end", "DATE", "REQUIRED"),
        ("decisions_json", "STRING", "REQUIRED"),
        ("coefficients_json", "STRING", "REQUIRED"),
        ("metrics_json", "STRING", "REQUIRED"),
    ],
}

PARTITIONING: Dict[str, Tuple[Optional[str], List[str]]] = {
    TABLE_LAYER_A: ("bin_ts", ["direction", "bin_ts"]),
    TABLE_FORECASTS: ("origin_ts", ["direction", "origin_ts"]),
    TABLE_BACKTEST: ("origin_ts", ["run_id", "direction"]),
    TABLE_REGISTRY: (None, []),
}


class Warehouse:
    """Thin wrapper around the BigQuery client: parameterised queries, table creation, batch loads."""

    def __init__(self, cfg: Config, client: Any = None) -> None:
        from google.cloud import bigquery  # noqa: PLC0415 - lazy, so `show-sql` needs no credentials

        self._bq = bigquery
        self.cfg = cfg
        self.client = client or bigquery.Client(project=cfg.project, location=cfg.bq_location)

    def _param(self, name: str, value: Any) -> Any:
        bq = self._bq
        if isinstance(value, (list, tuple)):
            items = list(value)
            kind = "TIMESTAMP" if items and isinstance(items[0], dt.datetime) else "STRING"
            return bq.ArrayQueryParameter(name, kind, items)
        if isinstance(value, bool):
            return bq.ScalarQueryParameter(name, "BOOL", value)
        if isinstance(value, dt.datetime):
            return bq.ScalarQueryParameter(name, "TIMESTAMP", as_utc(value))
        if isinstance(value, int):
            return bq.ScalarQueryParameter(name, "INT64", value)
        if isinstance(value, float):
            return bq.ScalarQueryParameter(name, "FLOAT64", value)
        return bq.ScalarQueryParameter(name, "STRING", str(value))

    def query(self, sql: str, params: Optional[Mapping[str, Any]] = None, timeout_sec: float = 600.0) -> List[Dict[str, Any]]:
        job_config = self._bq.QueryJobConfig(
            query_parameters=[self._param(k, v) for k, v in (params or {}).items()],
            labels={"component": "layer-b-timesfm"},
        )
        job = self.client.query(sql, job_config=job_config, location=self.cfg.bq_location)
        return [dict(row.items()) for row in job.result(timeout=timeout_sec)]

    def schema(self, name: str) -> List[Any]:
        return [self._bq.SchemaField(col, kind, mode=mode) for col, kind, mode in SCHEMAS[name]]

    def ensure_tables(self) -> List[str]:
        created = []
        for name in SCHEMAS:
            table = self._bq.Table(self.cfg.table(name), schema=self.schema(name))
            partition_col, cluster_cols = PARTITIONING[name]
            if partition_col:
                table.time_partitioning = self._bq.TimePartitioning(type_=self._bq.TimePartitioningType.DAY, field=partition_col)
            if cluster_cols:
                table.clustering_fields = cluster_cols
            self.client.create_table(table, exists_ok=True)
            created.append(self.cfg.table(name))
        return created

    def append(self, name: str, rows: Sequence[Mapping[str, Any]]) -> int:
        """Append rows with a batch load job (free, and never leaves rows in a streaming buffer)."""
        if not rows:
            return 0
        columns = [c for c, _, _ in SCHEMAS[name]]
        payload = [{c: _jsonable(row.get(c)) for c in columns} for row in rows]
        job_config = self._bq.LoadJobConfig(
            schema=self.schema(name),
            write_disposition=self._bq.WriteDisposition.WRITE_APPEND,
            source_format=self._bq.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        job = self.client.load_table_from_json(payload, self.cfg.table(name), job_config=job_config, location=self.cfg.bq_location)
        job.result(timeout=300)
        return len(payload)


# ----------------------------------------------------------------------------------------------
# SQL builders (one query serves both the live forecast and the rolling-origin backtest)
# ----------------------------------------------------------------------------------------------


def forecast_sql(cfg: Config, stride_minutes: int = BIN_MINUTES) -> str:
    """TimesFM 2.5 forecasts for every origin in [@origin_start, @origin_end] and both directions.

    Each (direction, origin) becomes its own series, built only from bins at or before the origin,
    so a single ``AI.FORECAST`` call produces a leak-free rolling-origin backtest; the live forecast
    is the same query with one origin. Gaps of up to ``max_fill_bins`` are carried forward; history
    before a longer gap is dropped; an origin whose own bin is missing is skipped (stale data).
    """
    if stride_minutes % BIN_MINUTES or stride_minutes <= 0:
        raise ConfigError("stride_minutes must be a positive multiple of 10")
    span = (cfg.context_window - 1) * BIN_MINUTES
    lookback = max(span, 7 * 24 * 60)  # the 7-day seasonal baselines need a week before the target
    ahead = cfg.horizon_steps * BIN_MINUTES
    model = TIMESFM_MODEL.replace("'", "")
    return f"""
WITH origins AS (
  SELECT origin_ts
  FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(@origin_start, @origin_end, INTERVAL {int(stride_minutes)} MINUTE)) AS origin_ts
),
obs AS (
  SELECT direction, bin_ts, dur_min
  FROM `{cfg.view}`
  WHERE direction IN UNNEST(@directions)
    AND dur_min IS NOT NULL
    AND bin_ts BETWEEN TIMESTAMP_SUB(@origin_start, INTERVAL {lookback} MINUTE)
                   AND TIMESTAMP_ADD(@origin_end, INTERVAL {ahead} MINUTE)
),
grid AS (
  SELECT d AS direction, o.origin_ts, b AS bin_ts
  FROM origins AS o
  CROSS JOIN UNNEST(@directions) AS d
  CROSS JOIN UNNEST(GENERATE_TIMESTAMP_ARRAY(TIMESTAMP_SUB(o.origin_ts, INTERVAL {span} MINUTE), o.origin_ts, INTERVAL {BIN_MINUTES} MINUTE)) AS b
),
filled AS (
  SELECT g.direction, g.origin_ts, g.bin_ts, x.dur_min AS raw,
         LAST_VALUE(x.dur_min IGNORE NULLS) OVER (
           PARTITION BY g.direction, g.origin_ts ORDER BY g.bin_ts
           ROWS BETWEEN {int(cfg.max_fill_bins)} PRECEDING AND CURRENT ROW) AS dur_min
  FROM grid AS g
  LEFT JOIN obs AS x USING (direction, bin_ts)
),
holes AS (
  SELECT direction, origin_ts,
         MAX(IF(dur_min IS NULL, bin_ts, NULL)) AS last_hole,
         MAX(IF(bin_ts = origin_ts, raw, NULL)) AS origin_raw
  FROM filled
  GROUP BY direction, origin_ts
),
segment AS (
  SELECT f.direction, f.origin_ts, f.bin_ts, f.dur_min
  FROM filled AS f
  JOIN holes AS h USING (direction, origin_ts)
  WHERE h.origin_raw IS NOT NULL AND (h.last_hole IS NULL OR f.bin_ts > h.last_hole)
),
eligible AS (
  SELECT direction, origin_ts, COUNT(*) AS n_context_bins
  FROM segment
  GROUP BY direction, origin_ts
  HAVING COUNT(*) >= @min_history_bins
),
history AS (
  SELECT CONCAT(s.direction, '|', FORMAT_TIMESTAMP('%Y%m%dT%H%M%SZ', s.origin_ts)) AS series_id, s.bin_ts, s.dur_min
  FROM segment AS s
  JOIN eligible AS e USING (direction, origin_ts)
),
forecast AS (
  SELECT *
  FROM AI.FORECAST(
    TABLE history,
    model => '{model}',
    timestamp_col => 'bin_ts',
    data_col => 'dur_min',
    id_cols => ['series_id'],
    horizon => {int(cfg.horizon_steps)},
    confidence_level => {float(cfg.confidence_level)},
    context_window => {int(cfg.context_window)})
),
parsed AS (
  SELECT SPLIT(series_id, '|')[OFFSET(0)] AS direction,
         PARSE_TIMESTAMP('%Y%m%dT%H%M%SZ', SPLIT(series_id, '|')[OFFSET(1)]) AS origin_ts,
         forecast_timestamp AS target_ts,
         forecast_value, prediction_interval_lower_bound, prediction_interval_upper_bound,
         confidence_level, ai_forecast_status
  FROM forecast
),
tod7 AS (
  SELECT direction, bin_ts,
         AVG(dur_min) OVER (
           PARTITION BY direction, TIME(bin_ts) ORDER BY UNIX_SECONDS(bin_ts)
           RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING) AS mean_prev_7d
  FROM obs
)
SELECT p.direction, p.origin_ts,
       DIV(TIMESTAMP_DIFF(p.target_ts, p.origin_ts, MINUTE), {BIN_MINUTES}) AS step,
       p.target_ts,
       act.dur_min AS actual,
       per.dur_min AS persistence,
       d1.dur_min AS seasonal_d1,
       d7.dur_min AS seasonal_d7,
       t7.mean_prev_7d AS seasonal_tod7,
       p.forecast_value AS timesfm,
       p.prediction_interval_lower_bound AS timesfm_lower,
       p.prediction_interval_upper_bound AS timesfm_upper,
       p.confidence_level,
       p.ai_forecast_status,
       e.n_context_bins
FROM parsed AS p
JOIN eligible AS e ON e.direction = p.direction AND e.origin_ts = p.origin_ts
LEFT JOIN obs AS act ON act.direction = p.direction AND act.bin_ts = p.target_ts AND act.bin_ts <= @label_cutoff
LEFT JOIN obs AS per ON per.direction = p.direction AND per.bin_ts = p.origin_ts
LEFT JOIN obs AS d1 ON d1.direction = p.direction AND d1.bin_ts = TIMESTAMP_SUB(p.target_ts, INTERVAL 1 DAY)
LEFT JOIN obs AS d7 ON d7.direction = p.direction AND d7.bin_ts = TIMESTAMP_SUB(p.target_ts, INTERVAL 7 DAY)
LEFT JOIN tod7 AS t7 ON t7.direction = p.direction AND t7.bin_ts = p.target_ts
ORDER BY p.direction, p.origin_ts, step
""".strip()


def layer_a_features_sql(cfg: Config) -> str:
    """Latest Layer A observation per (bin, direction) with the slide-4 quality gate applied."""
    return f"""
WITH a AS (
  SELECT bin_ts, direction, vehicle_count, other_direction_count, total_count, congestion_ordinal,
         ROW_NUMBER() OVER (PARTITION BY bin_ts, direction ORDER BY obs_ts DESC, ingested_at DESC) AS rn
  FROM `{cfg.table(TABLE_LAYER_A)}`
  WHERE camera_id = @camera_id AND bin_ts BETWEEN @start AND @end
),
g AS (
  SELECT bin_ts, MAX(congestion_ratio) AS max_congestion_ratio
  FROM `{cfg.view}`
  WHERE bin_ts BETWEEN @start AND @end
  GROUP BY bin_ts
)
SELECT a.bin_ts, a.direction, a.vehicle_count, a.other_direction_count, a.total_count, a.congestion_ordinal,
       (a.total_count = 0 AND IFNULL(g.max_congestion_ratio, 0) >= @congested_ratio) AS rejected
FROM a
LEFT JOIN g USING (bin_ts)
WHERE a.rn = 1
ORDER BY a.bin_ts, a.direction
""".strip()


def layer_a_existing_bins_sql(cfg: Config) -> str:
    return f"""
SELECT DISTINCT bin_ts
FROM `{cfg.table(TABLE_LAYER_A)}`
WHERE camera_id = @camera_id AND bin_ts BETWEEN @start AND @end
""".strip()


def latest_registry_sql(cfg: Config) -> str:
    return f"""
SELECT run_id, created_at, decisions_json, coefficients_json
FROM `{cfg.table(TABLE_REGISTRY)}`
WHERE model = @model AND context_window = @context_window
ORDER BY created_at DESC
LIMIT 1
""".strip()


def forecast_params(cfg: Config, origin_start: dt.datetime, origin_end: dt.datetime, label_cutoff: dt.datetime) -> Dict[str, Any]:
    return {
        "origin_start": as_utc(origin_start),
        "origin_end": as_utc(origin_end),
        "label_cutoff": as_utc(label_cutoff),
        "directions": list(DIRECTIONS),
        "min_history_bins": int(cfg.min_bins),
    }


# ----------------------------------------------------------------------------------------------
# Layer A client
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LayerAObservation:
    camera_id: str
    obs_ts: dt.datetime
    counts: Mapping[str, int]
    congestion: Mapping[str, Optional[str]]
    unknown_count: int
    total_count: int
    min_confidence: Optional[float]
    source_image: Optional[str]

    @property
    def bin_ts(self) -> dt.datetime:
        return floor_bin(self.obs_ts)

    def to_rows(self, ingested_at: dt.datetime) -> List[Dict[str, Any]]:
        rows = []
        for direction in DIRECTIONS:
            other = DIRECTIONS[1] if direction == DIRECTIONS[0] else DIRECTIONS[0]
            label = self.congestion.get(direction)
            rows.append({
                "camera_id": self.camera_id,
                "obs_ts": self.obs_ts,
                "bin_ts": self.bin_ts,
                "direction": direction,
                "vehicle_count": int(self.counts[direction]),
                "other_direction_count": int(self.counts[other]),
                "unknown_count": int(self.unknown_count),
                "total_count": int(self.total_count),
                "congestion_level": label,
                "congestion_ordinal": CONGESTION_ORDINAL.get(label) if label else None,
                "min_confidence": self.min_confidence,
                "source_image": self.source_image,
                "source": "swiftbackend",
                "ingested_at": as_utc(ingested_at),
            })
        return rows


def _count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or int(value) != value:
        raise LayerAError(f"{field} is not a non-negative integer: {value!r}")
    return int(value)


def parse_layer_a_payload(payload: Any, camera_id: str) -> LayerAObservation:
    """Validate a swiftbackend ``format=json`` response (camdetect/README.md contract)."""
    if not isinstance(payload, Mapping):
        raise LayerAError("payload is not a JSON object")
    directions = payload.get("directions")
    if not isinstance(directions, Mapping):
        raise LayerAError("payload has no 'directions' object")
    if not directions.get("available"):
        raise LayerAError(f"camera {camera_id} has no dividing line; per-direction counts are unavailable")
    counts, congestion = {}, {}
    for direction, key in LAYER_A_KEYS.items():
        block = directions.get(key)
        if not isinstance(block, Mapping):
            raise LayerAError(f"payload has no directions.{key}")
        counts[direction] = _count(block.get("count"), f"directions.{key}.count")
        label = block.get("congestion")
        congestion[direction] = label if isinstance(label, str) else None
    unknown = directions.get("unknown") or {}
    unknown_count = _count(unknown.get("count", 0), "directions.unknown.count") if isinstance(unknown, Mapping) else 0
    total = _count(payload.get("vehicle_count", sum(counts.values()) + unknown_count), "vehicle_count")
    if total != sum(counts.values()) + unknown_count:
        raise LayerAError("vehicle_count does not equal sg_my + my_sg + unknown")
    returned_camera = str(payload.get("camera_id", camera_id))
    if returned_camera != str(camera_id):
        raise LayerAError(f"asked for camera {camera_id}, got {returned_camera}")
    confidence = payload.get("min_confidence")
    return LayerAObservation(
        camera_id=str(camera_id),
        obs_ts=parse_layer_a_time(payload.get("date_time")),
        counts=counts,
        congestion=congestion,
        unknown_count=unknown_count,
        total_count=total,
        min_confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        source_image=payload.get("source_image") if isinstance(payload.get("source_image"), str) else None,
    )


def fetch_layer_a(
    cfg: Config,
    session: Any,
    when: Optional[dt.datetime] = None,
    *,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> LayerAObservation:
    """One swiftbackend call. ``when`` (any time zone) requests a historical frame.

    Only failures that cannot have reached Roboflow are retried (connection errors, 429 and 503).
    Timeouts and other 5xx responses may come after a billed inference call, so they fail at once.
    """
    if not cfg.layer_a_url:
        raise ConfigError("LAYER_A_URL is not set")
    params = {"camera_id": cfg.camera_id, "format": "json", "confidence": f"{cfg.layer_a_confidence:g}"}
    if when is not None:
        params["date_time"] = as_utc(when).astimezone(SGT).strftime("%Y-%m-%dT%H:%M:%S")
    delay, last = 2.0, "no attempt"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(cfg.layer_a_url, params=params, timeout=cfg.http_timeout_sec)
        except requests.ConnectionError as exc:
            last = f"connection error: {exc.__class__.__name__}"
        except requests.RequestException as exc:
            raise LayerAError(f"swiftbackend request failed: {exc.__class__.__name__}") from exc
        else:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    raise LayerAError("swiftbackend returned non-JSON") from exc
                return parse_layer_a_payload(payload, cfg.camera_id)
            if resp.status_code == 404:
                raise LayerANoFrame(f"no frame for camera {cfg.camera_id} at {params.get('date_time', 'now')}")
            if resp.status_code not in (429, 503):
                raise LayerAError(f"swiftbackend HTTP {resp.status_code}")
            last = f"HTTP {resp.status_code}"
        if attempt < max_attempts:
            sleep(delay)
            delay *= 2
    raise LayerAError(f"swiftbackend failed after {max_attempts} attempts ({last})")


def _bin_logged(cfg: Config, wh: Warehouse, bin_ts: dt.datetime) -> bool:
    return bool(wh.query(layer_a_existing_bins_sql(cfg), {"camera_id": cfg.camera_id, "start": bin_ts, "end": bin_ts}))


def ingest_layer_a(cfg: Config, wh: Warehouse, session: Any, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Log one live Layer A observation (idempotent per camera and bin).

    The table is checked before swiftbackend is called, so a bin another job has already logged
    costs no Roboflow inference; it is checked again if the frame lands in a different bin.
    """
    now = now or utc_now()
    expected = floor_bin(now)
    if _bin_logged(cfg, wh, expected):
        return {"status": "already_logged", "bin_ts": expected}
    obs = fetch_layer_a(cfg, session)
    if obs.bin_ts != expected and _bin_logged(cfg, wh, obs.bin_ts):
        return {"status": "already_logged", "bin_ts": obs.bin_ts}
    wh.append(TABLE_LAYER_A, obs.to_rows(now))
    return {"status": "logged", "bin_ts": obs.bin_ts, "counts": dict(obs.counts), "total_count": obs.total_count}


def backfill_layer_a(
    cfg: Config,
    wh: Warehouse,
    session: Any,
    start: dt.date,
    end: dt.date,
    *,
    every_minutes: int,
    max_calls: int,
    execute: bool,
    now: Optional[dt.datetime] = None,
    sleep: Callable[[float], None] = time.sleep,
    batch_size: int = 50,
    max_consecutive_failures: int = 5,
) -> Dict[str, Any]:
    """Score historical frames through swiftbackend so Layer A overlaps the travel-time labels.

    Resumable (bins already logged are skipped), budgeted (``max_calls``), and a dry run unless
    ``execute``. Each call costs Roboflow credits.
    """
    if every_minutes % BIN_MINUTES or every_minutes <= 0:
        raise ConfigError("--every-minutes must be a positive multiple of 10")
    if max_calls < 1:
        raise ConfigError("--max-calls must be at least 1")
    now = now or utc_now()
    first, last = sgt_day_bounds(start, end)
    last = min(last, last_complete_bin(now))
    if last < first:
        raise ConfigError("the backfill window is entirely in the future")
    wanted, ts = [], first
    while ts <= last:
        wanted.append(ts)
        ts += dt.timedelta(minutes=every_minutes)
    existing = {as_utc(r["bin_ts"]) for r in wh.query(layer_a_existing_bins_sql(cfg), {"camera_id": cfg.camera_id, "start": first, "end": last})}
    todo = [b for b in wanted if b not in existing]
    plan = todo[:max_calls]
    summary: Dict[str, Any] = {"wanted": len(wanted), "already_logged": len(wanted) - len(todo), "planned_calls": len(plan), "execute": execute}
    if not execute or not plan:
        return summary
    rows: List[Dict[str, Any]] = []
    logged = no_frame = failed = consecutive = 0
    stopped_early = False
    for bin_ts in plan:
        try:
            obs = fetch_layer_a(cfg, session, when=bin_ts + dt.timedelta(minutes=5), sleep=sleep)
        except LayerANoFrame:
            no_frame += 1
            consecutive = 0
            continue
        except LayerAError as exc:
            failed += 1
            consecutive += 1
            log("WARNING", "layer A backfill call failed", bin_ts=bin_ts, error=str(exc))
            if consecutive >= max_consecutive_failures:
                log("ERROR", "stopping backfill after repeated failures", consecutive=consecutive)
                stopped_early = True
                break
            continue
        consecutive = 0
        if obs.bin_ts != bin_ts:
            log("WARNING", "frame time fell outside the requested bin; skipped", requested=bin_ts, frame=obs.obs_ts)
            continue
        rows.extend(obs.to_rows(utc_now()))
        logged += 1
        if len(rows) >= batch_size * len(DIRECTIONS):
            wh.append(TABLE_LAYER_A, rows)
            rows = []
    wh.append(TABLE_LAYER_A, rows)
    summary.update(logged=logged, no_frame=no_frame, failed=failed, stopped_early=stopped_early)
    return summary


# ----------------------------------------------------------------------------------------------
# Forecast rows, features and the residual models
# ----------------------------------------------------------------------------------------------


def run_forecast_query(cfg: Config, wh: Warehouse, origin_start: dt.datetime, origin_end: dt.datetime, *,
                       label_cutoff: dt.datetime, stride_minutes: int = BIN_MINUTES) -> List[Dict[str, Any]]:
    rows = wh.query(forecast_sql(cfg, stride_minutes), forecast_params(cfg, origin_start, origin_end, label_cutoff))
    good = []
    for row in rows:
        status = row.get("ai_forecast_status") or ""
        if status:
            log("WARNING", "TimesFM returned a status for a series", direction=row.get("direction"), origin_ts=row.get("origin_ts"), status=status)
            continue
        row["origin_ts"], row["target_ts"] = as_utc(row["origin_ts"]), as_utc(row["target_ts"])
        row["step"] = int(row["step"])
        good.append(row)
    return good


def load_layer_a_features(cfg: Config, wh: Warehouse, start: dt.datetime, end: dt.datetime) -> Dict[Tuple[str, dt.datetime], Dict[str, Any]]:
    rows = wh.query(layer_a_features_sql(cfg), {
        "camera_id": cfg.camera_id,
        "start": as_utc(start) - cfg.layer_a_max_age_bins * BIN,
        "end": as_utc(end),
        "congested_ratio": float(cfg.congested_ratio),
    })
    return {(r["direction"], as_utc(r["bin_ts"])): r for r in rows}


def vision_features(cfg: Config, layer_a: Mapping[Tuple[str, dt.datetime], Mapping[str, Any]], direction: str,
                    origin: dt.datetime) -> Tuple[Optional[Dict[str, float]], str]:
    """Most recent accepted Layer A frame at or up to ``layer_a_max_age_bins`` before the origin."""
    rejected_seen = False
    for age in range(cfg.layer_a_max_age_bins + 1):
        rec = layer_a.get((direction, origin - age * BIN))
        if rec is None:
            continue
        if rec.get("rejected"):
            rejected_seen = True
            continue
        ordinal = rec.get("congestion_ordinal")
        return {
            "vehicle_count": float(rec["vehicle_count"]),
            "other_direction_count": float(rec["other_direction_count"]),
            "congestion_ordinal": float(ordinal) if ordinal is not None else 0.0,
        }, "ok"
    return None, ("rejected_quality_gate" if rejected_seen else "missing")


def base_features(row: Mapping[str, Any]) -> Dict[str, float]:
    return {"timesfm_minus_persistence": float(row["timesfm"]) - float(row["persistence"])}


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting for the small ridge normal equations."""
    n = len(b)
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


@dataclasses.dataclass
class RidgeResidual:
    """Ridge regression of the TimesFM residual on standardised features (intercept unpenalised)."""

    features: Tuple[str, ...]
    mean: List[float]
    std: List[float]
    beta: List[float]
    n: int

    def __post_init__(self) -> None:
        k = len(self.features)
        if not (len(self.mean) == len(self.std) == k and len(self.beta) == k + 1):
            raise ValueError("ridge coefficients do not match the feature list")
        if any(sd <= 0 for sd in self.std):
            raise ValueError("ridge feature scales must be positive")

    @classmethod
    def fit(cls, features: Sequence[str], xs: Sequence[Mapping[str, float]], residuals: Sequence[float], lam: float) -> RidgeResidual:
        if len(xs) != len(residuals) or not xs:
            raise ValueError("need matching, non-empty training rows")
        k = len(features)
        cols = [[float(x[f]) for x in xs] for f in features]
        mean = [sum(c) / len(c) for c in cols]
        std = []
        for c, mu in zip(cols, mean):
            var = sum((v - mu) ** 2 for v in c) / len(c)
            std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        design = [[1.0] + [(cols[j][i] - mean[j]) / std[j] for j in range(k)] for i in range(len(xs))]
        a = [[sum(row[p] * row[q] for row in design) for q in range(k + 1)] for p in range(k + 1)]
        for j in range(1, k + 1):
            a[j][j] += lam
        b = [sum(row[p] * r for row, r in zip(design, residuals)) for p in range(k + 1)]
        return cls(tuple(features), mean, std, solve_linear(a, b), len(xs))

    def predict(self, x: Mapping[str, float]) -> float:
        z = [(float(x[f]) - mu) / sd for f, mu, sd in zip(self.features, self.mean, self.std)]
        return self.beta[0] + sum(b * v for b, v in zip(self.beta[1:], z))

    def to_json(self) -> Dict[str, Any]:
        return {"features": list(self.features), "mean": self.mean, "std": self.std, "beta": self.beta, "n": self.n}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RidgeResidual:
        return cls(tuple(data["features"]), list(data["mean"]), list(data["std"]), list(data["beta"]), int(data["n"]))


# ----------------------------------------------------------------------------------------------
# Evaluation: metrics and a paired day-block bootstrap
# ----------------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default), q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def error_metrics(errors: Sequence[float]) -> Dict[str, Any]:
    abs_err = [abs(e) for e in errors]
    n = len(abs_err)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "mae": sum(abs_err) / n,
        "median_ae": percentile(abs_err, 50),
        "p90_ae": percentile(abs_err, 90),
        "rmse": math.sqrt(sum(e * e for e in errors) / n),
        "bias": sum(errors) / n,
        f"within_{int(WITHIN_MINUTES)}min": sum(1 for e in abs_err if e <= WITHIN_MINUTES) / n,
    }


def paired_day_bootstrap(pairs: Sequence[Tuple[dt.date, float]], reps: int, seed: int, level: float = 0.95) -> Dict[str, Any]:
    """CI of the mean paired difference, resampling whole SGT days (of the forecast origin).

    ``pairs`` holds (day, difference). Pass both directions' rows together to resample a calendar
    day as one block, so the two directions' correlated errors on the same day are never counted
    as independent evidence.
    """
    by_day: Dict[dt.date, List[float]] = {}
    for day, diff in pairs:
        by_day.setdefault(day, []).append(diff)
    days = sorted(by_day)
    if not days:
        return {"n": 0, "days": 0, "mean": None, "ci_low": None, "ci_high": None}
    sums = [sum(by_day[d]) for d in days]
    counts = [len(by_day[d]) for d in days]
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        idx = [rng.randrange(len(days)) for _ in days]
        means.append(sum(sums[i] for i in idx) / sum(counts[i] for i in idx))
    tail = (1.0 - level) / 2.0 * 100.0
    return {
        "n": sum(counts),
        "days": len(days),
        "mean": sum(sums) / sum(counts),
        "ci_low": percentile(means, tail),
        "ci_high": percentile(means, 100.0 - tail),
    }


def better(result: Mapping[str, Any], min_days: int = MIN_VALIDATION_DAYS) -> bool:
    """Challenger better than reference: enough days and the whole interval below zero."""
    return bool(result.get("days", 0) >= min_days and result.get("ci_high") is not None and result["ci_high"] < 0)


def summarise(rows: Sequence[Mapping[str, Any]], models: Sequence[str], steps: Sequence[int]) -> Dict[str, Any]:
    """Metrics per model, step, direction, period and day type (slide 15)."""
    out: Dict[str, Any] = {}
    for step in steps:
        step_rows = [r for r in rows if r["step"] == step and r.get("actual") is not None]
        for model in models:
            cell: Dict[str, Any] = {}
            groups = {"all": step_rows}
            for d in DIRECTIONS:
                groups[d] = [r for r in step_rows if r["direction"] == d]
            for key in ("morning_peak", "evening_peak", "off_peak"):
                groups[key] = [r for r in step_rows if period_of(r["target_ts"]) == key]
            for key in ("weekday", "weekend"):
                groups[key] = [r for r in step_rows if day_type_of(r["target_ts"]) == key]
            for name, subset in groups.items():
                errs = [float(r[model]) - float(r["actual"]) for r in subset if r.get(model) is not None]
                cell[name] = error_metrics(errs)
            out.setdefault(f"step_{step}", {})[model] = cell
        cov = [r for r in step_rows if r.get("timesfm_lower") is not None and r.get("timesfm_upper") is not None]
        if cov:
            inside = sum(1 for r in cov if r["timesfm_lower"] <= r["actual"] <= r["timesfm_upper"])
            out[f"step_{step}"]["timesfm_interval_coverage"] = {"n": len(cov), "coverage": inside / len(cov)}
    return out


def compare(rows: Sequence[Mapping[str, Any]], challenger: str, reference: str, cfg: Config) -> Dict[str, Any]:
    pairs = [
        (sgt_date(r["origin_ts"]), abs(float(r[challenger]) - float(r["actual"])) - abs(float(r[reference]) - float(r["actual"])))
        for r in rows
        if r.get("actual") is not None and r.get(challenger) is not None and r.get(reference) is not None
    ]
    return paired_day_bootstrap(pairs, cfg.bootstrap_reps, cfg.seed)


# ----------------------------------------------------------------------------------------------
# Backtest and training
# ----------------------------------------------------------------------------------------------


def backtest_rows(cfg: Config, wh: Warehouse, start: dt.date, end: dt.date, *, stride_minutes: int, chunk_days: int,
                  now: Optional[dt.datetime] = None) -> List[Dict[str, Any]]:
    """Rolling-origin TimesFM forecasts for complete SGT days, chunked to bound query size."""
    now = now or utc_now()
    if end >= sgt_date(now):
        raise ConfigError(f"end date {end} is not a complete day yet in Singapore")
    if chunk_days < 1:
        raise ConfigError("--chunk-days must be at least 1")
    first, last = sgt_day_bounds(start, end)
    label_cutoff = last  # labels after the window's last bin are never scored
    rows: List[Dict[str, Any]] = []
    day = start
    while day <= end:
        chunk_end = min(end, day + dt.timedelta(days=chunk_days - 1))
        c_first, c_last = sgt_day_bounds(day, chunk_end)
        log("INFO", "TimesFM backtest chunk", start=day, end=chunk_end)
        rows.extend(run_forecast_query(cfg, wh, c_first, c_last, label_cutoff=label_cutoff, stride_minutes=stride_minutes))
        day = chunk_end + dt.timedelta(days=1)
    log("INFO", "TimesFM backtest done", rows=len(rows), first_origin=first, last_origin=last)
    return rows


def attach_layer_a(cfg: Config, rows: List[Dict[str, Any]], layer_a: Mapping[Tuple[str, dt.datetime], Mapping[str, Any]]) -> None:
    for r in rows:
        feats, status = vision_features(cfg, layer_a, r["direction"], r["origin_ts"])
        r["layer_a_status"] = status
        r["vision"] = feats


def apply_models(rows: List[Dict[str, Any]], coefficients: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    """Add timesfm_calibrated and timesfm_layer_a predictions where the models and inputs exist."""
    cache: Dict[Tuple[str, str, str], RidgeResidual] = {}
    for r in rows:
        r["timesfm_calibrated"] = None
        r["timesfm_layer_a"] = None
        if r.get("timesfm") is None or r.get("persistence") is None:
            continue
        d, s = r["direction"], str(r["step"])
        for name, key in (("timesfm_calibrated", "calibrated"), ("timesfm_layer_a", "layer_a")):
            spec = coefficients.get(d, {}).get(s, {}).get(key)
            if not spec:
                continue
            x = base_features(r)
            if key == "layer_a":
                if not r.get("vision"):
                    continue
                x.update(r["vision"])
            try:
                model = cache.get((d, s, key))
                if model is None:
                    model = cache[(d, s, key)] = RidgeResidual.from_json(spec)
                value = float(r["timesfm"]) + model.predict(x)
            except (KeyError, TypeError, ValueError) as exc:
                log("ERROR", "unusable residual model; falling back", direction=d, step=s, model=key, error=str(exc))
                continue
            if math.isfinite(value):
                r[name] = value


def served_value(row: Mapping[str, Any], decision: str, fallback: Optional[str] = None) -> Tuple[str, Optional[float]]:
    """Value for the registry decision, falling back when an input is missing.

    For ``timesfm_layer_a`` the fallback is the policy chosen before the vision gate (``fallback``),
    never a model the calibration gate rejected.
    """
    if decision == "timesfm_layer_a":
        middle = (fallback,) if fallback in ("timesfm_calibrated", "timesfm") else ("timesfm",)
        chain: Tuple[str, ...] = ("timesfm_layer_a", *middle, "timesfm", "persistence")
    else:
        chain = {
            "timesfm_calibrated": ("timesfm_calibrated", "timesfm", "persistence"),
            "timesfm": ("timesfm", "persistence"),
            "persistence": ("persistence",),
        }[decision]
    for model in dict.fromkeys(chain):
        if row.get(model) is not None:
            return model, float(row[model])
    return "persistence", None


def train(cfg: Config, rows: List[Dict[str, Any]], layer_a: Mapping[Tuple[str, dt.datetime], Mapping[str, Any]],
          validation_days: int) -> Dict[str, Any]:
    """Fit the residual models on training days and apply the decision rules on validation days."""
    usable = [r for r in rows if r.get("actual") is not None and r.get("timesfm") is not None and r.get("persistence") is not None]
    if not usable:
        raise ConfigError("no backtest rows with labels; check the window and the data")
    days = sorted({sgt_date(r["origin_ts"]) for r in usable})
    if validation_days < MIN_VALIDATION_DAYS or validation_days >= len(days):
        raise ConfigError(f"--validation-days must be at least {MIN_VALIDATION_DAYS} and leave training days (window has {len(days)} days)")
    val_days = set(days[-validation_days:])
    attach_layer_a(cfg, usable, layer_a)
    # a training row whose label falls on a validation day would leak across the boundary
    train_rows = [r for r in usable if sgt_date(r["origin_ts"]) not in val_days and sgt_date(r["target_ts"]) not in val_days]
    val_rows = [r for r in usable if sgt_date(r["origin_ts"]) in val_days]

    coefficients: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for d in DIRECTIONS:
        for step in range(1, cfg.horizon_steps + 1):
            tr = [r for r in train_rows if r["direction"] == d and r["step"] == step]
            spec: Dict[str, Any] = {}
            if len(tr) >= MIN_TRAIN_ROWS:
                spec["calibrated"] = RidgeResidual.fit(
                    BASE_FEATURES, [base_features(r) for r in tr], [r["actual"] - r["timesfm"] for r in tr], cfg.ridge_lambda).to_json()
            tv = [r for r in tr if r.get("vision")]
            if len(tv) >= MIN_TRAIN_ROWS:
                spec["layer_a"] = RidgeResidual.fit(
                    BASE_FEATURES + VISION_FEATURES,
                    [{**base_features(r), **r["vision"]} for r in tv],
                    [r["actual"] - r["timesfm"] for r in tv], cfg.ridge_lambda).to_json()
            coefficients.setdefault(d, {})[str(step)] = spec
    apply_models(usable, coefficients)

    decisions: Dict[str, Dict[str, Any]] = {}
    for d in DIRECTIONS:
        for step in range(1, cfg.horizon_steps + 1):
            v = [r for r in val_rows if r["direction"] == d and r["step"] == step]
            note: Dict[str, Any] = {}
            policy = "timesfm"
            note["timesfm_vs_persistence"] = compare(v, "timesfm", "persistence", cfg)
            if all(r.get("timesfm_calibrated") is not None for r in v) and v:
                note["calibrated_vs_timesfm"] = compare(v, "timesfm_calibrated", "timesfm", cfg)
                if better(note["calibrated_vs_timesfm"]):
                    policy = "timesfm_calibrated"
            fallback = policy
            vision_rows = [r for r in v if r.get("timesfm_layer_a") is not None]
            coverage = len(vision_rows) / len(v) if v else 0.0
            note["layer_a_coverage"] = coverage
            if vision_rows:
                # slide 15: B1 vs the identical residual model without the vision features ...
                same_spec = "timesfm_calibrated" if all(r.get("timesfm_calibrated") is not None for r in vision_rows) else "timesfm"
                note["layer_a_vs_no_vision"] = compare(vision_rows, "timesfm_layer_a", same_spec, cfg)
                # ... and vs what would otherwise be served, so a rejected model is never the bar
                note["layer_a_vs_fallback"] = compare(vision_rows, "timesfm_layer_a", fallback, cfg)
                if coverage >= MIN_VISION_COVERAGE and better(note["layer_a_vs_no_vision"]) and better(note["layer_a_vs_fallback"]):
                    policy = "timesfm_layer_a"
            note["fallback"] = fallback
            as_served = [{**r, "policy": served_value(r, policy, fallback)[1]} for r in v]
            note["policy_vs_persistence"] = compare(as_served, "policy", "persistence", cfg)
            note["candidate"] = policy
            note["served"] = policy if better(note["policy_vs_persistence"]) else "persistence"
            decisions.setdefault(d, {})[str(step)] = note

    steps = sorted({r["step"] for r in usable})
    metrics = {
        "validation": summarise(val_rows, MODELS, steps),
        "validation_layer_a_rows": summarise([r for r in val_rows if r.get("vision")], MODELS, steps),
        "pooled_timesfm_vs_persistence": {
            str(s): compare([r for r in val_rows if r["step"] == s], "timesfm", "persistence", cfg) for s in steps
        },
    }
    train_days = [x for x in days if x not in val_days]
    return {
        "train_start": train_days[0], "train_end": train_days[-1],
        "validation_start": min(val_days), "validation_end": max(val_days),
        "coefficients": coefficients, "decisions": decisions, "metrics": metrics,
    }


def registry_row(cfg: Config, result: Mapping[str, Any], run_id: str, now: dt.datetime) -> Dict[str, Any]:
    slim = {d: {s: {"served": n["served"], "candidate": n["candidate"], "fallback": n["fallback"]} for s, n in steps.items()}
            for d, steps in result["decisions"].items()}
    return {
        "run_id": run_id,
        "created_at": now,
        "model": TIMESFM_MODEL,
        "context_window": cfg.context_window,
        "train_start": result["train_start"],
        "train_end": result["train_end"],
        "validation_start": result["validation_start"],
        "validation_end": result["validation_end"],
        "decisions_json": json.dumps(_jsonable(slim), sort_keys=True),
        "coefficients_json": json.dumps(_jsonable(result["coefficients"]), sort_keys=True),
        "metrics_json": json.dumps(_jsonable({"decisions": result["decisions"], **result["metrics"]}), sort_keys=True),
    }


# ----------------------------------------------------------------------------------------------
# Live cycle
# ----------------------------------------------------------------------------------------------


def run_cycle(cfg: Config, wh: Warehouse, session: Any, *, now: Optional[dt.datetime] = None, use_layer_a: bool = True,
              write: bool = True) -> Dict[str, Any]:
    """Ingest Layer A, forecast with TimesFM 2.5, apply the registered policy, log the forecasts."""
    now = now or utc_now()
    run_id = f"cycle-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    summary: Dict[str, Any] = {"run_id": run_id, "model": TIMESFM_MODEL}
    if write:
        wh.ensure_tables()  # idempotent; avoids a failed first run before `ensure-tables`

    if use_layer_a and cfg.layer_a_url:
        try:
            summary["layer_a_ingest"] = ingest_layer_a(cfg, wh, session, now) if write else {"status": "skipped_dry_run"}
        except Exception as exc:  # noqa: BLE001 - Layer B must still forecast when Layer A is down
            summary["layer_a_ingest"] = {"status": "failed", "error": str(exc)}
            log("WARNING", "Layer A ingest failed; forecasting without it", error=repr(exc))
    else:
        summary["layer_a_ingest"] = {"status": "disabled"}

    origin = last_complete_bin(now)
    summary["origin_ts"] = origin
    rows = run_forecast_query(cfg, wh, origin, origin, label_cutoff=origin)
    if not rows:
        raise NoForecastError(f"no direction had fresh data for origin {origin.isoformat()}")

    decisions: Mapping[str, Any] = {}
    coefficients: Mapping[str, Any] = {}
    registry_run_id = None
    try:
        registry = wh.query(latest_registry_sql(cfg), {"model": TIMESFM_MODEL, "context_window": cfg.context_window})
        if registry:
            decisions = json.loads(registry[0]["decisions_json"])
            coefficients = json.loads(registry[0]["coefficients_json"])
            registry_run_id = registry[0]["run_id"]
        else:
            log("WARNING", "no registry entry; serving persistence until `train --register` has run")
    except Exception as exc:  # noqa: BLE001 - an unreadable registry means the conservative default
        decisions, coefficients = {}, {}
        log("ERROR", "registry unreadable; serving persistence", error=repr(exc))
    summary["registry_run_id"] = registry_run_id

    layer_a: Dict[Tuple[str, dt.datetime], Dict[str, Any]] = {}
    if use_layer_a:
        try:
            layer_a = load_layer_a_features(cfg, wh, origin, origin)
        except Exception as exc:  # noqa: BLE001 - forecast without vision rather than not at all
            log("ERROR", "Layer A features unavailable; forecasting without them", error=repr(exc))
    attach_layer_a(cfg, rows, layer_a)
    apply_models(rows, coefficients)

    out_rows = []
    for r in rows:
        entry = decisions.get(r["direction"], {}).get(str(r["step"]), {})
        decision = entry.get("served", "persistence") if isinstance(entry, Mapping) else "persistence"
        if decision not in SERVABLE:
            decision = "persistence"
        model, value = served_value(r, decision, entry.get("fallback") if isinstance(entry, Mapping) else None)
        out_rows.append({
            "run_id": run_id, "created_at": now, "origin_ts": r["origin_ts"], "direction": r["direction"],
            "step": r["step"], "horizon_min": r["step"] * BIN_MINUTES, "target_ts": r["target_ts"],
            "persistence": r.get("persistence"), "timesfm": r.get("timesfm"),
            "timesfm_lower": r.get("timesfm_lower"), "timesfm_upper": r.get("timesfm_upper"),
            "confidence_level": r.get("confidence_level"),
            "timesfm_calibrated": r.get("timesfm_calibrated"), "timesfm_layer_a": r.get("timesfm_layer_a"),
            "served_model": model, "served_value": value, "layer_a_status": r.get("layer_a_status", "missing"),
            "registry_run_id": registry_run_id, "model": TIMESFM_MODEL, "context_window": cfg.context_window,
        })
    if write:
        wh.append(TABLE_FORECASTS, out_rows)
    summary["forecasts"] = [
        {k: row[k] for k in ("direction", "horizon_min", "target_ts", "served_model", "served_value", "timesfm", "persistence", "layer_a_status")}
        for row in out_rows
    ]
    summary["directions"] = sorted({row["direction"] for row in out_rows})
    return summary


def http_entrypoint(request: Any) -> Tuple[str, int, Dict[str, str]]:
    """Optional HTTP target (functions-framework / Cloud Run service): runs one cycle."""
    headers = {"Content-Type": "application/json"}
    try:
        cfg = Config.from_env()
        summary = run_cycle(cfg, Warehouse(cfg), requests.Session())
        return json.dumps(_jsonable(summary)), 200, headers
    except NoForecastError as exc:
        log("WARNING", "no forecast", error=str(exc))
        return json.dumps({"error": "no fresh data to forecast"}), 503, headers
    except Exception as exc:  # never leak internals to the caller
        log("ERROR", "cycle failed", error=repr(exc))
        return json.dumps({"error": "layer B cycle failed"}), 500, headers


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------


def _print(obj: Any) -> None:
    print(json.dumps(_jsonable(obj), sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SwiftBorder Layer B: TimesFM 2.5 fused with Layer A counts.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ensure-tables", help="create the Layer B tables if missing")

    p = sub.add_parser("run-cycle", help="ingest Layer A, forecast, log (Cloud Run Job entry point)")
    p.add_argument("--no-layer-a", action="store_true", help="skip Layer A ingestion and features")
    p.add_argument("--dry-run", action="store_true", help="forecast and print without writing to BigQuery")

    sub.add_parser("ingest-layer-a", help="log one live Layer A observation")

    p = sub.add_parser("backfill-layer-a", help="score historical frames (costs Roboflow credits)")
    p.add_argument("--start", required=True, help="first SGT date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last SGT date, YYYY-MM-DD")
    p.add_argument("--every-minutes", type=int, default=10)
    p.add_argument("--max-calls", type=int, default=200)
    p.add_argument("--execute", action="store_true", help="actually call swiftbackend (default: dry run)")

    for name, help_text in (("backtest", "rolling-origin TimesFM backtest with baselines"),
                            ("train", "fit residual models and apply the decision rules")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--start", required=True, help="first SGT date, YYYY-MM-DD")
        p.add_argument("--end", required=True, help="last SGT date, YYYY-MM-DD")
        p.add_argument("--stride-minutes", type=int, default=10, help="origin spacing (multiple of 10)")
        p.add_argument("--chunk-days", type=int, default=2, help="days of origins per AI.FORECAST call")
        p.add_argument("--allow-protected-window", action="store_true")
        if name == "backtest":
            p.add_argument("--write", action="store_true", help=f"append rows to {TABLE_BACKTEST}")
        else:
            p.add_argument("--validation-days", type=int, default=5)
            p.add_argument("--register", action="store_true", help=f"append the decision to {TABLE_REGISTRY}")

    sub.add_parser("show-sql", help="print the generated SQL (no credentials needed)")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, warehouse_factory: Optional[Callable[[Config], Warehouse]] = None,
         session: Any = None) -> int:
    if argv is None:  # Cloud Run Jobs: the command can come from LAYER_B_ARGS instead of the Procfile
        argv = sys.argv[1:] or shlex.split(os.environ.get("LAYER_B_ARGS", "run-cycle"))
    args = build_parser().parse_args(argv)
    try:
        cfg = Config.from_env()
        if args.command == "show-sql":
            for title, sql in (("forecast (live and backtest)", forecast_sql(cfg)), ("layer A features", layer_a_features_sql(cfg)),
                               ("latest registry", latest_registry_sql(cfg))):
                print(f"-- {title}\n{sql}\n")
            return EXIT_OK
        wh = (warehouse_factory or Warehouse)(cfg)
        http = session or requests.Session()
        if args.command == "ensure-tables":
            _print({"tables": wh.ensure_tables()})
        elif args.command == "run-cycle":
            _print(run_cycle(cfg, wh, http, use_layer_a=not args.no_layer_a, write=not args.dry_run))
        elif args.command == "ingest-layer-a":
            _print(ingest_layer_a(cfg, wh, http))
        elif args.command == "backfill-layer-a":
            result = backfill_layer_a(cfg, wh, http, parse_sgt_date(args.start), parse_sgt_date(args.end),
                                      every_minutes=args.every_minutes, max_calls=args.max_calls, execute=args.execute)
            _print(result)
            if result.get("stopped_early"):
                return EXIT_ERROR
        elif args.command in ("backtest", "train"):
            start, end = parse_sgt_date(args.start), parse_sgt_date(args.end)
            first, last = sgt_day_bounds(start, end)
            check_protected_window(cfg, first, last, args.allow_protected_window)
            rows = backtest_rows(cfg, wh, start, end, stride_minutes=args.stride_minutes, chunk_days=args.chunk_days)
            now = utc_now()
            run_id = f"{args.command}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            if args.command == "backtest":
                steps = sorted({r["step"] for r in rows})
                _print({"run_id": run_id, "rows": len(rows), "metrics": summarise(rows, MODELS[:5], steps),
                        "timesfm_vs_persistence": {str(s): compare([r for r in rows if r["step"] == s], "timesfm", "persistence", cfg) for s in steps}})
                if args.write:
                    wh.append(TABLE_BACKTEST, [{**r, "run_id": run_id, "created_at": now, "model": TIMESFM_MODEL,
                                                "context_window": cfg.context_window} for r in rows])
            else:
                layer_a: Dict[Tuple[str, dt.datetime], Dict[str, Any]] = {}
                try:
                    layer_a = load_layer_a_features(cfg, wh, first, last)
                except Exception as exc:  # noqa: BLE001 - e.g. the table does not exist yet; train without vision
                    log("WARNING", "Layer A features unavailable; training without them", error=repr(exc))
                result = train(cfg, rows, layer_a, args.validation_days)
                _print({"run_id": run_id, **result})
                if args.register:
                    wh.append(TABLE_REGISTRY, [registry_row(cfg, result, run_id, now)])
        return EXIT_OK
    except ConfigError as exc:
        log("ERROR", "invalid configuration or arguments", error=str(exc))
        return EXIT_CONFIG
    except NoForecastError as exc:
        log("ERROR", "no usable forecast", error=str(exc))
        return EXIT_NO_FORECAST
    except LayerAError as exc:
        log("ERROR", "Layer A failed", error=str(exc))
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - last-resort guard so the job logs a structured error
        log("ERROR", "unexpected failure", error=repr(exc))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
