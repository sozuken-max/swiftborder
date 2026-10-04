#!/usr/bin/env python3
"""SwiftBorder Layer B: Fuzzy C-Means + multilayer perceptron forecasts fused with Layer A counts.

Layer B forecasts the Google Maps travel time in traffic on the Woodlands Causeway, per direction,
10 to 60 minutes ahead. This module is the proposal's fuzzy hybrid ("rules over heavy / moderate /
light congestion, interpretable") built as a Fuzzy C-Means (FCM) front end feeding a multilayer
perceptron (MLP), with the Layer A (vision) counts as optional extra inputs.

How the model works
-------------------
1. Fuzzification (unsupervised). FCM with fuzzifier m = 2 clusters the traffic state at the
   forecast origin into c = 3 regimes. The state has three coordinates: the travel time now, its
   change over the last 30 minutes, and yesterday's change over the coming hour (a hint that a peak
   is about to build or clear). Regimes are ordered by their centre travel time and named light,
   moderate and heavy. Every origin gets a degree of membership in each regime and the degrees sum
   to one, so a state near a boundary is, say, 0.6 moderate and 0.4 heavy instead of being forced
   into one class. Beyond the outermost centres the degrees saturate, so a longer jam is never less
   "heavy" than a shorter one. Centres are stored in minutes, so a reader can see what each regime
   means for each direction. The regime of the origin state uses all three coordinates, so a long
   but clearing queue can sit closer to the middle regime than a shorter one that is building; the
   regime reported for a forecast uses its travel time alone.
2. Regression (supervised). An MLP (ReLU hidden layers, Adam, L2 penalty) maps the lag, seasonal
   and calendar features plus the membership degrees to the change in travel time at every horizon
   step at once. Several networks that differ only in their random seed are averaged: on a few
   weeks of data a seed ensemble is cheaper insurance against an unlucky initialisation than a
   bigger network, and the spread across members is logged as a rough uncertainty signal.
3. Fusion. The same network with the Layer A vehicle counts and congestion level appended.

    persistence, seasonal_d1, seasonal_d7, seasonal_tod7   baselines
    mlp              ablation: the identical network without the FCM membership degrees
    fcm_mlp          B0: FCM memberships + MLP
    fcm_mlp_layer_a  B1: B0 plus the Layer A features

Decision rules (proposal slides 11, 14 and 15), fixed in code and applied per direction and per
horizon step on validation days only:

1. fcm_mlp is the candidate.
2. fcm_mlp_layer_a replaces it only if Layer A covered at least half of the validation rows and, on
   those rows, it beats both the identical network trained on the same rows without the vision
   features and the model that would otherwise be served. When a live frame is missing, fcm_mlp is
   served if it passed rule 3 on its own, and persistence otherwise.
3. The candidate is served only if it beats persistence; otherwise persistence is served.

"Beats" means that the 95% paired day-block bootstrap interval of the difference in absolute error
lies entirely below zero, over at least three days. The FCM ablation (fcm_mlp against mlp) is
reported but not gated: it answers whether the fuzzy front end earns its place.

Splits and leakage control
--------------------------
Days are split in time order, never shuffled: ``[training days][validation days][test days]``.
The last two training days are held out for early stopping and for choosing the L2 penalty, so the
validation days only ever feed the decision rules, and the test days are scored once, after the
decisions are fixed. A label that falls on a later split's day is masked, so no row learns across
a boundary. Feature scaling, FCM centres and network weights are fitted on training days only.

The FCM centres and network weights are stored as plain JSON in the registry table, not as
pickles. The live cycle therefore needs numpy alone (scikit-learn is used for training only), and
anyone with read access can audit a stored model.

Course techniques covered (docs/grading): unsupervised learning (FCM), deep learning (MLP), a
hybrid and ensemble design (fuzzy front end feeding a seed ensemble of networks), and intelligent
sensing (Layer A vehicle counts).

Layer A data contract
---------------------
Layer A is the ``swiftbackend`` Cloud Run service (``camdetect/`` in this repository). It is called
with ``format=json`` and returns per-direction vehicle counts and a congestion level. Its labels
``SG-MY`` / ``MY-SG`` map to ``SG_TO_MY`` / ``MY_TO_SG``. Counts measure density (vehicles in view),
not flow, so they are used as features, never as the target. A frame with zero vehicles while
Google reports congestion (slide 4: haze or heavy rain) is rejected.

The historical detections live in ``asia-southeast1`` while the travel times live in the BigQuery
``US`` location, and BigQuery cannot join across locations. Layer A observations are therefore
logged into ``<dataset>.layer_a_counts`` next to the travel times. The table and its schema are
shared with ``layer_b_timesfm.py``. Ingestion checks the table before calling swiftbackend and
skips a bin that is already logged, but two jobs starting at the same moment can still both pay
for a Roboflow call, so set ``LAYER_A_URL`` on one Layer B job only.

Time conventions
----------------
Everything runs on the 10-minute bins of ``traffic_prediction.v_bins_10min`` (bin start in UTC).
The forecast origin is the last complete bin ``t``; step ``k`` targets bin ``t + 10k`` minutes, so
steps 3 and 6 match ``y_30`` and ``y_60`` of ``v_training_set`` (exactly, when there are no gaps:
that view counts rows, this module counts time). ``horizon_min`` is measured from the start of the
origin bin; the job runs as soon as that bin closes, so the target bin's readings arrive 10k-10 to
10k-5 minutes after the forecast is issued. Persistence is the value of bin ``t``. Features are
computed in Python from the bins by one function used for both training and the live cycle, so
the two cannot drift apart. Calendar dates on the command line are Singapore
dates (UTC+8, no daylight saving). A non-working day is a Saturday or Sunday, as in ``eval/``, or
any date listed in ``NONWORK_DAYS``.

Commands
--------
    python layer_b_fcm_mlp.py ensure-tables
    python layer_b_fcm_mlp.py run-cycle                     # Cloud Run Job, every 10 min
    python layer_b_fcm_mlp.py ingest-layer-a                # one live Layer A observation
    python layer_b_fcm_mlp.py backfill-layer-a --start 2026-09-13 --end 2026-09-30 --max-calls 200
    python layer_b_fcm_mlp.py train --start 2026-09-06 --end 2026-09-30 --validation-days 7 --test-days 5
    python layer_b_fcm_mlp.py train --lookback-days 28 --test-days 0 --register   # weekly refresh
    python layer_b_fcm_mlp.py show-sql

``backfill-layer-a`` is a dry run unless ``--execute`` is given; every call costs Roboflow credits.
``train`` refuses the team's protected confirmation window (default 1-19 Oct 2026) unless
``--allow-protected-window`` is given, so October labels are not scored before the pre-registered
frozen run. ``--register`` stores the fitted model and its decisions; ``--write-eval`` stores the
validation and test predictions row by row for the report's tables and plots.

Deployment (Cloud Run Jobs triggered by Cloud Scheduler)
--------------------------------------------------------
``Causeway/requirements.txt`` needs::

    google-cloud-bigquery==3.45.2
    requests==2.34.2
    numpy==2.4.6
    scikit-learn==1.9.1

Buildpacks start the command named in ``Causeway/Procfile``. One line serves this module, the
TimesFM module and the training job, chosen per job with environment variables (started without
arguments, either module also reads its command from ``LAYER_B_ARGS``, default ``run-cycle``)::

    web: python3 ${LAYER_B_MODULE:-layer_b_fcm_mlp.py} ${LAYER_B_ARGS:-run-cycle}

A ``Causeway/.python-version`` file containing ``3.11`` pins the interpreter to the team lock files;
without it buildpacks use the latest Python release.

Environment variables (all optional)::

    SWIFTBORDER_PROJECT   GCP project (default: swiftborder)
    BQ_LOCATION           location of the travel-time datasets (default: US)
    LAYER_B_DATASET       dataset for the Layer B tables (default: traffic_prediction)
    BINS_VIEW             10-minute bins view (default: <project>.traffic_prediction.v_bins_10min)
    LAYER_A_URL           swiftbackend URL; Layer A ingestion is skipped when unset
    LAYER_A_CAMERA_ID     default 2701 (the only camera with a dividing line)
    LAYER_A_CONFIDENCE    minimum detection confidence sent to swiftbackend (default 0.25)
    FCM_CLUSTERS          number of regimes, 2 to 5 (default 3: light, moderate, heavy)
    FCM_FUZZIFIER         fuzzifier m, 1.1 to 5 (default 2.0)
    MLP_HIDDEN            hidden layer widths, e.g. 32,16 (default)
    MLP_ENSEMBLE_SIZE     networks averaged per model (default 5)
    NONWORK_DAYS          extra non-working SGT dates, comma-separated, e.g. 2026-09-16 (Malaysia
                          Day); set the same value on the training and the 10-minute job
    PROTECTED_WINDOW      SGT dates START..END that train refuses (default 2026-10-01..2026-10-19)

Example::

    gcloud run jobs deploy layer-b-fcm-mlp --source Causeway --region asia-southeast1 \\
        --max-retries 1 --task-timeout 600s --memory 1Gi \\
        --service-account layer-b@swiftborder.iam.gserviceaccount.com \\
        --set-env-vars SWIFTBORDER_PROJECT=swiftborder,LAYER_B_MODULE=layer_b_fcm_mlp.py,LAYER_A_URL=https://<swiftbackend-url>
    gcloud scheduler jobs create http layer-b-fcm-mlp-10min --location asia-southeast1 \\
        --schedule "*/10 * * * *" --time-zone Asia/Singapore --http-method POST \\
        --uri https://run.googleapis.com/v2/projects/swiftborder/locations/asia-southeast1/jobs/layer-b-fcm-mlp:run \\
        --oauth-service-account-email layer-b@swiftborder.iam.gserviceaccount.com
    gcloud run jobs deploy layer-b-fcm-mlp-train --source Causeway --region asia-southeast1 \\
        --max-retries 0 --task-timeout 1800s --memory 2Gi \\
        --service-account layer-b@swiftborder.iam.gserviceaccount.com \\
        --set-env-vars "SWIFTBORDER_PROJECT=swiftborder,LAYER_B_MODULE=layer_b_fcm_mlp.py,LAYER_B_ARGS=train --lookback-days 28 --test-days 0 --register"

The weekly job fits on its lookback minus the validation days (the last two training days steer
early stopping and the L2 choice), and the validation days still decide what is served;
``--test-days 0`` because the one-off test belongs to the research run.

The training job can run weekly from a second scheduler entry. While its lookback overlaps the
protected window (with 28 days, until mid-November) it exits with code 2 by design, so register a
model trained on September data first; after the frozen run, set ``PROTECTED_WINDOW`` to an empty
value on the training job to lift the guard. Seven validation days give the decision rules every
day of the week, weekend included.
The Firebase front end can read the latest forecasts with::

    SELECT * FROM `swiftborder.traffic_prediction.layer_b_fcm_forecasts`
    WHERE origin_ts = (SELECT MAX(origin_ts) FROM `swiftborder.traffic_prediction.layer_b_fcm_forecasts`)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY direction, step ORDER BY created_at DESC) = 1

``run-cycle`` creates the Layer B tables on first use (``ensure-tables`` does the same on demand).
The jobs' service account needs ``roles/bigquery.jobUser`` on the project,
``roles/bigquery.dataEditor`` on the Layer B dataset and ``roles/bigquery.dataViewer`` on
``causeway``; the scheduler's account needs ``roles/run.invoker`` on the jobs.

Exit codes: 0 success, 1 unexpected error, 2 invalid configuration, arguments or too little data,
3 no usable forecast (stale data).

References
----------
J. C. Bezdek, R. Ehrlich, W. Full, "FCM: The fuzzy c-means clustering algorithm", Computers &
Geosciences 10(2-3), 191-203, 1984. X. L. Xie, G. Beni, "A validity measure for fuzzy clustering",
IEEE TPAMI 13(8), 841-847, 1991. D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization",
ICLR 2015. B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive
uncertainty estimation using deep ensembles", NeurIPS 2017. H. R. Kunsch, "The jackknife and the
bootstrap for general stationary observations", Annals of Statistics 17(3), 1217-1241, 1989.
F. Pedregosa et al., "Scikit-learn: Machine learning in Python", JMLR 12, 2825-2830, 2011.
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

import numpy as np
import requests

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------

UTC = dt.timezone.utc
SGT = dt.timezone(dt.timedelta(hours=8), "SGT")  # Singapore has no daylight saving time
SGT_OFFSET_SEC = 8 * 3600
BIN = dt.timedelta(minutes=10)
BIN_MINUTES = 10
BIN_SECONDS = 600
BINS_PER_DAY = 144
BINS_PER_WEEK = 7 * BINS_PER_DAY
EPOCH_DATE = dt.date(1970, 1, 1)

DIRECTIONS: Tuple[str, str] = ("SG_TO_MY", "MY_TO_SG")
LAYER_A_KEYS: Mapping[str, str] = {"SG_TO_MY": "sg_my", "MY_TO_SG": "my_sg"}
CONGESTION_ORDINAL: Mapping[str, int] = {"Free Flow": 0, "Quarter Way": 1, "Half Way": 2, "Back to Back": 3}

MODEL_NAME = "FCM+MLP"
MAX_HORIZON_STEPS = 36  # 6 hours; must stay below one day so the D-1 features remain causal

MORNING_PEAK_HOURS = range(6, 11)   # 06:00-10:59 SGT
EVENING_PEAK_HOURS = range(16, 22)  # 16:00-21:59 SGT
WITHIN_MINUTES = 15.0               # slide 15: share of forecasts within +/-15 minutes

LAG_STEPS: Tuple[int, ...] = (1, 2, 3, 6)  # 10, 20, 30 and 60 minutes before the origin
ROLL_BINS = 6                              # the last hour, origin included
BASE_FEATURES: Tuple[str, ...] = (
    "now", "delta_10", "delta_20", "delta_30", "delta_60", "roll_mean_60_minus_now", "roll_std_60",
    "d1_gap", "d1_rise_mid", "d1_rise_end", "d1_missing",
    "d7_gap", "d7_rise_mid", "d7_rise_end", "d7_missing",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "nonworkday",
)
FCM_FEATURES: Tuple[str, ...] = ("now", "delta_30", "d1_rise_end")  # "now" first: regimes are ordered by it
VISION_FEATURES: Tuple[str, ...] = ("vehicle_count", "other_direction_count", "congestion_ordinal")
REGIME_NAMES: Mapping[int, Tuple[str, ...]] = {
    2: ("light", "heavy"),
    3: ("light", "moderate", "heavy"),
    4: ("light", "moderate", "heavy", "severe"),
    5: ("free_flow", "light", "moderate", "heavy", "severe"),
}

MODELS: Tuple[str, ...] = ("persistence", "seasonal_d1", "seasonal_d7", "seasonal_tod7", "mlp", "fcm_mlp", "fcm_mlp_layer_a")
SERVABLE: Tuple[str, ...] = ("persistence", "fcm_mlp", "fcm_mlp_layer_a")
SERVE_CHAIN: Mapping[str, Tuple[str, ...]] = {  # fcm_mlp_layer_a is followed by its stored fallback
    "fcm_mlp": ("fcm_mlp", "persistence"),
    "persistence": ("persistence",),
}

MIN_TRAIN_ROWS = 288   # two days of origins to fit a network
MIN_INNER_ROWS = 72    # half a day of origins for early stopping
MIN_FIT_DAYS = 3
MIN_VALIDATION_DAYS = 3
MIN_VISION_COVERAGE = 0.5

TABLE_LAYER_A = "layer_a_counts"
TABLE_FORECASTS = "layer_b_fcm_forecasts"
TABLE_EVALUATION = "layer_b_fcm_evaluation"
TABLE_REGISTRY = "layer_b_fcm_registry"

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_NO_FORECAST = 0, 1, 2, 3

_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_DATASET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,1023}$")
_TABLE_REF_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """Invalid configuration, invalid arguments, or too little data to train."""


class LayerAError(RuntimeError):
    """The Layer A service failed or returned an unusable payload."""


class LayerANoFrame(LayerAError):
    """No camera frame exists for the requested time (HTTP 404 from swiftbackend)."""


class NoForecastError(RuntimeError):
    """No direction had fresh enough data to forecast."""


def log(severity: str, message: str, **fields: Any) -> None:
    """Structured log line; Cloud Logging reads ``severity`` and ``message`` from JSON on stdout."""
    record = {"severity": severity, "message": message, "component": "layer_b_fcm_mlp"}
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
    horizon_steps: int = 6
    max_fill_bins: int = 3               # gaps up to 30 minutes are carried forward in the features
    congested_ratio: float = 1.5         # Google congestion ratio above which a zero count is rejected
    layer_a_max_age_bins: int = 2        # a Layer A frame up to 20 minutes before the origin is usable
    fcm_clusters: int = 3
    fcm_fuzzifier: float = 2.0
    fcm_restarts: int = 5
    mlp_hidden: Tuple[int, ...] = (32, 16)
    mlp_alphas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)  # L2 penalties tried; chosen on inner validation
    ensemble_size: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 20
    inner_validation_days: int = 2
    bootstrap_reps: int = 2000
    seed: int = 20261004
    http_timeout_sec: float = 150.0      # swiftbackend can take ~120 s (frame, download, Roboflow)
    protected_window: Tuple[str, str] = ("2026-10-01", "2026-10-19")
    nonwork_days: Tuple[str, ...] = ()

    @property
    def view(self) -> str:
        return self.bins_view or f"{self.project}.traffic_prediction.v_bins_10min"

    @property
    def nonwork_epoch_days(self) -> List[int]:
        return sorted(date_to_epoch_day(parse_sgt_date(d)) for d in self.nonwork_days)

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
                fcm_clusters=int(env.get("FCM_CLUSTERS", cls.fcm_clusters)),
                fcm_fuzzifier=float(env.get("FCM_FUZZIFIER", cls.fcm_fuzzifier)),
                mlp_hidden=_csv_tuple(env["MLP_HIDDEN"], int) if env.get("MLP_HIDDEN") else cls.mlp_hidden,
                ensemble_size=int(env.get("MLP_ENSEMBLE_SIZE", cls.ensemble_size)),
                nonwork_days=_csv_tuple(env.get("NONWORK_DAYS", ""), str),
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
        if not 0.0 <= self.layer_a_confidence <= 1.0:
            raise ConfigError("LAYER_A_CONFIDENCE must be between 0 and 1")
        if not 1 <= self.horizon_steps <= MAX_HORIZON_STEPS:
            raise ConfigError(f"horizon_steps must be between 1 and {MAX_HORIZON_STEPS}")
        if not 0 <= self.max_fill_bins <= 12:
            raise ConfigError("max_fill_bins must be between 0 and 12")
        if self.fcm_clusters not in REGIME_NAMES:
            raise ConfigError(f"FCM_CLUSTERS must be one of {sorted(REGIME_NAMES)}")
        if not 1.1 <= self.fcm_fuzzifier <= 5.0:
            raise ConfigError("FCM_FUZZIFIER must be between 1.1 and 5")
        if self.fcm_restarts < 1:
            raise ConfigError("fcm_restarts must be at least 1")
        if not self.mlp_hidden or len(self.mlp_hidden) > 4 or any(not 1 <= h <= 512 for h in self.mlp_hidden):
            raise ConfigError("MLP_HIDDEN must list 1 to 4 layer widths between 1 and 512")
        if not self.mlp_alphas or any(not (math.isfinite(a) and a > 0) for a in self.mlp_alphas):
            raise ConfigError("mlp_alphas must be positive")
        if not 1 <= self.ensemble_size <= 20:
            raise ConfigError("MLP_ENSEMBLE_SIZE must be between 1 and 20")
        if not 0.0 < self.learning_rate < 1.0:
            raise ConfigError("learning_rate must be between 0 and 1")
        if self.batch_size < 8 or self.max_epochs < 1 or self.patience < 1 or self.inner_validation_days < 1:
            raise ConfigError("batch_size >= 8, max_epochs >= 1, patience >= 1 and inner_validation_days >= 1 are required")
        if self.bootstrap_reps < 100:
            raise ConfigError("bootstrap_reps must be at least 100")
        for day in self.nonwork_days:
            parse_sgt_date(day)
        if any(self.protected_window):
            start, end = (parse_sgt_date(d) for d in self.protected_window)
            if end < start:
                raise ConfigError("PROTECTED_WINDOW end is before its start")


def _csv_tuple(raw: str, cast: Callable[[str], Any]) -> Tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


# ----------------------------------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------------------------------


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def as_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        raise ValueError(f"naive datetime {ts!r}; timestamps must carry a time zone")
    return ts.astimezone(UTC)


def unix(ts: dt.datetime) -> int:
    return int(as_utc(ts).timestamp())


def from_unix(seconds: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(seconds), UTC)


def floor_bin(ts: dt.datetime) -> dt.datetime:
    """Start of the 10-minute bin containing ``ts`` (UTC), as ``v_bins_10min`` computes it."""
    return from_unix(unix(ts) // BIN_SECONDS * BIN_SECONDS)


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


def sgt_epoch_day(unix_seconds: Any) -> Any:
    """Days since 1970-01-01 in Singapore time; accepts an int or a numpy array."""
    return (unix_seconds + SGT_OFFSET_SEC) // 86400


def date_to_epoch_day(day: dt.date) -> int:
    return (day - EPOCH_DATE).days


def epoch_day_to_date(day: int) -> dt.date:
    return EPOCH_DATE + dt.timedelta(days=int(day))


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
    # labels are capped at the window's last bin (build_frame: label_cutoff), so origins and
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
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
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


def _num(value: Any) -> Optional[float]:
    """A finite float, or None for missing / NaN / infinite values."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _canon(values: Any) -> np.ndarray:
    """Round to 8 significant digits, the precision stored in the registry.

    Models are rounded right after fitting, so the weights that are evaluated are exactly the
    weights that are stored and served (predictions then agree to floating-point rounding).
    """
    arr = np.asarray(values, dtype=float)
    flat = [float(f"{v:.8g}") for v in arr.ravel().tolist()]
    return np.array(flat, dtype=float).reshape(arr.shape)


SCHEMAS: Dict[str, List[Tuple[str, str, str]]] = {
    TABLE_LAYER_A: [  # identical to layer_b_timesfm.py: the table is shared
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
        ("seasonal_d1", "FLOAT64", "NULLABLE"),
        ("seasonal_d7", "FLOAT64", "NULLABLE"),
        ("fcm_mlp", "FLOAT64", "NULLABLE"),
        ("fcm_mlp_spread", "FLOAT64", "NULLABLE"),
        ("fcm_mlp_layer_a", "FLOAT64", "NULLABLE"),
        ("served_model", "STRING", "REQUIRED"),
        ("served_value", "FLOAT64", "NULLABLE"),
        ("regime", "STRING", "NULLABLE"),
        ("regime_degree", "FLOAT64", "NULLABLE"),
        ("memberships_json", "STRING", "NULLABLE"),
        ("forecast_regime", "STRING", "NULLABLE"),
        ("forecast_regime_degree", "FLOAT64", "NULLABLE"),
        ("layer_a_status", "STRING", "REQUIRED"),
        ("registry_run_id", "STRING", "NULLABLE"),
        ("model", "STRING", "REQUIRED"),
    ],
    TABLE_EVALUATION: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("split", "STRING", "REQUIRED"),
        ("origin_ts", "TIMESTAMP", "REQUIRED"),
        ("direction", "STRING", "REQUIRED"),
        ("step", "INT64", "REQUIRED"),
        ("target_ts", "TIMESTAMP", "REQUIRED"),
        ("actual", "FLOAT64", "NULLABLE"),
        ("persistence", "FLOAT64", "NULLABLE"),
        ("seasonal_d1", "FLOAT64", "NULLABLE"),
        ("seasonal_d7", "FLOAT64", "NULLABLE"),
        ("seasonal_tod7", "FLOAT64", "NULLABLE"),
        ("mlp", "FLOAT64", "NULLABLE"),
        ("fcm_mlp", "FLOAT64", "NULLABLE"),
        ("fcm_mlp_layer_a", "FLOAT64", "NULLABLE"),
        ("served_model", "STRING", "NULLABLE"),
        ("served_value", "FLOAT64", "NULLABLE"),
        ("regime", "STRING", "NULLABLE"),
        ("layer_a_status", "STRING", "REQUIRED"),
        ("model", "STRING", "REQUIRED"),
    ],
    TABLE_REGISTRY: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("model", "STRING", "REQUIRED"),
        ("train_start", "DATE", "REQUIRED"),
        ("train_end", "DATE", "REQUIRED"),
        ("validation_start", "DATE", "REQUIRED"),
        ("validation_end", "DATE", "REQUIRED"),
        ("test_start", "DATE", "NULLABLE"),
        ("test_end", "DATE", "NULLABLE"),
        ("config_json", "STRING", "REQUIRED"),
        ("decisions_json", "STRING", "REQUIRED"),
        ("artefacts_json", "STRING", "REQUIRED"),
        ("metrics_json", "STRING", "REQUIRED"),
    ],
}

PARTITIONING: Dict[str, Tuple[Optional[str], List[str]]] = {
    TABLE_LAYER_A: ("bin_ts", ["direction", "bin_ts"]),
    TABLE_FORECASTS: ("origin_ts", ["direction", "origin_ts"]),
    TABLE_EVALUATION: ("origin_ts", ["run_id", "direction"]),
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
            labels={"component": "layer-b-fcm-mlp"},
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
# SQL builders
# ----------------------------------------------------------------------------------------------


def bins_sql(cfg: Config) -> str:
    """Raw 10-minute travel-time bins; all feature engineering happens in Python (build_frame)."""
    return f"""
SELECT direction, bin_ts, dur_min
FROM `{cfg.view}`
WHERE direction IN UNNEST(@directions)
  AND dur_min IS NOT NULL
  AND bin_ts BETWEEN @start AND @end
ORDER BY direction, bin_ts
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
SELECT run_id, created_at, config_json, decisions_json, artefacts_json
FROM `{cfg.table(TABLE_REGISTRY)}`
WHERE model = @model
ORDER BY created_at DESC
LIMIT 1
""".strip()


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


# ----------------------------------------------------------------------------------------------
# Series and features
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Grid:
    """Regular 10-minute grid: index ``i`` is the bin that starts ``10 i`` minutes after ``start``."""

    start_unix: int
    n: int

    @classmethod
    def spanning(cls, first: dt.datetime, last: dt.datetime) -> Grid:
        a, b = unix(floor_bin(first)), unix(floor_bin(last))
        if b < a:
            raise ConfigError("grid end is before its start")
        return cls(a, (b - a) // BIN_SECONDS + 1)

    def index(self, ts: dt.datetime) -> int:
        offset = unix(ts) - self.start_unix
        if offset % BIN_SECONDS or not 0 <= offset // BIN_SECONDS < self.n:
            raise ValueError(f"{ts!r} is not a bin of this grid")
        return offset // BIN_SECONDS

    def time(self, i: int) -> dt.datetime:
        return from_unix(self.start_unix + int(i) * BIN_SECONDS)


def series_from_rows(rows: Sequence[Mapping[str, Any]], grid: Grid) -> Dict[str, np.ndarray]:
    """One array per direction on the grid; NaN marks a bin without a travel time."""
    series = {d: np.full(grid.n, np.nan) for d in DIRECTIONS}
    for row in rows:
        direction, value = row.get("direction"), row.get("dur_min")
        if direction not in series or value is None:
            continue
        offset = unix(row["bin_ts"]) - grid.start_unix
        if offset % BIN_SECONDS or not 0 <= offset < grid.n * BIN_SECONDS:
            continue
        number = float(value)
        if math.isfinite(number) and number > 0:
            series[direction][offset // BIN_SECONDS] = number
    return series


def load_series(cfg: Config, wh: Warehouse, first: dt.datetime, last: dt.datetime) -> Tuple[Grid, Dict[str, np.ndarray]]:
    grid = Grid.spanning(first, last)
    rows = wh.query(bins_sql(cfg), {"directions": list(DIRECTIONS), "start": grid.time(0), "end": grid.time(grid.n - 1)})
    return grid, series_from_rows(rows, grid)


def history_start(cfg: Config, first_origin: dt.datetime) -> dt.datetime:
    """Earliest bin the features of ``first_origin`` can read (one week back, plus the fill limit)."""
    return as_utc(first_origin) - (BINS_PER_WEEK + cfg.max_fill_bins + 1) * BIN


def ffill_limited(x: np.ndarray, limit: int) -> np.ndarray:
    """Carry each value forward over at most ``limit`` missing bins; longer gaps stay missing."""
    n = len(x)
    if n == 0:
        return x.copy()
    positions = np.arange(n)
    last_seen = np.maximum.accumulate(np.where(np.isnan(x), -1, positions))
    out = np.full(n, np.nan)
    ok = (last_seen >= 0) & (positions - last_seen <= limit)
    out[ok] = x[last_seen[ok]]
    return out


def _take(arr: np.ndarray, positions: Any) -> np.ndarray:
    """``arr[positions]`` with NaN for positions outside the array."""
    pos = np.asarray(positions, dtype=np.int64)
    out = np.full(pos.shape, np.nan)
    ok = (pos >= 0) & (pos < len(arr))
    out[ok] = arr[pos[ok]]
    return out


@dataclasses.dataclass
class Frame:
    """Features, labels and baselines for a set of forecast origins of one direction."""

    direction: str
    origin_unix: np.ndarray      # (n,) int64 seconds
    X: np.ndarray                # (n, len(BASE_FEATURES)); NaN rows where complete is False
    complete: np.ndarray         # (n,) bool: every base feature is available
    now: np.ndarray              # (n,) travel time at the origin, i.e. persistence
    actual: np.ndarray           # (n, H) label per step; NaN if missing or after the label cutoff
    seasonal_d1: np.ndarray      # (n, H) same bin one day earlier
    seasonal_d7: np.ndarray      # (n, H) same bin seven days earlier
    seasonal_tod7: np.ndarray    # (n, H) mean of the same time of day over the previous seven days
    vision: np.ndarray           # (n, len(VISION_FEATURES)); NaN without an accepted frame
    vision_status: List[str]

    @property
    def n(self) -> int:
        return len(self.origin_unix)

    @property
    def horizon(self) -> int:
        return int(self.actual.shape[1])

    @property
    def origin_day(self) -> np.ndarray:
        return sgt_epoch_day(self.origin_unix)

    def target_day(self) -> np.ndarray:
        steps = np.arange(1, self.horizon + 1, dtype=np.int64)
        return sgt_epoch_day(self.origin_unix[:, None] + BIN_SECONDS * steps[None, :])

    def origin_ts(self, i: int) -> dt.datetime:
        return from_unix(int(self.origin_unix[i]))


def build_frame(cfg: Config, grid: Grid, raw: np.ndarray, direction: str, origin_idx: Any, *,
                label_cutoff: dt.datetime) -> Frame:
    """Causal features at every origin index; labels after ``label_cutoff`` are masked.

    Features read only bins at or before the origin. Lags and the rolling hour use values carried
    forward over gaps of up to ``max_fill_bins``; the origin's own bin must be observed. The D-1
    and D-7 features describe the same clock time one day and one week earlier, as the gap between
    then and now and as the rise over the coming half horizon and full horizon; when a reference
    day is missing its features are zero and a flag is set. Baselines use raw bins only.
    """
    i = np.asarray(origin_idx, dtype=np.int64)
    horizon = cfg.horizon_steps
    mid = horizon // 2
    filled = ffill_limited(raw, cfg.max_fill_bins)
    now = _take(raw, i)
    lags = {k: _take(filled, i - k) for k in LAG_STEPS}
    window = np.stack([_take(filled, i - j) for j in range(ROLL_BINS)], axis=1)
    roll_ok = ~np.isnan(window).any(axis=1)
    safe = np.where(roll_ok[:, None], window, 0.0)
    roll_mean = np.where(roll_ok, safe.mean(axis=1), np.nan)
    roll_std = np.where(roll_ok, safe.std(axis=1), np.nan)

    def seasonal(back: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base, at_mid, at_end = (_take(filled, i - back + s) for s in (0, mid, horizon))
        missing = np.isnan(base) | np.isnan(at_mid) | np.isnan(at_end)
        return (np.where(missing, 0.0, now - base), np.where(missing, 0.0, at_mid - base),
                np.where(missing, 0.0, at_end - base), missing.astype(float))

    d1 = seasonal(BINS_PER_DAY)
    d7 = seasonal(BINS_PER_WEEK)
    origin_unix = grid.start_unix + i * BIN_SECONDS
    local = origin_unix + SGT_OFFSET_SEC
    tod = 2.0 * math.pi * (local % 86400) / 86400.0
    day = local // 86400
    dow = (day + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    nonwork = (dow >= 5) | np.isin(day, cfg.nonwork_epoch_days)
    columns = {
        "now": now,
        "delta_10": now - lags[1],
        "delta_20": now - lags[2],
        "delta_30": now - lags[3],
        "delta_60": now - lags[6],
        "roll_mean_60_minus_now": roll_mean - now,
        "roll_std_60": roll_std,
        "d1_gap": d1[0], "d1_rise_mid": d1[1], "d1_rise_end": d1[2], "d1_missing": d1[3],
        "d7_gap": d7[0], "d7_rise_mid": d7[1], "d7_rise_end": d7[2], "d7_missing": d7[3],
        "tod_sin": np.sin(tod), "tod_cos": np.cos(tod),
        "dow_sin": np.sin(2.0 * math.pi * dow / 7.0), "dow_cos": np.cos(2.0 * math.pi * dow / 7.0),
        "nonworkday": nonwork.astype(float),
    }
    X = np.column_stack([np.asarray(columns[name], dtype=float) for name in BASE_FEATURES])
    complete = np.isfinite(X).all(axis=1)
    X[~complete] = np.nan

    steps = np.arange(1, horizon + 1, dtype=np.int64)
    target = i[:, None] + steps[None, :]
    actual = _take(raw, target)
    actual[grid.start_unix + target * BIN_SECONDS > unix(label_cutoff)] = np.nan
    previous = np.stack([_take(raw, target - j * BINS_PER_DAY) for j in range(1, 8)], axis=0)
    seen = (~np.isnan(previous)).sum(axis=0)
    tod7 = np.where(seen > 0, np.nansum(previous, axis=0) / np.maximum(seen, 1), np.nan)
    return Frame(
        direction=direction,
        origin_unix=origin_unix,
        X=X,
        complete=complete,
        now=now,
        actual=actual,
        seasonal_d1=_take(raw, target - BINS_PER_DAY),
        seasonal_d7=_take(raw, target - BINS_PER_WEEK),
        seasonal_tod7=tod7,
        vision=np.full((len(i), len(VISION_FEATURES)), np.nan),
        vision_status=["missing"] * len(i),
    )


def attach_vision(cfg: Config, frame: Frame, layer_a: Mapping[Tuple[str, dt.datetime], Mapping[str, Any]]) -> None:
    if not layer_a:
        return
    for i in range(frame.n):
        feats, status = vision_features(cfg, layer_a, frame.direction, frame.origin_ts(i))
        frame.vision_status[i] = status
        if feats is not None:
            frame.vision[i] = [feats[name] for name in VISION_FEATURES]


# ----------------------------------------------------------------------------------------------
# Fuzzy C-Means
# ----------------------------------------------------------------------------------------------


def _sq_dist(z: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.asarray(((z[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))


def _memberships_from_sq_dist(d2: np.ndarray, m: float) -> np.ndarray:
    """FCM membership u_ij = 1 / sum_k (d_ij / d_ik)^(2 / (m - 1)), computed without overflow."""
    d2 = np.maximum(d2, 1e-12)  # a point on a centre gets (almost) full membership of it
    ratio = d2 / d2.min(axis=1, keepdims=True)  # >= 1, so the negative power cannot overflow
    weights = ratio ** (-1.0 / (m - 1.0))
    return np.asarray(weights / weights.sum(axis=1, keepdims=True))


def _kmeans_plus_plus(z: np.ndarray, c: int, rng: np.random.Generator) -> np.ndarray:
    centres = [z[rng.integers(len(z))]]
    for _ in range(1, c):
        d2 = _sq_dist(z, np.asarray(centres)).min(axis=1)
        total = float(d2.sum())
        idx = int(rng.integers(len(z))) if total <= 0 else int(rng.choice(len(z), p=d2 / total))
        centres.append(z[idx])
    return np.asarray(centres, dtype=float)


@dataclasses.dataclass
class FuzzyCMeans:
    """Fuzzy C-Means on standardised features; centres ordered by the first feature (travel time)."""

    features: Tuple[str, ...]
    labels: Tuple[str, ...]
    m: float
    mean: np.ndarray
    scale: np.ndarray
    centres: np.ndarray  # (c, p) in standardised units
    n: int
    objective: float
    iterations: int

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.scale = np.asarray(self.scale, dtype=float)
        self.centres = np.asarray(self.centres, dtype=float)
        p, c = len(self.features), len(self.labels)
        if self.mean.shape != (p,) or self.scale.shape != (p,) or self.centres.shape != (c, p):
            raise ValueError("FCM parameters do not match the feature and regime lists")
        if c < 2 or not self.m > 1.0:
            raise ValueError("FCM needs at least two regimes and a fuzzifier above 1")
        if not (np.isfinite(self.mean).all() and np.isfinite(self.centres).all() and (self.scale > 0).all()):
            raise ValueError("FCM parameters must be finite with positive scales")

    @classmethod
    def fit(cls, x: np.ndarray, *, features: Sequence[str], c: int, m: float, seed: int, restarts: int = 5,
            max_iter: int = 300, tol: float = 1e-6) -> FuzzyCMeans:
        """Bezdek's alternating optimisation, best of ``restarts`` k-means++ starts by objective J_m."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(features):
            raise ValueError("FCM input does not match the feature list")
        if len(x) < 10 * c or not np.isfinite(x).all():
            raise ValueError("FCM needs at least ten finite rows per regime")
        mean = _canon(x.mean(axis=0))
        std = x.std(axis=0)
        scale = _canon(np.where(std > 1e-9, std, 1.0))
        z = (x - mean) / scale
        rng = np.random.default_rng(seed)
        best: Optional[Tuple[float, np.ndarray, int]] = None
        for _ in range(restarts):
            centres = _kmeans_plus_plus(z, c, rng)
            u = _memberships_from_sq_dist(_sq_dist(z, centres), m)
            iterations = 0
            for iterations in range(1, max_iter + 1):  # noqa: B007 - the final count is reported
                um = u ** m
                centres = (um.T @ z) / um.sum(axis=0)[:, None]
                u_next = _memberships_from_sq_dist(_sq_dist(z, centres), m)
                shift = float(np.abs(u_next - u).max())
                u = u_next
                if shift < tol:
                    break
            objective = float(((u ** m) * _sq_dist(z, centres)).sum())
            if best is None or objective < best[0]:
                best = (objective, centres, iterations)
        assert best is not None  # restarts >= 1
        objective, centres, iterations = best
        order = np.argsort(centres[:, 0], kind="stable")
        return cls(tuple(features), REGIME_NAMES[c], float(m), mean, scale, _canon(centres[order]), len(x), objective, iterations)

    def memberships(self, x: np.ndarray, *, saturate: bool = True) -> np.ndarray:
        """(n, c) membership degrees; each row sums to one. Rows with missing inputs give NaN.

        Plain FCM degrees drift towards 1/c for points far outside the centres, so a 120-minute
        jam would look less "heavy" than a 40-minute one. With ``saturate`` (the default, used for
        both training and serving) each coordinate is first clipped to the range spanned by the
        centres, which gives the outer regimes shoulder-shaped memberships, as in a Ruspini
        partition. ``saturate=False`` gives the textbook degrees used by the validity indices.
        """
        x = np.asarray(x, dtype=float)
        out = np.full((len(x), len(self.labels)), np.nan)
        ok = np.isfinite(x).all(axis=1)
        if ok.any():
            z = (x[ok] - self.mean) / self.scale
            if saturate:
                z = np.clip(z, self.centres.min(axis=0), self.centres.max(axis=0))
            out[ok] = _memberships_from_sq_dist(_sq_dist(z, self.centres), self.m)
        return out

    def centres_in_units(self) -> np.ndarray:
        return np.asarray(self.centres * self.scale + self.mean)

    def level_memberships(self, minutes: np.ndarray) -> np.ndarray:
        """Membership of travel times in each regime, using the centres' travel-time coordinate only.

        This is the projection of the clustering onto the travel-time axis; it labels a forecast
        (for which no trend exists yet) as, for example, 0.7 heavy and 0.3 moderate.
        """
        levels = self.centres_in_units()[:, 0]
        values = np.clip(np.asarray(minutes, dtype=float).reshape(-1), levels.min(), levels.max())  # shoulders
        return _memberships_from_sq_dist((values[:, None] - levels[None, :]) ** 2, self.m)

    def validity(self, x: np.ndarray) -> Dict[str, Any]:
        """Partition coefficient and entropy (Bezdek) and the Xie-Beni index (lower is better)."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x).all(axis=1)]
        if not len(x):
            return {"n": 0}
        u = self.memberships(x, saturate=False)
        z = (x - self.mean) / self.scale
        separation = min(float(((self.centres[a] - self.centres[b]) ** 2).sum())
                         for a in range(len(self.labels)) for b in range(a + 1, len(self.labels)))
        compactness = float(((u ** self.m) * _sq_dist(z, self.centres)).sum())
        sizes = np.bincount(u.argmax(axis=1), minlength=len(self.labels))
        return {
            "n": len(x),
            "partition_coefficient": float((u ** 2).sum(axis=1).mean()),
            "partition_entropy": float(-(u * np.log(np.maximum(u, 1e-300))).sum(axis=1).mean()),
            "xie_beni": compactness / (len(x) * separation) if separation > 0 else None,
            "crisp_sizes": {label: int(s) for label, s in zip(self.labels, sizes)},
        }

    def describe(self) -> Dict[str, Dict[str, float]]:
        """Regime centres in original units, for the report and for reviewers."""
        units = self.centres_in_units()
        return {label: {f: round(float(v), 3) for f, v in zip(self.features, row)} for label, row in zip(self.labels, units)}

    def to_json(self) -> Dict[str, Any]:
        return {
            "features": list(self.features), "labels": list(self.labels), "m": self.m,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(), "centres": self.centres.tolist(),
            "n": self.n, "objective": self.objective, "iterations": self.iterations,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> FuzzyCMeans:
        return cls(tuple(data["features"]), tuple(data["labels"]), float(data["m"]), np.asarray(data["mean"], dtype=float),
                   np.asarray(data["scale"], dtype=float), np.asarray(data["centres"], dtype=float), int(data["n"]),
                   float(data["objective"]), int(data["iterations"]))


# ----------------------------------------------------------------------------------------------
# Multilayer perceptron (seed ensemble)
# ----------------------------------------------------------------------------------------------


def _forward(z: np.ndarray, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> np.ndarray:
    """ReLU hidden layers and a linear output, as scikit-learn's MLPRegressor computes them."""
    a = z
    for w, b in zip(weights[:-1], biases[:-1]):
        a = np.maximum(a @ w + b, 0.0)
    return np.asarray(a @ weights[-1] + biases[-1])


@dataclasses.dataclass
class MLPEnsemble:
    """Networks that differ only in their seed, predicting the change in travel time per step."""

    features: Tuple[str, ...]
    hidden: Tuple[int, ...]
    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    floor: float  # shortest travel time seen in training; forecasts are not allowed below it
    members: List[Tuple[List[np.ndarray], List[np.ndarray]]]
    best_epochs: List[int]
    n_train: int

    def __post_init__(self) -> None:
        self.x_mean, self.x_scale = np.asarray(self.x_mean, dtype=float), np.asarray(self.x_scale, dtype=float)
        self.y_mean, self.y_scale = np.asarray(self.y_mean, dtype=float), np.asarray(self.y_scale, dtype=float)
        p, horizon = len(self.features), len(self.y_mean)
        if self.x_mean.shape != (p,) or self.x_scale.shape != (p,) or self.y_scale.shape != (horizon,) or horizon < 1:
            raise ValueError("network scaling does not match the feature list or horizon")
        if not ((self.x_scale > 0).all() and (self.y_scale > 0).all() and math.isfinite(self.floor)):
            raise ValueError("network scales must be positive and the floor finite")
        if not self.members or len(self.best_epochs) != len(self.members):
            raise ValueError("an ensemble needs at least one member and one best epoch per member")
        sizes = [p, *self.hidden, horizon]
        checked = []
        for weights, biases in self.members:
            ws = [np.asarray(w, dtype=float) for w in weights]
            bs = [np.asarray(b, dtype=float) for b in biases]
            if len(ws) != len(sizes) - 1 or len(bs) != len(ws):
                raise ValueError("network depth does not match the hidden layer list")
            for j, (w, b) in enumerate(zip(ws, bs)):
                if w.shape != (sizes[j], sizes[j + 1]) or b.shape != (sizes[j + 1],):
                    raise ValueError(f"layer {j} has the wrong shape")
                if not (np.isfinite(w).all() and np.isfinite(b).all()):
                    raise ValueError(f"layer {j} has non-finite weights")
            checked.append((ws, bs))
        self.members = checked

    @property
    def horizon(self) -> int:
        return len(self.y_mean)

    def member_forecasts(self, X: np.ndarray, now: np.ndarray) -> np.ndarray:
        """(members, n, H) travel-time forecasts in minutes."""
        z = (np.asarray(X, dtype=float) - self.x_mean) / self.x_scale
        deltas = np.stack([_forward(z, w, b) for w, b in self.members]) * self.y_scale + self.y_mean
        return np.maximum(np.asarray(now, dtype=float)[None, :, None] + deltas, self.floor)

    def predict(self, X: np.ndarray, now: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble mean and spread (standard deviation across members), both (n, H)."""
        forecasts = self.member_forecasts(X, now)
        return forecasts.mean(axis=0), forecasts.std(axis=0)

    def to_json(self) -> Dict[str, Any]:
        return {
            "features": list(self.features), "hidden": list(self.hidden), "alpha": self.alpha,
            "x_mean": self.x_mean.tolist(), "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean.tolist(), "y_scale": self.y_scale.tolist(), "floor": self.floor,
            "members": [{"weights": [w.tolist() for w in ws], "biases": [b.tolist() for b in bs]} for ws, bs in self.members],
            "best_epochs": list(self.best_epochs), "n_train": self.n_train,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MLPEnsemble:
        members = [([np.asarray(w, dtype=float) for w in mem["weights"]], [np.asarray(b, dtype=float) for b in mem["biases"]])
                   for mem in data["members"]]
        return cls(tuple(data["features"]), tuple(int(h) for h in data["hidden"]), float(data["alpha"]),
                   np.asarray(data["x_mean"], dtype=float), np.asarray(data["x_scale"], dtype=float),
                   np.asarray(data["y_mean"], dtype=float), np.asarray(data["y_scale"], dtype=float),
                   float(data["floor"]), members, [int(e) for e in data["best_epochs"]], int(data["n_train"]))


def _scaling(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    std = a.std(axis=0)
    return _canon(a.mean(axis=0)), _canon(np.where(std > 1e-9, std, 1.0))


def fit_mlp_ensemble(cfg: Config, features: Sequence[str], *, x_fit: np.ndarray, y_fit: np.ndarray, x_inner: np.ndarray,
                     y_inner: np.ndarray, now_fit: np.ndarray, alpha: float, seed: int) -> Tuple[MLPEnsemble, float]:
    """Train ``cfg.ensemble_size`` networks, each kept at its best epoch on the inner-validation days.

    ``y_*`` hold the change in travel time (minutes) at each step. Returns the ensemble and the
    inner-validation MAE (minutes) of its mean prediction, used to choose the L2 penalty.
    """
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415 - training only; serving needs numpy alone

    x_mean, x_scale = _scaling(x_fit)
    y_mean, y_scale = _scaling(y_fit)
    z_fit, z_inner = (x_fit - x_mean) / x_scale, (x_inner - x_mean) / x_scale
    t_fit, t_inner = (y_fit - y_mean) / y_scale, (y_inner - y_mean) / y_scale
    target = t_fit if t_fit.shape[1] > 1 else t_fit[:, 0]  # a single output must be 1-D for scikit-learn
    members: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
    epochs: List[int] = []
    for k in range(cfg.ensemble_size):
        # a RandomState object (not an int) so each partial_fit call draws a new minibatch order;
        # with an int, scikit-learn would re-seed on every call and repeat the same order each epoch
        net = MLPRegressor(hidden_layer_sizes=cfg.mlp_hidden, activation="relu", solver="adam", alpha=alpha,
                           batch_size=int(min(cfg.batch_size, len(z_fit))), learning_rate_init=cfg.learning_rate,
                           shuffle=True, random_state=np.random.RandomState(seed + k))
        best_score, best_epoch, stale = math.inf, 0, 0
        best_state: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None
        for epoch in range(1, cfg.max_epochs + 1):
            net.partial_fit(z_fit, target)  # one shuffled pass over the data
            weights = [np.asarray(w, dtype=float) for w in net.coefs_]
            biases = [np.asarray(b, dtype=float) for b in net.intercepts_]
            score = float(np.mean(np.abs(_forward(z_inner, weights, biases) - t_inner) * y_scale))
            if score < best_score - 1e-9:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = ([w.copy() for w in weights], [b.copy() for b in biases])
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        assert best_state is not None  # max_epochs >= 1, and the first epoch always improves on inf
        members.append(([_canon(w) for w in best_state[0]], [_canon(b) for b in best_state[1]]))
        epochs.append(best_epoch)
    ensemble = MLPEnsemble(tuple(features), tuple(cfg.mlp_hidden), float(alpha), x_mean, x_scale, y_mean, y_scale,
                           float(_canon(np.min(now_fit))), members, epochs, len(x_fit))
    inner_deltas = np.mean([_forward(z_inner, w, b) for w, b in ensemble.members], axis=0) * y_scale + y_mean
    return ensemble, float(np.mean(np.abs(inner_deltas - y_inner)))


def fuzzy_feature_names(labels: Sequence[str]) -> Tuple[str, ...]:
    return BASE_FEATURES + tuple(f"mu_{label}" for label in labels)


@dataclasses.dataclass
class DirectionModels:
    """Everything the live cycle needs for one direction."""

    fcm: FuzzyCMeans
    fcm_mlp: MLPEnsemble
    fcm_mlp_layer_a: Optional[MLPEnsemble] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "fcm": self.fcm.to_json(),
            "fcm_mlp": self.fcm_mlp.to_json(),
            "fcm_mlp_layer_a": self.fcm_mlp_layer_a.to_json() if self.fcm_mlp_layer_a else None,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], horizon: int) -> DirectionModels:
        """Rebuild and check that the stored model matches the features this code computes."""
        fcm = FuzzyCMeans.from_json(data["fcm"])
        if fcm.features != FCM_FEATURES:
            raise ValueError(f"stored FCM uses features {fcm.features}, this code computes {FCM_FEATURES}")
        names = fuzzy_feature_names(fcm.labels)
        fcm_mlp = MLPEnsemble.from_json(data["fcm_mlp"])
        layer_a = MLPEnsemble.from_json(data["fcm_mlp_layer_a"]) if data.get("fcm_mlp_layer_a") else None
        for model, expected in ((fcm_mlp, names), (layer_a, names + VISION_FEATURES)):
            if model is None:
                continue
            if model.features != expected:
                raise ValueError("stored network features differ from the features this code computes")
            if model.horizon != horizon:
                raise ValueError(f"stored network forecasts {model.horizon} steps, configuration expects {horizon}")
        return cls(fcm, fcm_mlp, layer_a)


def fcm_inputs(frame: Frame, fcm: FuzzyCMeans) -> np.ndarray:
    columns = [BASE_FEATURES.index(f) for f in fcm.features]
    out = np.full((frame.n, len(fcm.labels)), np.nan)
    if frame.complete.any():
        out[frame.complete] = fcm.memberships(frame.X[frame.complete][:, columns])
    return out


def _predict_rows(model: MLPEnsemble, X: np.ndarray, now: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.full((len(X), model.horizon), np.nan)
    spread = np.full((len(X), model.horizon), np.nan)
    if mask.any():
        mean[mask], spread[mask] = model.predict(X[mask], now[mask])
    return mean, spread


def predict_frame(models: DirectionModels, frame: Frame) -> Dict[str, np.ndarray]:
    """Memberships and network forecasts for every complete origin of ``frame``."""
    mu = fcm_inputs(frame, models.fcm)
    x_fuzzy = np.hstack([frame.X, mu])
    fcm_mlp, spread = _predict_rows(models.fcm_mlp, x_fuzzy, frame.now, frame.complete)
    layer_a = np.full_like(fcm_mlp, np.nan)
    if models.fcm_mlp_layer_a is not None:
        with_vision = frame.complete & np.isfinite(frame.vision).all(axis=1)
        layer_a, _ = _predict_rows(models.fcm_mlp_layer_a, np.hstack([x_fuzzy, frame.vision]), frame.now, with_vision)
    return {"memberships": mu, "fcm_mlp": fcm_mlp, "fcm_mlp_spread": spread, "fcm_mlp_layer_a": layer_a}


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
    """Metrics per model, step, direction, period and day type, with support counts (slide 15)."""
    out: Dict[str, Any] = {}
    for step in steps:
        step_rows = [r for r in rows if r["step"] == step and r.get("actual") is not None]
        groups: Dict[str, List[Mapping[str, Any]]] = {"all": step_rows}
        for d in DIRECTIONS:
            groups[d] = [r for r in step_rows if r["direction"] == d]
        for key in ("morning_peak", "evening_peak", "off_peak"):
            groups[key] = [r for r in step_rows if period_of(r["target_ts"]) == key]
        for key in ("weekday", "weekend"):
            groups[key] = [r for r in step_rows if day_type_of(r["target_ts"]) == key]
        for model in models:
            out.setdefault(f"step_{step}", {})[model] = {
                name: error_metrics([float(r[model]) - float(r["actual"]) for r in subset if r.get(model) is not None])
                for name, subset in groups.items()
            }
    return out


def compare(rows: Sequence[Mapping[str, Any]], challenger: str, reference: str, cfg: Config) -> Dict[str, Any]:
    pairs = [
        (sgt_date(r["origin_ts"]), abs(float(r[challenger]) - float(r["actual"])) - abs(float(r[reference]) - float(r["actual"])))
        for r in rows
        if r.get("actual") is not None and r.get(challenger) is not None and r.get(reference) is not None
    ]
    return paired_day_bootstrap(pairs, cfg.bootstrap_reps, cfg.seed)


def served_value(row: Mapping[str, Any], decision: str, fallback: Optional[str] = None) -> Tuple[str, Optional[float]]:
    """Value for a registry decision, falling back when an input is missing.

    Without a live Layer A frame, ``fcm_mlp_layer_a`` falls back to ``fallback``: ``fcm_mlp`` only
    if that model beat persistence on its own, otherwise persistence. A model that failed the
    persistence gate is therefore never served, whichever input is missing.
    """
    if not isinstance(fallback, str) or fallback not in SERVE_CHAIN:
        fallback = "persistence"  # missing or malformed in the registry: the conservative choice
    if decision == "fcm_mlp_layer_a":
        chain: Tuple[str, ...] = ("fcm_mlp_layer_a", *SERVE_CHAIN[fallback])
    else:
        chain = SERVE_CHAIN.get(decision, ("persistence",)) if isinstance(decision, str) else ("persistence",)
    for model in chain:
        if row.get(model) is not None:
            return model, float(row[model])
    return "persistence", None


# ----------------------------------------------------------------------------------------------
# Training: day-disjoint splits, fitting, decision rules and the one-off test
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DaySplit:
    """Time-ordered, day-disjoint split of SGT epoch days.

    Rank 0: fit the networks; 1: inner validation (early stopping, L2 penalty); 2: validation
    (decision rules); 3: test (scored once).
    """

    fit_days: Tuple[int, ...]
    inner_days: Tuple[int, ...]
    validation_days: Tuple[int, ...]
    test_days: Tuple[int, ...]

    def rank(self, day: Any) -> np.ndarray:
        day = np.asarray(day)
        out = np.zeros(day.shape, dtype=np.int64)
        thresholds = [(self.inner_days[0], 1), (self.validation_days[0], 2)]
        if self.test_days:
            thresholds.append((self.test_days[0], 3))
        for first_day, value in thresholds:
            out = np.where(day >= first_day, value, out)
        return out


def split_days(days: Sequence[int], *, validation_days: int, test_days: int, inner_days: int) -> DaySplit:
    ordered = sorted({int(d) for d in days})
    if validation_days < MIN_VALIDATION_DAYS:
        raise ConfigError(f"--validation-days must be at least {MIN_VALIDATION_DAYS}")
    if test_days and test_days < MIN_VALIDATION_DAYS:
        raise ConfigError(f"--test-days must be 0 or at least {MIN_VALIDATION_DAYS}")
    n_train = len(ordered) - validation_days - test_days
    if n_train < inner_days + MIN_FIT_DAYS:
        raise ConfigError(
            f"the window has {len(ordered)} days with data; {validation_days} validation + {test_days} test days "
            f"leave {n_train} training days, fewer than the {inner_days + MIN_FIT_DAYS} needed"
        )
    train = ordered[:n_train]
    return DaySplit(tuple(train[:-inner_days]), tuple(train[-inner_days:]),
                    tuple(ordered[n_train:n_train + validation_days]), tuple(ordered[n_train + validation_days:]))


def boundary_masked_labels(frame: Frame, split: DaySplit) -> np.ndarray:
    """Labels whose target falls on a later split's day are set to NaN (no learning across a boundary)."""
    origin_rank = split.rank(frame.origin_day)
    target_rank = split.rank(frame.target_day())
    labels = frame.actual.copy()
    labels[target_rank > origin_rank[:, None]] = np.nan
    return labels


def train_direction(cfg: Config, frame: Frame, split: DaySplit, direction_index: int) -> Dict[str, Any]:
    """Fit FCM and the three networks for one direction and predict every complete origin."""
    labels = boundary_masked_labels(frame, split)
    rank = split.rank(frame.origin_day)
    labelled = np.isfinite(labels).all(axis=1)
    fit_rows = frame.complete & labelled & (rank == 0)
    inner_rows = frame.complete & labelled & (rank == 1)
    if fit_rows.sum() < MIN_TRAIN_ROWS or inner_rows.sum() < MIN_INNER_ROWS:
        raise ConfigError(
            f"{frame.direction}: {int(fit_rows.sum())} fitting and {int(inner_rows.sum())} inner-validation rows; "
            f"need at least {MIN_TRAIN_ROWS} and {MIN_INNER_ROWS}"
        )
    seed = cfg.seed + 1000 * direction_index
    fcm_columns = [BASE_FEATURES.index(f) for f in FCM_FEATURES]
    fcm_rows = frame.complete & (rank <= 1)  # unsupervised: all training days, labels not needed
    fcm = FuzzyCMeans.fit(frame.X[fcm_rows][:, fcm_columns], features=FCM_FEATURES, c=cfg.fcm_clusters,
                          m=cfg.fcm_fuzzifier, seed=seed, restarts=cfg.fcm_restarts)
    mu = fcm_inputs(frame, fcm)
    names = fuzzy_feature_names(fcm.labels)
    x_fuzzy = np.hstack([frame.X, mu])
    deltas = labels - frame.now[:, None]

    def fit(x: np.ndarray, feature_names: Sequence[str], rows_fit: np.ndarray, rows_inner: np.ndarray,
            alpha: float) -> Tuple[MLPEnsemble, float]:
        return fit_mlp_ensemble(cfg, feature_names, x_fit=x[rows_fit], y_fit=deltas[rows_fit], x_inner=x[rows_inner],
                                y_inner=deltas[rows_inner], now_fit=frame.now[rows_fit], alpha=alpha, seed=seed)

    def tuned(x: np.ndarray, feature_names: Sequence[str], label: str) -> Tuple[MLPEnsemble, float, Dict[str, float]]:
        """Best L2 penalty on the inner-validation days (never on validation or test days)."""
        search: Dict[str, float] = {}
        chosen: Optional[Tuple[MLPEnsemble, float, float]] = None
        for alpha in cfg.mlp_alphas:
            model, inner_mae = fit(x, feature_names, fit_rows, inner_rows, alpha)
            search[f"{alpha:g}"] = inner_mae
            log("INFO", "candidate network", model=label, direction=frame.direction, alpha=alpha, inner_mae=inner_mae,
                best_epochs=model.best_epochs)
            if chosen is None or inner_mae < chosen[1]:
                chosen = (model, inner_mae, alpha)
        assert chosen is not None  # mlp_alphas is never empty (Config.validate)
        return chosen[0], chosen[2], search

    fcm_mlp, alpha, search = tuned(x_fuzzy, names, "fcm_mlp")
    mlp, mlp_alpha, mlp_search = tuned(frame.X, BASE_FEATURES, "mlp")  # the ablation gets the same tuning budget

    with_vision = frame.complete & np.isfinite(frame.vision).all(axis=1)
    layer_a = same_rows = None
    if (fit_rows & with_vision).sum() >= MIN_TRAIN_ROWS and (inner_rows & with_vision).sum() >= MIN_INNER_ROWS:
        x_vision = np.hstack([x_fuzzy, frame.vision])
        layer_a, _ = fit(x_vision, names + VISION_FEATURES, fit_rows & with_vision, inner_rows & with_vision, alpha)
        # slide 14: the identical network and rows, minus the vision features
        same_rows, _ = fit(x_fuzzy, names, fit_rows & with_vision, inner_rows & with_vision, alpha)

    predictions = {"memberships": mu}
    predictions["fcm_mlp"], predictions["fcm_mlp_spread"] = _predict_rows(fcm_mlp, x_fuzzy, frame.now, frame.complete)
    predictions["mlp"], _ = _predict_rows(mlp, frame.X, frame.now, frame.complete)
    empty = np.full_like(predictions["fcm_mlp"], np.nan)
    predictions["fcm_mlp_layer_a"] = predictions["fcm_mlp_same_rows"] = empty
    if layer_a is not None and same_rows is not None:
        predictions["fcm_mlp_layer_a"], _ = _predict_rows(layer_a, np.hstack([x_fuzzy, frame.vision]), frame.now, with_vision)
        predictions["fcm_mlp_same_rows"], _ = _predict_rows(same_rows, x_fuzzy, frame.now, with_vision)

    diagnostics = {
        "fcm": {"centres": fcm.describe(), "validity": fcm.validity(frame.X[fcm_rows][:, fcm_columns]),
                "iterations": fcm.iterations, "objective": fcm.objective},
        "alpha_search_inner_mae": {"fcm_mlp": search, "mlp": mlp_search},
        "alpha": {"fcm_mlp": alpha, "mlp": mlp_alpha},
        "best_epochs": {"fcm_mlp": fcm_mlp.best_epochs, "mlp": mlp.best_epochs,
                        "fcm_mlp_layer_a": layer_a.best_epochs if layer_a else None},
        "rows": {"fit": int(fit_rows.sum()), "inner_validation": int(inner_rows.sum()),
                 "validation": int((frame.complete & (rank == 2)).sum()), "test": int((frame.complete & (rank == 3)).sum()),
                 "with_layer_a": int(with_vision.sum())},
    }
    return {
        "models": DirectionModels(fcm, fcm_mlp, layer_a),
        "predictions": predictions,
        "labels": labels,
        "diagnostics": diagnostics,
    }


def evaluation_rows(frame: Frame, split: DaySplit, labels: np.ndarray, predictions: Mapping[str, np.ndarray],
                    regime_labels: Sequence[str]) -> List[Dict[str, Any]]:
    """One dict per (origin, step) on validation and test days, with every model's forecast."""
    rank = split.rank(frame.origin_day)
    mu = predictions["memberships"]
    rows: List[Dict[str, Any]] = []
    for i in np.flatnonzero(frame.complete & (rank >= 2)):
        origin = frame.origin_ts(int(i))
        regime = regime_labels[int(np.argmax(mu[i]))] if np.isfinite(mu[i]).all() else None
        for k in range(frame.horizon):
            actual = _num(labels[i, k])
            if actual is None:
                continue
            rows.append({
                "split": "validation" if rank[i] == 2 else "test",
                "direction": frame.direction,
                "origin_ts": origin,
                "step": k + 1,
                "target_ts": origin + (k + 1) * BIN,
                "actual": actual,
                "persistence": _num(frame.now[i]),
                "seasonal_d1": _num(frame.seasonal_d1[i, k]),
                "seasonal_d7": _num(frame.seasonal_d7[i, k]),
                "seasonal_tod7": _num(frame.seasonal_tod7[i, k]),
                "mlp": _num(predictions["mlp"][i, k]),
                "fcm_mlp": _num(predictions["fcm_mlp"][i, k]),
                "fcm_mlp_layer_a": _num(predictions["fcm_mlp_layer_a"][i, k]),
                "fcm_mlp_same_rows": _num(predictions["fcm_mlp_same_rows"][i, k]),
                "regime": regime,
                "layer_a_status": frame.vision_status[int(i)],
            })
    return rows


def decide(cfg: Config, validation_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Apply decision rules 1-3 per direction and step (validation days only)."""
    decisions: Dict[str, Dict[str, Any]] = {}
    for d in DIRECTIONS:
        for step in range(1, cfg.horizon_steps + 1):
            v = [r for r in validation_rows if r["direction"] == d and r["step"] == step and r.get("fcm_mlp") is not None]
            note: Dict[str, Any] = {"rows": len(v)}
            note["fcm_mlp_vs_persistence"] = compare(v, "fcm_mlp", "persistence", cfg)
            note["fcm_mlp_vs_mlp"] = compare(v, "fcm_mlp", "mlp", cfg)  # FCM ablation, reported only
            # what is served when no camera frame is available: fcm_mlp only if it passed rule 3 alone
            fallback = "fcm_mlp" if better(note["fcm_mlp_vs_persistence"]) else "persistence"
            policy = "fcm_mlp"
            vision_rows = [r for r in v if r.get("fcm_mlp_layer_a") is not None and r.get("fcm_mlp_same_rows") is not None]
            coverage = len(vision_rows) / len(v) if v else 0.0
            note["layer_a_coverage"] = coverage
            if vision_rows:
                note["layer_a_vs_no_vision"] = compare(vision_rows, "fcm_mlp_layer_a", "fcm_mlp_same_rows", cfg)
                note["layer_a_vs_fallback"] = compare(vision_rows, "fcm_mlp_layer_a", fallback, cfg)
                if coverage >= MIN_VISION_COVERAGE and better(note["layer_a_vs_no_vision"]) and better(note["layer_a_vs_fallback"]):
                    policy = "fcm_mlp_layer_a"
            as_served = [{**r, "policy": served_value(r, policy, fallback)[1]} for r in v]
            note["policy_vs_persistence"] = compare(as_served, "policy", "persistence", cfg)
            note["candidate"] = policy
            note["fallback"] = fallback
            note["served"] = policy if better(note["policy_vs_persistence"]) else "persistence"
            decisions.setdefault(d, {})[str(step)] = note
    return decisions


def apply_decisions(rows: Sequence[Dict[str, Any]], decisions: Mapping[str, Mapping[str, Any]]) -> None:
    for r in rows:
        entry = decisions.get(r["direction"], {}).get(str(r["step"]), {})
        r["served_model"], r["served_value"] = served_value(r, entry.get("served", "persistence"), entry.get("fallback"))


def pooled_comparisons(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Tuple[str, str]], cfg: Config) -> Dict[str, Any]:
    """Comparisons per step with both directions in one day block (joint day resampling)."""
    steps = sorted({r["step"] for r in rows})
    return {f"{a}_vs_{b}": {str(s): compare([r for r in rows if r["step"] == s], a, b, cfg) for s in steps} for a, b in pairs}


def train(cfg: Config, grid: Grid, series: Mapping[str, np.ndarray], layer_a: Mapping[Tuple[str, dt.datetime], Mapping[str, Any]],
          *, first: dt.datetime, last: dt.datetime, validation_days: int, test_days: int) -> Dict[str, Any]:
    """Fit on training days, decide on validation days, then score the test days once.

    ``first`` and ``last`` are the first and last origin bins; labels after ``last`` are never used.
    """
    origins = np.arange(grid.index(first), grid.index(last) + 1)
    frames: Dict[str, Frame] = {}
    for d in DIRECTIONS:
        frames[d] = build_frame(cfg, grid, series[d], d, origins, label_cutoff=last)
        attach_vision(cfg, frames[d], layer_a)
    days = {int(day) for fr in frames.values() for day in fr.origin_day[fr.complete & np.isfinite(fr.actual).any(axis=1)]}
    split = split_days(sorted(days), validation_days=validation_days, test_days=test_days, inner_days=cfg.inner_validation_days)

    artefacts: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for index, d in enumerate(DIRECTIONS):
        log("INFO", "training direction", direction=d, model=MODEL_NAME)
        fitted = train_direction(cfg, frames[d], split, index)
        models: DirectionModels = fitted["models"]
        artefacts[d] = models.to_json()
        diagnostics[d] = fitted["diagnostics"]
        rows.extend(evaluation_rows(frames[d], split, fitted["labels"], fitted["predictions"], models.fcm.labels))

    validation = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    decisions = decide(cfg, validation)  # fixed before the test rows are looked at
    apply_decisions(rows, decisions)
    steps = list(range(1, cfg.horizon_steps + 1))
    pairs = [("fcm_mlp", "persistence"), ("fcm_mlp", "mlp"), ("served_value", "persistence")]
    metrics: Dict[str, Any] = {
        "validation": summarise(validation, MODELS, steps),
        "validation_layer_a_rows": summarise([r for r in validation if r.get("fcm_mlp_layer_a") is not None], MODELS, steps),
        "validation_pooled": pooled_comparisons(validation, pairs, cfg),
        "test": summarise(test, (*MODELS, "served_value"), steps) if test else None,
        "test_pooled": pooled_comparisons(test, pairs, cfg) if test else None,
    }

    def iso(days_: Sequence[int], pick: int) -> Optional[dt.date]:
        return epoch_day_to_date(days_[pick]) if days_ else None

    return {
        "model": MODEL_NAME,
        "train_start": iso(split.fit_days, 0), "train_end": iso(split.inner_days, -1),
        "inner_validation_start": iso(split.inner_days, 0),
        "validation_start": iso(split.validation_days, 0), "validation_end": iso(split.validation_days, -1),
        "test_start": iso(split.test_days, 0), "test_end": iso(split.test_days, -1),
        "decisions": decisions,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "artefacts": artefacts,
        "rows": rows,
    }


def model_config(cfg: Config) -> Dict[str, Any]:
    keys = ("horizon_steps", "max_fill_bins", "layer_a_max_age_bins", "fcm_clusters", "fcm_fuzzifier", "fcm_restarts",
            "mlp_hidden", "mlp_alphas", "ensemble_size", "learning_rate", "batch_size", "max_epochs", "patience",
            "inner_validation_days", "bootstrap_reps", "seed", "nonwork_days", "camera_id")
    config = {k: getattr(cfg, k) for k in keys}
    config.update(base_features=list(BASE_FEATURES), fcm_features=list(FCM_FEATURES), vision_features=list(VISION_FEATURES))
    return config


def registry_row(cfg: Config, result: Mapping[str, Any], run_id: str, now: dt.datetime) -> Dict[str, Any]:
    slim = {d: {s: {"served": n["served"], "candidate": n["candidate"], "fallback": n["fallback"]} for s, n in steps.items()}
            for d, steps in result["decisions"].items()}
    report = {"decisions": result["decisions"], "diagnostics": result["diagnostics"], **result["metrics"]}
    return {
        "run_id": run_id,
        "created_at": now,
        "model": MODEL_NAME,
        "train_start": result["train_start"],
        "train_end": result["train_end"],
        "validation_start": result["validation_start"],
        "validation_end": result["validation_end"],
        "test_start": result["test_start"],
        "test_end": result["test_end"],
        "config_json": json.dumps(_jsonable(model_config(cfg)), sort_keys=True),
        "decisions_json": json.dumps(_jsonable(slim), sort_keys=True),
        "artefacts_json": json.dumps(_jsonable(result["artefacts"]), sort_keys=True),
        "metrics_json": json.dumps(_jsonable(report), sort_keys=True),
    }


def evaluation_table_rows(rows: Sequence[Mapping[str, Any]], run_id: str, now: dt.datetime) -> List[Dict[str, Any]]:
    return [{**r, "run_id": run_id, "created_at": now, "model": MODEL_NAME} for r in rows]


SERVING_CONFIG_KEYS: Tuple[str, ...] = ("horizon_steps", "max_fill_bins", "layer_a_max_age_bins", "nonwork_days", "camera_id")


def config_drift(cfg: Config, stored: Any) -> List[str]:
    """Feature settings that differ between the training run and this job (e.g. NONWORK_DAYS)."""
    if not isinstance(stored, Mapping):
        return []
    current = _jsonable(model_config(cfg))

    def norm(key: str, value: Any) -> Any:
        return sorted(value) if key == "nonwork_days" and isinstance(value, list) else value

    return [k for k in SERVING_CONFIG_KEYS if k in stored and norm(k, stored[k]) != norm(k, current[k])]


def load_registered(cfg: Config, wh: Warehouse) -> Tuple[Optional[str], Mapping[str, Any], Dict[str, DirectionModels]]:
    """Latest registered decisions and models; a direction whose model is unusable is left out."""
    found = wh.query(latest_registry_sql(cfg), {"model": MODEL_NAME})
    if not found:
        log("WARNING", "no registry entry; serving persistence until `train --register` has run")
        return None, {}, {}
    entry = found[0]
    try:
        drift = config_drift(cfg, json.loads(entry.get("config_json") or "{}"))
    except ValueError:
        drift = ["config_json unreadable"]
    if drift:  # the shapes still match, so serve, but say loudly that features may differ from training
        log("WARNING", "serving configuration differs from the training run", keys=drift, registry_run_id=entry.get("run_id"))
    decisions = json.loads(entry["decisions_json"])
    artefacts = json.loads(entry["artefacts_json"])
    if not isinstance(decisions, Mapping) or not isinstance(artefacts, Mapping):
        raise ValueError("registry entry is not a JSON object")
    models: Dict[str, DirectionModels] = {}
    for d in DIRECTIONS:
        try:
            models[d] = DirectionModels.from_json(artefacts[d], cfg.horizon_steps)
        except (KeyError, TypeError, ValueError) as exc:
            log("ERROR", "unusable registered model; serving persistence for this direction", direction=d, error=str(exc))
    return str(entry["run_id"]), decisions, models


# ----------------------------------------------------------------------------------------------
# Live cycle
# ----------------------------------------------------------------------------------------------


def run_cycle(cfg: Config, wh: Warehouse, session: Any, *, now: Optional[dt.datetime] = None, use_layer_a: bool = True,
              write: bool = True) -> Dict[str, Any]:
    """Ingest Layer A, compute features, apply the registered FCM + MLP policy, log the forecasts."""
    now = now or utc_now()
    run_id = f"cycle-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    summary: Dict[str, Any] = {"run_id": run_id, "model": MODEL_NAME}
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
    grid, series = load_series(cfg, wh, history_start(cfg, origin), origin)

    registry_run_id: Optional[str] = None
    decisions: Mapping[str, Any] = {}
    models: Dict[str, DirectionModels] = {}
    try:
        registry_run_id, decisions, models = load_registered(cfg, wh)
    except Exception as exc:  # noqa: BLE001 - an unreadable registry means the conservative default
        registry_run_id, decisions, models = None, {}, {}
        log("ERROR", "registry unreadable; serving persistence", error=repr(exc))
    summary["registry_run_id"] = registry_run_id

    layer_a: Dict[Tuple[str, dt.datetime], Dict[str, Any]] = {}
    if use_layer_a:
        try:
            layer_a = load_layer_a_features(cfg, wh, origin, origin)
        except Exception as exc:  # noqa: BLE001 - forecast without vision rather than not at all
            log("ERROR", "Layer A features unavailable; forecasting without them", error=repr(exc))

    origin_index = np.array([grid.index(origin)])
    out_rows: List[Dict[str, Any]] = []
    for d in DIRECTIONS:
        frame = build_frame(cfg, grid, series[d], d, origin_index, label_cutoff=origin)
        if not math.isfinite(frame.now[0]):
            log("WARNING", "no travel time in the origin bin; direction skipped", direction=d, origin_ts=origin)
            continue
        attach_vision(cfg, frame, layer_a)
        model = models.get(d)
        predictions: Dict[str, np.ndarray] = {}
        if model is not None:
            try:
                predictions = predict_frame(model, frame)
            except (ValueError, FloatingPointError) as exc:
                log("ERROR", "prediction failed; serving persistence", direction=d, error=str(exc))
        regime = regime_degree = memberships_json = None
        mu = predictions.get("memberships")
        if model is not None and mu is not None and np.isfinite(mu[0]).all():
            j = int(np.argmax(mu[0]))
            regime, regime_degree = model.fcm.labels[j], float(mu[0, j])
            memberships_json = json.dumps({label: round(float(u), 4) for label, u in zip(model.fcm.labels, mu[0])})
        for k in range(1, cfg.horizon_steps + 1):
            row = {
                "persistence": _num(frame.now[0]),
                "fcm_mlp": _num(predictions["fcm_mlp"][0, k - 1]) if predictions else None,
                "fcm_mlp_layer_a": _num(predictions["fcm_mlp_layer_a"][0, k - 1]) if predictions else None,
            }
            entry = decisions.get(d, {}).get(str(k), {}) if isinstance(decisions.get(d), Mapping) else {}
            if not isinstance(entry, Mapping):
                entry = {}
            decision = entry.get("served", "persistence")
            if decision not in SERVABLE:
                decision = "persistence"
            served_model, value = served_value(row, decision, entry.get("fallback"))
            forecast_regime = forecast_degree = None
            if model is not None and value is not None:
                level = model.fcm.level_memberships(np.array([value]))[0]
                forecast_regime, forecast_degree = model.fcm.labels[int(np.argmax(level))], float(level.max())
            out_rows.append({
                "run_id": run_id, "created_at": now, "origin_ts": origin, "direction": d,
                "step": k, "horizon_min": k * BIN_MINUTES, "target_ts": origin + k * BIN,
                "persistence": row["persistence"],
                "seasonal_d1": _num(frame.seasonal_d1[0, k - 1]), "seasonal_d7": _num(frame.seasonal_d7[0, k - 1]),
                "fcm_mlp": row["fcm_mlp"],
                "fcm_mlp_spread": _num(predictions["fcm_mlp_spread"][0, k - 1]) if predictions else None,
                "fcm_mlp_layer_a": row["fcm_mlp_layer_a"],
                "served_model": served_model, "served_value": value,
                "regime": regime, "regime_degree": regime_degree, "memberships_json": memberships_json,
                "forecast_regime": forecast_regime, "forecast_regime_degree": forecast_degree,
                "layer_a_status": frame.vision_status[0], "registry_run_id": registry_run_id, "model": MODEL_NAME,
            })
    if not out_rows:
        raise NoForecastError(f"no direction had fresh data for origin {origin.isoformat()}")
    if write:
        wh.append(TABLE_FORECASTS, out_rows)
    summary["forecasts"] = [
        {k: row[k] for k in ("direction", "horizon_min", "target_ts", "served_model", "served_value", "fcm_mlp",
                             "persistence", "regime", "forecast_regime", "layer_a_status")}
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
    parser = argparse.ArgumentParser(description="SwiftBorder Layer B: Fuzzy C-Means + MLP fused with Layer A counts.")
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

    p = sub.add_parser("train", help="fit FCM + MLP, apply the decision rules, score the test days once")
    window = p.add_mutually_exclusive_group(required=True)
    window.add_argument("--start", help="first SGT date of forecast origins, YYYY-MM-DD")
    window.add_argument("--lookback-days", type=int, help="number of days ending at --end")
    p.add_argument("--end", help="last SGT date, YYYY-MM-DD (default: yesterday in Singapore)")
    p.add_argument("--validation-days", type=int, default=7, help="days for the decision rules (7 covers a full week)")
    p.add_argument("--test-days", type=int, default=5, help="0 trains for deployment without a test report")
    p.add_argument("--register", action="store_true", help=f"store the model and decisions in {TABLE_REGISTRY}")
    p.add_argument("--write-eval", action="store_true", help=f"store validation and test predictions in {TABLE_EVALUATION}")
    p.add_argument("--allow-protected-window", action="store_true")

    sub.add_parser("show-sql", help="print the generated SQL (no credentials needed)")
    return parser


def training_window(args: argparse.Namespace, now: dt.datetime) -> Tuple[dt.date, dt.date]:
    today = sgt_date(now)
    end = parse_sgt_date(args.end) if args.end else today - dt.timedelta(days=1)
    if end >= today:
        raise ConfigError(f"end date {end} is not a complete day yet in Singapore")
    if args.lookback_days is not None:
        if args.lookback_days < 1:
            raise ConfigError("--lookback-days must be at least 1")
        return end - dt.timedelta(days=args.lookback_days - 1), end
    return parse_sgt_date(args.start), end


def main(argv: Optional[Sequence[str]] = None, *, warehouse_factory: Optional[Callable[[Config], Warehouse]] = None,
         session: Any = None) -> int:
    if argv is None:  # Cloud Run Jobs: the command can come from LAYER_B_ARGS instead of the Procfile
        argv = sys.argv[1:] or shlex.split(os.environ.get("LAYER_B_ARGS", "run-cycle"))
    args = build_parser().parse_args(argv)
    try:
        cfg = Config.from_env()
        if args.command == "show-sql":
            for title, sql in (("travel-time bins", bins_sql(cfg)), ("layer A features", layer_a_features_sql(cfg)),
                               ("layer A bins already logged", layer_a_existing_bins_sql(cfg)),
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
        elif args.command == "train":
            now = utc_now()
            start, end = training_window(args, now)
            first, last = sgt_day_bounds(start, end)
            check_protected_window(cfg, first, last, args.allow_protected_window)
            grid, series = load_series(cfg, wh, history_start(cfg, first), last)
            layer_a: Dict[Tuple[str, dt.datetime], Dict[str, Any]] = {}
            try:
                layer_a = load_layer_a_features(cfg, wh, first, last)
            except Exception as exc:  # noqa: BLE001 - e.g. the table does not exist yet; train without vision
                log("WARNING", "Layer A features unavailable; training without them", error=repr(exc))
            result = train(cfg, grid, series, layer_a, first=first, last=last, validation_days=args.validation_days,
                           test_days=args.test_days)
            run_id = f"train-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            _print({"run_id": run_id, **{k: v for k, v in result.items() if k not in ("artefacts", "rows")}})
            if args.register or args.write_eval:
                wh.ensure_tables()
            if args.register:
                wh.append(TABLE_REGISTRY, [registry_row(cfg, result, run_id, now)])
            if args.write_eval:
                wh.append(TABLE_EVALUATION, evaluation_table_rows(result["rows"], run_id, now))
        return EXIT_OK
    except ConfigError as exc:
        log("ERROR", "invalid configuration, arguments or data", error=str(exc))
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
