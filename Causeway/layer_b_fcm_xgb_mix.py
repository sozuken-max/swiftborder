#!/usr/bin/env python3
"""SwiftBorder Layer B: equal mix of FCM + MLP and XGBoost on Maps features, fed by Layer A counts.

Layer B forecasts the Google Maps travel time in traffic across the Woodlands Causeway, per direction,
30 minutes ahead. This module serves the equal-weight mix of two learners that make different kinds of
errors (slide 10 of the 17-slide proposal deck: "if the model families prove complementary, a simple
ensemble will be tested, selected on validation data only"):

    mix = 0.5 * fcm_mlp + 0.5 * xgb[maps]

* ``fcm_mlp`` is the Fuzzy C-Means + multilayer perceptron model of ``layer_b_fcm_mlp.py`` with that
  module's defaults: c = 3 regimes, fuzzifier m = 2, and a 32-16 ReLU network averaged over five seeds
  that predicts the change in travel time at 10 to 60 minutes at once. Its 30-minute step is used.
* ``xgb[maps]`` is the harness XGBoost of ``eval/joined.py`` (300 trees, learning rate 0.05, depth 4,
  seed 42) on the 17 Maps features of ``eval/features.py`` plus a direction flag: one model for both
  directions that predicts the 30-minute travel time directly.

Both members are refitted for every Singapore day on every label observed before 00:00 SGT that
day, back to the first logged bin (``MIX_HISTORY_START``). That is the daily-refit policy that was
evaluated (docs/evaluation.md section 8, docs/adr/0004-serve-local-models.md); ``MIX_LOOKBACK_DAYS``
can cap the history, but a capped fit is no longer the evaluated member. The weights are fixed, not
learned: two members of nearly equal accuracy gain little from estimated weights, and fixed weights
leave nothing for a few days of data to overfit. Because the mix is an average, its absolute error is
never larger than the mean of the two members' absolute errors, and it is smaller whenever one member
overshoots while the other undershoots.

Evidence and status
-------------------
docs/evaluation.md section 8 scored the mix on 13-30 Sep 2026 (both directions, 5,184 rows, daily
refit): MAE 2.134 min against 2.276 for xgb[maps], 2.230 for fcm_mlp and 2.640 for persistence, and
0.142 min better than xgb[maps] (joint calendar-day 95% interval -0.211 to -0.090). The gain is below
the 0.5-minute practical threshold and the mix was defined after that window was seen, so it is not a
confirmed result. It is not one of the frozen-run claims C1-C8; docs/roadmap.md proposes it as C9 for
the October-only Run B, and until the team confirms that before 19 Oct 2026 23:59 SGT an October
score is exploratory. This module writes its own tables; it does not change the ``served`` selection
of ``forecast-api``, which ADR 0004 ties to the frozen run.

``backtest --start 2026-09-13 --end 2026-09-30 --allow-protected-window`` reruns that protocol on all
5,184 rows. Section 8 scored the six 30-minute labels at 00:00-00:20 on 1 Oct, inside the protected
window, so without the flag the same run scores 5,178 rows. Its intervals come from the paired
day-block bootstrap below, not from the Diebold-Mariano tests with Holm correction of
``eval/significance.py``, so they will not match section 8 to the last digit.

Layer A feeds Layer B
---------------------
Layer A, the ``swiftbackend`` Cloud Run service (``camdetect/``), turns an LTA camera 2701 frame into
per-direction vehicle counts and a congestion level. When ``LAYER_A_URL`` is set, ``run-cycle`` logs
one observation per 10-minute bin into the shared ``<dataset>.layer_a_counts`` table. Both members can
then read the latest accepted frame up to 20 minutes before the origin as three extra inputs: the
direction's count, the other direction's count and the congestion level (slide 10). Counts measure
density, not flow (slide 11), so they are inputs and never the target, and a frame with no vehicles
while Google reports congestion (haze, heavy rain) is rejected. The Layer A client, its table and its
quality gate are those of ``layer_b_fcm_mlp.py``, so both modules share one log of camera calls; set
``LAYER_A_URL`` on one job only.

The vision variants (``fcm_mlp_layer_a``, ``xgb_maps_layer_a`` and their mix ``mix_layer_a``) are
trained only when logged frames cover enough training rows, and served only when the decision rules
below allow it. Layer A is an unvalidated prototype until its detector metrics are scored. When this
module was written no logged frame overlapped the travel-time labels (docs/evaluation.md section 8),
so until frames are logged the Maps-only mix is the model that can be served.

Decision rules
--------------
Fixed in code and applied per direction, on validation days only (slides 10, 13 and 15). The
validation days are scored by a rolling daily refit, exactly as the live models are produced:

1. The Maps-only mix is the candidate.
2. The Layer A mix replaces it only if logged frames covered at least half of the validation rows and,
   on those rows, it beats both the identical mix trained on the same rows without the vision inputs
   (slides 13 and 15) and the model that would otherwise be served. When a live frame is missing, the
   Maps-only mix is served if it passed rule 3 on its own, and persistence otherwise.
3. The candidate is served only if it beats persistence (slide 11); otherwise persistence is served.

"Beats" means that the 95% paired day-block bootstrap interval of the difference in absolute error
lies entirely below zero, over at least three days. The mix's differences from each member are
reported but not gated. An origin that follows a data gap (``after_gap``: more than 25 minutes
between consecutive bins within the last seven bins, as in ``v_training_set``) is outside the
conditions the mix was evaluated on, so persistence is served for it.

Training protocol and leakage control
-------------------------------------
For serving day D, every member is fitted on the origins since ``MIX_HISTORY_START`` (or of the last
``MIX_LOOKBACK_DAYS`` days, when set) whose labels closed before 00:00 SGT on D. ``xgb[maps]`` uses
every such row with a 30-minute label and no gap (``eval/joined.scorable``), in the harness's row
order, because XGBoost's row subsampling depends on it. ``fcm_mlp`` needs all six of its labels and
holds out its last two training days for early stopping and the choice of its L2 penalty, as in its
own module, and no label crosses from a later day into an earlier fit. ``train`` first runs that
procedure for each of the ``validation_days`` days before D, each one fitted only on its own past,
applies the decision rules to those out-of-sample forecasts, and then fits the models it registers
for D.

``train``, ``backtest`` and ``monitor`` refuse a window that touches the team's protected confirmation
window (default 1-19 Oct 2026) unless ``--allow-protected-window`` is given. Without that flag no
label inside the protected window is scored either, so October labels stay unseen until the
pre-registered frozen run.

Time conventions
----------------
Everything runs on the 10-minute bins of ``traffic_prediction.v_bins_10min`` (bin start in UTC). The
origin is the last complete bin ``t``; the target is the bin that starts 30 minutes later,
``[t+30, t+40)``, which is ``y_30`` of ``v_training_set`` and step 3 of ``layer_b_fcm_mlp.py``.
Persistence is the value of bin ``t``. The Maps features use time-based lags, so a missing bin stays
missing, as in the harness and in ``forecast-api``; the FCM + MLP features come from
``layer_b_fcm_mlp.build_frame``. Calendar dates on the command line are Singapore dates (UTC+8).

Commands
--------
    python layer_b_fcm_xgb_mix.py ensure-tables
    python layer_b_fcm_xgb_mix.py run-cycle                       # Cloud Run Job, every 10 minutes
    python layer_b_fcm_xgb_mix.py train --register                # Cloud Run Job, daily at 00:15 SGT
    python layer_b_fcm_xgb_mix.py backtest --start 2026-09-13 --end 2026-09-30 --allow-protected-window
    python layer_b_fcm_xgb_mix.py monitor --days 7                # live error against persistence
    python layer_b_fcm_xgb_mix.py show-sql

``train`` defaults to ``--end`` = yesterday in Singapore and registers the models for today with
``--register``; ``--write-eval`` stores its validation forecasts row by row. ``backtest`` prints the
metrics and the paired comparisons of a rolling daily refit over any past window (``--write-eval``
stores its rows). ``monitor`` scores the forecasts this module logged against the bins that arrived
later (slide 11: monitor live error against persistence). ``run-cycle --dry-run`` forecasts without
writing. To backfill Layer A frames, use ``layer_b_fcm_mlp.py backfill-layer-a`` (the table is shared).

Deployment (Cloud Run Jobs triggered by Cloud Scheduler)
--------------------------------------------------------
``Causeway/requirements.txt`` adds ``xgboost==3.2.0``, the pin of ``eval/`` and ``forecastapi/``, to
``layer_b_fcm_mlp.py``'s dependencies. On Linux it also installs NVIDIA NCCL (about 0.5 GB) into the
image. ``xgboost-cpu==3.2.0`` is the CPU-only build of the same library and gave identical xgb[maps]
predictions on the 13-30 Sep bins in a check on 2026-10-10, so a deploy-only requirements file may use
it; never install both packages in one environment. Buildpacks start the command in
``Causeway/Procfile``, and ``Causeway/.python-version`` pins Python 3.11 (the team lock files); both
were added with this module. The Procfile line, documented in ``layer_b_fcm_mlp.py`` too, serves this
module when ``LAYER_B_MODULE`` names it (started without arguments, the module reads its command from
``LAYER_B_ARGS``, default ``run-cycle``)::

    web: python3 ${LAYER_B_MODULE:-layer_b_fcm_mlp.py} ${LAYER_B_ARGS:-run-cycle}

Example (replace the service account with the team's Layer B account)::

    gcloud run jobs deploy layer-b-mix --source Causeway --region asia-southeast1 \\
        --max-retries 1 --task-timeout 600s --memory 1Gi --service-account <layer-b-service-account> \\
        --set-env-vars SWIFTBORDER_PROJECT=swiftborder,LAYER_B_MODULE=layer_b_fcm_xgb_mix.py
    gcloud scheduler jobs create http layer-b-mix-10min --location asia-southeast1 \\
        --schedule "*/10 * * * *" --time-zone Asia/Singapore --http-method POST \\
        --uri https://run.googleapis.com/v2/projects/swiftborder/locations/asia-southeast1/jobs/layer-b-mix:run \\
        --oauth-service-account-email <layer-b-service-account>
    gcloud run jobs deploy layer-b-mix-train --source Causeway --region asia-southeast1 \\
        --max-retries 0 --task-timeout 3600s --memory 2Gi --service-account <layer-b-service-account> \\
        --set-env-vars "SWIFTBORDER_PROJECT=swiftborder,LAYER_B_MODULE=layer_b_fcm_xgb_mix.py,LAYER_B_ARGS=train --register"
    gcloud scheduler jobs create http layer-b-mix-daily --location asia-southeast1 \\
        --schedule "15 0 * * *" --time-zone Asia/Singapore --http-method POST \\
        --uri https://run.googleapis.com/v2/projects/swiftborder/locations/asia-southeast1/jobs/layer-b-mix-train:run \\
        --oauth-service-account-email <layer-b-service-account>

The daily job runs at 00:15 SGT: every label the new day's fit may use closed at 00:00, and the few
validation labels that are still open are left out. Until it finishes (about 90 seconds on the 25 days
of September history, growing with the history), ``run-cycle`` serves the previous day's model,
whereas ``backtest`` uses the new day's model from 00:00.

With the full history, every training window from 1 Oct onwards contains the protected confirmation
window, so the daily job exits with code 2 by design until ``PROTECTED_WINDOW`` is cleared on the
training job after the frozen run. Register a model trained on September data first (``train --end
2026-09-30 --register``); ``run-cycle`` keeps serving it, and warns that it is old, until the daily
job takes over. ``run-cycle`` serves the latest registered model. It reads the last seven days of the
registry first (the table is partitioned by day, so that read stays small) and the whole registry
only when those days hold no entry. The jobs' service account needs ``roles/bigquery.jobUser`` on the
project, ``roles/bigquery.dataEditor`` on the Layer B dataset and ``roles/bigquery.dataViewer`` on
``causeway``; the scheduler's account needs ``roles/run.invoker``.

Environment variables (all optional): everything ``layer_b_fcm_mlp.py`` reads (``SWIFTBORDER_PROJECT``,
``BQ_LOCATION``, ``LAYER_B_DATASET``, ``BINS_VIEW``, ``LAYER_A_URL``, ``LAYER_A_CAMERA_ID``,
``LAYER_A_CONFIDENCE``, ``FCM_CLUSTERS``, ``FCM_FUZZIFIER``, ``MLP_HIDDEN``, ``MLP_ENSEMBLE_SIZE``,
``NONWORK_DAYS``, ``PROTECTED_WINDOW``), plus::

    MIX_HISTORY_START     first SGT date of training data (default 2026-09-06, the first logged bin)
    MIX_LOOKBACK_DAYS     optional cap on the training days before each serving day (default: none)
    MIX_VALIDATION_DAYS   days the decision rules are applied on (default 7, a full week)

Changing the FCM or MLP settings changes ``fcm_mlp`` away from the evaluated member.

The front end can read the latest forecasts with::

    SELECT * FROM `swiftborder.traffic_prediction.layer_b_mix_forecasts`
    WHERE origin_ts = (SELECT MAX(origin_ts) FROM `swiftborder.traffic_prediction.layer_b_mix_forecasts`)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY direction ORDER BY created_at DESC) = 1

Exit codes: 0 success, 1 unexpected error, 2 invalid configuration, arguments or too little data,
3 no usable forecast (stale data).

Course techniques covered (docs/grading): supervised learning (XGBoost and MLP regression),
unsupervised learning (Fuzzy C-Means), a hybrid and ensemble design (a fuzzy front end feeding a seed
ensemble of networks, mixed with gradient-boosted trees) and intelligent sensing (Layer A counts).

References
----------
J. M. Bates, C. W. J. Granger, "The combination of forecasts", Operational Research Quarterly 20(4),
451-468, 1969. A. Timmermann, "Forecast combinations", Handbook of Economic Forecasting 1, 135-196,
2006. T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system", KDD 2016. J. C. Bezdek,
R. Ehrlich, W. Full, "FCM: The fuzzy c-means clustering algorithm", Computers & Geosciences 10(2-3),
191-203, 1984. H. R. Kunsch, "The jackknife and the bootstrap for general stationary observations",
Annals of Statistics 17(3), 1217-1241, 1989.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import shlex
import sys
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests

import layer_b_fcm_mlp as fcm

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------

DIRECTIONS = fcm.DIRECTIONS
BIN = fcm.BIN
BIN_SECONDS = fcm.BIN_SECONDS
BIN_MINUTES = fcm.BIN_MINUTES
DAY_SECONDS = 86400

MODEL_NAME = "mix:fcm_mlp+xgb[maps]"
COMPONENT = "layer_b_fcm_xgb_mix"
TARGET_STEP = 3                              # bin [t+30, t+40): y_30 of v_training_set
HORIZON_MIN = TARGET_STEP * BIN_MINUTES
LABEL_LAG_SECONDS = (TARGET_STEP + 1) * BIN_SECONDS  # the target bin closes 40 minutes after the origin starts
MIX_WEIGHTS: Mapping[str, float] = {"fcm_mlp": 0.5, "xgb_maps": 0.5}

# eval/features.py MAPS_FEATURES + eval/joined.py "is_my_to_sg" (forecastapi/local_models.py has the same list)
MAPS_FEATURES: Tuple[str, ...] = (
    "y_persistence", "congestion_ratio", "speed_kmh", "lag_10", "lag_20", "lag_30", "lag_60",
    "roll_mean_30", "roll_mean_60", "slope_30", "tod_sin", "tod_cos", "tod_block", "dow",
    "is_weekend", "is_morning_peak", "is_evening_peak", "is_my_to_sg",
)
VISION_FEATURES: Tuple[str, ...] = fcm.VISION_FEATURES
# eval/joined.py _xgb (forecastapi/local_models.py HARNESS_XGB_PARAMS)
XGB_PARAMS: Mapping[str, Any] = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 10, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 42, "n_jobs": 1,
}
DEFAULT_HISTORY_START = "2026-09-06"  # first SGT day of traffic_prediction.v_bins_10min (Maps logging began)
RECENT_REGISTRY_DAYS = 7  # run-cycle reads these days of the registry first (the table is partitioned by day)
GAP_MINUTES = 25.0    # v_training_set: gap_min > 25 marks a gap
AFTER_GAP_ROWS = 7    # ... and the gap flag covers that bin and the next six observed bins
MIN_XGB_ROWS = 500    # eval/joined.run_folds min_train_rows

ROW_MODELS: Tuple[str, ...] = ("fcm_mlp", "xgb_maps", "mix", "fcm_mlp_layer_a", "xgb_maps_layer_a", "mix_layer_a",
                               "mix_same_rows")
CORE_MODELS: Tuple[str, ...] = ("persistence", "fcm_mlp", "xgb_maps", "mix")
REPORT_MODELS: Tuple[str, ...] = (*CORE_MODELS, "fcm_mlp_layer_a", "xgb_maps_layer_a", "mix_layer_a")
COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("mix", "persistence"), ("mix", "xgb_maps"), ("mix", "fcm_mlp"), ("xgb_maps", "persistence"),
    ("fcm_mlp", "persistence"), ("mix_layer_a", "mix_same_rows"),
)
SERVABLE: Tuple[str, ...] = ("persistence", "mix", "mix_layer_a")
SERVE_CHAIN: Mapping[str, Tuple[str, ...]] = {  # mix_layer_a is followed by its stored fallback
    "mix": ("mix", "persistence"),
    "persistence": ("persistence",),
}

TABLE_LAYER_A = fcm.TABLE_LAYER_A
TABLE_FORECASTS = "layer_b_mix_forecasts"
TABLE_EVALUATION = "layer_b_mix_evaluation"
TABLE_REGISTRY = "layer_b_mix_registry"

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_NO_FORECAST = fcm.EXIT_OK, fcm.EXIT_ERROR, fcm.EXIT_CONFIG, fcm.EXIT_NO_FORECAST
ConfigError = fcm.ConfigError
NoForecastError = fcm.NoForecastError


def log(severity: str, message: str, **fields: Any) -> None:
    """Structured log line; Cloud Logging reads ``severity`` and ``message`` from JSON on stdout."""
    record = {"severity": severity, "message": message, "component": COMPONENT}
    record.update({k: _jsonable(v) for k, v in fields.items()})
    print(json.dumps(record, sort_keys=True), flush=True)


# The FCM module's helpers, shared so both modules serialise, round and index identically:
_jsonable = fcm._jsonable  # numpy, datetimes and NaN to JSON-safe values
_num = fcm._num            # a finite float, or None
_take = fcm._take          # arr[positions] with NaN outside the array


def _iso_day(day: int) -> str:
    return fcm.epoch_day_to_date(day).isoformat()


def day_start_unix(day: int) -> int:
    """UTC seconds of 00:00 SGT on SGT epoch day ``day``."""
    return int(day) * DAY_SECONDS - fcm.SGT_OFFSET_SEC


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MixConfig:
    """Settings of this module; ``base`` holds those shared with ``layer_b_fcm_mlp.py``."""

    base: fcm.Config = dataclasses.field(default_factory=fcm.Config)
    lookback_days: Optional[int] = None   # None: every label since history_start (the evaluated daily refit)
    validation_days: int = 7
    min_xgb_rows: int = MIN_XGB_ROWS
    history_start: str = DEFAULT_HISTORY_START

    def table(self, name: str) -> str:
        return self.base.table(name)

    @property
    def history_start_day(self) -> int:
        return fcm.date_to_epoch_day(fcm.parse_sgt_date(self.history_start))

    def first_training_day(self, day: int) -> int:
        """Earliest SGT day a fit for serving day ``day`` reads: the history start, or the lookback cap."""
        if self.lookback_days is None:
            return self.history_start_day
        return max(self.history_start_day, day - self.lookback_days)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> MixConfig:
        env = os.environ if env is None else env
        base = fcm.Config.from_env(env)
        lookback = env.get("MIX_LOOKBACK_DAYS", "").strip()
        try:
            cfg = cls(base=base, lookback_days=int(lookback) if lookback else None,
                      validation_days=int(env.get("MIX_VALIDATION_DAYS", cls.validation_days)),
                      history_start=env.get("MIX_HISTORY_START", "").strip() or DEFAULT_HISTORY_START)
        except ValueError as exc:
            raise ConfigError(f"invalid numeric environment variable: {exc}") from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.base.validate()
        if self.base.horizon_steps < TARGET_STEP:
            raise ConfigError(f"the FCM + MLP horizon must reach step {TARGET_STEP} ({HORIZON_MIN} minutes)")
        fcm.parse_sgt_date(self.history_start)
        least = self.base.inner_validation_days + fcm.MIN_FIT_DAYS
        if self.lookback_days is not None and not least <= self.lookback_days <= 366:
            raise ConfigError(f"MIX_LOOKBACK_DAYS must be empty or between {least} and 366")
        if not fcm.MIN_VALIDATION_DAYS <= self.validation_days <= 28:
            raise ConfigError(f"MIX_VALIDATION_DAYS must be between {fcm.MIN_VALIDATION_DAYS} and 28")
        if self.min_xgb_rows < 100:
            raise ConfigError("min_xgb_rows must be at least 100")


def model_config(cfg: MixConfig) -> Dict[str, Any]:
    return {
        "fcm": fcm.model_config(cfg.base),
        "history_start": cfg.history_start,
        "lookback_days": cfg.lookback_days,
        "validation_days": cfg.validation_days,
        "min_xgb_rows": cfg.min_xgb_rows,
        "target_step": TARGET_STEP,
        "mix_weights": dict(MIX_WEIGHTS),
        "maps_features": list(MAPS_FEATURES),
        "xgb_params": dict(XGB_PARAMS),
    }


# ----------------------------------------------------------------------------------------------
# BigQuery tables and SQL
# ----------------------------------------------------------------------------------------------

_FORECAST_COLUMNS: List[Tuple[str, str, str]] = [
    ("persistence", "FLOAT64", "NULLABLE"),
    ("fcm_mlp", "FLOAT64", "NULLABLE"),
    ("xgb_maps", "FLOAT64", "NULLABLE"),
    ("mix", "FLOAT64", "NULLABLE"),
    ("fcm_mlp_layer_a", "FLOAT64", "NULLABLE"),
    ("xgb_maps_layer_a", "FLOAT64", "NULLABLE"),
    ("mix_layer_a", "FLOAT64", "NULLABLE"),
]

SCHEMAS: Dict[str, List[Tuple[str, str, str]]] = {
    TABLE_FORECASTS: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("origin_ts", "TIMESTAMP", "REQUIRED"),
        ("direction", "STRING", "REQUIRED"),
        ("horizon_min", "INT64", "REQUIRED"),
        ("target_ts", "TIMESTAMP", "REQUIRED"),
        *_FORECAST_COLUMNS,
        ("fcm_mlp_spread", "FLOAT64", "NULLABLE"),
        ("served_model", "STRING", "REQUIRED"),
        ("served_value", "FLOAT64", "NULLABLE"),
        ("decision", "STRING", "REQUIRED"),
        ("regime", "STRING", "NULLABLE"),
        ("regime_degree", "FLOAT64", "NULLABLE"),
        ("memberships_json", "STRING", "NULLABLE"),
        ("forecast_regime", "STRING", "NULLABLE"),
        ("forecast_regime_degree", "FLOAT64", "NULLABLE"),
        ("after_gap", "BOOL", "REQUIRED"),
        ("layer_a_status", "STRING", "REQUIRED"),
        ("labels_before", "TIMESTAMP", "NULLABLE"),
        ("registry_run_id", "STRING", "NULLABLE"),
        ("model", "STRING", "REQUIRED"),
    ],
    TABLE_EVALUATION: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("split", "STRING", "REQUIRED"),
        ("origin_ts", "TIMESTAMP", "REQUIRED"),
        ("direction", "STRING", "REQUIRED"),
        ("target_ts", "TIMESTAMP", "REQUIRED"),
        ("actual", "FLOAT64", "REQUIRED"),
        *_FORECAST_COLUMNS,
        ("mix_same_rows", "FLOAT64", "NULLABLE"),
        ("served_model", "STRING", "NULLABLE"),
        ("served_value", "FLOAT64", "NULLABLE"),
        ("layer_a_status", "STRING", "REQUIRED"),
        ("labels_before", "TIMESTAMP", "REQUIRED"),
        ("model", "STRING", "REQUIRED"),
    ],
    TABLE_REGISTRY: [
        ("run_id", "STRING", "REQUIRED"),
        ("created_at", "TIMESTAMP", "REQUIRED"),
        ("model", "STRING", "REQUIRED"),
        ("labels_before", "TIMESTAMP", "REQUIRED"),
        ("train_start", "DATE", "REQUIRED"),
        ("train_end", "DATE", "REQUIRED"),
        ("validation_start", "DATE", "REQUIRED"),
        ("validation_end", "DATE", "REQUIRED"),
        ("config_json", "STRING", "REQUIRED"),
        ("decisions_json", "STRING", "REQUIRED"),
        ("artefacts_json", "STRING", "REQUIRED"),
        ("metrics_json", "STRING", "REQUIRED"),
    ],
}
PARTITIONING: Dict[str, Tuple[Optional[str], List[str]]] = {
    TABLE_FORECASTS: ("origin_ts", ["direction", "origin_ts"]),
    TABLE_EVALUATION: ("origin_ts", ["run_id", "direction"]),
    TABLE_REGISTRY: ("created_at", ["model"]),
}
# the Layer A table is shared with layer_b_fcm_mlp.py and layer_b_timesfm.py, so its schema is theirs
ALL_SCHEMAS: Dict[str, List[Tuple[str, str, str]]] = {TABLE_LAYER_A: fcm.SCHEMAS[TABLE_LAYER_A], **SCHEMAS}
ALL_PARTITIONING: Dict[str, Tuple[Optional[str], List[str]]] = {
    TABLE_LAYER_A: fcm.PARTITIONING[TABLE_LAYER_A], **PARTITIONING,
}


class Warehouse(fcm.Warehouse):
    """The BigQuery wrapper of ``layer_b_fcm_mlp.py`` with this module's tables and query label."""

    def __init__(self, cfg: MixConfig, client: Any = None) -> None:
        super().__init__(cfg.base, client)

    def query(self, sql: str, params: Optional[Mapping[str, Any]] = None, timeout_sec: float = 600.0) -> List[Dict[str, Any]]:
        job_config = self._bq.QueryJobConfig(
            query_parameters=[self._param(k, v) for k, v in (params or {}).items()],
            labels={"component": "layer-b-fcm-xgb-mix"},
        )
        job = self.client.query(sql, job_config=job_config, location=self.cfg.bq_location)
        return [dict(row.items()) for row in job.result(timeout=timeout_sec)]

    def schema(self, name: str) -> List[Any]:
        return [self._bq.SchemaField(col, kind, mode=mode) for col, kind, mode in ALL_SCHEMAS[name]]

    def ensure_tables(self) -> List[str]:
        created = []
        for name in ALL_SCHEMAS:
            table = self._bq.Table(self.cfg.table(name), schema=self.schema(name))
            partition_col, cluster_cols = ALL_PARTITIONING[name]
            if partition_col:
                table.time_partitioning = self._bq.TimePartitioning(type_=self._bq.TimePartitioningType.DAY,
                                                                    field=partition_col)
            if cluster_cols:
                table.clustering_fields = cluster_cols
            self.client.create_table(table, exists_ok=True)
            created.append(self.cfg.table(name))
        return created

    def append(self, name: str, rows: Sequence[Mapping[str, Any]]) -> int:
        """Append rows with a batch load job (free, and never leaves rows in a streaming buffer)."""
        if not rows:
            return 0
        columns = [c for c, _, _ in ALL_SCHEMAS[name]]
        payload = [{c: _jsonable(row.get(c)) for c in columns} for row in rows]
        job_config = self._bq.LoadJobConfig(
            schema=self.schema(name),
            write_disposition=self._bq.WriteDisposition.WRITE_APPEND,
            source_format=self._bq.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        job = self.client.load_table_from_json(payload, self.cfg.table(name), job_config=job_config,
                                               location=self.cfg.bq_location)
        job.result(timeout=300)
        return len(payload)


def bins_sql(cfg: MixConfig) -> str:
    """10-minute bins with the Maps columns xgb[maps] reads; one route per direction, averaged to be safe."""
    return f"""
SELECT direction, bin_ts, AVG(dur_min) AS dur_min, AVG(congestion_ratio) AS congestion_ratio, AVG(speed_kmh) AS speed_kmh
FROM `{cfg.base.view}`
WHERE direction IN UNNEST(@directions)
  AND dur_min IS NOT NULL
  AND bin_ts BETWEEN @start AND @end
GROUP BY direction, bin_ts
ORDER BY direction, bin_ts
""".strip()


def latest_registry_sql(cfg: MixConfig) -> str:
    return f"""
SELECT run_id, created_at, labels_before, config_json, decisions_json, artefacts_json
FROM `{cfg.table(TABLE_REGISTRY)}`
WHERE model = @model AND created_at >= @since
ORDER BY created_at DESC
LIMIT 1
""".strip()


def monitor_sql(cfg: MixConfig) -> str:
    """The latest logged forecast per origin and direction, joined to the bin that later arrived."""
    return f"""
WITH f AS (
  SELECT origin_ts, direction, target_ts, persistence, fcm_mlp, xgb_maps, mix, served_model, served_value,
         ROW_NUMBER() OVER (PARTITION BY origin_ts, direction ORDER BY created_at DESC) AS rn
  FROM `{cfg.table(TABLE_FORECASTS)}`
  WHERE model = @model AND origin_ts BETWEEN @start AND @end
),
a AS (
  SELECT direction, bin_ts, AVG(dur_min) AS actual
  FROM `{cfg.base.view}`
  WHERE dur_min IS NOT NULL AND bin_ts BETWEEN @start AND @label_end
  GROUP BY direction, bin_ts
)
SELECT f.origin_ts, f.direction, f.target_ts, f.persistence, f.fcm_mlp, f.xgb_maps, f.mix, f.served_model,
       f.served_value, a.actual
FROM f
JOIN a ON a.direction = f.direction AND a.bin_ts = f.target_ts
WHERE f.rn = 1
ORDER BY f.origin_ts, f.direction
""".strip()


# ----------------------------------------------------------------------------------------------
# Data: bins on a regular grid, Maps features, per-direction feature sets
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Bins:
    """Travel time, congestion ratio and speed per direction on one 10-minute grid (NaN = no bin)."""

    grid: fcm.Grid
    dur: Mapping[str, np.ndarray]
    congestion: Mapping[str, np.ndarray]
    speed: Mapping[str, np.ndarray]
    cut: bool = False  # the read starts after the history start, so the first bin's predecessor is unknown


def bins_from_rows(rows: Sequence[Mapping[str, Any]], grid: fcm.Grid, *, cut: bool = False) -> Bins:
    """Place query rows on the grid; a row without a positive travel time is left out entirely."""
    dur = {d: np.full(grid.n, np.nan) for d in DIRECTIONS}
    congestion = {d: np.full(grid.n, np.nan) for d in DIRECTIONS}
    speed = {d: np.full(grid.n, np.nan) for d in DIRECTIONS}
    for row in rows:
        direction = row.get("direction")
        value = _num(row.get("dur_min"))
        if direction not in dur or value is None or value <= 0:
            continue
        offset = fcm.unix(row["bin_ts"]) - grid.start_unix
        if offset % BIN_SECONDS or not 0 <= offset < grid.n * BIN_SECONDS:
            continue
        i = offset // BIN_SECONDS
        dur[direction][i] = value
        ratio, kmh = _num(row.get("congestion_ratio")), _num(row.get("speed_kmh"))
        congestion[direction][i] = np.nan if ratio is None else ratio
        speed[direction][i] = np.nan if kmh is None else kmh
    return Bins(grid, dur, congestion, speed, cut)


def load_bins(cfg: MixConfig, wh: Any, first: dt.datetime, last: dt.datetime) -> Bins:
    grid = fcm.Grid.spanning(first, last)
    rows = wh.query(bins_sql(cfg), {"directions": list(DIRECTIONS), "start": grid.time(0), "end": grid.time(grid.n - 1)})
    return bins_from_rows(rows, grid, cut=grid.start_unix > day_start_unix(cfg.history_start_day))


def after_gap_flags(dur: np.ndarray, *, first_is_gap: bool = False) -> np.ndarray:
    """``after_gap`` of ``v_training_set`` on the grid, True at an observed bin near a gap.

    A gap is more than 25 minutes since the previous observed bin; the flag covers that bin and the
    six observed bins after it. The first bin read has no previous bin: when the read starts at the
    beginning of the data it is not a gap (``eval/features.maps_features``), and when the read is cut
    from a longer history (``first_is_gap``) it counts as one, as ``forecast-api`` does, so an outage
    longer than the read window cannot hide a gap.
    """
    flags = np.zeros(len(dur), dtype=bool)
    observed = np.flatnonzero(np.isfinite(dur))
    if not observed.size:
        return flags
    gap_min = np.zeros(len(observed))
    gap_min[0] = math.inf if first_is_gap else 0.0
    gap_min[1:] = np.diff(observed) * BIN_MINUTES
    big = np.concatenate([[0], np.cumsum(gap_min > GAP_MINUTES)])
    j = np.arange(len(observed))
    lo = np.maximum(0, j - (AFTER_GAP_ROWS - 1))
    flags[observed] = (big[j + 1] - big[lo]) > 0
    return flags


def _row_mean(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Mean over the finite values of each row (NaN when none), like pandas ``mean(skipna=True)``."""
    stack = np.column_stack(columns)
    finite = np.isfinite(stack)
    count = finite.sum(axis=1)
    total = np.where(finite, stack, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def maps_features(bins: Bins, direction: str, origin_idx: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The 18 xgb[maps] inputs at each origin and its ``after_gap`` flag.

    Reproduces ``eval/features.maps_features`` (and ``forecastapi/local_models.features_from_bins``):
    lags and labels are time-based, so a missing bin is NaN rather than the previous row, and the lag
    columns are blanked after a gap. XGBoost treats NaN as missing.
    """
    i = np.asarray(origin_idx, dtype=np.int64)
    dur = bins.dur[direction]
    lag = {k: _take(dur, i - k) for k in range(1, 7)}
    after_gap = after_gap_flags(dur, first_is_gap=bins.cut)[i] & np.isfinite(_take(dur, i))
    roll_30 = _row_mean([lag[1], lag[2], lag[3]])
    roll_60 = _row_mean([lag[k] for k in range(1, 7)])
    slope_30 = lag[1] - lag[4]
    blank = after_gap
    origin_unix = bins.grid.start_unix + i * BIN_SECONDS
    local = origin_unix + fcm.SGT_OFFSET_SEC
    tod_min = (local % DAY_SECONDS) // 60
    hour = tod_min // 60
    monday0 = (local // DAY_SECONDS + 3) % 7  # 1970-01-01 was a Thursday
    dow = (monday0 + 1) % 7 + 1               # BigQuery DAYOFWEEK: Sunday = 1 .. Saturday = 7
    columns = {
        "y_persistence": _take(dur, i),
        "congestion_ratio": _take(bins.congestion[direction], i),
        "speed_kmh": _take(bins.speed[direction], i),
        "lag_10": np.where(blank, np.nan, lag[1]),
        "lag_20": np.where(blank, np.nan, lag[2]),
        "lag_30": np.where(blank, np.nan, lag[3]),
        "lag_60": np.where(blank, np.nan, lag[6]),
        "roll_mean_30": np.where(blank, np.nan, roll_30),
        "roll_mean_60": np.where(blank, np.nan, roll_60),
        "slope_30": np.where(blank, np.nan, slope_30),
        "tod_sin": np.sin(2.0 * np.pi * tod_min / 1440.0),
        "tod_cos": np.cos(2.0 * np.pi * tod_min / 1440.0),
        "tod_block": tod_min / 5.0,
        "dow": dow.astype(float),
        "is_weekend": np.isin(dow, (1, 7)).astype(float),
        "is_morning_peak": ((hour >= 6) & (hour <= 10)).astype(float),
        "is_evening_peak": ((hour >= 16) & (hour <= 21)).astype(float),
        "is_my_to_sg": np.full(len(i), 1.0 if direction == "MY_TO_SG" else 0.0),
    }
    return np.column_stack([np.asarray(columns[name], dtype=float) for name in MAPS_FEATURES]), after_gap


@dataclasses.dataclass
class DirectionData:
    """Both members' inputs, labels and Layer A frames for the origins of one direction."""

    direction: str
    frame: fcm.Frame          # FCM + MLP features, labels for steps 1..H, baselines, vision
    maps: np.ndarray          # (n, len(MAPS_FEATURES))
    after_gap: np.ndarray     # (n,) bool

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.frame.now)

    @property
    def y30(self) -> np.ndarray:
        return self.frame.actual[:, TARGET_STEP - 1]


def protected_unix(cfg: MixConfig) -> Optional[Tuple[int, int]]:
    """First and last bin start (UTC seconds) of the protected window, or None when it is disabled."""
    if not all(cfg.base.protected_window):
        return None
    first, last = fcm.sgt_day_bounds(*(fcm.parse_sgt_date(d) for d in cfg.base.protected_window))
    return fcm.unix(first), fcm.unix(last)


def mask_protected_labels(cfg: MixConfig, frame: fcm.Frame) -> int:
    """Blank every label whose target bin lies in the protected window; returns how many were blanked."""
    bounds = protected_unix(cfg)
    if bounds is None:
        return 0
    steps = np.arange(1, frame.horizon + 1, dtype=np.int64)
    target = frame.origin_unix[:, None] + BIN_SECONDS * steps[None, :]
    inside = (target >= bounds[0]) & (target <= bounds[1]) & np.isfinite(frame.actual)
    frame.actual[inside] = np.nan
    return int(inside.sum())


def build_data(cfg: MixConfig, bins: Bins, origin_idx: Any, *, label_cutoff: dt.datetime,
               layer_a: Mapping[Tuple[str, dt.datetime], Mapping[str, Any]], protect_labels: bool) -> Dict[str, DirectionData]:
    """Both members' features at every origin of both directions; labels after ``label_cutoff`` are NaN."""
    data: Dict[str, DirectionData] = {}
    for d in DIRECTIONS:
        frame = fcm.build_frame(cfg.base, bins.grid, bins.dur[d], d, origin_idx, label_cutoff=label_cutoff)
        fcm.attach_vision(cfg.base, frame, layer_a)
        if protect_labels:
            mask_protected_labels(cfg, frame)
        maps, after_gap = maps_features(bins, d, origin_idx)
        data[d] = DirectionData(d, frame, maps, after_gap)
    return data


def subset_frame(frame: fcm.Frame, idx: Any) -> fcm.Frame:
    """The rows ``idx`` of a frame (every array and list field is sliced the same way)."""
    idx = np.asarray(idx, dtype=np.int64)
    values: Dict[str, Any] = {}
    for field in dataclasses.fields(frame):
        value = getattr(frame, field.name)
        if isinstance(value, np.ndarray):
            values[field.name] = value[idx]
        elif isinstance(value, list):
            values[field.name] = [value[int(i)] for i in idx]
        else:
            values[field.name] = value
    return fcm.Frame(**values)


# ----------------------------------------------------------------------------------------------
# Members: XGBoost wrapper, the daily fit and the mix
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class XGBMember:
    """An XGBoost regressor with the harness settings, stored as XGBoost's own JSON (no pickles)."""

    features: Tuple[str, ...]
    booster: Any  # xgboost.Booster
    n_train: int

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, features: Sequence[str]) -> XGBMember:
        from xgboost import XGBRegressor  # noqa: PLC0415 - lazy, so `show-sql` and the tests of other parts need no xgboost

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(features) or len(x) != len(y) or not len(y):
            raise ValueError("XGBoost training data do not match the feature list")
        if not np.isfinite(y).all():
            raise ValueError("XGBoost labels must be finite")
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(x, y)
        return cls(tuple(features), model.get_booster(), len(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        import xgboost  # noqa: PLC0415

        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} XGBoost inputs, got shape {x.shape}")
        if not len(x):
            return np.empty(0)
        return np.asarray(self.booster.predict(xgboost.DMatrix(x, missing=np.nan)), dtype=float)

    def to_json(self) -> Dict[str, Any]:
        raw = self.booster.save_raw(raw_format="json")
        return {"features": list(self.features), "n_train": self.n_train, "params": dict(XGB_PARAMS),
                "booster_json": bytes(raw).decode("utf-8")}

    @classmethod
    def from_json(cls, data: Mapping[str, Any], expected: Sequence[str]) -> XGBMember:
        import xgboost  # noqa: PLC0415

        features = tuple(data["features"])
        if features != tuple(expected):
            raise ValueError(f"stored XGBoost uses features {features}, this code computes {tuple(expected)}")
        booster = xgboost.Booster()
        booster.load_model(bytearray(str(data["booster_json"]).encode("utf-8")))
        if booster.num_features() != len(features):
            raise ValueError("stored XGBoost has the wrong number of inputs")
        return cls(features, booster, int(data["n_train"]))


@dataclasses.dataclass
class MixModels:
    """Everything a forecast needs: the FCM + MLP networks per direction and the XGBoost members."""

    labels_before: dt.datetime                     # 00:00 SGT of the serving day, in UTC
    networks: Dict[str, fcm.DirectionModels]       # fcm_mlp (and fcm_mlp_layer_a) per direction
    xgb_maps: Optional[XGBMember]
    xgb_maps_layer_a: Optional[XGBMember] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "labels_before": fcm.as_utc(self.labels_before).isoformat(),
            "networks": {d: m.to_json() for d, m in self.networks.items()},
            "xgb_maps": self.xgb_maps.to_json() if self.xgb_maps else None,
            "xgb_maps_layer_a": self.xgb_maps_layer_a.to_json() if self.xgb_maps_layer_a else None,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], horizon: int) -> Tuple[MixModels, List[str]]:
        """Rebuild the usable parts and list the unusable ones (a missing part serves persistence)."""
        problems: List[str] = []
        labels_before = fcm.as_utc(dt.datetime.fromisoformat(str(data["labels_before"])))
        networks: Dict[str, fcm.DirectionModels] = {}
        stored = data.get("networks")
        stored = stored if isinstance(stored, Mapping) else {}
        for d in DIRECTIONS:
            try:
                networks[d] = fcm.DirectionModels.from_json(stored[d], horizon)
            except (KeyError, TypeError, ValueError) as exc:
                problems.append(f"networks.{d}: {exc}")
        members: Dict[str, Optional[XGBMember]] = {}
        for key, expected in (("xgb_maps", MAPS_FEATURES), ("xgb_maps_layer_a", MAPS_FEATURES + VISION_FEATURES)):
            members[key] = None
            if data.get(key) is None:
                continue
            try:
                members[key] = XGBMember.from_json(data[key], expected)
            except (KeyError, TypeError, ValueError) as exc:  # xgboost raises XGBoostError (a ValueError)
                problems.append(f"{key}: {exc}")
        return cls(labels_before, networks, members["xgb_maps"], members["xgb_maps_layer_a"]), problems


@dataclasses.dataclass
class SameRows:
    """Twins of the vision members trained on the same Layer A rows without the vision inputs (slide 15)."""

    networks: Dict[str, fcm.MLPEnsemble]
    xgb_maps: Optional[XGBMember]


def mix_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equal-weight mix where both members have a forecast, NaN elsewhere."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    return np.where(both, MIX_WEIGHTS["fcm_mlp"] * a + MIX_WEIGHTS["xgb_maps"] * b, np.nan)


def fit_networks(cfg: MixConfig, dd: DirectionData, day: int, direction_index: int
                 ) -> Tuple[fcm.DirectionModels, Optional[fcm.MLPEnsemble], Dict[str, Any]]:
    """FCM + MLP for serving day ``day``, on the labels that closed before 00:00 SGT that day.

    The same procedure as ``layer_b_fcm_mlp.train_direction`` (FCM on the training days, the L2
    penalty and early stopping on the last two of them, seeds per direction), restricted to the
    lookback window and without the ablation network, which this module does not serve.
    """
    base, frame = cfg.base, dd.frame
    steps = np.arange(1, frame.horizon + 1, dtype=np.int64)
    target_unix = frame.origin_unix[:, None] + BIN_SECONDS * steps[None, :]
    origin_day = frame.origin_day
    in_window = (origin_day >= cfg.first_training_day(day)) & (origin_day < day)
    labels = frame.actual.copy()
    labels[target_unix > day_start_unix(day) - BIN_SECONDS] = np.nan  # the target bin must close before the day starts
    eligible = frame.complete & in_window & np.isfinite(labels).all(axis=1)
    days = sorted({int(x) for x in origin_day[eligible]})
    inner_n = base.inner_validation_days
    if len(days) < inner_n + fcm.MIN_FIT_DAYS:
        raise ConfigError(f"{dd.direction}: {len(days)} training days before {_iso_day(day)}; "
                          f"need at least {inner_n + fcm.MIN_FIT_DAYS}")
    split = fcm.DaySplit(tuple(days[:-inner_n]), tuple(days[-inner_n:]), (day,), ())
    origin_rank = split.rank(origin_day)
    labels[split.rank(fcm.sgt_epoch_day(target_unix)) > origin_rank[:, None]] = np.nan  # no label crosses a split
    labelled = np.isfinite(labels).all(axis=1)
    fit_rows = frame.complete & in_window & labelled & (origin_rank == 0)
    inner_rows = frame.complete & in_window & labelled & (origin_rank == 1)
    if fit_rows.sum() < fcm.MIN_TRAIN_ROWS or inner_rows.sum() < fcm.MIN_INNER_ROWS:
        raise ConfigError(f"{dd.direction}: {int(fit_rows.sum())} fitting and {int(inner_rows.sum())} inner-validation "
                          f"rows before {_iso_day(day)}; need at least {fcm.MIN_TRAIN_ROWS} and {fcm.MIN_INNER_ROWS}")
    seed = base.seed + 1000 * direction_index
    fcm_columns = [fcm.BASE_FEATURES.index(f) for f in fcm.FCM_FEATURES]
    fcm_rows = frame.complete & in_window & (origin_rank <= 1)  # unsupervised: labels are not needed
    clusters = fcm.FuzzyCMeans.fit(frame.X[fcm_rows][:, fcm_columns], features=fcm.FCM_FEATURES, c=base.fcm_clusters,
                                   m=base.fcm_fuzzifier, seed=seed, restarts=base.fcm_restarts)
    names = fcm.fuzzy_feature_names(clusters.labels)
    x_fuzzy = np.hstack([frame.X, fcm.fcm_inputs(frame, clusters)])
    deltas = labels - frame.now[:, None]

    def fit(x: np.ndarray, feature_names: Sequence[str], rows_fit: np.ndarray, rows_inner: np.ndarray,
            alpha: float) -> Tuple[fcm.MLPEnsemble, float]:
        return fcm.fit_mlp_ensemble(base, feature_names, x_fit=x[rows_fit], y_fit=deltas[rows_fit], x_inner=x[rows_inner],
                                    y_inner=deltas[rows_inner], now_fit=frame.now[rows_fit], alpha=alpha, seed=seed)

    search: Dict[str, float] = {}
    best: Optional[Tuple[fcm.MLPEnsemble, float, float]] = None
    for alpha in base.mlp_alphas:  # the L2 penalty is chosen on the inner-validation days only
        network, inner_mae = fit(x_fuzzy, names, fit_rows, inner_rows, alpha)
        search[f"{alpha:g}"] = inner_mae
        if best is None or inner_mae < best[1]:
            best = (network, inner_mae, alpha)
    assert best is not None  # mlp_alphas is never empty (Config.validate)
    network, _, alpha = best

    with_vision = frame.complete & np.isfinite(frame.vision).all(axis=1)
    vision_network: Optional[fcm.MLPEnsemble] = None
    same_rows: Optional[fcm.MLPEnsemble] = None
    if (fit_rows & with_vision).sum() >= fcm.MIN_TRAIN_ROWS and (inner_rows & with_vision).sum() >= fcm.MIN_INNER_ROWS:
        vision_network, _ = fit(np.hstack([x_fuzzy, frame.vision]), names + VISION_FEATURES, fit_rows & with_vision,
                                inner_rows & with_vision, alpha)
        same_rows, _ = fit(x_fuzzy, names, fit_rows & with_vision, inner_rows & with_vision, alpha)
    diagnostics = {
        "fcm_centres": clusters.describe(),
        "alpha": alpha,
        "alpha_search_inner_mae": search,
        "best_epochs": network.best_epochs,
        "rows": {"fit": int(fit_rows.sum()), "inner_validation": int(inner_rows.sum()),
                 "with_layer_a": int((fit_rows & with_vision).sum())},
        "fit_days": [_iso_day(split.fit_days[0]), _iso_day(split.fit_days[-1])],
        "inner_days": [_iso_day(x) for x in split.inner_days],
    }
    return fcm.DirectionModels(clusters, network, vision_network), same_rows, diagnostics


def xgb_training_set(cfg: MixConfig, data: Mapping[str, DirectionData], day: int, *, vision: bool
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of ``eval/joined.fold_split`` for serving day ``day``, in the harness's row order.

    A row qualifies when its origin is observed, it is not after a gap, its 30-minute label exists and
    the label's bin closed before 00:00 SGT on ``day``. With ``vision``, only rows with an accepted
    Layer A frame qualify and the frame's three inputs are appended.
    """
    cutoff = day_start_unix(day)
    first_day = cfg.first_training_day(day)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for d in sorted(DIRECTIONS):  # eval/joined.scorable sorts by direction, then time
        dd = data[d]
        rows = (dd.observed & ~dd.after_gap & np.isfinite(dd.y30)
                & (dd.frame.origin_unix + LABEL_LAG_SECONDS <= cutoff) & (dd.frame.origin_day >= first_day))
        x = dd.maps
        if vision:
            rows &= np.isfinite(dd.frame.vision).all(axis=1)
            x = np.hstack([dd.maps, dd.frame.vision])
        xs.append(x[rows])
        ys.append(dd.y30[rows])
    return np.vstack(xs), np.concatenate(ys)


def fit_day(cfg: MixConfig, data: Mapping[str, DirectionData], day: int) -> Tuple[MixModels, SameRows, Dict[str, Any]]:
    """Fit every member for serving day ``day`` (labels closed before 00:00 SGT that day)."""
    networks: Dict[str, fcm.DirectionModels] = {}
    same_networks: Dict[str, fcm.MLPEnsemble] = {}
    diagnostics: Dict[str, Any] = {"labels_before": fcm.from_unix(day_start_unix(day))}
    for index, d in enumerate(DIRECTIONS):
        try:
            models, same, note = fit_networks(cfg, data[d], day, index)
        except ConfigError as exc:
            log("WARNING", "FCM + MLP not fitted for this day and direction, so the mix is unavailable there",
                direction=d, day=_iso_day(day), error=str(exc))
            diagnostics[d] = {"error": str(exc)}
            continue
        networks[d] = models
        diagnostics[d] = note
        if same is not None:
            same_networks[d] = same
    if not networks:
        raise ConfigError(f"FCM + MLP could not be fitted for either direction before {_iso_day(day)}")
    x, y = xgb_training_set(cfg, data, day, vision=False)
    if len(y) < cfg.min_xgb_rows:
        raise ConfigError(f"xgb[maps]: {len(y)} training rows before {_iso_day(day)}; need at least {cfg.min_xgb_rows}")
    xgb_maps = XGBMember.fit(x, y, MAPS_FEATURES)
    xgb_vision: Optional[XGBMember] = None
    xgb_same: Optional[XGBMember] = None
    xv, yv = xgb_training_set(cfg, data, day, vision=True)
    if len(yv) >= cfg.min_xgb_rows:
        xgb_vision = XGBMember.fit(xv, yv, MAPS_FEATURES + VISION_FEATURES)
        xgb_same = XGBMember.fit(xv[:, :len(MAPS_FEATURES)], yv, MAPS_FEATURES)  # slide 15: same rows, no vision
    diagnostics["xgb_rows"] = {"maps": len(y), "layer_a": len(yv)}
    labels_before = fcm.from_unix(day_start_unix(day))
    return MixModels(labels_before, networks, xgb_maps, xgb_vision), SameRows(same_networks, xgb_same), diagnostics


PREDICTION_KEYS: Tuple[str, ...] = ("fcm_mlp", "fcm_mlp_spread", "xgb_maps", "mix", "fcm_mlp_layer_a",
                                    "xgb_maps_layer_a", "mix_layer_a", "mix_same_rows")


def predict_direction(models: MixModels, dd: DirectionData, idx: Any, same: Optional[SameRows] = None) -> Dict[str, Any]:
    """Every member's 30-minute forecast at the origins ``idx`` of one direction (NaN when unavailable).

    xgb[maps] is not used after a data gap, so neither is the mix there: the evaluated protocol never
    trained on or scored such origins.
    """
    idx = np.asarray(idx, dtype=np.int64)
    out: Dict[str, Any] = {key: np.full(len(idx), np.nan) for key in PREDICTION_KEYS}
    out["memberships"] = None
    sub = subset_frame(dd.frame, idx)
    step = TARGET_STEP - 1
    network = models.networks.get(dd.direction)
    if network is not None:
        p = fcm.predict_frame(network, sub)
        out["fcm_mlp"], out["fcm_mlp_spread"] = p["fcm_mlp"][:, step], p["fcm_mlp_spread"][:, step]
        out["fcm_mlp_layer_a"], out["memberships"] = p["fcm_mlp_layer_a"][:, step], p["memberships"]
    maps = dd.maps[idx]
    usable = np.isfinite(sub.now) & ~dd.after_gap[idx]
    with_vision = usable & np.isfinite(sub.vision).all(axis=1)
    if models.xgb_maps is not None and usable.any():
        out["xgb_maps"][usable] = models.xgb_maps.predict(maps[usable])
    if models.xgb_maps_layer_a is not None and with_vision.any():
        out["xgb_maps_layer_a"][with_vision] = models.xgb_maps_layer_a.predict(np.hstack([maps, sub.vision])[with_vision])
    out["mix"] = mix_of(out["fcm_mlp"], out["xgb_maps"])
    out["mix_layer_a"] = mix_of(out["fcm_mlp_layer_a"], out["xgb_maps_layer_a"])
    twin = same.networks.get(dd.direction) if same is not None else None
    if same is not None and twin is not None and same.xgb_maps is not None and out["memberships"] is not None:
        rows = sub.complete & np.isfinite(sub.vision).all(axis=1)
        fcm_twin = np.full(len(idx), np.nan)
        if rows.any():
            mean, _ = twin.predict(np.hstack([sub.X, out["memberships"]])[rows], sub.now[rows])
            fcm_twin[rows] = mean[:, step]
        xgb_twin = np.full(len(idx), np.nan)
        if with_vision.any():
            xgb_twin[with_vision] = same.xgb_maps.predict(maps[with_vision])
        out["mix_same_rows"] = mix_of(fcm_twin, xgb_twin)
    return out


# ----------------------------------------------------------------------------------------------
# Rolling daily refit, decision rules and reports
# ----------------------------------------------------------------------------------------------


def evaluation_rows(data: Mapping[str, DirectionData], models: MixModels, same: Optional[SameRows],
                    day: int) -> List[Dict[str, Any]]:
    """One row per scorable origin of ``day`` (observed, not after a gap, 30-minute label present)."""
    rows: List[Dict[str, Any]] = []
    for d in DIRECTIONS:
        dd = data[d]
        sel = np.flatnonzero((dd.frame.origin_day == day) & dd.observed & ~dd.after_gap & np.isfinite(dd.y30))
        if not sel.size:
            continue
        p = predict_direction(models, dd, sel, same)
        for j, i in enumerate(sel):
            origin = fcm.from_unix(int(dd.frame.origin_unix[i]))
            rows.append({
                "direction": d,
                "origin_ts": origin,
                "step": TARGET_STEP,
                "target_ts": origin + TARGET_STEP * BIN,
                "actual": float(dd.y30[i]),
                "persistence": float(dd.frame.now[i]),
                **{key: _num(p[key][j]) for key in ROW_MODELS},
                "layer_a_status": dd.frame.vision_status[int(i)],
                "labels_before": models.labels_before,
            })
    return rows


def rolling_refit(cfg: MixConfig, data: Mapping[str, DirectionData], days: Sequence[int]
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fit every member for each day on that day's past only, and score the day's origins."""
    rows: List[Dict[str, Any]] = []
    folds: Dict[str, Any] = {}
    for day in days:
        try:
            models, same, diagnostics = fit_day(cfg, data, day)
        except ConfigError as exc:
            log("WARNING", "day skipped: too little history", day=_iso_day(day), error=str(exc))
            folds[_iso_day(day)] = {"error": str(exc)}
            continue
        fold = evaluation_rows(data, models, same, day)
        rows.extend(fold)
        folds[_iso_day(day)] = {**diagnostics, "rows": len(fold)}
        log("INFO", "day scored", day=_iso_day(day), rows=len(fold))
    return rows, folds


def served_value(row: Mapping[str, Any], decision: Any, fallback: Any = None) -> Tuple[str, Optional[float]]:
    """Value for a registered decision, falling back when an input is missing.

    Without a live Layer A frame, ``mix_layer_a`` falls back to ``fallback``: the Maps-only mix only if
    it beat persistence on its own, otherwise persistence. A model that failed the persistence gate is
    therefore never served, whichever input is missing.
    """
    if not isinstance(fallback, str) or fallback not in SERVE_CHAIN:
        fallback = "persistence"  # missing or malformed in the registry: the conservative choice
    if decision == "mix_layer_a":
        chain: Tuple[str, ...] = ("mix_layer_a", *SERVE_CHAIN[fallback])
    else:
        chain = SERVE_CHAIN.get(decision, ("persistence",)) if isinstance(decision, str) else ("persistence",)
    for model in chain:
        if row.get(model) is not None:
            return model, float(row[model])
    return "persistence", None


def decide(cfg: MixConfig, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Decision rules 1-3 per direction, on out-of-sample validation rows only."""
    decisions: Dict[str, Dict[str, Any]] = {}
    for d in DIRECTIONS:
        v = [r for r in rows if r["direction"] == d]
        note: Dict[str, Any] = {"rows": len(v)}
        note["mix_vs_persistence"] = fcm.compare(v, "mix", "persistence", cfg.base)
        note["mix_vs_xgb_maps"] = fcm.compare(v, "mix", "xgb_maps", cfg.base)  # reported, not gated
        note["mix_vs_fcm_mlp"] = fcm.compare(v, "mix", "fcm_mlp", cfg.base)    # reported, not gated
        # what is served when no camera frame is available: the mix only if it passed rule 3 alone
        fallback = "mix" if fcm.better(note["mix_vs_persistence"]) else "persistence"
        policy = "mix"
        vision_rows = [r for r in v if r.get("mix_layer_a") is not None and r.get("mix_same_rows") is not None]
        coverage = len(vision_rows) / len(v) if v else 0.0
        note["layer_a_coverage"] = coverage
        if vision_rows:
            note["mix_layer_a_vs_same_rows"] = fcm.compare(vision_rows, "mix_layer_a", "mix_same_rows", cfg.base)
            note["mix_layer_a_vs_fallback"] = fcm.compare(vision_rows, "mix_layer_a", fallback, cfg.base)
            if (coverage >= fcm.MIN_VISION_COVERAGE and fcm.better(note["mix_layer_a_vs_same_rows"])
                    and fcm.better(note["mix_layer_a_vs_fallback"])):
                policy = "mix_layer_a"
        as_served = [{**r, "policy": served_value(r, policy, fallback)[1]} for r in v]
        note["policy_vs_persistence"] = fcm.compare(as_served, "policy", "persistence", cfg.base)
        note["candidate"] = policy
        note["fallback"] = fallback
        note["served"] = policy if fcm.better(note["policy_vs_persistence"]) else "persistence"
        decisions[d] = note
    return decisions


def apply_decisions(rows: Sequence[Dict[str, Any]], decisions: Mapping[str, Mapping[str, Any]]) -> None:
    for r in rows:
        entry = decisions.get(r["direction"], {})
        r["served_model"], r["served_value"] = served_value(r, entry.get("served", "persistence"), entry.get("fallback"))


def report(cfg: MixConfig, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Metrics per model and subgroup (slide 15) and paired day-block comparisons, pooled and per direction.

    ``metrics`` scores persistence, both members and the mix on the same rows (those where all four
    exist); ``metrics_all_rows`` scores every model on every row it has, so its counts differ.
    """
    common = [r for r in rows if all(r.get(k) is not None for k in CORE_MODELS)]
    models = REPORT_MODELS + (("served_value",) if any(r.get("served_value") is not None for r in rows) else ())
    return {
        "rows": len(rows),
        "common_rows": len(common),
        "days": len({fcm.sgt_date(r["origin_ts"]) for r in rows}),
        "metrics": fcm.summarise(common, CORE_MODELS, [TARGET_STEP]).get(f"step_{TARGET_STEP}", {}),
        "metrics_all_rows": fcm.summarise(rows, models, [TARGET_STEP]).get(f"step_{TARGET_STEP}", {}),
        "pooled": {f"{a}_vs_{b}": fcm.compare(rows, a, b, cfg.base) for a, b in COMPARISONS},
        "by_direction": {d: {f"{a}_vs_{b}": fcm.compare([r for r in rows if r["direction"] == d], a, b, cfg.base)
                             for a, b in COMPARISONS} for d in DIRECTIONS},
    }


def headline(result: Mapping[str, Any]) -> Dict[str, Any]:
    """MAE per model on the common rows (both directions) and the main comparisons, for a short printout."""
    metrics = result.get("metrics", {})
    mae = {m: metrics[m]["all"].get("mae") for m in metrics if metrics[m].get("all", {}).get("n")}
    served = result.get("metrics_all_rows", {}).get("served_value", {}).get("all", {})
    if served.get("n"):
        mae["served_value (all rows)"] = served.get("mae")
    pooled = result.get("pooled", {})
    keep = ("mix_vs_persistence", "mix_vs_xgb_maps", "mix_vs_fcm_mlp")
    return {"rows": result.get("rows"), "common_rows": result.get("common_rows"), "days": result.get("days"), "mae_min": mae,
            "comparisons": {k: {f: pooled[k].get(f) for f in ("mean", "ci_low", "ci_high", "days")} for k in keep if k in pooled}}


# ----------------------------------------------------------------------------------------------
# Windows: what to read for a set of scored days
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Window:
    first_origin: dt.datetime   # first origin any fit may use (history start, or the lookback cap)
    last_origin: dt.datetime    # last origin scored
    data_first: dt.datetime     # first bin read (one week of feature history before first_origin)
    data_last: dt.datetime      # last bin read; labels after it are unknown


def plan_window(cfg: MixConfig, first_day: int, last_day: int, now: dt.datetime) -> Window:
    first_origin = fcm.from_unix(day_start_unix(cfg.first_training_day(first_day)))
    last_origin = fcm.from_unix(day_start_unix(last_day + 1)) - BIN
    if last_origin < first_origin:
        raise ConfigError(f"the window ends before the history start {cfg.history_start}")
    data_last = min(last_origin + TARGET_STEP * BIN, fcm.last_complete_bin(now))
    if data_last < first_origin:
        raise ConfigError("the window is entirely in the future")
    return Window(first_origin, last_origin, fcm.history_start(cfg.base, first_origin), data_last)


def load_window(cfg: MixConfig, wh: Any, first_day: int, last_day: int, now: dt.datetime, *,
                allow_protected: bool) -> Tuple[Dict[str, DirectionData], Window]:
    """Read the bins and Layer A frames to score days ``first_day``..``last_day`` and fit each on its past."""
    window = plan_window(cfg, first_day, last_day, now)
    fcm.check_protected_window(cfg.base, window.first_origin, window.last_origin, allow_protected)
    bins = load_bins(cfg, wh, window.data_first, window.data_last)
    layer_a: Dict[Tuple[str, dt.datetime], Dict[str, Any]] = {}
    try:
        layer_a = fcm.load_layer_a_features(cfg.base, wh, window.first_origin, window.last_origin)
    except Exception as exc:  # noqa: BLE001 - e.g. the table does not exist yet; fit without vision
        log("WARNING", "Layer A features unavailable; fitting without them", error=repr(exc))
    grid = bins.grid
    origins = np.arange(grid.index(window.first_origin), grid.index(min(window.last_origin, window.data_last)) + 1)
    data = build_data(cfg, bins, origins, label_cutoff=window.data_last, layer_a=layer_a,
                      protect_labels=not allow_protected)
    return data, window


# ----------------------------------------------------------------------------------------------
# Commands: train, backtest, monitor
# ----------------------------------------------------------------------------------------------


def train(cfg: MixConfig, wh: Any, *, end: dt.date, now: dt.datetime, allow_protected: bool = False) -> Dict[str, Any]:
    """Score the validation days by rolling refit, apply the decision rules, fit the serving models."""
    end_day = fcm.date_to_epoch_day(end)
    serve_day = end_day + 1
    days = tuple(range(end_day - cfg.validation_days + 1, end_day + 1))
    data, _ = load_window(cfg, wh, days[0], end_day, now, allow_protected=allow_protected)
    rows, folds = rolling_refit(cfg, data, days)
    decisions = decide(cfg, rows)  # fixed before the serving models are fitted
    apply_decisions(rows, decisions)
    models, _, final = fit_day(cfg, data, serve_day)
    return {
        "model": MODEL_NAME,
        "labels_before": models.labels_before,
        "train_start": fcm.epoch_day_to_date(cfg.first_training_day(serve_day)),
        "train_end": end,
        "validation_start": fcm.epoch_day_to_date(days[0]),
        "validation_end": end,
        "decisions": decisions,
        "validation": report(cfg, rows),
        "folds": folds,
        "final_fit": final,
        "models": models,
        "rows": rows,
    }


def backtest(cfg: MixConfig, wh: Any, *, start: dt.date, end: dt.date, now: dt.datetime,
             allow_protected: bool = False) -> Dict[str, Any]:
    """Rolling daily refit over ``start``..``end``: the protocol of docs/evaluation.md section 8."""
    if end < start:
        raise ConfigError(f"end date {end} is before start date {start}")
    first_day, last_day = fcm.date_to_epoch_day(start), fcm.date_to_epoch_day(end)
    data, window = load_window(cfg, wh, first_day, last_day, now, allow_protected=allow_protected)
    rows, folds = rolling_refit(cfg, data, range(first_day, last_day + 1))
    if not rows:
        raise ConfigError(f"no origin could be scored from {start} to {end} (too little history or data)")
    return {"model": MODEL_NAME, "start": start, "end": end, "first_training_origin": window.first_origin,
            "backtest": report(cfg, rows), "folds": folds, "rows": rows}


def monitor(cfg: MixConfig, wh: Any, *, days: int, now: dt.datetime, allow_protected: bool = False) -> Dict[str, Any]:
    """Live error of the logged forecasts against persistence and the members (slide 11)."""
    if not 1 <= days <= 90:
        raise ConfigError("--days must be between 1 and 90")
    last_origin = fcm.last_complete_bin(now) - TARGET_STEP * BIN  # its target bin has closed
    first_origin = last_origin - days * dt.timedelta(days=1) + BIN
    fcm.check_protected_window(cfg.base, first_origin, last_origin, allow_protected)
    found = wh.query(monitor_sql(cfg), {"model": MODEL_NAME, "start": first_origin, "end": last_origin,
                                        "label_end": last_origin + TARGET_STEP * BIN})
    bounds = None if allow_protected else protected_unix(cfg)
    rows = []
    for r in found:
        target = fcm.as_utc(r["target_ts"])
        actual = _num(r.get("actual"))
        if actual is None or (bounds is not None and bounds[0] <= fcm.unix(target) <= bounds[1]):
            continue
        rows.append({"direction": r["direction"], "origin_ts": fcm.as_utc(r["origin_ts"]), "target_ts": target,
                     "step": TARGET_STEP, "actual": actual, "persistence": _num(r.get("persistence")),
                     "fcm_mlp": _num(r.get("fcm_mlp")), "xgb_maps": _num(r.get("xgb_maps")), "mix": _num(r.get("mix")),
                     "served_value": _num(r.get("served_value"))})
    result = report(cfg, rows)
    result["served_vs_persistence"] = fcm.compare(rows, "served_value", "persistence", cfg.base)
    return {"model": MODEL_NAME, "first_origin": first_origin, "last_origin": last_origin, **result}


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------


def registry_row(cfg: MixConfig, result: Mapping[str, Any], run_id: str, now: dt.datetime) -> Dict[str, Any]:
    slim = {d: {k: n[k] for k in ("served", "candidate", "fallback")} for d, n in result["decisions"].items()}
    metrics = {"decisions": result["decisions"], "validation": result["validation"], "folds": result["folds"],
               "final_fit": result["final_fit"]}
    return {
        "run_id": run_id,
        "created_at": now,
        "model": MODEL_NAME,
        "labels_before": result["labels_before"],
        "train_start": result["train_start"],
        "train_end": result["train_end"],
        "validation_start": result["validation_start"],
        "validation_end": result["validation_end"],
        "config_json": json.dumps(_jsonable(model_config(cfg)), sort_keys=True),
        "decisions_json": json.dumps(_jsonable(slim), sort_keys=True),
        "artefacts_json": json.dumps(result["models"].to_json(), sort_keys=True),
        "metrics_json": json.dumps(_jsonable(metrics), sort_keys=True),
    }


def evaluation_table_rows(rows: Sequence[Mapping[str, Any]], split: str, run_id: str, now: dt.datetime) -> List[Dict[str, Any]]:
    return [{**r, "split": split, "run_id": run_id, "created_at": now, "model": MODEL_NAME} for r in rows]


def load_registered(cfg: MixConfig, wh: Any, now: dt.datetime) -> Tuple[Optional[str], Mapping[str, Any], Optional[MixModels]]:
    """Latest registered decisions and models; an unusable part is left out (and persistence served).

    The recent partitions are read first; the whole registry only when they are empty, so a model
    registered before a long pause of the daily job (the protected window) is still served.
    """
    recent = fcm.as_utc(now) - dt.timedelta(days=RECENT_REGISTRY_DAYS)
    found = wh.query(latest_registry_sql(cfg), {"model": MODEL_NAME, "since": recent})
    if not found:
        log("WARNING", f"no registry entry in the last {RECENT_REGISTRY_DAYS} days; reading the whole registry")
        found = wh.query(latest_registry_sql(cfg), {"model": MODEL_NAME, "since": fcm.from_unix(0)})
    if not found:
        log("WARNING", "no registry entry; serving persistence until `train --register` has run")
        return None, {}, None
    entry = found[0]
    run_id = str(entry["run_id"])
    try:
        stored = json.loads(entry.get("config_json") or "{}")
        drift = fcm.config_drift(cfg.base, stored.get("fcm") if isinstance(stored, Mapping) else None)
    except ValueError:
        drift = ["config_json unreadable"]
    if drift:  # the shapes still match, so serve, but say loudly that features may differ from training
        log("WARNING", "serving configuration differs from the training run", keys=drift, registry_run_id=run_id)
    decisions = json.loads(entry["decisions_json"])
    artefacts = json.loads(entry["artefacts_json"])
    if not isinstance(decisions, Mapping) or not isinstance(artefacts, Mapping):
        raise ValueError("registry entry is not a JSON object")
    models, problems = MixModels.from_json(artefacts, cfg.base.horizon_steps)
    for problem in problems:
        log("ERROR", "unusable registered member; the mix falls back where it is needed", problem=problem,
            registry_run_id=run_id)
    return run_id, decisions, models


# ----------------------------------------------------------------------------------------------
# Live cycle
# ----------------------------------------------------------------------------------------------


def _regime(models: Optional[MixModels], direction: str, memberships: Any, value: Optional[float]
            ) -> Dict[str, Any]:
    """Fuzzy regime of the origin state and of the served forecast (slide 10: interpretable levels)."""
    out: Dict[str, Any] = {"regime": None, "regime_degree": None, "memberships_json": None,
                           "forecast_regime": None, "forecast_regime_degree": None}
    network = models.networks.get(direction) if models is not None else None
    if network is None:
        return out
    labels = network.fcm.labels
    if memberships is not None and np.isfinite(memberships[0]).all():
        j = int(np.argmax(memberships[0]))
        out.update(regime=labels[j], regime_degree=float(memberships[0, j]),
                   memberships_json=json.dumps({label: round(float(u), 4)
                                                for label, u in zip(labels, memberships[0], strict=True)}))
    if value is not None:
        level = network.fcm.level_memberships(np.array([value]))[0]
        out.update(forecast_regime=labels[int(np.argmax(level))], forecast_regime_degree=float(level.max()))
    return out


def run_cycle(cfg: MixConfig, wh: Any, session: Any, *, now: Optional[dt.datetime] = None, use_layer_a: bool = True,
              write: bool = True) -> Dict[str, Any]:
    """Log a Layer A observation, forecast both directions with the registered policy, log the forecasts."""
    now = now or fcm.utc_now()
    run_id = f"cycle-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    summary: Dict[str, Any] = {"run_id": run_id, "model": MODEL_NAME}
    if write:
        wh.ensure_tables()  # idempotent; avoids a failed first run before `ensure-tables`

    if use_layer_a and cfg.base.layer_a_url:
        try:
            summary["layer_a_ingest"] = fcm.ingest_layer_a(cfg.base, wh, session, now) if write else {"status": "skipped_dry_run"}
        except Exception as exc:  # noqa: BLE001 - Layer B must still forecast when Layer A is down
            summary["layer_a_ingest"] = {"status": "failed", "error": str(exc)}
            log("WARNING", "Layer A ingest failed; forecasting without it", error=repr(exc))
    else:
        summary["layer_a_ingest"] = {"status": "disabled"}

    origin = fcm.last_complete_bin(now)
    summary["origin_ts"] = origin
    bins = load_bins(cfg, wh, fcm.history_start(cfg.base, origin), origin)

    registry_run_id: Optional[str] = None
    decisions: Mapping[str, Any] = {}
    models: Optional[MixModels] = None
    try:
        registry_run_id, decisions, models = load_registered(cfg, wh, now)
    except Exception as exc:  # noqa: BLE001 - an unreadable registry means the conservative default
        registry_run_id, decisions, models = None, {}, None
        log("ERROR", "registry unreadable; serving persistence", error=repr(exc))
    summary["registry_run_id"] = registry_run_id
    if models is not None:
        age_days = (day_start_unix(int(fcm.sgt_epoch_day(fcm.unix(now)))) - fcm.unix(models.labels_before)) / DAY_SECONDS
        summary["model_age_days"] = age_days
        if age_days > 1:
            log("WARNING", "registered model is more than a day old; is the daily train job running?",
                labels_before=models.labels_before, age_days=age_days)

    layer_a: Dict[Tuple[str, dt.datetime], Dict[str, Any]] = {}
    if use_layer_a:
        try:
            layer_a = fcm.load_layer_a_features(cfg.base, wh, origin, origin)
        except Exception as exc:  # noqa: BLE001 - forecast without vision rather than not at all
            log("ERROR", "Layer A features unavailable; forecasting without them", error=repr(exc))

    origin_index = np.array([bins.grid.index(origin)])
    data = build_data(cfg, bins, origin_index, label_cutoff=origin, layer_a=layer_a, protect_labels=False)
    out_rows: List[Dict[str, Any]] = []
    for d in DIRECTIONS:
        dd = data[d]
        if not dd.observed[0]:
            log("WARNING", "no travel time in the origin bin; direction skipped", direction=d, origin_ts=origin)
            continue
        predictions: Dict[str, Any] = {}
        if models is not None:
            try:
                predictions = predict_direction(models, dd, [0])
            except (ValueError, FloatingPointError) as exc:
                log("ERROR", "prediction failed; serving persistence", direction=d, error=str(exc))
        row: Dict[str, Any] = {"persistence": _num(dd.frame.now[0])}
        for key in ("fcm_mlp", "fcm_mlp_spread", "xgb_maps", "mix", "fcm_mlp_layer_a", "xgb_maps_layer_a", "mix_layer_a"):
            row[key] = _num(predictions[key][0]) if predictions else None
        entry = decisions.get(d)
        if not isinstance(entry, Mapping):  # a direction missing from the registry is served persistence
            entry = {}
        decision = entry.get("served", "persistence")
        if decision not in SERVABLE:
            decision = "persistence"
        served_model, value = served_value(row, decision, entry.get("fallback"))
        out_rows.append({
            "run_id": run_id, "created_at": now, "origin_ts": origin, "direction": d, "horizon_min": HORIZON_MIN,
            "target_ts": origin + TARGET_STEP * BIN, **row,
            "served_model": served_model, "served_value": value, "decision": decision,
            **_regime(models, d, predictions.get("memberships"), value),
            "after_gap": bool(dd.after_gap[0]), "layer_a_status": dd.frame.vision_status[0],
            "labels_before": models.labels_before if models is not None else None,
            "registry_run_id": registry_run_id, "model": MODEL_NAME,
        })
    if not out_rows:
        raise NoForecastError(f"no direction had fresh data for origin {origin.isoformat()}")
    if write:
        wh.append(TABLE_FORECASTS, out_rows)
    summary["forecasts"] = [
        {k: r[k] for k in ("direction", "horizon_min", "target_ts", "served_model", "served_value", "mix", "fcm_mlp",
                           "xgb_maps", "persistence", "forecast_regime", "after_gap", "layer_a_status")}
        for r in out_rows
    ]
    return summary


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------


def _print(obj: Any) -> None:
    print(json.dumps(_jsonable(obj), sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SwiftBorder Layer B: equal mix of FCM + MLP and xgb[maps], fed by Layer A.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ensure-tables", help="create this module's tables and the shared Layer A table if missing")

    p = sub.add_parser("run-cycle", help="log Layer A, forecast, log the forecasts (Cloud Run Job entry point)")
    p.add_argument("--no-layer-a", action="store_true", help="skip Layer A ingestion and features")
    p.add_argument("--dry-run", action="store_true", help="forecast and print without writing to BigQuery")

    p = sub.add_parser("train", help="rolling validation, decision rules, fit and register the serving models")
    p.add_argument("--end", help="last SGT date of labels, YYYY-MM-DD (default: yesterday in Singapore)")
    p.add_argument("--lookback-days", type=int, help="training days before each serving day (default MIX_LOOKBACK_DAYS)")
    p.add_argument("--validation-days", type=int, help="days for the decision rules (default MIX_VALIDATION_DAYS)")
    p.add_argument("--register", action="store_true", help=f"store the models and decisions in {TABLE_REGISTRY}")
    p.add_argument("--write-eval", action="store_true", help=f"store the validation forecasts in {TABLE_EVALUATION}")
    p.add_argument("--allow-protected-window", action="store_true")

    p = sub.add_parser("backtest", help="rolling daily refit over past days (docs/evaluation.md section 8)")
    p.add_argument("--start", required=True, help="first SGT date of forecast origins, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last SGT date of forecast origins, YYYY-MM-DD")
    p.add_argument("--lookback-days", type=int, help="training days before each day (default MIX_LOOKBACK_DAYS)")
    p.add_argument("--write-eval", action="store_true", help=f"store the forecasts in {TABLE_EVALUATION}")
    p.add_argument("--allow-protected-window", action="store_true")

    p = sub.add_parser("monitor", help="score the logged forecasts against the bins that arrived later")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--allow-protected-window", action="store_true")

    sub.add_parser("show-sql", help="print the generated SQL (no credentials needed)")
    return parser


def _complete_day(value: Optional[str], now: dt.datetime) -> dt.date:
    """``value`` (default: yesterday in Singapore), which must be a complete day."""
    today = fcm.sgt_date(now)
    day = fcm.parse_sgt_date(value) if value else today - dt.timedelta(days=1)
    if day >= today:
        raise ConfigError(f"{day} is not a complete day yet in Singapore")
    return day


def _with_overrides(cfg: MixConfig, args: argparse.Namespace) -> MixConfig:
    changes = {k: getattr(args, k) for k in ("lookback_days", "validation_days") if getattr(args, k, None) is not None}
    if not changes:
        return cfg
    cfg = dataclasses.replace(cfg, **changes)
    cfg.validate()
    return cfg


def main(argv: Optional[Sequence[str]] = None, *, warehouse_factory: Optional[Callable[[MixConfig], Any]] = None,
         session: Any = None, now: Optional[dt.datetime] = None) -> int:
    if argv is None:  # Cloud Run Jobs: the command can come from LAYER_B_ARGS instead of the Procfile
        argv = sys.argv[1:] or shlex.split(os.environ.get("LAYER_B_ARGS", "run-cycle"))
    args = build_parser().parse_args(argv)
    try:
        cfg = _with_overrides(MixConfig.from_env(), args)
        if args.command == "show-sql":
            for title, sql in (("travel-time bins", bins_sql(cfg)), ("layer A features", fcm.layer_a_features_sql(cfg.base)),
                               ("layer A bins already logged", fcm.layer_a_existing_bins_sql(cfg.base)),
                               ("latest registry", latest_registry_sql(cfg)), ("monitor", monitor_sql(cfg))):
                print(f"-- {title}\n{sql}\n")
            return EXIT_OK
        wh = (warehouse_factory or Warehouse)(cfg)
        clock = now or fcm.utc_now()
        if args.command == "ensure-tables":
            _print({"tables": wh.ensure_tables()})
        elif args.command == "run-cycle":
            _print(run_cycle(cfg, wh, session or requests.Session(), now=clock, use_layer_a=not args.no_layer_a,
                             write=not args.dry_run))
        elif args.command == "train":
            result = train(cfg, wh, end=_complete_day(args.end, clock), now=clock,
                           allow_protected=args.allow_protected_window)
            run_id = f"train-{clock.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            served = {d: n["served"] for d, n in result["decisions"].items()}
            _print({"run_id": run_id, "labels_before": result["labels_before"], "served": served,
                    "validation": headline(result["validation"]), "final_fit": result["final_fit"]})
            if args.register or args.write_eval:
                wh.ensure_tables()
            if args.register:
                wh.append(TABLE_REGISTRY, [registry_row(cfg, result, run_id, clock)])
            if args.write_eval:
                wh.append(TABLE_EVALUATION, evaluation_table_rows(result["rows"], "validation", run_id, clock))
        elif args.command == "backtest":
            result = backtest(cfg, wh, start=fcm.parse_sgt_date(args.start), end=_complete_day(args.end, clock),
                              now=clock, allow_protected=args.allow_protected_window)
            run_id = f"backtest-{clock.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            _print({"run_id": run_id, "headline": headline(result["backtest"]),
                    **{k: v for k, v in result.items() if k != "rows"}})
            if args.write_eval:
                wh.ensure_tables()
                wh.append(TABLE_EVALUATION, evaluation_table_rows(result["rows"], "backtest", run_id, clock))
        elif args.command == "monitor":
            _print(monitor(cfg, wh, days=args.days, now=clock, allow_protected=args.allow_protected_window))
        return EXIT_OK
    except ConfigError as exc:
        log("ERROR", "invalid configuration, arguments or data", error=str(exc))
        return EXIT_CONFIG
    except NoForecastError as exc:
        log("ERROR", "no usable forecast", error=str(exc))
        return EXIT_NO_FORECAST
    except fcm.LayerAError as exc:
        log("ERROR", "Layer A failed", error=str(exc))
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - last-resort guard so the job logs a structured error
        log("ERROR", "unexpected failure", error=repr(exc))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
