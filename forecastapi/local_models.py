"""Local 30-minute models that forecast-api fits itself (ADR 0004).

The service cannot import ``eval/`` (Cloud Build packs ``forecastapi/`` only), so the definitions are
copied here. ``eval/tests/test_forecastapi_models.py`` asserts they equal ``eval/joined.py`` and
``eval/bq_replica.py``, and that a fit here predicts exactly what the harness predicts.

Training rows follow the harness folds:

- **daily** models (``ridge[maps]``, ``xgb[maps]``, ``lin_bq[daily]``, ``xgb_bq[daily]``) are fitted on
  every ``v_training_set`` row whose 30-minute label was observed before 00:00 SGT of the serving day
  (``joined.fold_split``). They change once a day.
- **frozen** replicas (``lin_bq[frozen]``, ``xgb_bq[frozen]``) use BQML's 12 Sep training rows: labels
  observed by 2026-09-12 06:15:33 UTC, 11-12 Sep SGT excluded.
- The ``*_bq`` models train only on rows with ``lag_60`` (BQML's filter); the ``[maps]`` models on
  every row with a label and no gap (``joined.scorable``).

Nothing here imports google-cloud; ``main.py`` supplies the rows.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SGT = dt.timezone(dt.timedelta(hours=8))
LABEL_LAG = pd.Timedelta(minutes=40)

# eval/features.py MAPS_FEATURES + joined "is_my_to_sg"
MAPS_FEATURES: List[str] = [
    "y_persistence", "congestion_ratio", "speed_kmh", "lag_10", "lag_20", "lag_30", "lag_60",
    "roll_mean_30", "roll_mean_60", "slope_30", "tod_sin", "tod_cos", "tod_block", "dow",
    "is_weekend", "is_morning_peak", "is_evening_peak", "is_my_to_sg",
]
# eval/bq_replica.py
BQ_LIN_FEATURES: List[str] = [
    "y_persistence", "congestion_ratio", "speed_kmh", "lag_10", "lag_20", "lag_30", "lag_60",
    "roll_mean_30", "roll_mean_60", "slope_30", "tod_sin", "tod_cos", "dow", "is_weekend",
    "is_morning_peak", "is_evening_peak", "is_my_to_sg",
]
BQ_XGB_FEATURES: List[str] = BQ_LIN_FEATURES[:12] + ["tod_block"] + BQ_LIN_FEATURES[12:]
VIEW_COLUMNS: List[str] = sorted({c for c in MAPS_FEATURES + BQ_XGB_FEATURES if c != "is_my_to_sg"})

# eval/joined.py _xgb / _ridge
HARNESS_XGB_PARAMS: Dict[str, float] = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 10, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 42, "n_jobs": 1,
}
HARNESS_RIDGE_ALPHA = 1.0
# eval/bq_replica.py
BQ_XGB_PARAMS: Dict[str, float] = {
    "n_estimators": 28, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8, "reg_lambda": 1.0,
    "reg_alpha": 0.0, "min_child_weight": 1, "gamma": 0.0, "colsample_bytree": 1.0, "base_score": 0.5,
}
BQ_XGB_SEEDS = (0, 1, 2, 3, 4)
BQ_LIN_L2 = 0.1
BQML_TRAINED_AT = pd.Timestamp("2026-09-12 06:15:33", tz="UTC")
BQML_VAL_DAYS = ("2026-09-11", "2026-09-12")

# id -> how it is trained. ``eval_id`` is the harness column it reproduces.
LOCAL_MODELS: Dict[str, Dict[str, str]] = {
    "ridge[maps]": {"kind": "harness_ridge", "rows": "daily", "eval_id": "ridge[maps]"},
    "xgb[maps]": {"kind": "harness_xgb", "rows": "daily", "eval_id": "xgb[maps]"},
    "lin_bq[daily]": {"kind": "bq_lin", "rows": "daily_bq", "eval_id": "lin_bq[daily]"},
    "xgb_bq[daily]": {"kind": "bq_xgb", "rows": "daily_bq", "eval_id": "xgb_bq[daily]"},
    "lin_bq[frozen]": {"kind": "bq_lin", "rows": "frozen", "eval_id": "lin_bq[frozen]"},
    "xgb_bq[frozen]": {"kind": "bq_xgb", "rows": "frozen", "eval_id": "xgb_bq[frozen]"},
}


def serving_day_start(now: Optional[dt.datetime] = None) -> pd.Timestamp:
    """00:00 SGT of the serving day, as UTC."""
    now = now or dt.datetime.now(dt.timezone.utc)
    local = now.astimezone(SGT)
    return pd.Timestamp(dt.datetime(local.year, local.month, local.day, tzinfo=SGT)).tz_convert("UTC")


def version_for(model_id: str, now: Optional[dt.datetime] = None) -> str:
    if LOCAL_MODELS[model_id]["rows"] == "frozen":
        return "bqml-replica-2026-09-12"
    return "labels_before=" + serving_day_start(now).tz_convert(SGT).isoformat()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """View rows -> float features, UTC bin_ts, SGT date, direction flag."""
    out = frame.copy()
    out["bin_ts"] = pd.to_datetime(out["bin_ts"], utc=True)
    for col in VIEW_COLUMNS + [c for c in ("y_30", "after_gap") if c in out.columns]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out["is_my_to_sg"] = (out["direction"] == "MY_TO_SG").astype(float)
    out["date_sgt"] = out["bin_ts"].dt.tz_convert("Asia/Singapore").dt.strftime("%Y-%m-%d")
    return out.sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def training_rows(frame: pd.DataFrame, rows: str, day_start: pd.Timestamp) -> pd.DataFrame:
    """Rows a model of this ``rows`` kind trains on (``frame`` already prepared)."""
    labelled = frame[frame["y_30"].notna() & (frame["after_gap"] == 0)]
    if rows == "daily":
        return labelled[labelled["bin_ts"] + LABEL_LAG <= day_start]
    bq_rows = labelled[labelled["lag_60"].notna()]
    if rows == "daily_bq":
        return bq_rows[bq_rows["bin_ts"] + LABEL_LAG <= day_start]
    if rows == "frozen":
        fr = bq_rows[bq_rows["bin_ts"] + LABEL_LAG <= BQML_TRAINED_AT]
        return fr[~fr["date_sgt"].isin(BQML_VAL_DAYS)]
    raise ValueError(rows)


class SeedMean:
    def __init__(self, models: Sequence):
        self.models = list(models)

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.models], axis=0)


class Fitted:
    def __init__(self, model, features: List[str], n_rows: int, seeds: Sequence[int]):
        self.model, self.features, self.n_rows, self.seeds = model, features, n_rows, list(seeds)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(frame[self.features]), dtype=float)


def fit(model_id: str, train: pd.DataFrame) -> Fitted:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    kind = LOCAL_MODELS[model_id]["kind"]
    y = train["y_30"].to_numpy()
    if kind == "harness_ridge":
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), Ridge(alpha=HARNESS_RIDGE_ALPHA))
        return Fitted(m.fit(train[MAPS_FEATURES], y), MAPS_FEATURES, len(train), [])
    if kind == "harness_xgb":
        m = XGBRegressor(**HARNESS_XGB_PARAMS).fit(train[MAPS_FEATURES], y)
        return Fitted(m, MAPS_FEATURES, len(train), [HARNESS_XGB_PARAMS["random_state"]])
    if kind == "bq_lin":
        m = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(), Ridge(alpha=BQ_LIN_L2))
        return Fitted(m.fit(train[BQ_LIN_FEATURES], y), BQ_LIN_FEATURES, len(train), [])
    if kind == "bq_xgb":
        x = train[BQ_XGB_FEATURES]
        m = SeedMean(XGBRegressor(**BQ_XGB_PARAMS, random_state=s, n_jobs=1).fit(x, y) for s in BQ_XGB_SEEDS)
        return Fitted(m, BQ_XGB_FEATURES, len(train), BQ_XGB_SEEDS)
    raise ValueError(kind)
