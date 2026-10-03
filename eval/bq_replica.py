"""Local replicas of the BigQuery ML models ``lin_h30`` and ``xgb_h30``.

Same features, settings and training rows as ``sql/bigquery/traffic_prediction/bqml_*_h30.sql``
(options read from ``bq show --model`` on 2026-10-03):

- ``lin_h30``: linear regression on standardised features, L2 0.1, normal equation, mean imputation.
- ``xgb_h30``: boosted trees, 28 trees (where BQML early-stopped), learning rate 0.1, depth 4,
  subsample 0.8, L2 1, min child weight 1. ``base_score = 0.5`` because BQML runs XGBoost 0.9, whose
  trees start from 0.5; current XGBoost starts from the label mean, which shifts the 28-tree model.
  Five seeds, averaged (subsampling is random).
- Rows: ``v_training_set`` with ``y_30``, ``after_gap = 0`` and ``lag_60``. The **frozen** replica uses
  rows whose label was observed by the BQML training time (2026-09-12 06:15 UTC) and, like BQML's
  CUSTOM split, does not train on 11-12 Sep SGT. The **daily** replica refits the same estimator on
  every earlier day (``joined`` rolling folds).

Direction enters as ``is_my_to_sg`` (0/1), equivalent to BQML's encoding of a two-level ``direction``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

LIN_FEATURES: List[str] = [
    "y_persistence", "congestion_ratio", "speed_kmh", "lag_10", "lag_20", "lag_30", "lag_60",
    "roll_mean_30", "roll_mean_60", "slope_30", "tod_sin", "tod_cos", "dow", "is_weekend",
    "is_morning_peak", "is_evening_peak", "is_my_to_sg",
]
XGB_FEATURES: List[str] = LIN_FEATURES[:12] + ["tod_block"] + LIN_FEATURES[12:]
BQML_TRAINED_AT = pd.Timestamp("2026-09-12 06:15:33", tz="UTC")  # lin_h30 creationTime (xgb_h30: 06:19)
BQML_VAL_DAYS = ("2026-09-11", "2026-09-12")
LABEL_LAG = pd.Timedelta(minutes=40)
XGB_SEEDS = (0, 1, 2, 3, 4)
XGB_PARAMS: Dict[str, float] = {
    "n_estimators": 28, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8, "reg_lambda": 1.0,
    "reg_alpha": 0.0, "min_child_weight": 1, "gamma": 0.0, "colsample_bytree": 1.0, "base_score": 0.5,
}
LIN_L2 = 0.1


def bqml_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows BQML trains on: label present, no gap, ``lag_60`` present."""
    return frame[frame["y_30"].notna() & (frame["after_gap"] == 0) & frame["lag_60"].notna()]


def frozen_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """BQML's training rows on 12 Sep: labels observed by the training time, 11-12 Sep held out."""
    rows = bqml_rows(frame)
    rows = rows[rows["bin_ts"] + LABEL_LAG <= BQML_TRAINED_AT]
    return rows[~rows["date_sgt"].isin(BQML_VAL_DAYS)]


def with_direction(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_my_to_sg" in frame.columns:
        return frame
    out = frame.copy()
    out["is_my_to_sg"] = (out["direction"] == "MY_TO_SG").astype(int)
    return out


def fit_lin(train: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = with_direction(train)
    return make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(), Ridge(alpha=LIN_L2)).fit(
        train[LIN_FEATURES], train["y_30"].to_numpy()
    )


class SeedMean:
    """Mean prediction of several fitted models (sklearn-style ``predict``)."""

    def __init__(self, models: Sequence):
        self.models = list(models)

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.models], axis=0)


def fit_xgb(train: pd.DataFrame, seeds: Sequence[int] = XGB_SEEDS) -> SeedMean:
    from xgboost import XGBRegressor

    train = with_direction(train)
    x, y = train[XGB_FEATURES], train["y_30"].to_numpy()
    return SeedMean(XGBRegressor(**XGB_PARAMS, random_state=int(s), n_jobs=1).fit(x, y) for s in seeds)


def predict_lin(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(with_direction(frame)[LIN_FEATURES])


def predict_xgb(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(with_direction(frame)[XGB_FEATURES])


def settings() -> Dict[str, object]:
    return {
        "lin": {"model": "Ridge on standardised features, mean imputation", "l2": LIN_L2, "features": LIN_FEATURES},
        "xgb": {"params": XGB_PARAMS, "seeds": list(XGB_SEEDS), "features": XGB_FEATURES},
        "frozen_training": {"labels_observed_by_utc": BQML_TRAINED_AT.isoformat(), "excluded_days_sgt": list(BQML_VAL_DAYS)},
        "source": "sql/bigquery/traffic_prediction/bqml_lin_h30.sql, bqml_xgb_h30.sql; bq show --model 2026-10-03",
    }
