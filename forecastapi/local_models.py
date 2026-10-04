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

Features are built here from ``v_bins_10min`` (``features_from_bins``) with the harness's time-based
lags, not read from ``v_training_set``, whose positional LAG/LEAD differ after a skipped bin.

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
    # exploratory (eval/horizon_study.py): the harness XGBoost plus the profile at the origin and target
    "xgb[maps+prof]": {"kind": "harness_xgb_prof", "rows": "daily", "eval_id": "horizon_study:xgb[maps+prof]"},
}

# Horizons (minutes) of eval/horizon_study.py. 30 is the evaluated horizon of the report; the others
# are exploratory. Only these models and the baselines take a horizon other than 30.
HORIZONS = tuple(range(30, 1441, 30))  # every 30 minutes, 30 min .. 24 h
MULTI_HORIZON_MODELS = ("xgb[maps]", "xgb[maps+prof]")
PROFILE_FEATURES = ["prof_now", "prof_h"]

# eval/camera_forecast.py FourierProfile, as eval/horizon_study.py fits it on Maps rows
PROFILE_HARMONICS = 8
PROFILE_ALPHA = 1.0
PROFILE_NOW_OFFSET = pd.Timedelta(minutes=5)


def label_column(horizon: int) -> str:
    return f"y_{int(horizon)}"


def _profile_calendar(ts) -> tuple:
    sgt = pd.to_datetime(pd.Series(ts).reset_index(drop=True), utc=True).dt.tz_convert("Asia/Singapore")
    tod = (sgt.dt.hour * 60 + sgt.dt.minute + sgt.dt.second / 60.0).to_numpy(dtype=float)
    weekend = (sgt.dt.dayofweek >= 5).to_numpy(dtype=float)
    return tod, weekend


def _fourier_design(tod_min: np.ndarray, weekend: np.ndarray, harmonics: int) -> np.ndarray:
    cols = [weekend]
    for k in range(1, harmonics + 1):
        a = 2 * np.pi * k * tod_min / 1440.0
        s, c = np.sin(a), np.cos(a)
        cols += [s, c, s * weekend, c * weekend]
    return np.column_stack(cols)


class Profile:
    """Calendar baseline: per-direction ridge on Fourier (K = 8) x weekend terms of the Maps duration.

    It ignores current traffic. Fitted on every closed bin before the serving day, as each fold of
    ``eval/horizon_study.py`` does, and read at the target time. A baseline for comparison, not a forecast.
    """

    def __init__(self, harmonics: int = PROFILE_HARMONICS, alpha: float = PROFILE_ALPHA):
        self.harmonics, self.alpha = harmonics, alpha
        self.models_: Dict[str, object] = {}
        self.fallback_: Dict[str, float] = {}
        self.n_rows = 0

    def fit(self, frame: pd.DataFrame) -> "Profile":
        from sklearn.linear_model import Ridge

        rows = frame[["direction", "bin_ts", "y_persistence"]].dropna()
        self.n_rows = int(len(rows))
        for d, g in rows.groupby("direction"):
            tod, we = _profile_calendar(g["bin_ts"] + PROFILE_NOW_OFFSET)
            y = g["y_persistence"].to_numpy(dtype=float)
            self.fallback_[d] = float(np.mean(y))
            if len(y) >= 10:
                self.models_[d] = Ridge(alpha=self.alpha).fit(_fourier_design(tod, we, self.harmonics), y)
        return self

    def predict(self, direction: Sequence[str], ts) -> np.ndarray:
        """Profile value of the bin starting at ``ts`` (the fit reads ``bin_ts + 5 min`` too)."""
        direction = np.asarray(direction)
        tod, we = _profile_calendar(pd.Series(ts).reset_index(drop=True) + PROFILE_NOW_OFFSET)
        out = np.full(len(direction), np.nan)
        for d in np.unique(direction):
            m = direction == d
            if d in self.models_:
                out[m] = self.models_[d].predict(_fourier_design(tod[m], we[m], self.harmonics))
            else:
                out[m] = self.fallback_.get(d, np.nan)
        return out


def with_profile(frame: pd.DataFrame, profile: Profile, horizon: int) -> pd.DataFrame:
    """Add ``prof_now`` (origin bin) and ``prof_h`` (target bin) for a horizon."""
    out = frame.copy()
    d = out["direction"].to_numpy()
    out["prof_now"] = profile.predict(d, out["bin_ts"])
    out["prof_h"] = profile.predict(d, out["bin_ts"] + pd.Timedelta(minutes=int(horizon)))
    return out


def serving_day_start(now: Optional[dt.datetime] = None) -> pd.Timestamp:
    """00:00 SGT of the serving day, as UTC."""
    now = now or dt.datetime.now(dt.timezone.utc)
    local = now.astimezone(SGT)
    return pd.Timestamp(dt.datetime(local.year, local.month, local.day, tzinfo=SGT)).tz_convert("UTC")


def version_for(model_id: str, now: Optional[dt.datetime] = None) -> str:
    if LOCAL_MODELS[model_id]["rows"] == "frozen":
        return "bqml-replica-2026-09-12"
    return "labels_before=" + serving_day_start(now).tz_convert(SGT).isoformat()


BIN = pd.Timedelta(minutes=10)


def _shift_time(g: pd.DataFrame, col: str, steps: int) -> pd.Series:
    """Value of ``col`` at bin_ts + steps*10 min in the same route (NaN when that bin is missing)."""
    lookup = g.set_index("bin_ts")[col]
    return pd.Series(lookup.reindex(g["bin_ts"] + steps * BIN).to_numpy(), index=g.index)


def features_from_bins(bins: pd.DataFrame, *, unknown_first_gap: bool = False) -> pd.DataFrame:
    """``v_bins_10min`` rows -> ``v_training_set`` columns with **time-based** lags and labels.

    A copy of ``eval/features.maps_features`` (the harness), not of the view: the view uses positional
    LAG/LEAD, so one skipped bin shifts its lags and its ``y_30`` by ten minutes. Here a missing bin is
    NaN, exactly as in the harness. ``gap_min`` is recomputed from the rows given, so pass every bin of
    the period. The first row has no gap, as in the harness export; with ``unknown_first_gap`` (a
    window cut from a longer history) it counts as a gap, so the six rows after it are ``after_gap``.
    Columns: ``direction``, ``bin_ts``, ``VIEW_COLUMNS``, ``after_gap``, ``y_30``.
    """
    b = bins.copy()
    b["bin_ts"] = pd.to_datetime(b["bin_ts"], utc=True)
    for col in ("dur_min", "congestion_ratio", "speed_kmh"):
        b[col] = pd.to_numeric(b[col], errors="coerce").astype(float)
    parts = []
    for _, g in b.groupby("route_id", sort=False):
        g = g.sort_values("bin_ts").copy()
        gap_min = g["bin_ts"].diff().dt.total_seconds() / 60.0
        if unknown_first_gap:
            gap_min.iloc[0] = np.inf
        lag = {k: _shift_time(g, "dur_min", -k) for k in (1, 2, 3, 4, 5, 6)}
        f = pd.DataFrame(index=g.index)
        f["direction"] = g["direction"]
        f["bin_ts"] = g["bin_ts"]
        f["y_persistence"] = g["dur_min"]
        f["congestion_ratio"] = g["congestion_ratio"]
        f["speed_kmh"] = g["speed_kmh"]
        f["lag_10"], f["lag_20"], f["lag_30"], f["lag_60"] = lag[1], lag[2], lag[3], lag[6]
        f["roll_mean_30"] = pd.concat([lag[1], lag[2], lag[3]], axis=1).mean(axis=1, skipna=True)
        f["roll_mean_60"] = pd.concat([lag[k] for k in range(1, 7)], axis=1).mean(axis=1, skipna=True)
        f["slope_30"] = lag[1] - lag[4]
        big_gap = (gap_min > 25).astype(int)
        f["after_gap"] = big_gap.rolling(7, min_periods=1).max().astype(int).to_numpy()
        for h in HORIZONS:
            f[label_column(h)] = _shift_time(g, "dur_min", h // 10)
        parts.append(f)
    labels = [label_column(h) for h in HORIZONS]
    if not parts:
        return pd.DataFrame(columns=["direction", "bin_ts", *VIEW_COLUMNS, "after_gap", *labels])
    out = pd.concat(parts).sort_values(["direction", "bin_ts"]).reset_index(drop=True)
    for col in ("lag_10", "lag_20", "lag_30", "lag_60", "roll_mean_30", "roll_mean_60", "slope_30"):
        out.loc[out["after_gap"] == 1, col] = np.nan
    sgt = out["bin_ts"].dt.tz_convert("Asia/Singapore")
    tod_min = sgt.dt.hour * 60 + sgt.dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * tod_min / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * tod_min / 1440)
    out["tod_block"] = tod_min / 5.0
    out["dow"] = (sgt.dt.dayofweek + 1) % 7 + 1  # BigQuery DAYOFWEEK: Sunday=1 .. Saturday=7
    out["is_weekend"] = out["dow"].isin([1, 7]).astype(int)
    out["is_morning_peak"] = sgt.dt.hour.between(6, 10).astype(int)
    out["is_evening_peak"] = sgt.dt.hour.between(16, 21).astype(int)
    return out[["direction", "bin_ts", *VIEW_COLUMNS, "after_gap", *labels]]


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """View rows -> float features, UTC bin_ts, SGT date, direction flag."""
    out = frame.copy()
    out["bin_ts"] = pd.to_datetime(out["bin_ts"], utc=True)
    for col in VIEW_COLUMNS + [label_column(h) for h in HORIZONS] + ["after_gap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out["is_my_to_sg"] = (out["direction"] == "MY_TO_SG").astype(float)
    out["date_sgt"] = out["bin_ts"].dt.tz_convert("Asia/Singapore").dt.strftime("%Y-%m-%d")
    return out.sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def training_rows(frame: pd.DataFrame, rows: str, day_start: pd.Timestamp, horizon: int = 30) -> pd.DataFrame:
    """Rows a model of this ``rows`` kind trains on (``frame`` already prepared).

    The label of horizon ``h`` is the bin ``[t+h, t+h+10)``; a row counts once that bin closed before
    ``day_start`` (``joined.LABEL_LAG`` at 30 minutes, ``horizon_study`` otherwise).
    """
    if horizon != 30 and rows != "daily":
        raise ValueError("only daily models take a horizon other than 30")
    ycol = label_column(horizon)
    labelled = frame[frame[ycol].notna() & (frame["after_gap"] == 0)]
    if rows == "daily":
        return labelled[labelled["bin_ts"] + pd.Timedelta(minutes=int(horizon) + 10) <= day_start]
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


def fit(model_id: str, train: pd.DataFrame, horizon: int = 30) -> Fitted:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    kind = LOCAL_MODELS[model_id]["kind"]
    if horizon != 30 and model_id not in MULTI_HORIZON_MODELS:
        raise ValueError(f"{model_id} is fitted for 30 minutes only")
    y = train[label_column(horizon)].to_numpy()
    if kind == "harness_ridge":
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), Ridge(alpha=HARNESS_RIDGE_ALPHA))
        return Fitted(m.fit(train[MAPS_FEATURES], y), MAPS_FEATURES, len(train), [])
    if kind == "harness_xgb":
        m = XGBRegressor(**HARNESS_XGB_PARAMS).fit(train[MAPS_FEATURES], y)
        return Fitted(m, MAPS_FEATURES, len(train), [HARNESS_XGB_PARAMS["random_state"]])
    if kind == "harness_xgb_prof":  # train must carry with_profile(...) columns for this horizon
        cols = MAPS_FEATURES + PROFILE_FEATURES
        m = XGBRegressor(**HARNESS_XGB_PARAMS).fit(train[cols], y)
        return Fitted(m, cols, len(train), [HARNESS_XGB_PARAMS["random_state"]])
    if kind == "bq_lin":
        m = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(), Ridge(alpha=BQ_LIN_L2))
        return Fitted(m.fit(train[BQ_LIN_FEATURES], y), BQ_LIN_FEATURES, len(train), [])
    if kind == "bq_xgb":
        x = train[BQ_XGB_FEATURES]
        m = SeedMean(XGBRegressor(**BQ_XGB_PARAMS, random_state=s, n_jobs=1).fit(x, y) for s in BQ_XGB_SEEDS)
        return Fitted(m, BQ_XGB_FEATURES, len(train), BQ_XGB_SEEDS)
    raise ValueError(kind)
