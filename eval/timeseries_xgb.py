"""Exploratory sklearn XGBoost on causeway Maps durations (not the BQML serve path).

Scope: **one Maps route only**: ``route_id = jb_to_woodlands`` (JB -> Woodlands / inbound to SG).
The reverse leg (SG -> JB) is not implemented here; score both directions with ``layer_b.py`` (BQML).

Pipeline (all causal; every model input is observed at or before the forecast origin):

1. ``prepare_route_frame``: keep ``status = OK`` rows for one route. Raw values, no filling.
2. ``regular_grid``: 5-minute bins ``[ts, ts + 5 min)``. ``y_raw`` is the bin mean (NaN when the bin
   has no observation). ``y_ffill`` forward-fills at most ``max_ffill_steps`` bins (no bfill). The
   model-input column (``config.target_col``) is ``y_ffill`` after the causal slew-rate cap.
3. ``engineer_features``: calendar fields per bin plus D-1 / D-7 lookups of the input series.
4. ``build_supervised_frame``: one row per forecast origin ``t`` with label ``y = y_raw[t + h]``.
   Lags come from the input series at ``t, t-1, ...``; ``target_lag_1`` is the origin value.
   ``duration_sec`` is the Maps typical duration at the origin. Calendar and D-1/D-7 describe the
   target time (known in advance because ``h`` is less than a day). ``persistence`` is the last
   observed value at the origin (``y_ffill[t]``), not a model feature.
5. ``split_supervised``: chronological split on target time. Backtest days are full calendar days on
   or after the split boundary, so no backtest label was seen in training.

Scores use raw labels only: bins without an observation are never scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# JB -> Woodlands (one row in causeway.travel_times). Not the SG -> JB route.
DEFAULT_ROUTE_ID = "jb_to_woodlands"
OFFLINE_ROUTE_LABEL = "JB -> SG (jb_to_woodlands only)"
TARGET_COL = "duration_in_traffic_sec"
STEP_MINUTES = 5
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES
CANONICAL_TABLE = "causeway.travel_times"
DEFAULT_MAX_SLEW_STEP_SEC = 300
DEFAULT_MAX_FFILL_STEPS = 6  # 30 minutes
DWT_WAVELET = "db2"
DWT_LEVEL = 3
CALENDAR_COLS = ("hour", "minute", "dayofweek", "day", "month", "nonworkday")
META_COLS = ("origin_ts", "target_ts", "y", "persistence")


@dataclass(frozen=True)
class XGBTrainConfig:
    """Defaults match teammate notebook (2026-09-26); DWT off in production path."""

    n_estimators: int = 200
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: int = 50
    subsample: float = 0.6
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0


@dataclass(frozen=True)
class TimeSeriesConfig:
    window_size: int = 36
    horizon_steps: int = 12  # 5-minute steps -> 60 minutes ahead
    keep_lags: int = 12
    train_fraction: float = 0.8
    route_id: str = DEFAULT_ROUTE_ID
    target_col: str = TARGET_COL
    max_slew_step_sec: int = DEFAULT_MAX_SLEW_STEP_SEC
    max_ffill_steps: int = DEFAULT_MAX_FFILL_STEPS
    use_dwt: bool = False
    xgb: XGBTrainConfig = field(default_factory=XGBTrainConfig)

    @property
    def horizon_minutes(self) -> int:
        return self.horizon_steps * STEP_MINUTES


@dataclass
class SupervisedSplit:
    """Chronological train/test split of a supervised frame (see ``build_supervised_frame``)."""

    train: pd.DataFrame
    test: pd.DataFrame
    boundary: pd.Timestamp
    feature_cols: List[str]


def apply_slew_rate_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    """Causal cap on step-to-step change (notebook: 300 s per 5-min bin).

    NaN entries stay NaN and the filter restarts at the next observation, so a gap does not drag
    the first value after it toward the value before it.
    """
    vals = np.asarray(values, dtype=float)
    if max_step <= 0:
        return vals.copy()
    filtered = np.full_like(vals, np.nan)
    prev = np.nan
    for i, v in enumerate(vals):
        if np.isnan(v):
            prev = np.nan
            continue
        if np.isnan(prev):
            filtered[i] = v
        else:
            filtered[i] = prev + np.clip(v - prev, -max_step, max_step)
        prev = filtered[i]
    return filtered


def dwt_features_from_windows(
    windows: np.ndarray,
    *,
    wavelet: str = DWT_WAVELET,
    level: int = DWT_LEVEL,
) -> pd.DataFrame:
    """Optional wavelet features per lag window (requires PyWavelets)."""
    import pywt

    scale = 2 ** (level / 2)
    rows: List[Dict[str, float]] = []
    for w in windows:
        c_a, *c_ds = pywt.wavedec(w, wavelet, level=level, mode="periodization")
        trend = c_a / scale
        row: Dict[str, float] = {
            "dwt_trend_last": float(trend[-1]),
            "dwt_trend_slope": float(trend[-1] - trend[-2]),
        }
        for lvl, c_d in zip(range(level, 0, -1), c_ds):
            row[f"dwt_noise_L{lvl}"] = float(np.std(c_d))
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_observed_at_sgt(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `observed_at_sgt` when the export only has `observed_at` (UTC)."""
    out = frame.copy()
    if "observed_at_sgt" not in out.columns and "observed_at" in out.columns:
        ts = pd.to_datetime(out["observed_at"], utc=True).dt.tz_convert("Asia/Singapore")
        out["observed_at_sgt"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def _parse_sgt(values: pd.Series) -> pd.Series:
    """Naive SGT timestamps from strings or tz-aware values."""
    ts = pd.to_datetime(values, errors="coerce", format="mixed")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    return ts


def sync_canonical_travel_times(
    cache_path: Union[str, Path],
    project: str = "swiftborder",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load the canonical BigQuery table `causeway.travel_times`.

    Uses a local CSV cache unless `refresh=True`, then re-downloads from project
    `swiftborder` (read-only query).
    """
    path = Path(cache_path)
    if path.exists() and not refresh:
        return ensure_observed_at_sgt(pd.read_csv(path))

    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    query = f"""
    SELECT *
    FROM `{project}.{CANONICAL_TABLE}`
    ORDER BY observed_at
    """
    frame = client.query(query).to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return ensure_observed_at_sgt(frame)


def prepare_route_frame(
    export: pd.DataFrame,
    config: Optional[TimeSeriesConfig] = None,
) -> pd.DataFrame:
    """Filter a full-table export to one route. Raw observations; nothing is filled."""
    config = config or TimeSeriesConfig()
    return _filter_route(ensure_observed_at_sgt(export), config)


def load_from_csv(path: str, config: Optional[TimeSeriesConfig] = None) -> pd.DataFrame:
    """Load a local full-table export and filter to the configured route."""
    config = config or TimeSeriesConfig()
    return prepare_route_frame(pd.read_csv(path), config)


def load_from_bigquery(
    project: str = "swiftborder",
    config: Optional[TimeSeriesConfig] = None,
) -> pd.DataFrame:
    """Read `causeway.travel_times` for one route (read-only)."""
    from google.cloud import bigquery

    config = config or TimeSeriesConfig()
    client = bigquery.Client(project=project)
    query = f"""
    SELECT
      observed_at,
      route_id,
      status,
      duration_sec,
      duration_in_traffic_sec,
      error_message
    FROM `{project}.causeway.travel_times`
    WHERE route_id = @route_id
    ORDER BY observed_at
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("route_id", "STRING", config.route_id),
        ]
    )
    frame = client.query(query, job_config=job_config).to_dataframe()
    return prepare_route_frame(frame, config)


def _filter_route(frame: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    required = {
        "observed_at_sgt",
        "route_id",
        "status",
        "duration_sec",
        config.target_col,
        "error_message",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = frame[list(required)].copy()
    out = out[
        (out["status"] == "OK")
        & (out["error_message"].isna())
        & (out["route_id"] == config.route_id)
    ]
    out["observed_at_sgt"] = _parse_sgt(out["observed_at_sgt"])
    out[config.target_col] = pd.to_numeric(out[config.target_col], errors="coerce").astype(float)
    out["duration_sec"] = pd.to_numeric(out["duration_sec"], errors="coerce").astype(float)
    out = out.dropna(subset=["observed_at_sgt", config.target_col])
    out = out.sort_values("observed_at_sgt")
    return out[["observed_at_sgt", "route_id", "duration_sec", config.target_col]].reset_index(drop=True)


def regular_grid(raw: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """5-minute grid indexed by bin start (naive SGT).

    Columns: ``y_raw`` (bin mean, NaN if empty), ``y_ffill`` (causal fill, at most
    ``max_ffill_steps`` bins), ``duration_sec`` (causal fill), and the model-input column
    ``config.target_col`` (``y_ffill`` after the causal slew cap).
    """
    step = f"{STEP_MINUTES}min"
    frame = raw.copy()
    frame["ts"] = _parse_sgt(frame["observed_at_sgt"]).dt.floor(step)
    binned = frame.groupby("ts")[[config.target_col, "duration_sec"]].mean()
    index = pd.date_range(binned.index.min(), binned.index.max(), freq=step, name="ts")
    binned = binned.reindex(index)

    limit = config.max_ffill_steps if config.max_ffill_steps > 0 else None
    grid = pd.DataFrame(index=index)
    grid["y_raw"] = binned[config.target_col]
    grid["y_ffill"] = grid["y_raw"].ffill(limit=limit)
    grid["duration_sec"] = binned["duration_sec"].ffill(limit=limit)
    grid[config.target_col] = apply_slew_rate_limit(grid["y_ffill"].to_numpy(), float(config.max_slew_step_sec))
    return grid


def engineer_features(raw: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """Grid plus calendar fields and D-1 / D-7 lookups of the input series (per bin time)."""
    grid = regular_grid(raw, config)
    ts = grid.index
    grid["hour"] = ts.hour
    grid["minute"] = ts.minute
    grid["dayofweek"] = ts.dayofweek
    grid["day"] = ts.day
    grid["month"] = ts.month
    grid["nonworkday"] = ts.dayofweek.isin([5, 6]).astype(int)
    series = grid[config.target_col]
    grid["d1"] = series.reindex(ts - pd.Timedelta(days=1)).to_numpy()
    grid["d7"] = series.reindex(ts - pd.Timedelta(days=7)).to_numpy()
    return grid


def build_supervised_frame(features: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """One row per forecast origin with a raw label ``h`` steps ahead (see module docstring).

    Rows need a complete input window (``window_size`` bins ending at the origin) and an observed
    label. Columns: ``origin_ts``, ``target_ts``, feature columns, ``y``, ``persistence``.
    """
    h = config.horizon_steps
    if h * STEP_MINUTES >= 24 * 60:
        raise ValueError("horizon must be shorter than one day (D-1 features must be known at the origin)")
    w = config.window_size
    if config.keep_lags > w:
        raise ValueError("keep_lags cannot exceed window_size")

    inputs = features[config.target_col].to_numpy(dtype=float)
    y_raw = features["y_raw"].to_numpy(dtype=float)
    y_ffill = features["y_ffill"].to_numpy(dtype=float)
    dur = features["duration_sec"].to_numpy(dtype=float)
    ts = features.index
    n = len(features)
    if n < w + h:
        return pd.DataFrame(columns=list(META_COLS))

    origins = np.arange(w - 1, n - h)
    windows = np.stack([inputs[o - w + 1 : o + 1] for o in origins]) if len(origins) else np.empty((0, w))
    targets = origins + h
    valid = ~np.isnan(windows).any(axis=1) & ~np.isnan(y_raw[targets])
    origins, targets, windows = origins[valid], targets[valid], windows[valid]

    out = pd.DataFrame(
        {
            "origin_ts": ts[origins],
            "target_ts": ts[targets],
            "duration_sec": dur[origins],
        }
    )
    for col in CALENDAR_COLS + ("d1", "d7"):
        out[col] = features[col].to_numpy()[targets]
    for k in range(config.keep_lags, 0, -1):
        out[f"target_lag_{k}"] = windows[:, w - k]
    if config.use_dwt and len(windows):
        out = pd.concat([out, dwt_features_from_windows(windows)], axis=1)
    out["y"] = y_raw[targets]
    out["persistence"] = y_ffill[origins]
    return out.reset_index(drop=True)


def feature_columns(supervised: pd.DataFrame) -> List[str]:
    return [c for c in supervised.columns if c not in META_COLS]


def split_boundary(supervised: pd.DataFrame, train_fraction: float) -> pd.Timestamp:
    """Target time at which the test set starts (``train_fraction`` of rows before it)."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    ordered = supervised["target_ts"].sort_values().reset_index(drop=True)
    return pd.Timestamp(ordered.iloc[int(len(ordered) * train_fraction)])


def split_supervised(
    features: pd.DataFrame,
    config: TimeSeriesConfig,
    *,
    train_fraction: Optional[float] = None,
) -> SupervisedSplit:
    """Chronological split on target time: train labels all precede test labels."""
    train_fraction = train_fraction if train_fraction is not None else config.train_fraction
    sup = build_supervised_frame(features, config)
    boundary = split_boundary(sup, train_fraction)
    train = sup[sup["target_ts"] < boundary].reset_index(drop=True)
    test = sup[sup["target_ts"] >= boundary].reset_index(drop=True)
    return SupervisedSplit(train=train, test=test, boundary=boundary, feature_cols=feature_columns(sup))


def build_supervised_matrices(
    features: pd.DataFrame,
    config: TimeSeriesConfig,
    *,
    train_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Compatibility wrapper: (x_train, x_test, y_train, y_test) from ``split_supervised``."""
    split = split_supervised(features, config, train_fraction=train_fraction)
    cols = split.feature_cols
    return (
        split.train[cols],
        split.test[cols],
        split.train["y"].to_numpy(),
        split.test["y"].to_numpy(),
    )


def train_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    config: Optional[TimeSeriesConfig] = None,
    random_state: int = 42,
):
    from xgboost import XGBRegressor

    xgb_cfg = (config or TimeSeriesConfig()).xgb
    model = XGBRegressor(
        n_estimators=xgb_cfg.n_estimators,
        learning_rate=xgb_cfg.learning_rate,
        max_depth=xgb_cfg.max_depth,
        min_child_weight=xgb_cfg.min_child_weight,
        subsample=xgb_cfg.subsample,
        colsample_bytree=xgb_cfg.colsample_bytree,
        reg_lambda=xgb_cfg.reg_lambda,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def rmse_seconds(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def rmse_minutes(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return rmse_seconds(y_true, y_pred) / 60.0


def mae_minutes(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))) / 60.0


def holdout_persistence_predictions(test: pd.DataFrame, config: Optional[TimeSeriesConfig] = None) -> np.ndarray:
    """Persistence T-H: last observed value at the forecast origin (``persistence`` column)."""
    if "persistence" not in test.columns:
        raise ValueError("expected a supervised frame with a 'persistence' column")
    return test["persistence"].to_numpy(dtype=float)


def holdout_metrics(model, split: SupervisedSplit) -> Dict[str, float]:
    """MAE / RMSE (minutes) for the model and persistence on the chronological hold-out."""
    y = split.test["y"].to_numpy()
    pred = model.predict(split.test[split.feature_cols])
    persist = holdout_persistence_predictions(split.test)
    return {
        "n": int(len(y)),
        "model_mae_min": mae_minutes(y, pred),
        "model_rmse_min": rmse_minutes(y, pred),
        "persistence_mae_min": mae_minutes(y, persist),
        "persistence_rmse_min": rmse_minutes(y, persist),
    }


def holdout_significance_vs_persistence(
    model,
    split: SupervisedSplit,
    config: Optional[TimeSeriesConfig] = None,
    *,
    alpha: float = 0.05,
    random_state: int = 0,
    **kwargs,
):
    """Paired absolute-error comparison (minutes) on the hold-out vs persistence T-H."""
    from significance import compare_absolute_errors

    config = config or TimeSeriesConfig()
    scale = 1.0 / 60.0
    y = split.test["y"].to_numpy() * scale
    pred = model.predict(split.test[split.feature_cols]) * scale
    persist = holdout_persistence_predictions(split.test) * scale
    return compare_absolute_errors(
        y,
        pred,
        persist,
        label_challenger="XGB (sklearn)",
        label_reference=f"Persistence T-{config.horizon_minutes}",
        horizon_steps=config.horizon_steps,
        timestamps=list(split.test["target_ts"]),
        alpha=alpha,
        random_state=random_state,
        **kwargs,
    )


def backtest_days(split: SupervisedSplit, *, min_coverage: float = 0.9) -> List[str]:
    """Full calendar days (SGT) starting at or after the split boundary with enough labels."""
    test = split.test
    if test.empty:
        return []
    per_day = test.groupby(test["target_ts"].dt.normalize()).size()
    days = [
        d.strftime("%Y-%m-%d")
        for d, count in per_day.items()
        if d >= split.boundary and count >= min_coverage * STEPS_PER_DAY
    ]
    return days


def score_forecast_days(
    model,
    features: pd.DataFrame,
    split: SupervisedSplit,
    dates: Sequence[str],
    config: TimeSeriesConfig,
    *,
    w_yday: float = 0.5,
    w_lastwk: float = 0.5,
) -> pd.DataFrame:
    """Per-day backtest on test rows only (raw labels; identical rows for every method).

    Methods: XGB on the actual lag window; XGB on a D-1/D-7 blended lag window; their average;
    persistence T-H; the naive D-1/D-7 blend at the target time.
    """
    series = features[config.target_col]
    step = pd.Timedelta(minutes=STEP_MINUTES)

    def blended(times: pd.DatetimeIndex) -> np.ndarray:
        """Weighted D-1/D-7 mean; uses whichever is available when the other is missing."""
        d1 = series.reindex(times - pd.Timedelta(days=1)).to_numpy()
        d7 = series.reindex(times - pd.Timedelta(days=7)).to_numpy()
        w1 = np.where(np.isnan(d1), 0.0, w_yday)
        w7 = np.where(np.isnan(d7), 0.0, w_lastwk)
        total = w1 + w7
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (np.nan_to_num(d1) * w1 + np.nan_to_num(d7) * w7) / total
        return np.where(total > 0, out, np.nan)

    scores: List[Dict[str, object]] = []
    for d in dates:
        rows = split.test[split.test["target_ts"].dt.strftime("%Y-%m-%d") == d].reset_index(drop=True)
        if rows.empty:
            continue
        x_actual = rows[split.feature_cols]
        x_blend = x_actual.copy()
        origins = pd.DatetimeIndex(rows["origin_ts"])
        for k in range(config.keep_lags, 0, -1):
            x_blend[f"target_lag_{k}"] = blended(origins - (k - 1) * step)
        fc_actual = model.predict(x_actual)
        fc_blend = model.predict(x_blend)
        methods = {
            "XGB actual window": fc_actual,
            "XGB blend window": fc_blend,
            "XGB average": (fc_actual + fc_blend) / 2,
            f"Persistence T-{config.horizon_minutes}": rows["persistence"].to_numpy(),
            "Naive blend": blended(pd.DatetimeIndex(rows["target_ts"])),
        }
        keep = np.ones(len(rows), dtype=bool)
        for pred in methods.values():
            keep &= ~np.isnan(np.asarray(pred, dtype=float))
        y = rows["y"].to_numpy()[keep]
        label = f"{d[5:]} {pd.Timestamp(d).day_name()[:3]}"
        for name, pred in methods.items():
            scores.append(
                {
                    "date": label,
                    "method": name,
                    "n": int(keep.sum()),
                    "RMSE_min": rmse_minutes(y, np.asarray(pred)[keep]),
                    "MAE_min": mae_minutes(y, np.asarray(pred)[keep]),
                }
            )
    return pd.DataFrame(scores)
