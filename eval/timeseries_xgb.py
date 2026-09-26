"""Exploratory sklearn XGBoost on causeway Maps durations (not the BQML serve path)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# `jb_to_woodlands` ~= direction `SG_TO_MY` in `traffic_prediction` views.
DEFAULT_ROUTE_ID = "jb_to_woodlands"
TARGET_COL = "duration_in_traffic_sec"
STEP_MINUTES = 5
CANONICAL_TABLE = "causeway.travel_times"


def ensure_observed_at_sgt(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `observed_at_sgt` when the export only has `observed_at` (UTC)."""
    out = frame.copy()
    if "observed_at_sgt" not in out.columns and "observed_at" in out.columns:
        ts = pd.to_datetime(out["observed_at"], utc=True).dt.tz_convert("Asia/Singapore")
        out["observed_at_sgt"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out


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
    """Filter a full-table export to one route and clean the target series."""
    config = config or TimeSeriesConfig()
    return _filter_route(ensure_observed_at_sgt(export), config)


@dataclass(frozen=True)
class TimeSeriesConfig:
    window_size: int = 36
    horizon_steps: int = 12  # 5-minute steps -> 60 minutes ahead
    keep_lags: int = 12
    train_fraction: float = 0.8
    route_id: str = DEFAULT_ROUTE_ID
    target_col: str = TARGET_COL

    @property
    def horizon_minutes(self) -> int:
        return self.horizon_steps * STEP_MINUTES


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

    out = frame[
        [
            "observed_at_sgt",
            "route_id",
            "status",
            "duration_sec",
            config.target_col,
            "error_message",
        ]
    ].copy()
    out = out[
        (out["status"] == "OK")
        & (out["error_message"].isna())
        & (out["route_id"] == config.route_id)
    ]
    out = out.drop(columns=["error_message", "status"])
    out = out.sort_values("observed_at_sgt")
    target = pd.to_numeric(out[config.target_col], errors="coerce").astype(float)
    out[config.target_col] = target.interpolate(method="linear").bfill().ffill()
    return out.reset_index(drop=True)


def engineer_features(frame: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """Calendar features and D-1 / D-7 lookups on a 5-minute grid."""
    features = frame.copy()
    dt_tmp = pd.to_datetime(features["observed_at_sgt"], errors="coerce")
    features["hour"] = dt_tmp.dt.hour
    features["minute"] = dt_tmp.dt.minute
    features["dayofweek"] = dt_tmp.dt.dayofweek
    features["day"] = dt_tmp.dt.day
    features["month"] = dt_tmp.dt.month
    features["nonworkday"] = features["dayofweek"].isin([5, 6]).astype(int)

    ts_grid = dt_tmp.dt.round(f"{STEP_MINUTES}min")
    lookup = pd.Series(features[config.target_col].values, index=ts_grid.values)
    features["d1"] = (ts_grid - pd.Timedelta(days=1)).map(lookup).values
    features["d7"] = (ts_grid - pd.Timedelta(days=7)).map(lookup).values
    features = features.drop(columns="observed_at_sgt")
    return features


def build_supervised_matrices(
    features: pd.DataFrame,
    config: TimeSeriesConfig,
    *,
    train_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Chronological split, lag windows, and aligned X/y for train and test."""
    train_fraction = train_fraction if train_fraction is not None else config.train_fraction
    split_index = int(len(features) * train_fraction)
    train_data = features.iloc[:split_index].copy()
    test_data = features.iloc[split_index:].copy()

    target = config.target_col
    train_target = train_data[target].values
    test_target = test_data[target].values

    h = config.horizon_steps
    w = config.window_size

    train_windows = sliding_window_view(train_target, window_shape=w)
    test_windows = sliding_window_view(test_target, window_shape=w)
    train_target_windows = train_windows[:-h]
    test_target_windows = test_windows[:-h]

    lag_cols = [f"target_lag_{i}" for i in range(w, 0, -1)]
    train_lags = pd.DataFrame(train_target_windows, columns=lag_cols)
    test_lags = pd.DataFrame(test_target_windows, columns=lag_cols)

    drop_cols = [target, "route_id"]
    offset = w + h - 1
    train_feats = train_data.drop(columns=drop_cols).iloc[offset:].reset_index(drop=True)
    test_feats = test_data.drop(columns=drop_cols).iloc[offset:].reset_index(drop=True)

    keep = [f"target_lag_{i}" for i in range(config.keep_lags, 0, -1)]
    x_train = pd.concat([train_feats, train_lags[keep]], axis=1)
    x_test = pd.concat([test_feats, test_lags[keep]], axis=1)
    y_train = train_target[offset:]
    y_test = test_target[offset:]
    return x_train, x_test, y_train, y_test


def train_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
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


def holdout_persistence_predictions(x_test: pd.DataFrame, config: TimeSeriesConfig) -> np.ndarray:
    """Persistence T-H: lag `horizon_steps` on the supervised test matrix."""
    lag_col = f"target_lag_{config.horizon_steps}"
    if lag_col not in x_test.columns:
        raise ValueError(f"missing {lag_col} for persistence baseline")
    return x_test[lag_col].to_numpy(dtype=float)


def holdout_significance_vs_persistence(
    model,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    config: Optional[TimeSeriesConfig] = None,
    *,
    block_size: int = 12,
    alpha: float = 0.05,
    random_state: int = 0,
):
    """Paired MAE significance (minutes) for chronological hold-out vs persistence T-H."""
    from significance import compare_absolute_errors

    config = config or TimeSeriesConfig()
    pred = model.predict(x_test)
    persist = holdout_persistence_predictions(x_test, config)
    scale = 1.0 / 60.0
    return compare_absolute_errors(
        y_test * scale,
        pred * scale,
        persist * scale,
        label_challenger="XGB (sklearn)",
        label_reference=f"Persistence T-{config.horizon_minutes}",
        block_size=block_size,
        alpha=alpha,
        random_state=random_state,
    )


def mae_minutes(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))) / 60.0


def regularized_series(
    frame: pd.DataFrame,
    config: TimeSeriesConfig,
) -> Tuple[pd.Series, float]:
    """5-minute grid of target and a free-flow proxy from median duration_sec."""
    s = frame[["observed_at_sgt", config.target_col, "duration_sec"]].copy()
    s["ts"] = pd.to_datetime(s["observed_at_sgt"]).dt.round(f"{STEP_MINUTES}min")
    s = s.groupby("ts")[[config.target_col, "duration_sec"]].mean()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=f"{STEP_MINUTES}min")
    s = s.reindex(full_index).interpolate(method="time")
    free_flow = float(s["duration_sec"].median())
    return s[config.target_col], free_flow


def score_forecast_days(
    model,
    y_series: pd.Series,
    free_flow: float,
    dates: Sequence[str],
    config: TimeSeriesConfig,
    *,
    w_yday: float = 0.5,
    w_lastwk: float = 0.5,
) -> pd.DataFrame:
    """Backtest fixed calendar days (notebook cell 2 logic)."""
    step = pd.Timedelta(minutes=STEP_MINUTES)
    h = config.horizon_steps
    w = config.window_size
    feature_cols = list(model.get_booster().feature_names)

    def blended_at(idx: pd.DatetimeIndex) -> np.ndarray:
        d1 = y_series.reindex(idx - pd.Timedelta(days=1)).values
        d7 = y_series.reindex(idx - pd.Timedelta(days=7)).values
        return w_yday * d1 + w_lastwk * d7

    def build_x(day_idx: pd.DatetimeIndex, source: str) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for t in day_idx:
            lag_times = pd.DatetimeIndex([t - (h + k - 1) * step for k in range(w, 0, -1)])
            if source == "actual":
                lags = y_series.reindex(lag_times).values
            else:
                lags = blended_at(lag_times)
            row = {
                "duration_sec": free_flow,
                "hour": t.hour,
                "minute": t.minute,
                "dayofweek": t.dayofweek,
                "day": t.day,
                "month": t.month,
                "nonworkday": int(t.dayofweek in (5, 6)),
                "d1": y_series.get(t - pd.Timedelta(days=1), np.nan),
                "d7": y_series.get(t - pd.Timedelta(days=7), np.nan),
            }
            row.update({f"target_lag_{k}": v for k, v in zip(range(w, 0, -1), lags)})
            rows.append(row)
        return pd.DataFrame(rows)[feature_cols]

    scores: List[Dict[str, object]] = []
    for d in dates:
        day_idx = pd.date_range(pd.Timestamp(d), periods=288, freq=f"{STEP_MINUTES}min")
        actual = y_series.reindex(day_idx)
        fc_actual = pd.Series(model.predict(build_x(day_idx, "actual")), index=day_idx)
        fc_blend = pd.Series(model.predict(build_x(day_idx, "blend")), index=day_idx)
        proxy = pd.Series(blended_at(day_idx), index=day_idx)
        persist = y_series.reindex(day_idx - h * step).set_axis(day_idx)
        fc_avg = (fc_actual + fc_blend) / 2
        ts = pd.Timestamp(d)
        label = f"{d[5:]} {ts.day_name()[:3]}"
        for name, pred in [
            ("XGB actual window", fc_actual),
            ("XGB blend window", fc_blend),
            ("XGB average", fc_avg),
            (f"Persistence T-{config.horizon_minutes}", persist),
            ("Naive blend", proxy),
        ]:
            scores.append(
                {
                    "date": label,
                    "method": name,
                    "RMSE_min": rmse_minutes(actual, pred),
                    "MAE_min": mae_minutes(actual, pred),
                }
            )
    return pd.DataFrame(scores)
