import numpy as np
import pandas as pd

from timeseries_xgb import (
    TimeSeriesConfig,
    XGBTrainConfig,
    apply_slew_rate_limit,
    build_supervised_matrices,
    engineer_features,
    rmse_seconds,
)


def _synthetic_frame(n: int = 120) -> pd.DataFrame:
    start = pd.Timestamp("2026-09-01 00:00:00")
    times = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    dur = 1500 + np.arange(n) * 2
    return pd.DataFrame(
        {
            "observed_at_sgt": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "route_id": ["jb_to_woodlands"] * n,
            "status": ["OK"] * n,
            "duration_sec": dur * 0.9,
            "duration_in_traffic_sec": dur,
            "error_message": [None] * n,
        }
    )


def test_engineer_and_matrix_shapes():
    config = TimeSeriesConfig(window_size=10, horizon_steps=3, keep_lags=4, train_fraction=0.8)
    raw = _synthetic_frame(80)
    features = engineer_features(raw, config)
    x_train, x_test, y_train, y_test = build_supervised_matrices(features, config)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert "target_lag_1" in x_train.columns
    assert "hour" in x_train.columns


def test_rmse_seconds():
    assert rmse_seconds([100, 200], [110, 190]) == 10.0


def test_slew_rate_limit_caps_steps():
    raw = np.array([0.0, 500.0, 0.0])
    capped = apply_slew_rate_limit(raw, 300.0)
    assert capped[1] == 300.0
    assert capped[2] == 0.0


def test_default_xgb_train_config_matches_teammate():
    cfg = TimeSeriesConfig().xgb
    assert cfg.n_estimators == 200
    assert cfg.max_depth == 4
    assert cfg.min_child_weight == 50
