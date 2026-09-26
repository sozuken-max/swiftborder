import numpy as np
import pandas as pd

from timeseries_xgb import TimeSeriesConfig, engineer_features
from timeseries_lstm import build_lstm_sequences, sequence_feature_columns


def _features(n: int = 100) -> pd.DataFrame:
    from timeseries_xgb import DEFAULT_ROUTE_ID, TARGET_COL

    start = pd.Timestamp("2026-09-01 00:00:00")
    times = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    dur = 1500 + np.arange(n) * 2
    raw = pd.DataFrame(
        {
            "observed_at_sgt": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "route_id": [DEFAULT_ROUTE_ID] * n,
            "duration_sec": dur * 0.9,
            TARGET_COL: dur,
        }
    )
    config = TimeSeriesConfig(window_size=8, horizon_steps=2, keep_lags=4)
    return engineer_features(raw, config), config


def test_lstm_sequence_shapes():
    features, config = _features(60)
    cols = sequence_feature_columns(features, config)
    x_train, x_test, y_train, y_test = build_lstm_sequences(features, config, sequence_cols=cols)
    assert x_train.ndim == 3
    assert x_train.shape[1] == config.window_size
    assert x_train.shape[2] == len(cols)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
