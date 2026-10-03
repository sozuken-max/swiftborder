import numpy as np
import pandas as pd
import pytest

import timeseries_xgb as tsx
from timeseries_xgb import TimeSeriesConfig, engineer_features
from timeseries_lstm import build_lstm_sequences, build_lstm_split, sequence_feature_columns


def _features(n: int = 100, *, slew: int = 0):
    from timeseries_xgb import DEFAULT_ROUTE_ID, TARGET_COL

    start = pd.Timestamp("2026-09-01 00:00:00")
    times = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    dur = 1500 + np.arange(n) * 2
    raw = pd.DataFrame(
        {
            "observed_at_sgt": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "route_id": [DEFAULT_ROUTE_ID] * n,
            "status": ["OK"] * n,
            "duration_sec": dur * 0.9,
            TARGET_COL: dur,
            "error_message": [None] * n,
        }
    )
    config = TimeSeriesConfig(window_size=8, horizon_steps=2, keep_lags=4, max_slew_step_sec=slew)
    return engineer_features(tsx.prepare_route_frame(raw, config), config), config


def test_lstm_sequence_shapes():
    features, config = _features(60)
    cols = sequence_feature_columns(features, config)
    x_train, x_test, y_train, y_test, cols_out = build_lstm_sequences(features, config, sequence_cols=cols)
    assert cols_out == cols
    assert x_train.ndim == 3
    assert x_train.shape[1] == config.window_size
    assert x_train.shape[2] == len(cols)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert not np.isnan(x_train).any()


def test_lstm_and_xgb_share_the_same_test_rows():
    features, config = _features(120)
    lstm = build_lstm_split(features, config)
    xgb = tsx.split_supervised(features, config)
    assert list(lstm.test_target_ts) == list(xgb.test["target_ts"])
    np.testing.assert_allclose(lstm.y_test, xgb.test["y"])
    assert lstm.boundary == xgb.boundary


def test_lstm_window_ends_at_the_origin():
    features, config = _features(120)
    split = build_lstm_split(features, config)
    target_idx = split.cols.index(config.target_col)
    # Last timestep is the origin, h steps before the label (ramp of 2 s per step).
    np.testing.assert_allclose(split.x_test[:, -1, target_idx], split.y_test - 2.0 * config.horizon_steps)
    np.testing.assert_allclose(split.test_persistence, split.y_test - 2.0 * config.horizon_steps)


def test_persistence_from_sequences():
    features, config = _features(60)
    cols = sequence_feature_columns(features, config)
    x_train, x_test, y_train, y_test, _ = build_lstm_sequences(features, config, sequence_cols=cols)
    from timeseries_lstm import persistence_predictions_from_sequences

    persist = persistence_predictions_from_sequences(x_test, cols, config)
    assert persist.shape == y_test.shape


@pytest.mark.slow
def test_build_recurrent_model_smoke():
    pytest.importorskip("tensorflow")
    from timeseries_lstm import LSTMTrainConfig, build_recurrent_model

    for arch in ("lstm", "gru", "bilstm_attention", "residual_gated"):
        model = build_recurrent_model((12, 5), LSTMTrainConfig(architecture=arch, lstm_units=16))
        assert model.output_shape[-1] == 1


@pytest.mark.slow
def test_train_lstm_end_to_end_reports_persistence():
    pytest.importorskip("tensorflow")
    from timeseries_lstm import LSTMTrainConfig, train_lstm

    features, config = _features(200)
    _, result = train_lstm(features, config, LSTMTrainConfig(lstm_units=8, epochs=2, patience=1))
    assert result.n_test > 0
    # Persistence on a 2 s/step ramp is off by exactly 2h seconds.
    assert result.persistence_metrics["mae_min"] == pytest.approx(2.0 * config.horizon_steps / 60.0)
