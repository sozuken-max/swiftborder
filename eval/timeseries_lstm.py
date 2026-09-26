"""Sequence tensors and Keras model factory for LSTM experiments (same canonical data as XGB)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from timeseries_xgb import TimeSeriesConfig, mae_minutes, rmse_minutes


def sequence_feature_columns(features: pd.DataFrame, config: TimeSeriesConfig) -> List[str]:
    """Default multivariate channels per timestep (extend for weather / vision joins later)."""
    candidates = [
        config.target_col,
        "duration_sec",
        "hour",
        "minute",
        "dayofweek",
        "month",
        "nonworkday",
        "d1",
        "d7",
    ]
    return [c for c in candidates if c in features.columns and c != "route_id"]


def build_lstm_sequences(
    features: pd.DataFrame,
    config: TimeSeriesConfig,
    *,
    sequence_cols: Optional[Sequence[str]] = None,
    train_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/test tensors for sequence models.

    Returns X with shape (n, window_size, n_features), y in target units (seconds).
    """
    train_fraction = train_fraction if train_fraction is not None else config.train_fraction
    cols = list(sequence_cols) if sequence_cols is not None else sequence_feature_columns(features, config)
    if config.target_col not in cols:
        raise ValueError(f"target column {config.target_col} must be in sequence_cols")
    target_idx = cols.index(config.target_col)

    matrix = features[cols].astype(np.float64).values
    window = config.window_size
    horizon = config.horizon_steps

    x_list: List[np.ndarray] = []
    y_list: List[float] = []
    for end in range(window + horizon - 1, len(matrix)):
        start = end - window - horizon + 1
        x_list.append(matrix[start : start + window])
        y_list.append(float(matrix[end, target_idx]))

    x_all = np.stack(x_list, axis=0)
    y_all = np.asarray(y_list, dtype=np.float64)
    split = int(len(x_all) * train_fraction)
    return x_all[:split], x_all[split:], y_all[:split], y_all[split:]


def build_lstm_model(
    input_shape: Tuple[int, int],
    *,
    lstm_units: int = 64,
    custom_recurrent_layer: Optional[Type] = None,
):
    """
    Build a regression head on top of a recurrent block.

    Pass `custom_recurrent_layer` as a `keras.layers.Layer` subclass (not an instance)
    to plug in a custom LSTM design; it must accept `units` and standard Keras kwargs.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="sequence")
    if custom_recurrent_layer is not None:
        recurrent = custom_recurrent_layer(units=lstm_units, name="custom_recurrent")
        x = recurrent(inputs)
    else:
        x = layers.LSTM(lstm_units, name="lstm")(inputs)
    x = layers.Dense(32, activation="relu", name="dense_hidden")(x)
    outputs = layers.Dense(1, name="duration_sec")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="causeway_duration_lstm")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Report errors in minutes for parity with the XGB notebook."""
    flat_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "rmse_min": rmse_minutes(y_true, flat_pred),
        "mae_min": mae_minutes(y_true, flat_pred),
    }
