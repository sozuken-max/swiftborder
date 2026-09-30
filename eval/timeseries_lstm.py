"""LSTM experiments on the same JB→SG canonical series as XGB (60 min horizon)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from timeseries_xgb import TimeSeriesConfig, mae_minutes, rmse_minutes, split_supervised

ARCHITECTURE_CHOICES: Tuple[str, ...] = (
    "lstm",
    "lstm_dropout",
    "stacked_lstm",
    "bilstm",
    "gru",
    "residual_gated",
    "bilstm_attention",
)


@dataclass(frozen=True)
class LSTMTrainConfig:
    architecture: str = "lstm"
    lstm_units: int = 64
    lstm_units_2: int = 32
    dropout: float = 0.2
    dense_units: int = 32
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 64
    patience: int = 6
    verbose: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURE_CHOICES:
            raise ValueError(f"unknown architecture {self.architecture}; choose from {ARCHITECTURE_CHOICES}")


@dataclass
class LSTMTrainResult:
    config: LSTMTrainConfig
    metrics: Dict[str, float]
    persistence_metrics: Dict[str, float]
    history: Dict[str, List[float]]
    n_train: int
    n_test: int


@dataclass
class LSTMTuneResult:
    best_config: LSTMTrainConfig
    best_metrics: Dict[str, float]
    trials: List[Dict[str, Any]]


def sequence_feature_columns(features: pd.DataFrame, config: TimeSeriesConfig) -> List[str]:
    """Multivariate channels per timestep (extend for weather / vision joins later)."""
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


@dataclass
class LSTMSplit:
    """Sequence tensors aligned to the same supervised rows as the XGB split."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    cols: List[str]
    test_target_ts: pd.Series
    test_persistence: np.ndarray
    boundary: pd.Timestamp
    train_persistence: Optional[np.ndarray] = None
    """Raw value at each training origin (anchor for residual targets)."""


def _sequence_matrix(features: pd.DataFrame, cols: Sequence[str], config: TimeSeriesConfig) -> np.ndarray:
    """Per-bin channel matrix. Missing side channels fall back to the input value at that bin."""
    frame = features[list(cols)].astype(np.float64).copy()
    fallback = features[config.target_col].astype(np.float64)
    for col in cols:
        if col != config.target_col:
            frame[col] = frame[col].fillna(fallback)
    return frame.to_numpy()


def build_lstm_split(
    features: pd.DataFrame,
    config: TimeSeriesConfig,
    *,
    sequence_cols: Optional[Sequence[str]] = None,
    train_fraction: Optional[float] = None,
) -> LSTMSplit:
    """Chronological train/test tensors built from ``timeseries_xgb.split_supervised``.

    Each sample is the ``window_size`` bins ending at the forecast origin; the label is the raw
    observation ``h`` steps later. Train and test rows (and the boundary) are identical to the
    XGB split, so LSTM and XGB scores are comparable.
    """
    cols = list(sequence_cols) if sequence_cols is not None else sequence_feature_columns(features, config)
    if config.target_col not in cols:
        raise ValueError(f"target column {config.target_col} must be in sequence_cols")
    split = split_supervised(features, config, train_fraction=train_fraction)
    matrix = _sequence_matrix(features, cols, config)
    w = config.window_size

    def tensor(rows: pd.DataFrame) -> np.ndarray:
        pos = features.index.get_indexer(pd.DatetimeIndex(rows["origin_ts"]))
        if len(pos) == 0:
            return np.empty((0, w, len(cols)))
        return np.stack([matrix[p - w + 1 : p + 1] for p in pos])

    return LSTMSplit(
        x_train=tensor(split.train),
        x_test=tensor(split.test),
        y_train=split.train["y"].to_numpy(dtype=np.float64),
        y_test=split.test["y"].to_numpy(dtype=np.float64),
        cols=cols,
        test_target_ts=split.test["target_ts"],
        test_persistence=split.test["persistence"].to_numpy(dtype=np.float64),
        boundary=split.boundary,
        train_persistence=split.train["persistence"].to_numpy(dtype=np.float64),
    )


def build_lstm_sequences(
    features: pd.DataFrame,
    config: TimeSeriesConfig,
    *,
    sequence_cols: Optional[Sequence[str]] = None,
    train_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Compatibility wrapper: (x_train, x_test, y_train, y_test, cols) from ``build_lstm_split``."""
    s = build_lstm_split(features, config, sequence_cols=sequence_cols, train_fraction=train_fraction)
    return s.x_train, s.x_test, s.y_train, s.y_test, s.cols


def persistence_predictions_from_sequences(
    x: np.ndarray,
    sequence_cols: Sequence[str],
    config: TimeSeriesConfig,
) -> np.ndarray:
    """Model-input value at the origin (last timestep). Equals persistence when the slew cap is off.

    The scored persistence baseline is ``LSTMSplit.test_persistence`` (last raw observation).
    """
    target_idx = list(sequence_cols).index(config.target_col)
    return x[:, -1, target_idx].astype(np.float64)


def scale_sequences(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    from sklearn.preprocessing import StandardScaler

    n_features = x_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, n_features))
    def transform(x: np.ndarray) -> np.ndarray:
        flat = scaler.transform(x.reshape(-1, n_features))
        return flat.reshape(x.shape)

    return transform(x_train), transform(x_val), transform(x_test), scaler


def _require_keras(seed: Optional[int] = None):
    """Import Keras. Seeds Python, NumPy and TensorFlow only when ``seed`` is given.

    Model builders pass their config seed once, before creating layers; later calls (for example
    to build callbacks) must not reset the generator, or every "seed" gives the same model.
    """
    import os

    # DirectML and CPU-only Windows wheels cannot register CudnnRNN.
    os.environ.setdefault("TF_DISABLE_CUDNN_RNN", "1")
    import tensorflow as tf

    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))
    return tf.keras


def _lstm_kwargs() -> Dict[str, Any]:
    """DirectML / CPU TensorFlow builds cannot use cuDNN RNN kernels."""
    import tensorflow as tf

    if not tf.test.is_built_with_cuda():
        return {"implementation": 1, "unroll": True}
    return {}


def build_residual_gated_lstm_layer(units: int):
    """Built-in custom recurrent block: LSTM + layer norm (notebook example)."""
    keras = _require_keras()
    layers = keras.layers

    class ResidualGatedLSTM(layers.Layer):
        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.lstm = layers.LSTM(units, return_sequences=False, **_lstm_kwargs())
            self.norm = layers.LayerNormalization()

        def call(self, inputs, training=None):
            return self.norm(self.lstm(inputs, training=training))

    return ResidualGatedLSTM(units=units)


class _TemporalAttentionPooling:
    """Lightweight attention over timesteps (factory returns a keras Layer)."""

    @staticmethod
    def layer(units: int):
        keras = _require_keras()
        layers = keras.layers

        class TemporalAttentionPooling(layers.Layer):
            def __init__(self, units: int, **kwargs):
                super().__init__(**kwargs)
                self.proj = layers.Dense(units, activation="tanh")
                self.score = layers.Dense(1)

            def call(self, inputs, training=None):
                import tensorflow as tf

                e = self.score(self.proj(inputs))
                weights = tf.nn.softmax(e, axis=1)
                return tf.reduce_sum(inputs * weights, axis=1)

        return TemporalAttentionPooling(units=units)


def build_recurrent_model(
    input_shape: Tuple[int, int],
    train_config: LSTMTrainConfig,
    *,
    custom_recurrent_layer: Optional[Type] = None,
):
    """Build and compile a regression model for the chosen architecture (seeded by ``train_config.seed``)."""
    keras = _require_keras(train_config.seed)
    layers = keras.layers

    inputs = keras.Input(shape=input_shape, name="sequence")
    arch = train_config.architecture
    u1 = train_config.lstm_units
    u2 = train_config.lstm_units_2
    drop = train_config.dropout
    lstm_kw = _lstm_kwargs()

    if custom_recurrent_layer is not None:
        x = custom_recurrent_layer(units=u1, name="custom_recurrent")(inputs)
    elif arch == "lstm":
        x = layers.LSTM(u1, name="lstm", **lstm_kw)(inputs)
    elif arch == "lstm_dropout":
        x = layers.LSTM(u1, dropout=drop, recurrent_dropout=0.0, name="lstm", **lstm_kw)(inputs)
    elif arch == "stacked_lstm":
        x = layers.LSTM(u1, return_sequences=True, name="lstm_1", **lstm_kw)(inputs)
        x = layers.Dropout(drop)(x)
        x = layers.LSTM(u2, name="lstm_2", **lstm_kw)(x)
    elif arch == "bilstm":
        x = layers.Bidirectional(
            layers.LSTM(u1 // 2, name="lstm_inner", **lstm_kw),
            name="bilstm",
        )(inputs)
    elif arch == "gru":
        x = layers.GRU(u1, name="gru")(inputs)
    elif arch == "residual_gated":
        x = build_residual_gated_lstm_layer(u1)(inputs)
    elif arch == "bilstm_attention":
        x = layers.Bidirectional(
            layers.LSTM(u1 // 2, return_sequences=True, name="lstm_inner", **lstm_kw),
            name="bilstm",
        )(inputs)
        x = _TemporalAttentionPooling.layer(u1)(x)
    else:
        raise ValueError(f"unsupported architecture: {arch}")

    if drop > 0 and arch not in ("lstm_dropout", "stacked_lstm"):
        x = layers.Dropout(drop)(x)
    x = layers.Dense(train_config.dense_units, activation="relu", name="dense_hidden")(x)
    outputs = layers.Dense(1, name="duration_sec")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"causeway_{arch}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_config.learning_rate),
        loss="mse",
    )
    return model


def build_lstm_model(
    input_shape: Tuple[int, int],
    *,
    lstm_units: int = 64,
    custom_recurrent_layer: Optional[Type] = None,
):
    """Backward-compatible wrapper around :func:`build_recurrent_model`."""
    cfg = LSTMTrainConfig(
        architecture="residual_gated" if custom_recurrent_layer else "lstm",
        lstm_units=lstm_units,
    )
    if custom_recurrent_layer is not None:
        return build_recurrent_model(input_shape, cfg, custom_recurrent_layer=custom_recurrent_layer)
    return build_recurrent_model(input_shape, cfg)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    flat_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "rmse_min": rmse_minutes(y_true, flat_pred),
        "mae_min": mae_minutes(y_true, flat_pred),
    }


def _chronological_val_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0 or len(x_train) < 20:
        return x_train, np.empty((0,) + x_train.shape[1:]), y_train, np.array([])
    n_val = max(1, int(len(x_train) * val_fraction))
    split = len(x_train) - n_val
    return x_train[:split], x_train[split:], y_train[:split], y_train[split:]


def train_lstm(
    features: pd.DataFrame,
    ts_config: TimeSeriesConfig,
    train_config: Optional[LSTMTrainConfig] = None,
    *,
    sequence_cols: Optional[Sequence[str]] = None,
) -> Tuple[Any, LSTMTrainResult]:
    """Train one architecture; returns (keras model, result bundle)."""
    train_config = train_config or LSTMTrainConfig()
    split = build_lstm_split(features, ts_config, sequence_cols=sequence_cols)
    x_train, x_test, y_train, y_test = split.x_train, split.x_test, split.y_train, split.y_test
    x_fit, x_val, y_fit, y_val = _chronological_val_split(x_train, y_train)
    x_fit, x_val, x_test_s, _ = scale_sequences(x_fit, x_val, x_test)

    model = build_recurrent_model((x_fit.shape[1], x_fit.shape[2]), train_config)
    keras = _require_keras()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if len(y_val) else "loss",
            patience=train_config.patience,
            restore_best_weights=True,
        )
    ]
    fit_kw: Dict[str, Any] = {
        "epochs": train_config.epochs,
        "batch_size": train_config.batch_size,
        "verbose": train_config.verbose,
        "callbacks": callbacks,
    }
    if len(y_val):
        fit_kw["validation_data"] = (x_val, y_val)
    history = model.fit(x_fit, y_fit, **fit_kw)
    pred = model.predict(x_test_s, verbose=0)
    metrics = evaluate_predictions(y_test, pred)
    persistence_metrics = evaluate_predictions(y_test, split.test_persistence)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    result = LSTMTrainResult(
        config=train_config,
        metrics=metrics,
        persistence_metrics=persistence_metrics,
        history=hist,
        n_train=len(x_train),
        n_test=len(x_test),
    )
    return model, result


def default_tune_grid() -> List[LSTMTrainConfig]:
    """Small search grid for course-scale data (~6k route rows)."""
    configs: List[LSTMTrainConfig] = []
    for arch, units, lr, drop in product(
        ("lstm", "stacked_lstm", "bilstm", "gru", "residual_gated", "bilstm_attention"),
        (48, 64),
        (1e-3, 5e-4),
        (0.0, 0.2),
    ):
        configs.append(
            LSTMTrainConfig(
                architecture=arch,
                lstm_units=units,
                dropout=drop,
                learning_rate=lr,
                epochs=25,
                patience=4,
                verbose=0,
            )
        )
    return configs


def tune_lstm_hyperparameters(
    features: pd.DataFrame,
    ts_config: TimeSeriesConfig,
    *,
    candidates: Optional[Sequence[LSTMTrainConfig]] = None,
    max_trials: Optional[int] = None,
) -> LSTMTuneResult:
    """Grid search on chronological val split; picks lowest val RMSE (minutes)."""
    candidates = list(candidates or default_tune_grid())
    if max_trials is not None:
        candidates = candidates[: max_trials]

    trials: List[Dict[str, Any]] = []
    best_config: Optional[LSTMTrainConfig] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_val_rmse = float("inf")

    x_train, x_test, y_train, y_test, _cols = build_lstm_sequences(features, ts_config)
    x_fit, x_val, y_fit, y_val = _chronological_val_split(x_train, y_train)
    x_fit_s, x_val_s, x_test_s, _ = scale_sequences(x_fit, x_val, x_test)

    for cfg in candidates:
        model = build_recurrent_model((x_fit_s.shape[1], x_fit_s.shape[2]), cfg)
        keras = _require_keras()
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                restore_best_weights=True,
            )
        ]
        model.fit(
            x_fit_s,
            y_fit,
            validation_data=(x_val_s, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        val_pred = model.predict(x_val_s, verbose=0)
        val_rmse = evaluate_predictions(y_val, val_pred)["rmse_min"]
        test_pred = model.predict(x_test_s, verbose=0)
        test_metrics = evaluate_predictions(y_test, test_pred)
        row = {
            "config": asdict(cfg),
            "val_rmse_min": val_rmse,
            "test_rmse_min": test_metrics["rmse_min"],
            "test_mae_min": test_metrics["mae_min"],
        }
        trials.append(row)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_config = cfg
            best_metrics = test_metrics

    if best_config is None or best_metrics is None:
        raise RuntimeError("tuning produced no trials")
    return LSTMTuneResult(best_config=best_config, best_metrics=best_metrics, trials=trials)


def compare_lstm_architectures(
    features: pd.DataFrame,
    ts_config: TimeSeriesConfig,
    architectures: Sequence[str] = ARCHITECTURE_CHOICES,
    *,
    base_config: Optional[LSTMTrainConfig] = None,
) -> pd.DataFrame:
    """Train each architecture with shared hyperparameters (except architecture name)."""
    base = base_config or LSTMTrainConfig(epochs=30, patience=5, verbose=0)
    rows: List[Dict[str, Any]] = []
    for arch in architectures:
        cfg = LSTMTrainConfig(
            architecture=arch,
            lstm_units=base.lstm_units,
            lstm_units_2=base.lstm_units_2,
            dropout=base.dropout,
            dense_units=base.dense_units,
            learning_rate=base.learning_rate,
            epochs=base.epochs,
            batch_size=base.batch_size,
            patience=base.patience,
            verbose=base.verbose,
        )
        _, result = train_lstm(features, ts_config, cfg)
        rows.append(
            {
                "architecture": arch,
                "rmse_min": result.metrics["rmse_min"],
                "mae_min": result.metrics["mae_min"],
                "persistence_rmse_min": result.persistence_metrics["rmse_min"],
                "persistence_mae_min": result.persistence_metrics["mae_min"],
                "n_test": result.n_test,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_min")


def save_tune_report(result: LSTMTuneResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_config": asdict(result.best_config),
        "best_test_metrics": result.best_metrics,
        "trials": result.trials,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
