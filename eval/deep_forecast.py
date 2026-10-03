#!/usr/bin/env python3
"""Deep sequence forecasters (LSTM, GRU, patch Transformer) scored on the offline 60-min split.

All models see the same rows as offline XGBoost (``timeseries_lstm.build_lstm_split`` over
``timeseries_xgb.split_supervised``: ``jb_to_woodlands``, 36-step window, label 60 min ahead) and
share one training protocol, so differences come from the architecture:

- inputs standardised on the fit part of the training rows; the last 15% of training rows (by time)
  are the validation set for early stopping;
- target: the change from the value at the forecast origin (``anchor=True``) or the raw duration
  (``anchor=False``, ablation), standardised on the fit rows;
- Huber loss, AdamW with weight decay, early stopping on validation loss with best weights restored,
  learning rate halved on plateaus;
- several seeds per model. The scored prediction is the mean over seeds (a seed ensemble); per-seed
  MAE is also recorded so the spread is visible.

Significance: Diebold-Mariano on absolute errors with day-block bootstrap CIs and Holm over the
whole family (``significance.compare_absolute_errors``), the same rules as the other components.

    python deep_forecast.py                   # print tables (needs requirements-notebook.txt)
    python generate_comparison_plots.py --deep   # write into a run folder / run.json
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from plots import write_series_csv

CACHE = Path(__file__).resolve().parent / "data" / "causeway_gdata.csv"
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
XGB_SEED = 42  # same as generate_comparison_plots.XGB_SEED, so the XGB row matches the offline component


@dataclass(frozen=True)
class SequenceFitConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    patience: int = 10
    reduce_lr_patience: int = 4
    val_fraction: float = 0.15


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    kind: str  # "recurrent" or "transformer"
    anchor: bool
    architecture: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    "lstm": ModelSpec("lstm", "LSTM(64), anchored", "recurrent", True, "lstm"),
    "gru": ModelSpec("gru", "GRU(64), anchored", "recurrent", True, "gru"),
    "transformer": ModelSpec("transformer", "Patch Transformer, anchored", "transformer", True),
    "transformer_raw": ModelSpec("transformer_raw", "Patch Transformer, raw target (ablation)", "transformer", False),
}
DEFAULT_MODELS: Tuple[str, ...] = ("lstm", "gru", "transformer", "transformer_raw")


def residual_target(y: np.ndarray, anchor: Optional[np.ndarray]) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return y if anchor is None else y - np.asarray(anchor, dtype=float)


def _builder(spec: ModelSpec, recurrent_units: int, recurrent_dropout: float, transformer_config) -> Callable:
    def build(shape: Tuple[int, int], seed: int):
        if spec.kind == "transformer":
            from timeseries_transformer import build_transformer_model

            return build_transformer_model(shape, replace_seed(transformer_config, seed))
        from timeseries_lstm import LSTMTrainConfig, build_recurrent_model

        cfg = LSTMTrainConfig(architecture=spec.architecture, lstm_units=recurrent_units, dropout=recurrent_dropout, seed=seed)
        return build_recurrent_model(shape, cfg)

    return build


def replace_seed(config, seed: int):
    from dataclasses import replace

    return replace(config, seed=int(seed))


def fit_sequence_model(
    build: Callable,
    split,
    *,
    seed: int,
    anchor: bool = True,
    fit_config: SequenceFitConfig = SequenceFitConfig(),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train one model on ``split`` (a ``timeseries_lstm.LSTMSplit``); return test predictions (seconds)."""
    from timeseries_lstm import _chronological_val_split, _require_keras, scale_sequences

    if anchor and split.train_persistence is None:
        raise ValueError("anchor=True needs split.train_persistence")
    n_val = max(1, int(len(split.x_train) * fit_config.val_fraction))
    cut = len(split.x_train) - n_val
    x_fit, x_val, y_fit, y_val = _chronological_val_split(split.x_train, split.y_train, fit_config.val_fraction)
    if len(x_fit) != cut:
        raise RuntimeError("validation split changed; keep deep_forecast and timeseries_lstm in step")
    x_fit_s, x_val_s, x_test_s, _ = scale_sequences(x_fit, x_val, split.x_test)

    a_fit = split.train_persistence[:cut] if anchor else None
    a_val = split.train_persistence[cut:] if anchor else None
    a_test = split.test_persistence if anchor else None
    r_fit = residual_target(y_fit, a_fit)
    r_val = residual_target(y_val, a_val)
    mu, sd = float(r_fit.mean()), float(r_fit.std() or 1.0)

    model = build((x_fit_s.shape[1], x_fit_s.shape[2]), seed)
    keras = _require_keras()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=fit_config.learning_rate, weight_decay=fit_config.weight_decay),
        loss=keras.losses.Huber(delta=fit_config.huber_delta),
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=fit_config.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=fit_config.reduce_lr_patience),
    ]
    t0 = time.time()
    history = model.fit(
        x_fit_s,
        (r_fit - mu) / sd,
        validation_data=(x_val_s, (r_val - mu) / sd),
        epochs=fit_config.epochs,
        batch_size=fit_config.batch_size,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )
    seconds = time.time() - t0
    z = model.predict(x_test_s, verbose=0).reshape(-1)
    pred = z * sd + mu
    if anchor:
        pred = pred + a_test
    val_loss = history.history.get("val_loss", [])
    info = {
        "seed": int(seed),
        "params": int(model.count_params()),
        "epochs_run": len(val_loss),
        "best_epoch": int(np.argmin(val_loss)) + 1 if val_loss else None,
        "best_val_loss": float(np.min(val_loss)) if val_loss else None,
        "train_seconds": round(seconds, 1),
    }
    return pred, info


def _tensorflow_build() -> Dict[str, Any]:
    """TensorFlow build facts that affect bit-for-bit reproducibility of the deep rows."""
    import os

    import tensorflow as tf

    info = dict(tf.sysconfig.get_build_info())
    return {
        "version": tf.__version__,
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "devices": [d.device_type for d in tf.config.list_physical_devices()],
        "build_info": {k: str(v) for k, v in info.items()},
        "env": {k: os.environ.get(k) for k in ("TF_ENABLE_ONEDNN_OPTS", "TF_DISABLE_CUDNN_RNN", "TF_DETERMINISTIC_OPS")},
    }


def plot_seed_mae(summary: Dict[str, Dict[str, Any]], references: Dict[str, float], path: Path, title: str) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        s = summary[name]
        ax.bar(i, s["seed_mean_prediction_mae_min"], color="#4C78A8", alpha=0.8)
        ax.scatter([i] * len(s["mae_by_seed"]), s["mae_by_seed"], color="black", s=14, zorder=3)
    styles = ["--", ":", "-."]
    for k, (label, value) in enumerate(references.items()):
        ax.axhline(value, linestyle=styles[k % len(styles)], color="#E45756" if k == 0 else "#54A24B", label=f"{label} ({value:.2f})")
    ax.set_xticks(range(len(names)), [summary[n]["label"] for n in names], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Hold-out MAE (min)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        s = summary[name]
        rows.append(
            {
                "kind": "bar",
                "model_index": i,
                "model": s["label"],
                "seed": "",
                "mae_min": s["seed_mean_prediction_mae_min"],
            }
        )
        seeds = [r.get("seed", j) for j, r in enumerate(s.get("runs") or [])]
        if len(seeds) != len(s["mae_by_seed"]):
            seeds = list(range(len(s["mae_by_seed"])))
        for seed, mae in zip(seeds, s["mae_by_seed"]):
            rows.append({"kind": "seed", "model_index": i, "model": s["label"], "seed": seed, "mae_min": mae})
    for label, value in references.items():
        rows.append({"kind": "reference", "model_index": "", "model": label, "seed": "", "mae_min": value})
    csv_path = write_series_csv(path, rows, ("kind", "model_index", "model", "seed", "mae_min"))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path, csv_path]


def deep_component(
    run_dir: Path,
    *,
    cache: Path = CACHE,
    project: str = "swiftborder",
    models: Sequence[str] = DEFAULT_MODELS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    alpha: float = 0.05,
    fit_config: SequenceFitConfig = SequenceFitConfig(),
    recurrent_units: int = 64,
    recurrent_dropout: float = 0.1,
    transformer_config=None,
    keep: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """``keep``: optional dict that receives per-row test predictions (seconds) for reuse by ``ensemble``."""
    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("the deep component needs TensorFlow: pip install -r requirements-notebook.txt") from exc

    import timeseries_xgb as tsx
    from plots import plot_mae_diff_forest
    from run_artifacts import artifact_relpath, file_fingerprint, metric_row, subdir
    from significance import apply_holm, compare_absolute_errors, format_comparison_table
    from timeseries_lstm import build_lstm_split
    from timeseries_transformer import TransformerConfig

    transformer_config = transformer_config or TransformerConfig()
    unknown = [m for m in models if m not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {sorted(MODEL_SPECS)}")
    if not seeds:
        raise ValueError("need at least one seed")

    out = subdir(run_dir, "deep")
    config = tsx.TimeSeriesConfig()
    export = tsx.sync_canonical_travel_times(cache, project=project, refresh=False)
    raw = tsx.prepare_route_frame(export, config)
    features = tsx.engineer_features(raw, config)
    xs = tsx.split_supervised(features, config)
    split = build_lstm_split(features, config)
    y = split.y_test
    if not np.array_equal(y, xs.test["y"].to_numpy()):
        raise RuntimeError("sequence split and XGB split disagree on test rows")

    xgb = tsx.train_xgb(xs.train[xs.feature_cols], xs.train["y"].to_numpy(), config=config, random_state=XGB_SEED)
    xgb_pred = xgb.predict(xs.test[xs.feature_cols])
    persist = split.test_persistence
    persist_label = f"Persistence T-{config.horizon_minutes}"
    xgb_label = "XGB (sklearn)"

    metrics: List[Dict[str, Any]] = [
        metric_row(persist_label, "holdout", len(y), tsx.mae_minutes(y, persist), tsx.rmse_minutes(y, persist)),
        metric_row(xgb_label, "holdout", len(y), tsx.mae_minutes(y, xgb_pred), tsx.rmse_minutes(y, xgb_pred)),
    ]
    summary: Dict[str, Dict[str, Any]] = {}
    ensemble: Dict[str, np.ndarray] = {}
    for name in models:
        spec = MODEL_SPECS[name]
        build = _builder(spec, recurrent_units, recurrent_dropout, transformer_config)
        preds, runs = [], []
        for seed in seeds:
            pred, info = fit_sequence_model(build, split, seed=seed, anchor=spec.anchor, fit_config=fit_config)
            info["mae_min"] = tsx.mae_minutes(y, pred)
            preds.append(pred)
            runs.append(info)
            metrics.append(metric_row(spec.label, f"holdout seed {seed}", len(y), info["mae_min"], tsx.rmse_minutes(y, pred)))
            print(f"{spec.label} seed {seed}: MAE {info['mae_min']:.3f} min ({info['epochs_run']} epochs, {info['train_seconds']} s)")
        mean_pred = np.mean(preds, axis=0)
        ensemble[name] = mean_pred  # seed average for this architecture
        maes = [r["mae_min"] for r in runs]
        metrics.append(metric_row(f"{spec.label}, seed mean", "holdout", len(y), tsx.mae_minutes(y, mean_pred), tsx.rmse_minutes(y, mean_pred)))
        summary[name] = {
            "label": spec.label,
            "anchor": spec.anchor,
            "params": runs[0]["params"],
            "mae_by_seed": maes,
            "mae_seed_mean_min": float(np.mean(maes)),
            "mae_seed_sd_min": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "seed_mean_prediction_mae_min": tsx.mae_minutes(y, mean_pred),
            "runs": runs,
        }

    scale = 1.0 / 60.0
    timestamps = list(split.test_target_ts)
    if keep is not None:
        keep["y"] = y * scale
        keep["timestamps"] = timestamps
        keep["xgb_label"] = xgb_label
        keep["preds"] = {xgb_label: xgb_pred * scale, **{MODEL_SPECS[n].label: ensemble[n] * scale for n in models}}

    def cmp(ch: np.ndarray, ref: np.ndarray, lc: str, lr: str):
        return compare_absolute_errors(
            y * scale, ch * scale, ref * scale, label_challenger=lc, label_reference=lr,
            horizon_steps=config.horizon_steps, timestamps=timestamps, alpha=alpha,
        )

    family = []
    for name in models:
        label = f"{MODEL_SPECS[name].label}, seed mean"
        family.append(cmp(ensemble[name], persist, label, persist_label))
        family.append(cmp(ensemble[name], xgb_pred, label, xgb_label))
    if "transformer" in ensemble and "transformer_raw" in ensemble:
        family.append(cmp(ensemble["transformer"], ensemble["transformer_raw"], "Patch Transformer, anchored", "Patch Transformer, raw target"))
    if "transformer" in ensemble and "lstm" in ensemble:
        family.append(cmp(ensemble["transformer"], ensemble["lstm"], "Patch Transformer, anchored", "LSTM(64), anchored"))
    comparisons = apply_holm(family)
    print("Deep significance (Holm over this family):")
    print(format_comparison_table(comparisons))

    written = [
        *plot_seed_mae(
            summary,
            {persist_label: tsx.mae_minutes(y, persist), xgb_label: tsx.mae_minutes(y, xgb_pred)},
            out / "deep-mae-by-seed.png",
            title=f"Offline 60 min, {tsx.OFFLINE_ROUTE_LABEL}: seed-mean MAE (bars) and per-seed MAE (dots)",
        ),
        *plot_mae_diff_forest(comparisons, out / "deep-mae-diff.png", title="Deep models: paired MAE difference (day-block bootstrap CI)"),
    ]
    meta: Dict[str, Any] = {
        "dataset": {
            "source": f"{project}.causeway.travel_times",
            "cache": file_fingerprint(cache),
            "route_id": config.route_id,
            "route_scope": tsx.OFFLINE_ROUTE_LABEL,
            "supervised_rows": {"train": int(len(split.y_train)), "test": int(len(y))},
            "sequence_channels": split.cols,
        },
        "window": {
            "timezone": "Asia/Singapore",
            "basis": "same rows as the offline component (target time; train labels < start <= test labels)",
            "start": str(split.boundary),
            "end": str(split.test_target_ts.max()),
        },
        "models": [MODEL_SPECS[m].label for m in models] + [xgb_label, persist_label],
        "horizon_minutes": config.horizon_minutes,
        "config": {
            "seeds": [int(s) for s in seeds],
            "fit": asdict(fit_config),
            "recurrent": {"units": recurrent_units, "dropout": recurrent_dropout, "head": "Dropout -> Dense(32, relu) -> Dense(1)"},
            "transformer": asdict(transformer_config),
            "window_size": config.window_size,
            "xgb_seed": XGB_SEED,
            "tensorflow": _tensorflow_build(),
        },
        "seed_summary": summary,
        "metrics": metrics,
        "significance": [c.to_dict() for c in comparisons],
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta


def main(argv=None) -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=sorted(MODEL_SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        _, meta = deep_component(Path(tmp), models=args.models, seeds=args.seeds)
    for name, s in meta["seed_summary"].items():
        print(f"{s['label']}: seed-mean MAE {s['seed_mean_prediction_mae_min']:.3f}; per seed {[round(m, 3) for m in s['mae_by_seed']]}; {s['params']} params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
