"""Legacy top-level module names (``import joined``) map to topic packages under ``eval/``."""

from __future__ import annotations

import importlib
import sys
from typing import Dict

# name -> dotted module (run with cwd=eval or PYTHONPATH=eval)
MODULE_ALIASES: Dict[str, str] = {
    "constants": "harness.constants",
    "metrics": "harness.metrics",
    "plots": "harness.plots",
    "significance": "harness.significance",
    "run_artifacts": "harness.run_artifacts",
    "promote_report_run": "harness.promote_report_run",
    "replay_series": "harness.replay_series",
    "generate_comparison_plots": "harness.generate_comparison_plots",
    "features": "lb.features",
    "joined": "lb.joined",
    "layer_b": "lb.layer_b",
    "bq_replica": "lb.bq_replica",
    "bqml_parity": "lb.bqml_parity",
    "camera_forecast": "lb.camera_forecast",
    "ensemble": "lb.ensemble",
    "fuzzy_traffic": "lb.fuzzy_traffic",
    "timeseries_xgb": "timeseries.timeseries_xgb",
    "timeseries_lstm": "timeseries.timeseries_lstm",
    "timeseries_transformer": "timeseries.timeseries_transformer",
    "deep_forecast": "timeseries.deep_forecast",
    "train_lstm": "timeseries.train_lstm",
    "check_tf_gpu": "timeseries.check_tf_gpu",
    "layer_a": "la.layer_a",
    "layer_a_count_lib": "la.layer_a_count_lib",
    "score_layer_a_local": "la.score_layer_a_local",
    "score_layer_a_swiftbackend": "la.score_layer_a_swiftbackend",
    "download_layer_a_frames": "la.download_layer_a_frames",
    "build_layer_a_artifact_summary": "la.build_layer_a_artifact_summary",
    "backfill_camera_counts": "la.backfill_camera_counts",
    "horizon_study": "studies.horizon_study",
    "counts_substitute": "studies.counts_substitute",
    "camera_current": "studies.camera_current",
    "layer_a_eval": "studies.layer_a_eval",
    "layer_b_fcm_mlp": "warehouse.layer_b_fcm_mlp",
    "layer_b_timesfm": "warehouse.layer_b_timesfm",
    "layer_b_fcm_xgb_mix": "warehouse.layer_b_fcm_xgb_mix",
}

_CLI_MODULES = tuple(sorted(MODULE_ALIASES.keys()))


def register_flat_aliases() -> None:
    for short, qualified in MODULE_ALIASES.items():
        if short in sys.modules:
            continue
        sys.modules[short] = importlib.import_module(qualified)


def cli_module_names() -> tuple[str, ...]:
    return _CLI_MODULES
