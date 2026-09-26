#!/usr/bin/env python3
"""Train or tune LSTM variants on jb_to_woodlands (JB→SG) canonical export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from timeseries_xgb import TimeSeriesConfig, engineer_features, prepare_route_frame, sync_canonical_travel_times
from timeseries_lstm import (
    ARCHITECTURE_CHOICES,
    LSTMTrainConfig,
    compare_lstm_architectures,
    save_tune_report,
    train_lstm,
    tune_lstm_hyperparameters,
)

CACHE = Path(__file__).resolve().parent / "data" / "causeway_gdata.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "runs"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-bq", action="store_true")
    parser.add_argument("--mode", choices=("train", "tune", "compare"), default="train")
    parser.add_argument("--architecture", default="lstm", choices=ARCHITECTURE_CHOICES)
    parser.add_argument("--max-trials", type=int, default=None, help="Cap tuning grid size")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "lstm_last_tune.json")
    args = parser.parse_args(argv)

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("tensorflow required: pip install -r requirements-notebook.txt", file=sys.stderr)
        return 2

    ts_config = TimeSeriesConfig()
    export = sync_canonical_travel_times(CACHE, refresh=args.refresh_bq)
    features = engineer_features(prepare_route_frame(export, ts_config), ts_config)

    if args.mode == "train":
        cfg = LSTMTrainConfig(architecture=args.architecture, lstm_units=args.lstm_units, epochs=args.epochs)
        _, result = train_lstm(features, ts_config, cfg)
        print(f"architecture={cfg.architecture}")
        print("test:", result.metrics)
        print("persistence:", result.persistence_metrics)
        return 0

    if args.mode == "compare":
        df = compare_lstm_architectures(features, ts_config)
        print(df.to_string(index=False))
        return 0

    tune_result = tune_lstm_hyperparameters(features, ts_config, max_trials=args.max_trials)
    save_tune_report(tune_result, args.out)
    print("best_config:", tune_result.best_config)
    print("best_test_metrics:", tune_result.best_metrics)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
