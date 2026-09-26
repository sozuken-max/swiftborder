#!/usr/bin/env python3
"""Generate comparison plots for offline (60 min) and optional BQML (30 min) eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_artifacts import (
    artifact_relpath,
    create_run_dir,
    subdir,
    update_latest_pointer,
    write_manifest,
    write_run_readme,
)

CACHE = Path(__file__).resolve().parent / "data" / "causeway_gdata.csv"
BACKTEST_DAYS = ("2026-09-22", "2026-09-23", "2026-09-24")


def _comparison_to_dict(result) -> Dict[str, Any]:
    return {
        "label_challenger": result.label_challenger,
        "label_reference": result.label_reference,
        "n": int(result.n),
        "mean_ae_diff_min": float(result.mean_ae_diff_min),
        "paired_t_pvalue": float(result.paired_t_pvalue),
        "bootstrap_ci_low_min": float(result.bootstrap_ci_low_min),
        "bootstrap_ci_high_min": float(result.bootstrap_ci_high_min),
        "challenger_better_at_alpha": bool(result.challenger_better_at_alpha),
    }


def _offline_plots(run_dir: Path, refresh_bq: bool) -> Tuple[List[Path], Dict[str, Any]]:
    import pandas as pd

    import timeseries_xgb as tsx
    from plots import plot_backtest_method_means, plot_holdout_forecast_sample, plot_mae_diff_forest
    from significance import format_comparison_table

    out = subdir(run_dir, "offline")
    config = tsx.TimeSeriesConfig()
    export = tsx.sync_canonical_travel_times(CACHE, refresh=refresh_bq)
    raw = tsx.prepare_route_frame(export, config)
    features = tsx.engineer_features(raw, config)
    x_train, x_test, y_train, y_test = tsx.build_supervised_matrices(features, config)
    model = tsx.train_xgb(x_train, y_train)
    pred = model.predict(x_test)
    persist = tsx.holdout_persistence_predictions(x_test, config)

    y_series, free_flow = tsx.regularized_series(raw, config)
    scores = tsx.score_forecast_days(model, y_series, free_flow, BACKTEST_DAYS, config)

    written: List[Path] = []
    written.append(
        plot_backtest_method_means(
            scores,
            out / "backtest-mae.png",
            title="Offline 60 min: mean MAE by method (22-24 Sep 2026)",
        )
    )

    split_index = int(len(features) * config.train_fraction)
    offset = config.window_size + config.horizon_steps - 1
    test_start = split_index + offset
    ts_raw = pd.to_datetime(raw["observed_at_sgt"], errors="coerce")
    test_times = ts_raw.iloc[test_start : test_start + len(y_test)]

    actual_min = y_test / 60.0
    written.append(
        plot_holdout_forecast_sample(
            test_times,
            actual_min,
            {
                "XGB": pred / 60.0,
                f"Persistence T-{config.horizon_minutes}": persist / 60.0,
            },
            out / "holdout-sample.png",
            title="Offline 60 min: hold-out tail (actual vs XGB vs persistence)",
            max_points=288,
        )
    )

    sig = tsx.holdout_significance_vs_persistence(model, x_test, y_test, config, block_size=12)
    written.append(
        plot_mae_diff_forest(
            [sig],
            out / "holdout-mae-diff.png",
            title="Offline hold-out: paired MAE diff vs persistence (bootstrap CI)",
        )
    )
    print("Offline significance:")
    print(format_comparison_table([sig]))

    try:
        cache_rel = CACHE.relative_to(Path(__file__).resolve().parent.parent).as_posix()
    except ValueError:
        cache_rel = CACHE.as_posix()
    meta: Dict[str, Any] = {
        "dataset": {
            "source": "swiftborder.causeway.travel_times",
            "cache_path": cache_rel,
            "refreshed_from_bq": refresh_bq,
            "canonical_rows": len(export),
            "route_id": config.route_id,
            "route_rows": len(raw),
        },
        "models": ["sklearn.XGBRegressor (eval/timeseries_xgb.train_xgb)"],
        "horizon_minutes": config.horizon_minutes,
        "train_fraction": config.train_fraction,
        "backtest_days": list(BACKTEST_DAYS),
        "holdout_rmse_min": float(tsx.rmse_minutes(y_test, pred)),
        "significance_vs_persistence": _comparison_to_dict(sig),
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta


def _bqml_plots(
    run_dir: Path,
    project: str,
    holdout_days: int,
    block_size: int,
    alpha: float,
) -> Tuple[List[Path], Dict[str, Any]]:
    from google.cloud import bigquery

    from layer_b import MODELS, _fetch_model, _fetch_persistence
    from metrics import score_slices
    from plots import plot_bqml_mae_comparison, plot_mae_diff_forest
    from significance import compare_scored_rows

    out = subdir(run_dir, "bqml")
    client = bigquery.Client(project=project)
    window_start, window_end, persist_rows = _fetch_persistence(client, project, holdout_days)
    slices_by_candidate: Dict[str, List[dict]] = {
        "Persistence": score_slices(persist_rows, "y_30", "predicted"),
    }
    model_row_sets: Dict[str, List[dict]] = {}
    for model in MODELS:
        model_row_sets[model] = _fetch_model(client, project, model, holdout_days)
        slices_by_candidate[model] = score_slices(model_row_sets[model], "y_30", "predicted")

    written: List[Path] = []
    written.append(
        plot_bqml_mae_comparison(
            slices_by_candidate,
            out / "mae-by-direction.png",
            title=f"BQML 30 min hold-out ({holdout_days}d): MAE by direction",
        )
    )

    comparisons = []
    for model in MODELS:
        for direction in ("both", "SG_TO_MY", "MY_TO_SG"):
            comparisons.append(
                compare_scored_rows(
                    persist_rows,
                    model_row_sets[model],
                    direction=direction,
                    time_of_day="all",
                    label_challenger=f"{model} ({direction})",
                    label_reference="Persistence",
                    block_size=block_size,
                    alpha=alpha,
                )
            )
    written.append(
        plot_mae_diff_forest(
            comparisons,
            out / "mae-diff-ci.png",
            title="BQML 30 min: paired MAE difference vs persistence (bootstrap CI)",
        )
    )

    both_all = next(
        (s for s in slices_by_candidate["Persistence"] if s["direction"] == "both" and s["time_of_day"] == "all"),
        {},
    )
    meta: Dict[str, Any] = {
        "dataset": {
            "project": project,
            "view": "traffic_prediction.v_training_set",
            "holdout_days": holdout_days,
            "window_start": window_start,
            "window_end": window_end,
            "holdout_rows": len(persist_rows),
        },
        "models": ["Persistence (y_persistence)", "lin_h30", "xgb_h30"],
        "horizon_minutes": 30,
        "significance_block_size": block_size,
        "significance_alpha": alpha,
        "significance_comparisons": [_comparison_to_dict(c) for c in comparisons],
        "mae_min_both_all": {
            "Persistence": both_all.get("mae"),
            **{
                m: next(
                    (
                        s["mae"]
                        for s in slices_by_candidate[m]
                        if s["direction"] == "both" and s["time_of_day"] == "all"
                    ),
                    None,
                )
                for m in MODELS
            },
        },
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Parent directory for run folders (default: eval/runs)",
    )
    parser.add_argument("--run-id", default=None, help="Explicit run folder name")
    parser.add_argument(
        "--reuse-run",
        action="store_true",
        help="Write into an existing run directory (must exist)",
    )
    parser.add_argument("--refresh-bq", action="store_true", help="Re-download travel_times before offline plots")
    parser.add_argument("--bqml", action="store_true", help="Include 30 min BQML plots (with offline by default)")
    parser.add_argument("--bqml-only", action="store_true", help="BQML plots only (no offline)")
    parser.add_argument("--project", default="swiftborder")
    parser.add_argument("--holdout-days", type=int, default=3)
    parser.add_argument("--significance-block-size", type=int, default=6)
    parser.add_argument("--significance-alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    if args.bqml_only:
        components = ["bqml"]
    elif args.bqml:
        components = ["offline", "bqml"]
    else:
        components = ["offline"]

    if args.run_id:
        from run_artifacts import RUNS_ROOT

        runs_root = args.runs_root or RUNS_ROOT
        run_dir = runs_root / args.run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
        elif not args.reuse_run:
            print(f"Run directory exists: {run_dir} (use --reuse-run to append)", file=sys.stderr)
            return 2
    else:
        run_dir = create_run_dir(
            components=components,
            runs_root=args.runs_root,
            exist_ok=args.reuse_run,
        )

    manifest: Dict[str, Any] = {"components": components}
    all_written: List[Path] = []

    run_offline = "offline" in components
    if run_offline:
        try:
            paths, offline_meta = _offline_plots(run_dir, args.refresh_bq)
            all_written.extend(paths)
            manifest["offline"] = offline_meta
        except FileNotFoundError as exc:
            print(f"Offline plots skipped: {exc}", file=sys.stderr)
            print("Place a cache at eval/data/causeway_gdata.csv or use --refresh-bq.", file=sys.stderr)

    if args.bqml:
        paths, bqml_meta = _bqml_plots(
            run_dir,
            args.project,
            args.holdout_days,
            args.significance_block_size,
            args.significance_alpha,
        )
        all_written.extend(paths)
        manifest["bqml"] = bqml_meta

    if not all_written:
        return 1

    write_manifest(run_dir, manifest)
    write_run_readme(run_dir, manifest)
    update_latest_pointer(run_dir, manifest)

    print(f"\nRun directory: {run_dir}")
    print("Wrote:")
    for p in all_written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
