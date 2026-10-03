#!/usr/bin/env python3
"""Offline (60 min, sklearn XGB) scoring and figures, plus optional BQML (30 min) via layer_b.

Writes one run folder under eval/runs/<run_id>/ with a schema-v2 run.json (see run_artifacts.py).

    python generate_comparison_plots.py                 # offline only (uses the CSV cache)
    python generate_comparison_plots.py --refresh-bq    # re-download travel_times first (read-only)
    python generate_comparison_plots.py --bqml          # offline + BQML fixed window
    python generate_comparison_plots.py --deep          # + LSTM / GRU / patch Transformer (TensorFlow)
    python generate_comparison_plots.py --fuzzy         # + fuzzy light / moderate / heavy classifier
    python generate_comparison_plots.py --bqml --joined --ensemble   # + ensembles / hybrids of Layer B models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_artifacts import (
    artifact_relpath,
    create_run_dir,
    file_fingerprint,
    metric_row,
    new_manifest,
    subdir,
    update_latest_pointer,
    write_manifest,
    write_run_readme,
)

CACHE = Path(__file__).resolve().parent / "data" / "causeway_gdata.csv"
XGB_SEED = 42


def offline_component(
    run_dir: Path,
    *,
    refresh_bq: bool = False,
    cache: Path = CACHE,
    project: str = "swiftborder",
    alpha: float = 0.05,
    data_cutoff=None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Train on the chronological 80% (by target time); score the rest and full backtest days."""
    import numpy as np

    import timeseries_xgb as tsx
    from plots import plot_backtest_method_means, plot_holdout_forecast_sample, plot_mae_diff_forest
    from significance import apply_holm, compare_absolute_errors, format_comparison_table

    out = subdir(run_dir, "offline")
    config = tsx.TimeSeriesConfig()
    export = tsx.sync_canonical_travel_times(cache, project=project, refresh=refresh_bq)
    export = tsx.apply_data_cutoff(export, data_cutoff)
    raw = tsx.prepare_route_frame(export, config)
    features = tsx.engineer_features(raw, config)
    split = tsx.split_supervised(features, config)
    model = tsx.train_xgb(split.train[split.feature_cols], split.train["y"].to_numpy(), config=config, random_state=XGB_SEED)

    test = split.test
    y = test["y"].to_numpy()
    pred = model.predict(test[split.feature_cols])
    persist = test["persistence"].to_numpy()
    maps_typical = test["duration_sec"].to_numpy()
    persist_label = f"Persistence T-{config.horizon_minutes}"
    maps_label = "Maps typical duration at origin"

    candidates = {"XGB (sklearn)": pred, persist_label: persist, maps_label: maps_typical}
    metrics: List[Dict[str, Any]] = []
    for name, p in candidates.items():
        ok = ~np.isnan(p)
        metrics.append(metric_row(name, "holdout", int(ok.sum()), tsx.mae_minutes(y[ok], p[ok]), tsx.rmse_minutes(y[ok], p[ok])))

    days = tsx.backtest_days(split)
    scores = tsx.score_forecast_days(model, features, split, days, config)
    for _, row in scores.iterrows():
        metrics.append(metric_row(row["method"], f"backtest {row['date']}", row["n"], row["MAE_min"], row["RMSE_min"]))
    if not scores.empty:
        for method, g in scores.groupby("method"):
            metrics.append(
                metric_row(method, "backtest mean of days", int(g["n"].sum()), float(g["MAE_min"].mean()), float(g["RMSE_min"].mean()))
            )

    ok = ~np.isnan(maps_typical)
    scale = 1.0 / 60.0
    timestamps = list(test["target_ts"])
    comparisons = apply_holm(
        [
            compare_absolute_errors(
                y * scale, pred * scale, persist * scale,
                label_challenger="XGB (sklearn)", label_reference=persist_label,
                horizon_steps=config.horizon_steps, timestamps=timestamps, alpha=alpha,
            ),
            compare_absolute_errors(
                y[ok] * scale, pred[ok] * scale, maps_typical[ok] * scale,
                label_challenger="XGB (sklearn)", label_reference=maps_label,
                horizon_steps=config.horizon_steps, timestamps=[t for t, k in zip(timestamps, ok) if k], alpha=alpha,
            ),
        ]
    )
    print("Offline significance (Holm over this family):")
    print(format_comparison_table(comparisons))

    written: List[Path] = []
    if not scores.empty:
        written.extend(
            plot_backtest_method_means(
                scores,
                out / "backtest-mae.png",
                title=f"Offline 60 min: mean MAE by method ({days[0]} .. {days[-1]}, after split)",
            )
        )
    written.extend(
        plot_holdout_forecast_sample(
            list(test["target_ts"]),
            y / 60.0,
            {"XGB": pred / 60.0, persist_label: persist / 60.0},
            out / "holdout-sample.png",
            title="Offline 60 min: hold-out tail (raw actual vs XGB vs persistence)",
            max_points=288,
        )
    )
    written.extend(
        plot_mae_diff_forest(
            comparisons,
            out / "holdout-mae-diff.png",
            title="Offline hold-out: paired MAE difference (day-block bootstrap CI)",
        )
    )

    meta: Dict[str, Any] = {
        "dataset": {
            "source": f"{project}.causeway.travel_times",
            "cache": file_fingerprint(cache),
            "refreshed_from_bq": refresh_bq,
            "data_cutoff_utc": None if data_cutoff is None else data_cutoff.isoformat(),
            "canonical_rows": int(len(export)),
            "route_id": config.route_id,
            "route_scope": tsx.OFFLINE_ROUTE_LABEL,
            "route_rows": int(len(raw)),
            "observed_min_sgt": str(raw["observed_at_sgt"].min()),
            "observed_max_sgt": str(raw["observed_at_sgt"].max()),
            "supervised_rows": {"train": int(len(split.train)), "test": int(len(test))},
        },
        "window": {
            "timezone": "Asia/Singapore",
            "basis": "target time; train labels < start <= test labels",
            "start": str(split.boundary),
            "end": str(test["target_ts"].max()),
            "backtest_days": days,
        },
        "models": ["sklearn.XGBRegressor (timeseries_xgb.train_xgb)", persist_label, maps_label],
        "horizon_minutes": config.horizon_minutes,
        "config": {
            "window_size": config.window_size,
            "keep_lags": config.keep_lags,
            "train_fraction": config.train_fraction,
            "max_slew_step_sec": config.max_slew_step_sec,
            "max_ffill_steps": config.max_ffill_steps,
            "use_dwt": config.use_dwt,
            "dwt_wavelet": config.dwt_wavelet,
            "dwt_level": config.dwt_level,
            "xgb": dict(config.xgb.__dict__),
            "xgb_seed": XGB_SEED,
            "features": split.feature_cols,
        },
        "metrics": metrics,
        "significance": [c.to_dict() for c in comparisons],
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta


def cutoff_window_end(cutoff, window_end: Optional[str]) -> Optional[str]:
    """BQML window end under a data cutoff: the last origin whose 30-min label is observed by the cutoff.

    Returns ``window_end`` unchanged without a cutoff. Raises ValueError when an explicit window end
    would score a label after the cutoff.
    """
    import pandas as pd

    from layer_b import parse_sgt
    from timeseries_xgb import CUTOFF_TZ, LABEL_LAG_30

    if cutoff is None:
        return window_end
    if window_end:
        end = pd.Timestamp(parse_sgt(window_end))
        if end + LABEL_LAG_30 > cutoff:
            raise ValueError(f"--window-end {window_end} scores labels after --data-cutoff")
        return window_end
    last = (cutoff - LABEL_LAG_30).floor("10min")
    return last.tz_convert(CUTOFF_TZ).strftime("%Y-%m-%d %H:%M")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=None, help="Parent directory for run folders (default: eval/runs)")
    parser.add_argument("--run-id", default=None, help="Explicit run folder name")
    parser.add_argument("--reuse-run", action="store_true", help="Write into an existing run directory")
    parser.add_argument("--refresh-bq", action="store_true", help="Re-download travel_times before offline scoring")
    parser.add_argument("--bqml", action="store_true", help="Also run the BQML fixed-window harness (layer_b)")
    parser.add_argument("--bqml-only", action="store_true", help="BQML only (no offline)")
    parser.add_argument("--project", default="swiftborder")
    parser.add_argument("--window-start", default=None, help="BQML window start (SGT); default in layer_b")
    parser.add_argument("--window-end", default=None, help="BQML window end (SGT); default latest labelled bin")
    parser.add_argument("--joined", action="store_true", help="Also run the joined-feature experiment (joined.py)")
    parser.add_argument("--joined-start", default=None, help="first joined test day (SGT, default joined.DEFAULT_TEST_START)")
    parser.add_argument("--joined-end", default=None, help="last joined test day (SGT, default joined.DEFAULT_TEST_END)")
    parser.add_argument("--deep", action="store_true", help="Also score LSTM / GRU / patch Transformer on the offline split (needs TensorFlow)")
    parser.add_argument("--deep-seeds", type=int, nargs="+", default=None, help="seeds per deep model (default deep_forecast.DEFAULT_SEEDS)")
    parser.add_argument("--fuzzy", action="store_true", help="Also score the fuzzy traffic-level classifier (60 min, both directions)")
    parser.add_argument("--ensemble", action="store_true", help="Also score ensembles / hybrids of the Layer B models (needs --bqml and --joined; uses --deep if given)")
    parser.add_argument("--data-cutoff", default=None, help="Drop observations after this time (SGT, e.g. '2026-10-19 23:59'); no scored label is later")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.ensemble and not ((args.bqml or args.bqml_only) and args.joined):
        parser.error("--ensemble needs --bqml and --joined (it reuses their out-of-sample rows)")
    import timeseries_xgb as tsx

    cutoff = tsx.parse_data_cutoff(args.data_cutoff)
    try:
        args.window_end = cutoff_window_end(cutoff, args.window_end)
    except ValueError as exc:
        parser.error(str(exc))
    if cutoff is not None and args.joined:
        import joined as _joined

        last_day = cutoff.tz_convert(tsx.CUTOFF_TZ).strftime("%Y-%m-%d")
        if (args.joined_end or _joined.DEFAULT_TEST_END) > last_day:
            parser.error(f"--joined-end is after the --data-cutoff day {last_day}")

    components = ["bqml"] if args.bqml_only else (["offline", "bqml"] if args.bqml else ["offline"])
    if args.joined:
        components.append("joined")
    if args.deep:
        components.append("deep")
    if args.fuzzy:
        components.append("fuzzy")
    if args.ensemble:
        components.append("ensemble")
    run_dir = create_run_dir(components=components, run_id=args.run_id, runs_root=args.runs_root, exist_ok=args.reuse_run)
    manifest = new_manifest(components)
    if cutoff is not None:
        manifest["data_cutoff"] = {
            "utc": cutoff.isoformat(),
            "sgt": cutoff.tz_convert(tsx.CUTOFF_TZ).isoformat(),
            "rule": "observations after the cutoff are dropped, so every scored label is at or before it",
            "bqml_window_end_sgt": args.window_end,
        }
    written: List[Path] = []
    kept: Dict[str, Dict[str, Any]] = {"bqml": {}, "joined": {}, "deep": {}}

    if "offline" in components:
        try:
            paths, meta = offline_component(run_dir, refresh_bq=args.refresh_bq, project=args.project, alpha=args.alpha, data_cutoff=cutoff)
        except FileNotFoundError as exc:
            print(f"Offline skipped: {exc}. Place eval/data/causeway_gdata.csv or use --refresh-bq.", file=sys.stderr)
            return 1
        written.extend(paths)
        manifest["offline"] = meta

    if "bqml" in components:
        from layer_b import bqml_component

        paths, meta = bqml_component(
            run_dir, project=args.project, window_start=args.window_start, window_end=args.window_end, alpha=args.alpha,
            keep=kept["bqml"],
        )
        written.extend(paths)
        manifest["bqml"] = meta

    if "joined" in components:
        import joined

        import features

        frame, info = joined.load_inputs(refresh_bq=False, data_cutoff=cutoff)  # offline step already refreshed the cache if asked
        parity = features.parity_with_live_view(frame, project=args.project)
        info["parity_with_live_v_training_set"] = parity
        print("Parity with live v_training_set (max abs diff per column):", parity)
        paths, meta = joined.joined_component(
            run_dir,
            frame,
            start=args.joined_start or joined.DEFAULT_TEST_START,
            end=args.joined_end or joined.DEFAULT_TEST_END,
            alpha=args.alpha,
            inputs=info,
            keep=kept["joined"],
        )
        written.extend(paths)
        manifest["joined"] = meta

    if "deep" in components:
        import deep_forecast

        seeds = tuple(args.deep_seeds) if args.deep_seeds else deep_forecast.DEFAULT_SEEDS
        paths, meta = deep_forecast.deep_component(run_dir, project=args.project, seeds=seeds, alpha=args.alpha, keep=kept["deep"], data_cutoff=cutoff)
        written.extend(paths)
        manifest["deep"] = meta

    if "fuzzy" in components:
        import fuzzy_traffic

        paths, meta = fuzzy_traffic.fuzzy_component(run_dir, project=args.project, alpha=args.alpha, data_cutoff=cutoff)
        written.extend(paths)
        manifest["fuzzy"] = meta

    if "ensemble" in components:
        import ensemble

        paths, meta = ensemble.ensemble_component(
            run_dir,
            bqml_result=kept["bqml"]["result"],
            joined_oof=kept["joined"]["oof"],
            deep_keep=kept["deep"] or None,
            alpha=args.alpha,
        )
        written.extend(paths)
        manifest["ensemble"] = meta

    from run_artifacts import multiplicity_summary

    manifest["multiplicity"] = multiplicity_summary(manifest)
    print(f"Multiplicity check (Holm over all {manifest['multiplicity']['n_comparisons']} comparisons):", manifest["multiplicity"]["transitions"])

    write_manifest(run_dir, manifest)
    write_run_readme(run_dir, manifest)
    update_latest_pointer(run_dir, manifest, runs_root=args.runs_root)
    print(f"\nRun directory: {run_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
