#!/usr/bin/env python3
"""Read-only Layer B harness: score persistence and BQML models on a hold-out window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import bigquery

from metrics import score_slices
from significance import compare_scored_rows, format_comparison_table

DEFAULT_PROJECT = "swiftborder"
DATASET = "traffic_prediction"
VIEW = "v_training_set"
MODELS = ("lin_h30", "xgb_h30")
HORIZON_LABEL = "30 min"


def _holdout_query(project: str, holdout_days: int) -> str:
    table = f"`{project}.{DATASET}.{VIEW}`"
    return f"""
WITH bounds AS (
  SELECT
    TIMESTAMP_SUB(MAX(bin_ts), INTERVAL @holdout_days DAY) AS holdout_start,
    MAX(bin_ts) AS holdout_end
  FROM {table}
  WHERE y_30 IS NOT NULL
),
holdout AS (
  SELECT t.*
  FROM {table} AS t
  CROSS JOIN bounds AS b
  WHERE t.y_30 IS NOT NULL AND t.bin_ts >= b.holdout_start
)
SELECT
  (SELECT holdout_start FROM bounds) AS holdout_start,
  (SELECT holdout_end FROM bounds) AS holdout_end,
  direction,
  bin_ts,
  is_morning_peak,
  is_evening_peak,
  y_30,
  y_persistence
FROM holdout
"""


def _predict_query(project: str, model: str, holdout_days: int) -> str:
    table = f"`{project}.{DATASET}.{VIEW}`"
    model_ref = f"`{project}.{DATASET}.{model}`"
    return f"""
WITH bounds AS (
  SELECT TIMESTAMP_SUB(MAX(bin_ts), INTERVAL @holdout_days DAY) AS holdout_start
  FROM {table}
  WHERE y_30 IS NOT NULL
),
holdout AS (
  SELECT t.*
  FROM {table} AS t
  CROSS JOIN bounds AS b
  WHERE t.y_30 IS NOT NULL AND t.bin_ts >= b.holdout_start
)
SELECT
  direction,
  bin_ts,
  is_morning_peak,
  is_evening_peak,
  y_30,
  predicted_y_30 AS predicted
FROM ML.PREDICT(MODEL {model_ref}, TABLE holdout)
"""


def _fetch_persistence(client: bigquery.Client, project: str, holdout_days: int) -> tuple[str, str, List[dict]]:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("holdout_days", "INT64", holdout_days),
        ]
    )
    rows = list(client.query(_holdout_query(project, holdout_days), job_config=job_config).result())
    if not rows:
        raise RuntimeError("hold-out query returned no rows")
    start = str(rows[0]["holdout_start"])
    end = str(rows[0]["holdout_end"])
    scored = [
        {
            "direction": r["direction"],
            "is_morning_peak": r["is_morning_peak"],
            "is_evening_peak": r["is_evening_peak"],
            "bin_ts": r["bin_ts"],
            "y_30": r["y_30"],
            "predicted": r["y_persistence"],
        }
        for r in rows
    ]
    return start, end, scored


def _fetch_model(client: bigquery.Client, project: str, model: str, holdout_days: int) -> List[dict]:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("holdout_days", "INT64", holdout_days),
        ]
    )
    rows = client.query(_predict_query(project, model, holdout_days), job_config=job_config).result()
    return [
        {
            "direction": r["direction"],
            "bin_ts": r["bin_ts"],
            "is_morning_peak": r["is_morning_peak"],
            "is_evening_peak": r["is_evening_peak"],
            "y_30": r["y_30"],
            "predicted": r["predicted"],
        }
        for r in rows
    ]


def _persistence_mae_lookup(persist_slices: List[dict]) -> Dict[tuple[str, str], float]:
    return {
        (s["direction"], s["time_of_day"]): s["mae"]
        for s in persist_slices
        if s["direction"] != "both" or s["time_of_day"] == "all"
    }


def _print_table(
    candidate: str,
    window: str,
    slices: List[dict],
    persist_mae: Dict[tuple[str, str], float],
) -> None:
    print(f"\n## {candidate} ({HORIZON_LABEL})")
    print("| Candidate | Horizon | Test window | MAE (min) | RMSE (min) | Persistence MAE | Direction | Time of day | n |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in slices:
        if s["time_of_day"] == "all" and s["direction"] == "both":
            persist_col = "—" if candidate == "Persistence" else f"{persist_mae.get(('both', 'all'), float('nan')):.3f}"
        elif candidate == "Persistence":
            persist_col = "—"
        else:
            persist_col = f"{persist_mae.get((s['direction'], s['time_of_day']), float('nan')):.3f}"
        print(
            f"| {candidate} | {HORIZON_LABEL} | {window} | {s['mae']:.3f} | {s['rmse']:.3f} | {persist_col} | {s['direction']} | {s['time_of_day']} | {s['n']} |"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--holdout-days", type=int, default=3, help="Length of the trailing hold-out window")
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Write model_registry (not implemented; flag reserved)",
    )
    parser.add_argument(
        "--significance",
        action="store_true",
        help="Print paired MAE significance vs persistence (block bootstrap + paired t-test)",
    )
    parser.add_argument(
        "--significance-block-size",
        type=int,
        default=6,
        help="Block length for bootstrap (6 bins = 60 min at 10-minute cadence)",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=0.05,
        help="Alpha for bootstrap confidence interval",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write BQML comparison PNGs under eval/runs/<run_id>/bqml/ (requires matplotlib)",
    )
    parser.add_argument("--run-id", default=None, help="Run folder name under eval/runs (default: auto)")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory (overrides --run-id)")
    parser.add_argument("--reuse-run", action="store_true", help="Write into an existing run directory")
    args = parser.parse_args(argv)

    if args.write_registry:
        print("model_registry writes are disabled by default; --write-registry is not implemented yet.", file=sys.stderr)
        return 2

    client = bigquery.Client(project=args.project)
    start, end, persist_rows = _fetch_persistence(client, args.project, args.holdout_days)
    window = f"{start} .. {end}"

    persist_slices = score_slices(persist_rows, "y_30", "predicted")
    persist_mae = {(s["direction"], s["time_of_day"]): s["mae"] for s in persist_slices}

    print("Layer B hold-out scores (skill on the Maps duration series; label y_30).")
    _print_table("Persistence", window, persist_slices, persist_mae)

    model_row_sets: Dict[str, List[dict]] = {}
    for model in MODELS:
        model_rows = _fetch_model(client, args.project, model, args.holdout_days)
        model_row_sets[model] = model_rows
        model_slices = score_slices(model_rows, "y_30", "predicted")
        _print_table(model, window, model_slices, persist_mae)

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
                    block_size=args.significance_block_size,
                    alpha=args.significance_alpha,
                )
            )

    if args.significance:
        print("\n## Significance vs persistence (paired absolute error; minutes)")
        print(
            "Block bootstrap mean AE difference (challenger - reference); "
            f"block_size={args.significance_block_size}, alpha={args.significance_alpha}. "
            "'Better at alpha' uses one-sided interpretation: CI entirely below 0 => challenger."
        )
        print(format_comparison_table(comparisons))

    if args.plots:
        from generate_comparison_plots import _bqml_plots
        from run_artifacts import (
            create_run_dir,
            update_latest_pointer,
            write_manifest,
            write_run_readme,
        )

        if args.run_dir is not None:
            run_dir = args.run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_run_dir(
                components=["bqml"],
                run_id=args.run_id,
                exist_ok=args.reuse_run,
            )
        _paths, bqml_meta = _bqml_plots(
            run_dir,
            args.project,
            args.holdout_days,
            args.significance_block_size,
            args.significance_alpha,
        )
        manifest = {"components": ["bqml"], "bqml": bqml_meta}
        write_manifest(run_dir, manifest)
        write_run_readme(run_dir, manifest)
        update_latest_pointer(run_dir, manifest)
        print(f"Wrote BQML comparison plots under {run_dir / 'bqml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
