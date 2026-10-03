#!/usr/bin/env python3
"""Write figure CSVs from an existing schema-v2 ``run.json``.

Does not fit models, read BigQuery, or redraw PNGs. The promoted snapshot
``eval/runs/report/`` is refused so a replay cannot overwrite the citation target.
``--output-dir`` writes the CSVs somewhere else and still refuses that snapshot.

``offline/holdout-sample`` is skipped: the tail of actual and forecast minutes
is not in the manifest.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from camera_forecast import DIRECTIONS
from fuzzy_traffic import LEVELS, LevelPartition
from plots import write_series_csv
from run_artifacts import EVAL_ROOT, load_manifest

REPORT_SNAPSHOT = EVAL_ROOT / "runs" / "report"
SNAPSHOTS = (REPORT_SNAPSHOT, EVAL_ROOT / "runs" / "report-confirm")
HOLDOUT_SAMPLE = "offline/holdout-sample.png"
DIRECTIONS_WITH_BOTH = ("SG_TO_MY", "MY_TO_SG", "both")
FOREST_COLUMNS = (
    "display_rank",
    "challenger",
    "reference",
    "n",
    "mean_ae_diff_min",
    "bootstrap_ci_low_min",
    "bootstrap_ci_high_min",
    "decision",
)
PLOTTED_CONFUSIONS = (
    "Persistence level",
    "Fuzzy rule base",
    "XGB forecast -> fuzzy level",
)
DEEP_REFERENCE_ORDER = ("Persistence T-60", "XGB (sklearn)")


@dataclass
class ReplayResult:
    written: List[Path] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def _writes_into_report(path: Path) -> bool:
    resolved = path.resolve()
    for snap in SNAPSHOTS:
        snapshot = snap.resolve()
        if resolved == snapshot or snapshot in resolved.parents:
            return True
    return False


def replay_series_csvs(
    run_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    *,
    output_dir: Optional[Path] = None,
) -> ReplayResult:
    """Write same-stem CSVs for figures whose series are in ``run.json``.

    ``output_dir`` receives the CSVs. When it is omitted, they are written under
    ``run_dir``. Either destination is refused when it is the promoted snapshot.
    """
    run_dir = Path(run_dir)
    dest = Path(output_dir) if output_dir is not None else run_dir
    if _writes_into_report(dest):
        raise ValueError(f"refusing to write into the promoted report snapshot: {dest}")
    data = dict(manifest) if manifest is not None else load_manifest(run_dir)
    result = ReplayResult()
    _replay_offline(dest, data.get("offline") or {}, result)
    _replay_bqml(dest, data.get("bqml") or {}, result)
    _replay_joined(dest, data.get("joined") or {}, result)
    _replay_deep(dest, data.get("deep") or {}, result)
    _replay_fuzzy(dest, data.get("fuzzy") or {}, result)
    _replay_ensemble(dest, data.get("ensemble") or {}, result)
    return result


def _dest(run_dir: Path, block: Mapping[str, Any], relative: str) -> Optional[Path]:
    """PNG path from ``artifacts`` when the component lists figures; else the conventional path."""
    name = Path(relative).name
    artifacts = [str(item) for item in (block.get("artifacts") or [])]
    if artifacts:
        for item in artifacts:
            if Path(item).name == name:
                return run_dir / Path(item)
        return None
    return run_dir / relative


def _write(result: ReplayResult, figure: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    if not rows:
        result.skipped.append(f"{figure.as_posix()}: no rows in run.json")
        return
    result.written.append(write_series_csv(figure, rows, columns))


def _forest_rows(comparisons: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Same column order as ``plot_mae_diff_forest`` (rank 0 is the top row)."""
    rows = []
    for rank, row in enumerate(comparisons):
        rows.append(
            {
                "display_rank": rank,
                "challenger": row.get("label_challenger"),
                "reference": row.get("label_reference"),
                "n": row.get("n"),
                "mean_ae_diff_min": row.get("mean_ae_diff_min"),
                "bootstrap_ci_low_min": row.get("bootstrap_ci_low_min"),
                "bootstrap_ci_high_min": row.get("bootstrap_ci_high_min"),
                "decision": row.get("decision", "not significant"),
            }
        )
    return rows


def _by_family(block: Mapping[str, Any], family: Optional[str]) -> List[Mapping[str, Any]]:
    rows = list(block.get("significance") or [])
    if family is None:
        untagged = [row for row in rows if not row.get("family")]
        return untagged or rows
    return [row for row in rows if row.get("family") == family]


def _replay_offline(run_dir: Path, block: Mapping[str, Any], result: ReplayResult) -> None:
    if not block:
        return
    backtest = _dest(run_dir, block, "offline/backtest-mae.png")
    if backtest is not None:
        means = [row for row in block.get("metrics") or [] if row.get("slice") == "backtest mean of days"]
        means.sort(key=lambda row: float(row["mae_min"]))
        _write(
            result,
            backtest,
            [
                {"order_from_bottom": i, "method": row["candidate"], "mean_mae_min": row["mae_min"]}
                for i, row in enumerate(means)
            ],
            ("order_from_bottom", "method", "mean_mae_min"),
        )
    listed = [str(item) for item in (block.get("artifacts") or [])]
    if any(Path(item).name == "holdout-sample.png" for item in listed):
        result.skipped.append(f"{HOLDOUT_SAMPLE}: point series (target_ts, actual, forecasts) is not in run.json")
    forest = _dest(run_dir, block, "offline/holdout-mae-diff.png")
    if forest is not None:
        _write(result, forest, _forest_rows(_by_family(block, None)), FOREST_COLUMNS)


def _replay_bqml(run_dir: Path, block: Mapping[str, Any], result: ReplayResult) -> None:
    if not block:
        return
    bars = _dest(run_dir, block, "bqml/mae-by-direction.png")
    if bars is not None:
        metrics = list(block.get("metrics") or [])
        candidates: List[str] = []
        for row in metrics:
            if row.get("slice") in {f"{direction}/all" for direction in DIRECTIONS_WITH_BOTH}:
                if row["candidate"] not in candidates:
                    candidates.append(row["candidate"])
        series = []
        for candidate in candidates:
            for direction in DIRECTIONS_WITH_BOTH:
                match = next(
                    (
                        row
                        for row in metrics
                        if row.get("candidate") == candidate and row.get("slice") == f"{direction}/all"
                    ),
                    None,
                )
                series.append(
                    {
                        "candidate": candidate,
                        "direction": direction,
                        "mae_min": None if match is None else match["mae_min"],
                    }
                )
        _write(result, bars, series, ("candidate", "direction", "mae_min"))
    forest = _dest(run_dir, block, "bqml/mae-diff-ci.png")
    if forest is not None:
        _write(result, forest, _forest_rows(_by_family(block, "headline")), FOREST_COLUMNS)


def _replay_joined(run_dir: Path, block: Mapping[str, Any], result: ReplayResult) -> None:
    if not block:
        return
    forest = _dest(run_dir, block, "joined/joined-mae-diff.png")
    if forest is not None:
        _write(result, forest, _forest_rows(_by_family(block, "joined")), FOREST_COLUMNS)
    bars = _dest(run_dir, block, "joined/joined-mae-by-feature-set.png")
    if bars is not None:
        rows = [row for row in block.get("metrics") or [] if row.get("slice") == "both/all"]
        rows.sort(key=lambda row: float(row["mae_min"]))
        _write(
            result,
            bars,
            [
                {"order_from_bottom": i, "candidate": row["candidate"], "slice": row["slice"], "mae_min": row["mae_min"]}
                for i, row in enumerate(rows)
            ],
            ("order_from_bottom", "candidate", "slice", "mae_min"),
        )
    camfc = _dest(run_dir, block, "joined/camfc-mae-diff.png")
    if camfc is not None:
        _write(result, camfc, _forest_rows(_by_family(block, "camfc")), FOREST_COLUMNS)
    profiles = _dest(run_dir, block, "joined/camfc-profiles.png")
    if profiles is not None:
        curves = ((block.get("dataset") or {}).get("camera_forecast") or {}).get("curves") or {}
        _write(result, profiles, _profile_rows(curves), (
            "tod_min",
            "hour_sgt",
            "direction",
            "day_type",
            "camera_forecast_vehicles",
            "maps_travel_time_min",
        ))


def _profile_rows(curves: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Camera profile only. The harness plots Maps profiles as ``None``."""
    if "tod_min" not in curves:
        return []
    rows = []
    for direction in DIRECTIONS:
        for day_type in ("weekday", "weekend"):
            camera = curves.get(f"{direction}/{day_type}") or []
            for tod, vehicles in zip(curves["tod_min"], camera):
                rows.append(
                    {
                        "tod_min": tod,
                        "hour_sgt": float(tod) / 60.0,
                        "direction": direction,
                        "day_type": day_type,
                        "camera_forecast_vehicles": vehicles,
                        "maps_travel_time_min": "",
                    }
                )
    return rows


def _replay_deep(run_dir: Path, block: Mapping[str, Any], result: ReplayResult) -> None:
    if not block:
        return
    seeds = _dest(run_dir, block, "deep/deep-mae-by-seed.png")
    if seeds is not None:
        _write(
            result,
            seeds,
            _seed_rows(block.get("seed_summary") or {}, list(block.get("metrics") or [])),
            ("kind", "model_index", "model", "seed", "mae_min"),
        )
    forest = _dest(run_dir, block, "deep/deep-mae-diff.png")
    if forest is not None:
        _write(result, forest, _forest_rows(_by_family(block, None)), FOREST_COLUMNS)


def _seed_rows(summary: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(summary.values()):
        rows.append(
            {
                "kind": "bar",
                "model_index": index,
                "model": item["label"],
                "seed": "",
                "mae_min": item["seed_mean_prediction_mae_min"],
            }
        )
        by_seed = list(item.get("mae_by_seed") or [])
        seeds = [run.get("seed", i) for i, run in enumerate(item.get("runs") or [])]
        if len(seeds) != len(by_seed):
            seeds = list(range(len(by_seed)))
        for seed, mae in zip(seeds, by_seed):
            rows.append({"kind": "seed", "model_index": index, "model": item["label"], "seed": seed, "mae_min": mae})
    holdout = {row["candidate"]: row["mae_min"] for row in metrics if row.get("slice") == "holdout"}
    for label in DEEP_REFERENCE_ORDER:
        if label in holdout:
            rows.append({"kind": "reference", "model_index": "", "model": label, "seed": "", "mae_min": holdout[label]})
    return rows


def _replay_fuzzy(run_dir: Path, block: Mapping[str, Any], result: ReplayResult) -> None:
    if not block:
        return
    memberships = _dest(run_dir, block, "fuzzy/fuzzy-memberships.png")
    if memberships is not None:
        partition = (block.get("config") or {}).get("partition")
        _write(
            result,
            memberships,
            [] if not partition else _membership_rows(partition),
            (
                "travel_time_min",
                "membership_light",
                "membership_moderate",
                "membership_heavy",
                "light_max_min",
                "heavy_min_min",
            ),
        )
    confusion = _dest(run_dir, block, "fuzzy/fuzzy-confusion.png")
    if confusion is not None:
        labels = list(block.get("confusion_labels") or LEVELS)
        stored = block.get("confusion") or {}
        names = [name for name in PLOTTED_CONFUSIONS if name in stored] or list(stored)
        rows = []
        for name in names:
            matrix = stored[name]
            for i, actual in enumerate(labels):
                for j, predicted in enumerate(labels):
                    rows.append({"model": name, "actual": actual, "predicted": predicted, "count": int(matrix[i][j])})
        _write(result, confusion, rows, ("model", "actual", "predicted", "count"))


def _membership_rows(partition: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Same 551-point grid as ``plot_memberships`` (travel time 5 to 60 minutes)."""
    cuts = LevelPartition(**dict(partition))
    minutes = np.linspace(5, 60, 551)
    degrees = cuts.memberships(minutes)
    return [
        {
            "travel_time_min": float(value),
            "membership_light": float(degrees[i, 0]),
            "membership_moderate": float(degrees[i, 1]),
            "membership_heavy": float(degrees[i, 2]),
            "light_max_min": cuts.light_max,
            "heavy_min_min": cuts.heavy_min,
        }
        for i, value in enumerate(minutes)
    ]


def _replay_ensemble(run_dir: Path, block: Mapping[str, Any], result: ReplayResult) -> None:
    if not block:
        return
    for relative, family in (
        ("ensemble/ensemble-30min-mae-diff.png", "30min"),
        ("ensemble/ensemble-60min-mae-diff.png", "60min"),
    ):
        figure = _dest(run_dir, block, relative)
        if figure is not None:
            _write(result, figure, _forest_rows(_by_family(block, family)), FOREST_COLUMNS)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory that contains run.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write CSVs here instead of the run directory (still refuses eval/runs/report/)",
    )
    args = parser.parse_args(argv)
    try:
        result = replay_series_csvs(args.run_dir, output_dir=args.output_dir)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for path in result.written:
        print(path)
    for note in result.skipped:
        print(f"skipped {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
