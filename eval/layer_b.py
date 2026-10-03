#!/usr/bin/env python3
"""Read-only Layer B harness: score persistence, BQML models, and their ensemble at 30 minutes.

Test window (fixed, out-of-sample): ``--window-start`` (default 2026-09-13 00:00 SGT, the day after
``lin_h30`` / ``xgb_h30`` were trained on 12 Sep) to ``--window-end`` (default: the latest bin with a
label, pinned at run time and recorded in run.json). The harness refuses to score a model whose
training started on or after the window start.

Rows scored: ``y_30 IS NOT NULL``, ``after_gap = 0``, and the bin three rows ahead is exactly 30
minutes later (checked in the query; ``v_training_set`` itself is not modified). Every candidate is
scored on the same rows.

Candidates: persistence (``y_persistence``), ``lin_h30``, ``xgb_h30``, and ``ensemble_mean`` (the mean
of the two BQML predictions; the hybrid/ensemble technique).

Slices: direction (both / SG_TO_MY / MY_TO_SG) crossed with one of time of day (morning peak,
evening peak, other), day type (weekday, weekend), light (day 07:00-18:59 SGT, night), or all.

Significance (significance.py): Diebold-Mariano + day-block bootstrap per comparison. Holm is
applied within two families: **headline** (each challenger vs persistence per direction on "all",
plus the ensemble vs each single model) and **slices** (challengers vs persistence on every other
slice; exploratory).

Nothing is written to BigQuery. ``model_registry`` is never changed by this script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

DEFAULT_PROJECT = "swiftborder"
DATASET = "traffic_prediction"
VIEW = "v_training_set"
MODELS = ("lin_h30", "xgb_h30")
ENSEMBLE = "ensemble_mean"
PERSISTENCE = "Persistence"
CANDIDATES = (PERSISTENCE,) + MODELS + (ENSEMBLE,)
HORIZON_MINUTES = 30
HORIZON_STEPS = 3  # 10-minute bins
SGT = timezone(timedelta(hours=8))
DEFAULT_WINDOW_START_SGT = "2026-09-13 00:00:00"
DIRECTIONS = ("SG_TO_MY", "MY_TO_SG")
DAY_START_HOUR, DAY_END_HOUR = 7, 19  # day = [07:00, 19:00) SGT


class Source(Protocol):
    """What the harness needs from BigQuery (a fake implements this in tests)."""

    def latest_labelled_bin(self) -> datetime: ...

    def fetch_rows(self, start: datetime, end: datetime) -> List[dict]: ...

    def fetch_predictions(self, model: str, start: datetime, end: datetime) -> List[dict]: ...

    def model_metadata(self, model: str) -> dict: ...


def parse_sgt(value: str) -> datetime:
    """Parse 'YYYY-MM-DD[ HH:MM[:SS]]'. Naive values are SGT; the result is tz-aware UTC."""
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SGT)
    return dt.astimezone(timezone.utc)


def _utc(ts: Any) -> datetime:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class BigQuerySource:
    """Read-only queries against project ``swiftborder`` (no DML, no DDL)."""

    def __init__(self, project: str = DEFAULT_PROJECT, client=None):
        from google.cloud import bigquery

        self._bq = bigquery
        self.project = project
        self.client = client or bigquery.Client(project=project)
        self.table = f"`{project}.{DATASET}.{VIEW}`"

    def _params(self, start: datetime, end: datetime):
        return self._bq.QueryJobConfig(
            query_parameters=[
                self._bq.ScalarQueryParameter("start", "TIMESTAMP", start),
                self._bq.ScalarQueryParameter("end", "TIMESTAMP", end),
            ]
        )

    def latest_labelled_bin(self) -> datetime:
        rows = list(self.client.query(f"SELECT MAX(bin_ts) AS mx FROM {self.table} WHERE y_30 IS NOT NULL").result())
        return _utc(rows[0]["mx"])

    def fetch_rows(self, start: datetime, end: datetime) -> List[dict]:
        query = f"""
WITH t AS (
  SELECT
    *,
    TIMESTAMP_DIFF(LEAD(bin_ts, {HORIZON_STEPS}) OVER (PARTITION BY route_id ORDER BY bin_ts), bin_ts, MINUTE)
      AS lead_gap_min
  FROM {self.table}
)
SELECT direction, bin_ts, is_morning_peak, is_evening_peak, is_weekend, after_gap, lead_gap_min,
       y_30, y_persistence
FROM t
WHERE y_30 IS NOT NULL AND bin_ts >= @start AND bin_ts <= @end
ORDER BY direction, bin_ts
"""
        return [dict(r.items()) for r in self.client.query(query, job_config=self._params(start, end)).result()]

    def fetch_predictions(self, model: str, start: datetime, end: datetime) -> List[dict]:
        query = f"""
SELECT direction, bin_ts, predicted_y_30 AS predicted
FROM ML.PREDICT(
  MODEL `{self.project}.{DATASET}.{model}`,
  (SELECT * FROM {self.table} WHERE y_30 IS NOT NULL AND bin_ts >= @start AND bin_ts <= @end)
)
"""
        return [dict(r.items()) for r in self.client.query(query, job_config=self._params(start, end)).result()]

    def model_metadata(self, model: str) -> dict:
        m = self.client.get_model(f"{self.project}.{DATASET}.{model}")
        runs = m.training_runs or []
        starts = []
        split = {}
        for run in runs:
            run = run if isinstance(run, dict) else getattr(run, "_properties", {})
            if run.get("startTime"):
                starts.append(_utc(run["startTime"]))
            opts = run.get("trainingOptions") or {}
            split = {k: opts[k] for k in ("dataSplitMethod", "dataSplitColumn") if k in opts} or split
        return {
            "model": model,
            "model_type": m.model_type,
            "created": _utc(m.created),
            "modified": _utc(m.modified),
            "training_run_starts": starts,
            "data_split": split,
        }


# --- row preparation --------------------------------------------------------------


@dataclass
class HoldoutRows:
    rows: List[dict]
    """Scored rows: direction, bin_ts, flags, y_30, and one prediction column per candidate."""
    excluded: Dict[str, int]
    window_start: datetime
    window_end: datetime


def _flag(v: Any) -> bool:
    return bool(v) and str(v) not in ("0", "False", "false")


def prepare_rows(
    base: Sequence[dict],
    predictions: Dict[str, Sequence[dict]],
    window_start: datetime,
    window_end: datetime,
) -> HoldoutRows:
    """Apply the gap filters and join predictions on (direction, bin_ts). Same rows for every candidate."""
    excluded = {"after_gap": 0, "lead_gap_not_30": 0, "missing_prediction": 0}
    pred_index: Dict[str, Dict[Tuple[str, datetime], float]] = {}
    for model, rows in predictions.items():
        idx = {}
        for r in rows:
            key = (str(r["direction"]), _utc(r["bin_ts"]))
            if key in idx:
                raise ValueError(f"duplicate prediction for {model} at {key}")
            idx[key] = r["predicted"]
        pred_index[model] = idx

    out: List[dict] = []
    for r in base:
        if _flag(r.get("after_gap")):
            excluded["after_gap"] += 1
            continue
        if r.get("lead_gap_min") != HORIZON_STEPS * 10:
            excluded["lead_gap_not_30"] += 1
            continue
        key = (str(r["direction"]), _utc(r["bin_ts"]))
        preds = {m: pred_index[m].get(key) for m in MODELS}
        if any(p is None for p in preds.values()):
            excluded["missing_prediction"] += 1
            continue
        row = {
            "direction": key[0],
            "bin_ts": key[1],
            "is_morning_peak": r.get("is_morning_peak"),
            "is_evening_peak": r.get("is_evening_peak"),
            "is_weekend": r.get("is_weekend"),
            "y_30": float(r["y_30"]),
            PERSISTENCE: float(r["y_persistence"]),
        }
        for m, p in preds.items():
            row[m] = float(p)
        row[ENSEMBLE] = (row["lin_h30"] + row["xgb_h30"]) / 2.0
        out.append(row)
    out.sort(key=lambda r: (r["direction"], r["bin_ts"]))
    return HoldoutRows(rows=out, excluded=excluded, window_start=window_start, window_end=window_end)


# --- slices -----------------------------------------------------------------------


def time_of_day(row: dict) -> str:
    if _flag(row.get("is_morning_peak")):
        return "morning peak"
    if _flag(row.get("is_evening_peak")):
        return "evening peak"
    return "other"


def day_type(row: dict) -> str:
    return "weekend" if _flag(row.get("is_weekend")) else "weekday"


def light(row: dict) -> str:
    hour = row["bin_ts"].astimezone(SGT).hour
    return "day" if DAY_START_HOUR <= hour < DAY_END_HOUR else "night"


DIMENSIONS = {
    "time_of_day": (time_of_day, ("morning peak", "evening peak", "other")),
    "day_type": (day_type, ("weekday", "weekend")),
    "light": (light, ("day", "night")),
}


def slice_definitions() -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    """(name, direction, dimension, value). Direction x (all | one value of one dimension)."""
    out = []
    for direction in ("both",) + DIRECTIONS:
        out.append((f"{direction}/all", direction, None, None))
        for dim, (_, values) in DIMENSIONS.items():
            for value in values:
                out.append((f"{direction}/{dim}={value}", direction, dim, value))
    return out


def in_slice(row: dict, direction: str, dim: Optional[str], value: Optional[str]) -> bool:
    if direction != "both" and row["direction"] != direction:
        return False
    if dim is None:
        return True
    fn, _ = DIMENSIONS[dim]
    return fn(row) == value


def score_all_slices(rows: Sequence[dict]) -> List[dict]:
    from metrics import mae, rmse
    from run_artifacts import metric_row

    out = []
    for name, direction, dim, value in slice_definitions():
        sub = [r for r in rows if in_slice(r, direction, dim, value)]
        if not sub:
            continue
        actual = [r["y_30"] for r in sub]
        for cand in CANDIDATES:
            pred = [r[cand] for r in sub]
            out.append(metric_row(cand, name, len(sub), mae(actual, pred), rmse(actual, pred), direction=direction))
    return out


# --- significance -----------------------------------------------------------------


def _compare(rows: Sequence[dict], challenger: str, reference: str, label: str, alpha: float, n_bootstrap: int):
    from significance import compare_absolute_errors

    return compare_absolute_errors(
        [r["y_30"] for r in rows],
        [r[challenger] for r in rows],
        [r[reference] for r in rows],
        label_challenger=f"{challenger} ({label})",
        label_reference=reference,
        timestamps=[r["bin_ts"] for r in rows],
        groups=[r["direction"] for r in rows],
        horizon_steps=HORIZON_STEPS,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        day_utc_offset_hours=8.0,
    )


def significance_families(rows: Sequence[dict], *, alpha: float = 0.05, n_bootstrap: int = 4999) -> Dict[str, list]:
    """Headline and slice families, each Holm-adjusted on its own."""
    from significance import apply_holm

    headline = []
    for direction in ("both",) + DIRECTIONS:
        sub = [r for r in rows if in_slice(r, direction, None, None)]
        if len(sub) < 3:
            continue
        for challenger in MODELS + (ENSEMBLE,):
            headline.append(_compare(sub, challenger, PERSISTENCE, f"{direction}/all", alpha, n_bootstrap))
    both = list(rows)
    if len(both) >= 3:
        for single in MODELS:
            headline.append(_compare(both, ENSEMBLE, single, "both/all", alpha, n_bootstrap))

    slices = []
    for name, direction, dim, value in slice_definitions():
        if dim is None:
            continue
        sub = [r for r in rows if in_slice(r, direction, dim, value)]
        if len(sub) < 3 or len({r["bin_ts"].astimezone(SGT).date() for r in sub}) < 2:
            continue
        for challenger in MODELS + (ENSEMBLE,):
            slices.append(_compare(sub, challenger, PERSISTENCE, name, alpha, n_bootstrap))
    return {"headline": apply_holm(headline), "slices": apply_holm(slices)}


# --- model provenance -------------------------------------------------------------


class LeakageError(RuntimeError):
    pass


def check_models_out_of_sample(meta: Iterable[dict], window_start: datetime) -> None:
    for m in meta:
        latest = max([m["created"], *m.get("training_run_starts", [])])
        if latest >= window_start:
            raise LeakageError(
                f"{m['model']} was trained at {latest.isoformat()}, not before the window start "
                f"{window_start.isoformat()}; choose a later --window-start"
            )


# --- run ----------------------------------------------------------------------------


def run_harness(
    source: Source,
    *,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 4999,
) -> Dict[str, Any]:
    start = parse_sgt(window_start or DEFAULT_WINDOW_START_SGT)
    end = parse_sgt(window_end) if window_end else source.latest_labelled_bin()
    if end <= start:
        raise ValueError(f"window end {end} is not after start {start}")
    meta = [source.model_metadata(m) for m in MODELS]
    check_models_out_of_sample(meta, start)
    base = source.fetch_rows(start, end)
    preds = {m: source.fetch_predictions(m, start, end) for m in MODELS}
    holdout = prepare_rows(base, preds, start, end)
    if not holdout.rows:
        raise RuntimeError("no scorable rows in the window")
    return {
        "holdout": holdout,
        "model_metadata": meta,
        "metrics": score_all_slices(holdout.rows),
        "families": significance_families(holdout.rows, alpha=alpha, n_bootstrap=n_bootstrap),
        "base_rows": len(base),
    }


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def component_from_result(result: Dict[str, Any], project: str, artifacts: List[str]) -> Dict[str, Any]:
    holdout: HoldoutRows = result["holdout"]
    rows = holdout.rows
    per_dir = {d: sum(1 for r in rows if r["direction"] == d) for d in DIRECTIONS}
    days = sorted({r["bin_ts"].astimezone(SGT).date().isoformat() for r in rows})
    sig = []
    for family, comps in result["families"].items():
        for c in comps:
            d = c.to_dict()
            d["family"] = family
            sig.append(d)
    return {
        "dataset": {
            "project": project,
            "view": f"{DATASET}.{VIEW}",
            "label": "y_30 (Maps duration_in_traffic, minutes, 10-min bin 3 rows ahead)",
            "rows_in_window": result["base_rows"],
            "rows_scored": len(rows),
            "rows_scored_per_direction": per_dir,
            "excluded": holdout.excluded,
            "days_sgt": days,
        },
        "window": {
            "timezone": "UTC (bin_ts); SGT = UTC+8",
            "basis": "forecast origin bin_ts; fixed window after model training",
            "start": _iso(holdout.window_start),
            "end": _iso(holdout.window_end),
            "start_sgt": holdout.window_start.astimezone(SGT).isoformat(),
            "end_sgt": holdout.window_end.astimezone(SGT).isoformat(),
        },
        "model_metadata": [
            {
                **{k: v for k, v in m.items() if k not in ("created", "modified", "training_run_starts")},
                "created": _iso(m["created"]),
                "modified": _iso(m["modified"]),
                "training_run_starts": [_iso(t) for t in m.get("training_run_starts", [])],
            }
            for m in result["model_metadata"]
        ],
        "models": ["Persistence (y_persistence)", "lin_h30", "xgb_h30", "ensemble_mean = (lin_h30 + xgb_h30) / 2"],
        "horizon_minutes": HORIZON_MINUTES,
        "metrics": result["metrics"],
        "significance": sig,
        "artifacts": artifacts,
    }


def _plot(run_dir: Path, result: Dict[str, Any]) -> List[Path]:
    from plots import plot_bqml_mae_comparison, plot_mae_diff_forest
    from run_artifacts import subdir

    out = subdir(run_dir, "bqml")
    slices_by_candidate: Dict[str, List[dict]] = {c: [] for c in CANDIDATES}
    for m in result["metrics"]:
        if m["slice"].endswith("/all"):
            slices_by_candidate[m["candidate"]].append(
                {"direction": m["direction"], "time_of_day": "all", "mae": m["mae_min"]}
            )
    start = result["holdout"].window_start.astimezone(SGT).strftime("%d %b")
    end = result["holdout"].window_end.astimezone(SGT).strftime("%d %b")
    return [
        *plot_bqml_mae_comparison(
            slices_by_candidate, out / "mae-by-direction.png", title=f"BQML 30 min ({start} - {end} SGT): MAE by direction"
        ),
        *plot_mae_diff_forest(
            result["families"]["headline"],
            out / "mae-diff-ci.png",
            title="BQML 30 min: paired MAE difference (day-block CI; Holm headline family)",
        ),
    ]


def bqml_component(
    run_dir: Path,
    *,
    project: str = DEFAULT_PROJECT,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    alpha: float = 0.05,
    source: Optional[Source] = None,
    n_bootstrap: int = 4999,
    keep: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """``keep``: optional dict that receives the raw harness result (``keep["result"]``) for reuse."""
    from run_artifacts import artifact_relpath

    result = run_harness(
        source or BigQuerySource(project),
        window_start=window_start,
        window_end=window_end,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )
    if keep is not None:
        keep["result"] = result
    print_report(result)
    paths = _plot(run_dir, result)
    return paths, component_from_result(result, project, [artifact_relpath(run_dir, p) for p in paths])


def print_report(result: Dict[str, Any]) -> None:
    from significance import format_comparison_table

    h: HoldoutRows = result["holdout"]
    print(
        f"Layer B 30 min, window {h.window_start.astimezone(SGT):%Y-%m-%d %H:%M} .. "
        f"{h.window_end.astimezone(SGT):%Y-%m-%d %H:%M} SGT; {len(h.rows)} rows scored; excluded {h.excluded}"
    )
    print("\n| Candidate | Slice | n | MAE (min) | RMSE (min) |\n| --- | --- | --- | --- | --- |")
    for m in result["metrics"]:
        if m["slice"].endswith("/all"):
            print(f"| {m['candidate']} | {m['slice']} | {m['n']} | {m['mae_min']:.3f} | {m['rmse_min']:.3f} |")
    for family, comps in result["families"].items():
        print(f"\n## Significance: {family} family (Holm within family, n={len(comps)})")
        print(format_comparison_table(comps))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--window-start", default=None, help=f"SGT (default {DEFAULT_WINDOW_START_SGT})")
    parser.add_argument("--window-end", default=None, help="SGT (default: latest labelled bin, pinned in run.json)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=4999)
    parser.add_argument("--significance", action="store_true", help="(kept for compatibility; always printed)")
    parser.add_argument("--plots", action="store_true", help="Write a run folder with figures and run.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--holdout-days", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.holdout_days is not None:
        print("--holdout-days was removed; use --window-start / --window-end (SGT).", file=sys.stderr)
        return 2

    if not args.plots:
        result = run_harness(
            BigQuerySource(args.project),
            window_start=args.window_start,
            window_end=args.window_end,
            alpha=args.alpha,
            n_bootstrap=args.n_bootstrap,
        )
        print_report(result)
        return 0

    from run_artifacts import create_run_dir, new_manifest, update_latest_pointer, write_manifest, write_run_readme

    run_dir = create_run_dir(components=["bqml"], run_id=args.run_id, runs_root=args.runs_root)
    _, meta = bqml_component(
        run_dir,
        project=args.project,
        window_start=args.window_start,
        window_end=args.window_end,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
    )
    manifest = new_manifest(["bqml"])
    manifest["bqml"] = meta
    write_manifest(run_dir, manifest)
    write_run_readme(run_dir, manifest)
    update_latest_pointer(run_dir, manifest, runs_root=args.runs_root)
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
