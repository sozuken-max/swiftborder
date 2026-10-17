"""Shared warehouse error summaries and paired comparisons via ``harness.significance``."""

from __future__ import annotations

import datetime as dt
import math
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from harness.significance import ComparisonResult, compare_absolute_errors

WITHIN_MINUTES = 15.0
MIN_VALIDATION_DAYS = 3


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default), q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def error_metrics(errors: Sequence[float]) -> Dict[str, Any]:
    abs_err = [abs(e) for e in errors]
    n = len(abs_err)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "mae": sum(abs_err) / n,
        "median_ae": percentile(abs_err, 50),
        "p90_ae": percentile(abs_err, 90),
        "rmse": math.sqrt(sum(e * e for e in errors) / n),
        "bias": sum(errors) / n,
        f"within_{int(WITHIN_MINUTES)}min": sum(1 for e in abs_err if e <= WITHIN_MINUTES) / n,
    }


def paired_day_bootstrap(
    pairs: Sequence[Tuple[dt.date, float]],
    reps: int,
    seed: int,
    level: float = 0.95,
) -> Dict[str, Any]:
    """Legacy day-sum bootstrap (tests and backward-compatible fields). Prefer ``compare`` for scoring."""
    by_day: Dict[dt.date, List[float]] = {}
    for day, diff in pairs:
        by_day.setdefault(day, []).append(diff)
    days = sorted(by_day)
    if not days:
        return {"n": 0, "days": 0, "mean": None, "ci_low": None, "ci_high": None}
    sums = [sum(by_day[d]) for d in days]
    counts = [len(by_day[d]) for d in days]
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        idx = [rng.randrange(len(days)) for _ in days]
        means.append(sum(sums[i] for i in idx) / sum(counts[i] for i in idx))
    tail = (1.0 - level) / 2.0 * 100.0
    return {
        "n": sum(counts),
        "days": len(days),
        "mean": sum(sums) / sum(counts),
        "ci_low": percentile(means, tail),
        "ci_high": percentile(means, 100.0 - tail),
    }


def comparison_to_legacy_dict(comp: ComparisonResult) -> Dict[str, Any]:
    if comp.pooled and comp.joint_ci_low_min is not None and comp.joint_ci_high_min is not None:
        ci_low, ci_high = comp.joint_ci_low_min, comp.joint_ci_high_min
    else:
        ci_low, ci_high = comp.bootstrap_ci_low_min, comp.bootstrap_ci_high_min
    days = comp.calendar_days
    if days is None and comp.blocks_per_resample is not None:
        days = comp.blocks_per_resample
    return {
        "n": comp.n,
        "days": days,
        "mean": comp.mean_ae_diff_min,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "decision": comp.decision,
        "significance": comp.to_dict(),
    }


def _row_pairs(
    rows: Sequence[Mapping[str, Any]],
    challenger: str,
    reference: str,
    sgt_date: Callable[[dt.datetime], dt.date],
) -> List[Tuple[dt.date, float]]:
    pairs: List[Tuple[dt.date, float]] = []
    for row in rows:
        if row.get("actual") is None or row.get(challenger) is None or row.get(reference) is None:
            continue
        origin = row["origin_ts"]
        if not isinstance(origin, dt.datetime):
            origin = dt.datetime.fromisoformat(str(origin).replace("Z", "+00:00"))
        pairs.append(
            (
                sgt_date(origin),
                abs(float(row[challenger]) - float(row["actual"]))
                - abs(float(row[reference]) - float(row["actual"])),
            )
        )
    return pairs


def compare(
    rows: Sequence[Mapping[str, Any]],
    challenger: str,
    reference: str,
    cfg: Any,
    *,
    sgt_date: Optional[Callable[[dt.datetime], dt.date]] = None,
) -> Dict[str, Any]:
    """Paired absolute-error comparison using ``harness.significance`` (DM, day-block bootstrap, Holm-ready)."""
    sgt_date = sgt_date or (lambda ts: ts.astimezone(dt.timezone(dt.timedelta(hours=8))).date())
    bundled: List[Tuple[Any, Any, float, float, float]] = []
    for row in rows:
        if row.get("actual") is None or row.get(challenger) is None or row.get(reference) is None:
            continue
        bundled.append(
            (
                row.get("direction", "all"),
                row["origin_ts"],
                float(row["actual"]),
                float(row[challenger]),
                float(row[reference]),
            )
        )
    if not bundled:
        return {"n": 0, "days": 0, "mean": None, "ci_low": None, "ci_high": None, "decision": "insufficient data"}
    bundled.sort(key=lambda t: (t[0], t[1]))
    groups = [t[0] for t in bundled]
    timestamps = [t[1] for t in bundled]
    actual = [t[2] for t in bundled]
    pred_c = [t[3] for t in bundled]
    pred_r = [t[4] for t in bundled]
    groups_arg: Optional[List[Any]] = groups if len(set(groups)) > 1 else None
    comp = compare_absolute_errors(
        actual,
        pred_c,
        pred_r,
        label_challenger=challenger,
        label_reference=reference,
        block="day",
        timestamps=timestamps,
        groups=groups_arg,
        n_bootstrap=int(cfg.bootstrap_reps),
        random_state=int(cfg.seed),
        day_utc_offset_hours=8.0,
    )
    out = comparison_to_legacy_dict(comp)
    legacy = paired_day_bootstrap(_row_pairs(rows, challenger, reference, sgt_date), int(cfg.bootstrap_reps), int(cfg.seed))
    out["legacy_bootstrap"] = legacy
    if comp.decision == "insufficient data" and legacy.get("days"):
        out["days"] = legacy["days"]
        out["mean"] = legacy["mean"]
        out["ci_low"] = legacy["ci_low"]
        out["ci_high"] = legacy["ci_high"]
    return out


def better(result: Mapping[str, Any], min_days: int = MIN_VALIDATION_DAYS) -> bool:
    """Challenger better than reference (harness rule when enough days; legacy CI gate otherwise)."""
    days = int(result.get("days") or 0)
    if days < min_days:
        return False
    decision = result.get("decision")
    if decision == "challenger":
        return True
    if decision in ("reference", "not significant"):
        return False
    return bool(result.get("ci_high") is not None and result["ci_high"] < 0)
