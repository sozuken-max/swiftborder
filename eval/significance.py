"""Paired significance tests for forecast error (MAE-oriented).

Hold-out rows are paired by time (and direction). For each row the loss differential is
``d = |y - y_hat_challenger| - |y - y_hat_reference|`` in minutes (negative => challenger better).

Methods (see docs/evaluation.md, "Significance"):

- **Diebold-Mariano** (Diebold & Mariano 1995) on ``d`` with a Newey-West (Bartlett) HAC variance
  (Newey & West 1987) and the Harvey-Leybourne-Newbold (1997) small-sample correction, compared to
  Student-t(n-1). The HAC lag is the largest of ``h - 1`` (overlapping h-step errors), the
  Newey-West (1994) rule ``floor(4 (n/100)^(2/9))``, and the bootstrap block length (one day of
  samples by default), capped at n/3: traffic loss differentials stay correlated for about a day.
  Autocovariances are computed within groups (directions) only.
- **Moving-block bootstrap** (Kunsch 1989): overlapping blocks drawn within each group, never across
  a group boundary. Default block length is one day of samples, which covers the daily traffic cycle.
- **Holm** (1979) step-down correction of the **two-sided** DM p-values across a family
  (``apply_holm``), which bounds the family-wise error of any directional claim at alpha.
- Paired t-test: supplementary only (assumes independent differences).

Decision rule: **challenger** when the (Holm-adjusted) two-sided DM p-value is below alpha, the
mean difference is negative, **and** the two-sided (1 - alpha) bootstrap CI lies entirely below 0;
**reference** by the mirror rule; **insufficient data** when a resample has fewer than
``MIN_BLOCKS`` blocks (the CI is too coarse to use); otherwise **not significant**.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from metrics import peak_time_of_day

SECONDS_PER_DAY = 86400.0
MIN_BLOCKS = 10
"""Fewer bootstrap blocks than this per resample gives a coarse CI; the decision is 'insufficient data'."""
Block = Union[int, str]


@dataclass(frozen=True)
class ComparisonResult:
    """Challenger vs reference on the same actuals."""

    label_challenger: str
    label_reference: str
    n: int
    mean_ae_diff_min: float
    """Mean of (|y - y_hat_c| - |y - y_hat_r|) in minutes; negative => challenger better."""
    paired_t_pvalue: float
    bootstrap_ci_low_min: float
    bootstrap_ci_high_min: float
    block_size: int
    alpha: float
    dm_stat: float = float("nan")
    dm_pvalue: float = float("nan")
    """One-sided p for H1: challenger has lower expected absolute error."""
    dm_pvalue_reference: float = float("nan")
    """One-sided p for H1: reference has lower expected absolute error."""
    hac_lag: int = 0
    horizon_steps: int = 1
    n_groups: int = 1
    block: str = "fixed"
    blocks_per_resample: Optional[int] = None
    """Blocks drawn per bootstrap resample. Below MIN_BLOCKS the decision is 'insufficient'."""
    holm_adjusted_p: Optional[float] = None
    """Holm-adjusted **two-sided** DM p-value over the comparison family."""
    holm_adjusted_p_reference: Optional[float] = None
    """Same value as holm_adjusted_p (kept for manifest compatibility)."""
    family_size: int = 1

    @property
    def dm_pvalue_two_sided(self) -> float:
        if math.isnan(self.dm_pvalue) or math.isnan(self.dm_pvalue_reference):
            return float("nan")
        return min(1.0, 2.0 * min(self.dm_pvalue, self.dm_pvalue_reference))

    def _p(self) -> float:
        return self.dm_pvalue_two_sided if self.holm_adjusted_p is None else self.holm_adjusted_p

    @property
    def enough_blocks(self) -> bool:
        return self.blocks_per_resample is None or self.blocks_per_resample >= MIN_BLOCKS

    @property
    def challenger_better_at_alpha(self) -> bool:
        return bool(self.enough_blocks and self._p() < self.alpha and self.mean_ae_diff_min < 0 and self.bootstrap_ci_high_min < 0)

    @property
    def reference_better_at_alpha(self) -> bool:
        return bool(self.enough_blocks and self._p() < self.alpha and self.mean_ae_diff_min > 0 and self.bootstrap_ci_low_min > 0)

    @property
    def decision(self) -> str:
        if not self.enough_blocks:
            return "insufficient data"
        if self.challenger_better_at_alpha:
            return "challenger"
        if self.reference_better_at_alpha:
            return "reference"
        return "not significant"

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["dm_pvalue_two_sided"] = self.dm_pvalue_two_sided
        out["challenger_better_at_alpha"] = self.challenger_better_at_alpha
        out["reference_better_at_alpha"] = self.reference_better_at_alpha
        out["decision"] = self.decision
        return {k: _jsonable(v) for k, v in out.items()}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def paired_absolute_error_diff_minutes(
    actual: Sequence[float],
    predicted_challenger: Sequence[float],
    predicted_reference: Sequence[float],
) -> List[float]:
    if not (len(actual) == len(predicted_challenger) == len(predicted_reference)):
        raise ValueError("actual, challenger, and reference must have the same length")
    a = np.asarray(actual, dtype=float)
    return list(np.abs(a - np.asarray(predicted_challenger, dtype=float)) - np.abs(a - np.asarray(predicted_reference, dtype=float)))


def paired_t_test_two_sided(differences: Sequence[float]) -> Tuple[float, float]:
    """Return (t_statistic, two-sided p-value). Supplementary: assumes independent differences."""
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    var = float(np.var(d, ddof=1))
    if var <= 0:
        return 0.0, 1.0
    t_stat = float(d.mean() / math.sqrt(var / n))
    return t_stat, float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))


# --- groups -------------------------------------------------------------------


def _group_slices(groups: Optional[Sequence[Any]], n: int) -> List[slice]:
    """Contiguous runs of equal group labels. Input must already be ordered by (group, time)."""
    if groups is None:
        return [slice(0, n)]
    if len(groups) != n:
        raise ValueError("groups must match the number of differences")
    slices: List[slice] = []
    start = 0
    seen = set()
    for i in range(1, n + 1):
        if i == n or groups[i] != groups[start]:
            label = groups[start]
            if label in seen:
                raise ValueError("rows must be sorted so each group is contiguous")
            seen.add(label)
            slices.append(slice(start, i))
            start = i
    return slices


def newey_west_auto_lag(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (n / 100)^(2/9))."""
    if n <= 0:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def default_hac_lag(n: int, horizon_steps: int) -> int:
    return max(horizon_steps - 1, newey_west_auto_lag(n), 0)


def hac_variance_of_mean(
    d: np.ndarray,
    lag: int,
    slices: Optional[List[slice]] = None,
) -> float:
    """Newey-West (Bartlett) long-run variance of d, divided by n (variance of the mean).

    Autocovariances pair observations only inside the same group slice.
    """
    n = len(d)
    slices = slices or [slice(0, n)]
    centred = d - d.mean()
    gamma0 = float(np.dot(centred, centred)) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        acc = 0.0
        for s in slices:
            seg = centred[s]
            if len(seg) > k:
                acc += float(np.dot(seg[k:], seg[:-k]))
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * acc / n
    return max(lrv, 0.0) / n


def diebold_mariano(
    differences: Sequence[float],
    *,
    horizon_steps: int = 1,
    hac_lag: Optional[int] = None,
    groups: Optional[Sequence[Any]] = None,
) -> Tuple[float, float, float, int]:
    """HLN-corrected Diebold-Mariano test on a loss differential.

    Returns (statistic, p_challenger_better, p_reference_better, lag). Negative statistic means the
    challenger's loss is lower. p-values are one-sided from Student-t(n - 1).
    """
    d = np.asarray(differences, dtype=float)
    n = len(d)
    h = max(int(horizon_steps), 1)
    lag = default_hac_lag(n, h) if hac_lag is None else int(hac_lag)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), lag
    var_mean = hac_variance_of_mean(d, lag, _group_slices(groups, n))
    if var_mean <= 0:
        return 0.0, 0.5, 0.5, lag
    dm = float(d.mean() / math.sqrt(var_mean))
    hln_arg = (n + 1 - 2 * h + h * (h - 1) / n) / n
    hln = math.sqrt(hln_arg) if hln_arg > 0 else 1.0
    stat = dm * hln
    return stat, float(stats.t.cdf(stat, df=n - 1)), float(stats.t.sf(stat, df=n - 1)), lag


# --- bootstrap ------------------------------------------------------------------


def _to_seconds(ts: Any) -> float:
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return float(ts)
    if isinstance(ts, np.datetime64):
        return float(ts.astype("datetime64[ns]").astype(np.int64)) / 1e9
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)  # naive = wall clock; never the machine's local zone
    if hasattr(ts, "timestamp"):
        return float(ts.timestamp())
    raise TypeError(f"unsupported timestamp type: {type(ts)!r}")


def day_block_length(
    timestamps: Sequence[Any],
    slices: Optional[List[slice]] = None,
    *,
    utc_offset_hours: float = 0.0,
) -> int:
    """Median number of samples per calendar day within groups (one block = one day of this slice).

    Counting per day (rather than dividing 24 h by the sampling step) keeps a day block meaningful for
    sub-daily slices such as "morning peak". ``utc_offset_hours`` sets the day boundary (8 for SGT
    when timestamps are UTC-aware); naive timestamps are treated as wall clock with offset 0.
    """
    secs = np.array([_to_seconds(t) for t in timestamps], dtype=float)
    days = np.floor((secs + utc_offset_hours * 3600.0) / SECONDS_PER_DAY).astype(np.int64)
    counts: List[int] = []
    for s in slices or [slice(0, len(secs))]:
        seg = days[s]
        if len(seg):
            _, c = np.unique(seg, return_counts=True)
            counts.extend(c.tolist())
    if not counts:
        raise ValueError("need timestamps for day-length blocks")
    return max(1, int(round(float(np.median(counts)))))


def block_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    block_size: int = 1,
    n_resamples: int = 4999,
    confidence: float = 0.95,
    random_state: Optional[int] = 0,
    groups: Optional[Sequence[Any]] = None,
) -> Tuple[float, float]:
    """Percentile CI for the mean using an overlapping moving-block bootstrap within groups.

    Each group keeps its size: ``ceil(m / b)`` blocks of length ``b`` (the last truncated) are drawn
    from the ``m - b + 1`` overlapping start positions of that group. A block never crosses a group.
    """
    d = np.asarray(values, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n == 1:
        return float(d[0]), float(d[0])
    rng = np.random.default_rng(random_state)
    totals = np.zeros(n_resamples)
    for s in _group_slices(groups, n):
        seg = d[s]
        m = len(seg)
        b = min(block_size, m)
        k = math.ceil(m / b)
        last = m - (k - 1) * b
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        starts = rng.integers(0, m - b + 1, size=(n_resamples, k))
        full = cs[starts[:, : k - 1] + b] - cs[starts[:, : k - 1]] if k > 1 else np.zeros((n_resamples, 0))
        tail = cs[starts[:, -1] + last] - cs[starts[:, -1]]
        totals += full.sum(axis=1) + tail
    means = np.sort(totals / n)
    tail_p = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail_p)), float(np.quantile(means, 1.0 - tail_p))


# --- comparisons ----------------------------------------------------------------


def compare_absolute_errors(
    actual: Sequence[float],
    predicted_challenger: Sequence[float],
    predicted_reference: Sequence[float],
    *,
    label_challenger: str = "challenger",
    label_reference: str = "reference",
    block: Optional[Block] = None,
    block_size: Optional[int] = None,
    timestamps: Optional[Sequence[Any]] = None,
    groups: Optional[Sequence[Any]] = None,
    horizon_steps: int = 1,
    hac_lag: Optional[int] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 4999,
    random_state: Optional[int] = 0,
    day_utc_offset_hours: float = 0.0,
) -> ComparisonResult:
    """Paired comparison. Rows must be ordered by (group, time).

    ``block``: an int block length, or ``"day"`` (needs ``timestamps``). Default is ``"day"`` when
    timestamps are given, else ``block_size`` (legacy argument), else 1.
    """
    diffs = np.asarray(paired_absolute_error_diff_minutes(actual, predicted_challenger, predicted_reference))
    n = len(diffs)
    slices = _group_slices(groups, n)
    if block is None:
        block = block_size if block_size is not None else ("day" if timestamps is not None else 1)
    if block == "day":
        if timestamps is None:
            raise ValueError('block="day" needs timestamps')
        b = day_block_length(timestamps, slices, utc_offset_hours=day_utc_offset_hours)
        block_label = "day"
    else:
        b = int(block)
        block_label = "fixed"
    mean_diff = float(diffs.mean()) if n else float("nan")
    _, t_p = paired_t_test_two_sided(diffs)
    if hac_lag is None:
        # Loss differentials on traffic series stay correlated for about a day, so the HAC
        # bandwidth is at least one bootstrap block (capped at a third of the sample).
        hac_lag = min(max(default_hac_lag(n, horizon_steps), b), max(1, n // 3))
    dm_stat, dm_p, dm_p_ref, lag = diebold_mariano(diffs, horizon_steps=horizon_steps, hac_lag=hac_lag, groups=groups)
    ci_low, ci_high = block_bootstrap_mean_ci(
        diffs,
        block_size=b,
        n_resamples=n_bootstrap,
        confidence=1.0 - alpha,
        random_state=random_state,
        groups=groups,
    )
    return ComparisonResult(
        label_challenger=label_challenger,
        label_reference=label_reference,
        n=n,
        mean_ae_diff_min=mean_diff,
        paired_t_pvalue=t_p,
        bootstrap_ci_low_min=ci_low,
        bootstrap_ci_high_min=ci_high,
        block_size=b,
        alpha=alpha,
        dm_stat=dm_stat,
        dm_pvalue=dm_p,
        dm_pvalue_reference=dm_p_ref,
        hac_lag=lag,
        horizon_steps=int(horizon_steps),
        n_groups=len(slices),
        block=block_label,
        blocks_per_resample=sum(math.ceil((s.stop - s.start) / min(b, s.stop - s.start)) for s in slices if s.stop > s.start),
    )


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm (1979) step-down adjusted p-values (monotone, capped at 1). NaN stays NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


def apply_holm(results: Sequence[ComparisonResult]) -> List[ComparisonResult]:
    """Holm-adjust the two-sided DM p-values over one comparison family.

    A single family of two-sided tests keeps the family-wise error of any directional claim at
    alpha (adjusting each one-sided side separately would allow 2 * alpha).
    """
    adj = holm_adjust([r.dm_pvalue_two_sided for r in results])
    return [replace(r, holm_adjusted_p=a, holm_adjusted_p_reference=a, family_size=len(results)) for r, a in zip(results, adj)]


def _row_matches_slice(row: dict, direction: str, time_of_day: str) -> bool:
    if direction != "both" and str(row["direction"]) != direction:
        return False
    tod = peak_time_of_day(row.get("is_morning_peak"), row.get("is_evening_peak"))
    if time_of_day != "all" and tod != time_of_day:
        return False
    return True


def filter_scored_rows(
    rows: Iterable[dict],
    *,
    direction: str = "both",
    time_of_day: str = "all",
    predicate=None,
) -> List[dict]:
    out = [r for r in rows if _row_matches_slice(r, direction, time_of_day)]
    if predicate is not None:
        out = [r for r in out if predicate(r)]
    return out


def align_scored_rows(
    reference_rows: List[dict],
    challenger_rows: List[dict],
    key_fields: Tuple[str, ...] = ("direction", "bin_ts"),
) -> Tuple[List[dict], List[dict]]:
    """Pair rows on key_fields (sorted by direction then time); raises on duplicate or unmatched keys."""
    ref_index = {tuple(r[k] for k in key_fields): r for r in reference_rows}
    ch_index = {tuple(r[k] for k in key_fields): r for r in challenger_rows}
    if len(ref_index) != len(reference_rows):
        raise ValueError("duplicate keys in reference rows")
    if len(ch_index) != len(challenger_rows):
        raise ValueError("duplicate keys in challenger rows")
    common = sorted(set(ref_index) & set(ch_index), key=lambda k: tuple(_sort_key(v) for v in k))
    if len(common) != len(ref_index) or len(common) != len(ch_index):
        raise ValueError("reference and challenger rows do not share the same keys")
    return [ref_index[k] for k in common], [ch_index[k] for k in common]


def _sort_key(v: Any) -> Any:
    if isinstance(v, str):
        return (0, v)
    try:
        return (1, _to_seconds(v))
    except TypeError:
        return (2, str(v))


def compare_scored_rows(
    reference_rows: List[dict],
    challenger_rows: List[dict],
    *,
    actual_key: str = "y_30",
    predicted_key: str = "predicted",
    direction: str = "both",
    time_of_day: str = "all",
    predicate=None,
    label_challenger: str,
    label_reference: str,
    horizon_steps: int = 3,
    block: Optional[Block] = "day",
    block_size: Optional[int] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 4999,
    random_state: Optional[int] = 0,
    day_utc_offset_hours: float = 0.0,
) -> ComparisonResult:
    """Filter a slice, pair on (direction, bin_ts), and compare. Directions are separate groups."""
    ref_f = filter_scored_rows(reference_rows, direction=direction, time_of_day=time_of_day, predicate=predicate)
    ch_f = filter_scored_rows(challenger_rows, direction=direction, time_of_day=time_of_day, predicate=predicate)
    ref_aligned, ch_aligned = align_scored_rows(ref_f, ch_f)
    if block_size is not None:
        block = block_size
    return compare_absolute_errors(
        [float(r[actual_key]) for r in ref_aligned],
        [float(r[predicted_key]) for r in ch_aligned],
        [float(r[predicted_key]) for r in ref_aligned],
        label_challenger=label_challenger,
        label_reference=label_reference,
        block=block,
        timestamps=[r["bin_ts"] for r in ref_aligned],
        groups=[str(r["direction"]) for r in ref_aligned],
        horizon_steps=horizon_steps,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        day_utc_offset_hours=day_utc_offset_hours,
    )


def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    return f"{p:.2g}" if p < 1e-3 else f"{p:.4f}"


def format_comparison_table(results: Sequence[ComparisonResult]) -> str:
    lines = [
        "| Challenger | Reference | n | Mean AE diff (min) | Block bootstrap CI | Block | DM p (two-sided) | Holm p | paired t p | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        ci = f"[{r.bootstrap_ci_low_min:.3f}, {r.bootstrap_ci_high_min:.3f}]"
        block = f"{r.block} ({r.block_size} x {r.blocks_per_resample})"
        lines.append(
            f"| {r.label_challenger} | {r.label_reference} | {r.n} | {r.mean_ae_diff_min:.3f} | {ci} | {block} | "
            f"{_fmt_p(r.dm_pvalue_two_sided)} | {_fmt_p(r.holm_adjusted_p)} | {_fmt_p(r.paired_t_pvalue)} | {r.decision} |"
        )
    return "\n".join(lines)
