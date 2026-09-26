"""Paired significance tests for forecast error (MAE-oriented).

Hold-out rows are paired by time (and direction). We compare per-observation
absolute errors against a reference forecast (usually persistence). Block
bootstrap respects serial correlation in the error-difference series; the
paired t-test is a supplementary check when differences are roughly independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from metrics import peak_time_of_day


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

    @property
    def challenger_better_at_alpha(self) -> bool:
        """One-sided: challenger beats reference if upper CI bound on mean diff is < 0."""
        return self.mean_ae_diff_min < 0 and self.bootstrap_ci_high_min < 0

    @property
    def reference_better_at_alpha(self) -> bool:
        return self.mean_ae_diff_min > 0 and self.bootstrap_ci_low_min > 0


def paired_absolute_error_diff_minutes(
    actual: Sequence[float],
    predicted_challenger: Sequence[float],
    predicted_reference: Sequence[float],
) -> List[float]:
    if not (len(actual) == len(predicted_challenger) == len(predicted_reference)):
        raise ValueError("actual, challenger, and reference must have the same length")
    out: List[float] = []
    for a, c, r in zip(actual, predicted_challenger, predicted_reference):
        out.append(abs(a - c) - abs(a - r))
    return out


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_t_test_two_sided(differences: Sequence[float]) -> Tuple[float, float]:
    """Return (t_statistic, two-sided p-value). Uses normal approximation if scipy is absent."""
    n = len(differences)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(differences) / n
    var = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if var <= 0:
        return 0.0, 1.0
    se = math.sqrt(var / n)
    t_stat = mean / se
    try:
        from scipy import stats  # type: ignore

        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    except ImportError:
        p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return t_stat, p_value


def block_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    block_size: int = 1,
    n_resamples: int = 4999,
    confidence: float = 0.95,
    random_state: Optional[int] = 0,
) -> Tuple[float, float]:
    """Percentile CI for the mean of `values` using moving-block bootstrap."""
    if not values:
        return float("nan"), float("nan")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    n = len(values)
    if n == 1:
        return values[0], values[0]

    import random

    rng = random.Random(random_state)
    blocks: List[List[float]] = []
    for start in range(0, n, block_size):
        blocks.append(list(values[start : start + block_size]))
    if not blocks:
        blocks = [list(values)]

    boot_means: List[float] = []
    for _ in range(n_resamples):
        sample: List[float] = []
        while len(sample) < n:
            block = blocks[rng.randrange(len(blocks))]
            sample.extend(block)
        sample = sample[:n]
        boot_means.append(sum(sample) / len(sample))

    boot_means.sort()
    tail = (1.0 - confidence) / 2.0
    lo_idx = int(tail * n_resamples)
    hi_idx = int((1.0 - tail) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, len(boot_means) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_means) - 1))
    return boot_means[lo_idx], boot_means[hi_idx]


def compare_absolute_errors(
    actual: Sequence[float],
    predicted_challenger: Sequence[float],
    predicted_reference: Sequence[float],
    *,
    label_challenger: str = "challenger",
    label_reference: str = "reference",
    block_size: int = 6,
    alpha: float = 0.05,
    n_bootstrap: int = 4999,
    random_state: Optional[int] = 0,
) -> ComparisonResult:
    diffs = paired_absolute_error_diff_minutes(actual, predicted_challenger, predicted_reference)
    mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
    _, p_value = paired_t_test_two_sided(diffs)
    ci_low, ci_high = block_bootstrap_mean_ci(
        diffs,
        block_size=block_size,
        n_resamples=n_bootstrap,
        confidence=1.0 - alpha,
        random_state=random_state,
    )
    return ComparisonResult(
        label_challenger=label_challenger,
        label_reference=label_reference,
        n=len(diffs),
        mean_ae_diff_min=mean_diff,
        paired_t_pvalue=p_value,
        bootstrap_ci_low_min=ci_low,
        bootstrap_ci_high_min=ci_high,
        block_size=block_size,
        alpha=alpha,
    )


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
) -> List[dict]:
    return [r for r in rows if _row_matches_slice(r, direction, time_of_day)]


def align_scored_rows(
    reference_rows: List[dict],
    challenger_rows: List[dict],
    key_fields: Tuple[str, ...] = ("direction", "bin_ts"),
) -> Tuple[List[dict], List[dict]]:
    """Pair rows on key_fields; raises if keys are duplicated or missing on either side."""
    ref_index = {tuple(r[k] for k in key_fields): r for r in reference_rows}
    ch_index = {tuple(r[k] for k in key_fields): r for r in challenger_rows}
    if len(ref_index) != len(reference_rows):
        raise ValueError("duplicate keys in reference rows")
    if len(ch_index) != len(challenger_rows):
        raise ValueError("duplicate keys in challenger rows")
    common = sorted(set(ref_index) & set(ch_index))
    if len(common) != len(ref_index) or len(common) != len(ch_index):
        raise ValueError("reference and challenger rows do not share the same keys")
    ref_aligned = [ref_index[k] for k in common]
    ch_aligned = [ch_index[k] for k in common]
    return ref_aligned, ch_aligned


def compare_scored_rows(
    reference_rows: List[dict],
    challenger_rows: List[dict],
    *,
    actual_key: str = "y_30",
    predicted_key: str = "predicted",
    direction: str = "both",
    time_of_day: str = "all",
    label_challenger: str,
    label_reference: str,
    block_size: int = 6,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
) -> ComparisonResult:
    ref_f = filter_scored_rows(reference_rows, direction=direction, time_of_day=time_of_day)
    ch_f = filter_scored_rows(challenger_rows, direction=direction, time_of_day=time_of_day)
    ref_aligned, ch_aligned = align_scored_rows(ref_f, ch_f)
    actual = [float(r[actual_key]) for r in ref_aligned]
    pred_ref = [float(r[predicted_key]) for r in ref_aligned]
    pred_ch = [float(r[predicted_key]) for r in ch_aligned]
    return compare_absolute_errors(
        actual,
        pred_ch,
        pred_ref,
        label_challenger=label_challenger,
        label_reference=label_reference,
        block_size=block_size,
        alpha=alpha,
        random_state=random_state,
    )


def format_comparison_table(results: Sequence[ComparisonResult]) -> str:
    lines = [
        "| Challenger | Reference | n | Mean AE diff (min) | Block bootstrap CI | paired t p | Better at alpha |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        ci = f"[{r.bootstrap_ci_low_min:.3f}, {r.bootstrap_ci_high_min:.3f}]"
        if r.challenger_better_at_alpha:
            verdict = "challenger"
        elif r.reference_better_at_alpha:
            verdict = "reference"
        else:
            verdict = "not significant"
        lines.append(
            f"| {r.label_challenger} | {r.label_reference} | {r.n} | {r.mean_ae_diff_min:.3f} | {ci} | {r.paired_t_pvalue:.4f} | {verdict} |"
        )
    return "\n".join(lines)
