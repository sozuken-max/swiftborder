import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from significance import (
    ComparisonResult,
    apply_holm,
    block_bootstrap_mean_ci,
    compare_absolute_errors,
    compare_scored_rows,
    day_block_length,
    default_hac_lag,
    diebold_mariano,
    format_comparison_table,
    hac_variance_of_mean,
    holm_adjust,
    newey_west_auto_lag,
    paired_absolute_error_diff_minutes,
    paired_t_test_two_sided,
)


def test_paired_ae_diff_negative_when_challenger_better():
    actual = [10.0, 20.0, 30.0]
    ref = [12.0, 22.0, 35.0]
    ch = [10.5, 19.0, 30.5]
    diffs = paired_absolute_error_diff_minutes(actual, ch, ref)
    assert all(d <= 0 for d in diffs)


def test_compare_absolute_errors_challenger_wins_clearly():
    rng = np.random.default_rng(3)
    n = 400
    actual = 50 + rng.normal(0, 1, n)
    ref = actual + rng.normal(0, 5, n)
    ch = actual + rng.normal(0, 1, n)
    result = compare_absolute_errors(actual, ch, ref, block=10, n_bootstrap=999, random_state=1)
    assert result.mean_ae_diff_min < 0
    assert result.challenger_better_at_alpha
    assert result.decision == "challenger"


def test_no_difference_is_not_significant():
    rng = np.random.default_rng(4)
    actual = rng.normal(0, 1, 500)
    a = actual + rng.normal(0, 1, 500)
    b = actual + rng.normal(0, 1, 500)
    result = compare_absolute_errors(actual, a, b, block=10, n_bootstrap=999)
    assert result.decision == "not significant"


def test_paired_t_test_zero_mean():
    t_stat, p_value = paired_t_test_two_sided([0.0, 0.0, 0.0, 0.0])
    assert t_stat == 0.0
    assert p_value == 1.0


def test_paired_t_test_matches_scipy():
    d = [0.3, -0.1, 0.5, 0.2, 0.9, -0.4, 0.1]
    t_stat, p = paired_t_test_two_sided(d)
    ref = stats.ttest_1samp(d, 0.0)
    assert t_stat == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue)


# --- Diebold-Mariano ------------------------------------------------------------------


def test_dm_with_h1_and_no_lag_equals_the_t_statistic():
    """Known answer: HLN correction with h=1 and lag 0 turns DM into the one-sample t-test."""
    d = [0.3, -0.1, 0.5, 0.2, 0.9, -0.4, 0.1, 0.6]
    stat, p_ch, p_ref, lag = diebold_mariano(d, horizon_steps=1, hac_lag=0)
    ref = stats.ttest_1samp(d, 0.0)
    assert lag == 0
    assert stat == pytest.approx(ref.statistic)
    assert p_ref == pytest.approx(ref.pvalue / 2)  # mean > 0 => reference better
    assert p_ch == pytest.approx(1 - ref.pvalue / 2)


def test_dm_hand_computed_with_one_lag():
    d = np.array([1.0, -1.0, 2.0, 0.0, 1.0, 3.0])
    n, h = len(d), 2
    c = d - d.mean()
    g0 = (c @ c) / n
    g1 = (c[1:] @ c[:-1]) / n
    lrv = g0 + 2 * 0.5 * g1  # Bartlett weight 1 - 1/2
    dm = d.mean() / math.sqrt(lrv / n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat, _, p_ref, lag = diebold_mariano(d, horizon_steps=h, hac_lag=1)
    assert lag == 1
    assert stat == pytest.approx(dm * hln)
    assert p_ref == pytest.approx(stats.t.sf(dm * hln, df=n - 1))


def test_hac_autocovariance_does_not_cross_groups():
    d = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    whole = hac_variance_of_mean(d, 1)
    grouped = hac_variance_of_mean(d, 1, [slice(0, 3), slice(3, 6)])
    c = d - d.mean()
    n = len(d)
    expected_lrv = (c @ c) / n + 2 * 0.5 * ((c[1:3] @ c[0:2]) + (c[4:6] @ c[3:5])) / n
    assert grouped == pytest.approx(expected_lrv / n)
    assert grouped != pytest.approx(whole)


def test_default_hac_lag_is_at_least_h_minus_1_and_the_nw_rule():
    assert newey_west_auto_lag(100) == 4
    assert default_hac_lag(100, 12) == 11
    assert default_hac_lag(3000, 3) == newey_west_auto_lag(3000)


def test_dm_is_more_conservative_than_iid_t_on_autocorrelated_losses():
    rng = np.random.default_rng(0)
    n, phi = 2000, 0.9
    e = rng.normal(0, 1, n)
    d = np.empty(n)
    d[0] = e[0]
    for i in range(1, n):
        d[i] = phi * d[i - 1] + e[i]
    d = d * 0.1 - 0.02
    _, p_t = paired_t_test_two_sided(d)
    _, p_dm, _, _ = diebold_mariano(d, horizon_steps=1, hac_lag=30)
    assert p_dm * 2 > p_t  # HAC widens the variance


# --- bootstrap -----------------------------------------------------------------------


def test_block_bootstrap_ci_contains_mean():
    values = [float(i) for i in range(20)]
    lo, hi = block_bootstrap_mean_ci(values, block_size=5, n_resamples=500, random_state=0)
    assert lo <= sum(values) / len(values) <= hi


def test_block_bootstrap_never_mixes_groups():
    # Group A is all -1, group B all +1. Each group keeps its size, so every resample mean is 0.
    values = [-1.0] * 30 + [1.0] * 30
    lo, hi = block_bootstrap_mean_ci(values, block_size=7, n_resamples=300, groups=["A"] * 30 + ["B"] * 30)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)
    # Without groups, blocks cross the boundary and the mean varies.
    lo2, hi2 = block_bootstrap_mean_ci(values, block_size=7, n_resamples=300)
    assert hi2 - lo2 > 0.1


def test_groups_must_be_contiguous():
    with pytest.raises(ValueError):
        block_bootstrap_mean_ci([1.0, 2.0, 3.0], block_size=1, groups=["A", "B", "A"])


def test_moving_block_coverage_on_ar1_is_near_nominal():
    rng = np.random.default_rng(7)
    reps, n, phi, covered = 150, 1000, 0.7, 0
    for r in range(reps):
        e = rng.normal(0, 1, n)
        x = np.empty(n)
        x[0] = e[0] / math.sqrt(1 - phi**2)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        lo, hi = block_bootstrap_mean_ci(x, block_size=40, n_resamples=999, random_state=r)
        covered += lo <= 0.0 <= hi
    assert 0.85 <= covered / reps <= 0.99


def test_moving_block_is_wider_than_iid_on_ar1():
    rng = np.random.default_rng(8)
    n, phi = 2000, 0.8
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    lo1, hi1 = block_bootstrap_mean_ci(x, block_size=1, n_resamples=999)
    lo50, hi50 = block_bootstrap_mean_ci(x, block_size=50, n_resamples=999)
    assert (hi50 - lo50) > 2 * (hi1 - lo1)


def test_day_block_length_from_timestamps():
    t0 = datetime(2026, 9, 13)
    ts = [t0 + timedelta(minutes=10 * i) for i in range(300)]
    assert day_block_length(ts) == 144
    assert day_block_length([t.isoformat() for t in ts]) == 144


def test_compare_defaults_to_day_blocks_with_timestamps():
    t0 = datetime(2026, 9, 13)
    n = 144 * 3
    ts = [t0 + timedelta(minutes=10 * i) for i in range(n)]
    actual = [20.0] * n
    result = compare_absolute_errors(actual, [21.0] * n, [22.0] * n, timestamps=ts, n_bootstrap=200)
    assert result.block == "day"
    assert result.block_size == 144
    assert result.blocks_per_resample == 3


# --- Holm ------------------------------------------------------------------------------


def test_holm_matches_hand_calculation():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> max-accumulate 0.06
    assert holm_adjust(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_nan_and_caps_at_one():
    out = holm_adjust([0.6, float("nan"), 0.7])
    assert math.isnan(out[1])
    assert out[0] == pytest.approx(1.0) and out[2] == pytest.approx(1.0)


def _result(p: float, ci_high: float = -0.1, ci_low: float = -0.3, blocks=None) -> ComparisonResult:
    """p is the one-sided 'challenger better' DM p-value; two-sided = 2p for p < 0.5."""
    return ComparisonResult(
        "c", "r", 100, -0.2, 0.01, ci_low, ci_high, 10, 0.05, dm_pvalue=p, dm_pvalue_reference=1 - p, blocks_per_resample=blocks
    )


def test_two_sided_p_is_twice_the_smaller_tail():
    assert _result(0.01).dm_pvalue_two_sided == pytest.approx(0.02)
    assert _result(0.9).dm_pvalue_two_sided == pytest.approx(0.2)


def test_decision_needs_both_dm_and_bootstrap():
    assert _result(0.01).decision == "challenger"
    assert _result(0.01, ci_high=0.05).decision == "not significant"
    assert _result(0.03).decision == "not significant"  # two-sided 0.06


def test_decision_is_insufficient_below_min_blocks():
    assert _result(0.001, blocks=5).decision == "insufficient data"
    assert _result(0.001, blocks=10).decision == "challenger"


def test_holm_can_remove_a_win():
    family = apply_holm([_result(0.015), _result(0.01), _result(0.25)])
    # two-sided 0.03, 0.02, 0.5 -> Holm: 0.02*3=0.06, 0.03*2=0.06, 0.5
    assert [r.holm_adjusted_p for r in family] == pytest.approx([0.06, 0.06, 0.5])
    assert all(r.decision == "not significant" for r in family)
    assert all(r.family_size == 3 for r in family)


def test_holm_bounds_both_directions_in_one_family():
    # One challenger-side and one reference-side result at one-sided 0.02 each: a single
    # two-sided family adjusts both (0.04 * 2 = 0.08), so neither directional claim survives.
    ch = _result(0.02)
    ref = ComparisonResult("c", "r", 100, 0.3, 0.01, 0.1, 0.5, 10, 0.05, dm_pvalue=0.98, dm_pvalue_reference=0.02)
    family = apply_holm([ch, ref])
    assert [r.decision for r in family] == ["not significant", "not significant"]


def test_reference_better_decision():
    r = ComparisonResult("c", "r", 100, 0.3, 0.01, 0.1, 0.5, 10, 0.05, dm_pvalue=0.99, dm_pvalue_reference=0.001)
    assert r.decision == "reference"


def test_default_hac_lag_covers_a_day_block():
    t0 = datetime(2026, 9, 13)
    n = 144 * 12
    ts = [t0 + timedelta(minutes=10 * i) for i in range(n)]
    rng = np.random.default_rng(0)
    actual = rng.normal(20, 1, n)
    r = compare_absolute_errors(actual, actual + rng.normal(0, 1, n), actual + rng.normal(0, 1, n), timestamps=ts, horizon_steps=3, n_bootstrap=100)
    assert r.hac_lag == 144
    assert r.blocks_per_resample == 12


def test_to_dict_is_json_safe():
    d = _result(0.01).to_dict()
    assert d["decision"] == "challenger"
    assert d["holm_adjusted_p"] is None
    assert isinstance(d["dm_pvalue"], float)


def test_format_table_has_dm_and_holm_columns():
    table = format_comparison_table(apply_holm([_result(0.01)]))
    assert "DM p" in table and "Holm p" in table and "challenger" in table


# --- scored rows -----------------------------------------------------------------------


def _row(direction, ts, y, pred, morning=1):
    return {"direction": direction, "bin_ts": ts, "is_morning_peak": morning, "is_evening_peak": 0, "y_30": y, "predicted": pred}


def test_compare_scored_rows_aligns_on_bin_ts():
    rows_ref = [_row("SG_TO_MY", "2026-01-01T10:00:00", 10.0, 11.0), _row("MY_TO_SG", "2026-01-01T10:10:00", 20.0, 22.0)]
    rows_ch = [_row("SG_TO_MY", "2026-01-01T10:00:00", 10.0, 10.2), _row("MY_TO_SG", "2026-01-01T10:10:00", 20.0, 19.0)]
    result = compare_scored_rows(
        rows_ref, rows_ch, label_challenger="model", label_reference="persist", block_size=1, random_state=0
    )
    assert result.n == 2
    assert result.n_groups == 2
    assert result.mean_ae_diff_min < 0


def test_compare_scored_rows_groups_by_direction_with_day_blocks():
    t0 = datetime(2026, 9, 13)
    ref, ch = [], []
    for direction in ("SG_TO_MY", "MY_TO_SG"):
        for i in range(144 * 2):
            ts = t0 + timedelta(minutes=10 * i)
            ref.append(_row(direction, ts, 20.0, 22.0))
            ch.append(_row(direction, ts, 20.0, 21.0))
    result = compare_scored_rows(ref, ch, label_challenger="m", label_reference="p", n_bootstrap=200)
    assert result.block == "day" and result.block_size == 144
    assert result.n_groups == 2
    assert result.horizon_steps == 3


def test_compare_scored_rows_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        compare_scored_rows(
            [_row("SG_TO_MY", "2026-01-01T10:00:00", 1.0, 1.0)],
            [_row("SG_TO_MY", "2026-01-01T10:10:00", 1.0, 1.0)],
            label_challenger="m",
            label_reference="p",
            block_size=1,
        )


def test_joint_day_bootstrap_keeps_shared_day_shocks_together():
    import significance as sig

    # Both directions share one shock per day; within a day the rows barely vary.
    rng = np.random.default_rng(1)
    t0 = datetime(2026, 9, 1)
    days, per_day = 12, 24
    shocks = rng.normal(-0.2, 1.0, size=days)
    ts, groups, d = [], [], []
    for g in ("A", "B"):
        for k in range(days):
            for j in range(per_day):
                ts.append(t0 + timedelta(days=k, hours=j))
                groups.append(g)
                d.append(shocks[k] + rng.normal(0, 0.05))
    zeros = np.zeros(len(d))
    r = sig.compare_absolute_errors(zeros, np.abs(d), zeros, timestamps=ts, groups=groups, n_bootstrap=199)
    day_ids = sig.calendar_day_ids(ts)
    lo, hi = sig.joint_day_bootstrap_mean_ci(d, day_ids, n_resamples=999)
    per_lo, per_hi = sig.block_bootstrap_mean_ci(d, block_size=per_day, groups=groups, n_resamples=999)
    # treating the two directions as independent understates the spread of a shared-day mean
    assert (hi - lo) > 1.2 * (per_hi - per_lo)
    assert r.calendar_days == days and r.pooled
    p = sig.day_cluster_pvalue(d, day_ids)
    assert 0.0 <= p <= 1.0


def test_single_group_comparison_keeps_the_block_gate():
    import significance as sig

    t0 = datetime(2026, 9, 1)
    ts = [t0 + timedelta(hours=h) for h in range(24 * 12)]
    rng = np.random.default_rng(2)
    y = np.zeros(len(ts))
    r = sig.compare_absolute_errors(y, y + 1.0 + rng.normal(0, 0.1, len(ts)), y + 2.0, timestamps=ts, n_bootstrap=199)
    assert not r.pooled and r.joint_ci_low_min is None and r.calendar_days == 12
    assert r.decision == "challenger"
