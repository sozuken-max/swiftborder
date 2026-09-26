import math

from significance import (
    block_bootstrap_mean_ci,
    compare_absolute_errors,
    compare_scored_rows,
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
    n = 80
    actual = [50.0] * n
    ref = [55.0] * n
    ch = [52.0] * n
    result = compare_absolute_errors(
        actual,
        ch,
        ref,
        block_size=4,
        n_bootstrap=400,
        random_state=1,
    )
    assert result.mean_ae_diff_min < 0
    assert result.challenger_better_at_alpha


def test_paired_t_test_zero_mean():
    diffs = [0.0, 0.0, 0.0, 0.0]
    t_stat, p_value = paired_t_test_two_sided(diffs)
    assert t_stat == 0.0
    assert p_value == 1.0


def test_block_bootstrap_ci_contains_mean():
    values = [float(i) for i in range(20)]
    mean = sum(values) / len(values)
    lo, hi = block_bootstrap_mean_ci(values, block_size=5, n_resamples=500, random_state=0)
    assert lo <= mean <= hi


def test_compare_scored_rows_aligns_on_bin_ts():
    rows_ref = [
        {
            "direction": "SG_TO_MY",
            "bin_ts": "2026-01-01T10:00:00",
            "is_morning_peak": 1,
            "is_evening_peak": 0,
            "y_30": 10.0,
            "predicted": 11.0,
        },
        {
            "direction": "MY_TO_SG",
            "bin_ts": "2026-01-01T10:10:00",
            "is_morning_peak": 1,
            "is_evening_peak": 0,
            "y_30": 20.0,
            "predicted": 22.0,
        },
    ]
    rows_ch = [
        {
            "direction": "SG_TO_MY",
            "bin_ts": "2026-01-01T10:00:00",
            "is_morning_peak": 1,
            "is_evening_peak": 0,
            "y_30": 10.0,
            "predicted": 10.2,
        },
        {
            "direction": "MY_TO_SG",
            "bin_ts": "2026-01-01T10:10:00",
            "is_morning_peak": 1,
            "is_evening_peak": 0,
            "y_30": 20.0,
            "predicted": 19.0,
        },
    ]
    result = compare_scored_rows(
        rows_ref,
        rows_ch,
        label_challenger="model",
        label_reference="persist",
        block_size=1,
        random_state=0,
    )
    assert result.n == 2
    assert result.mean_ae_diff_min < 0
