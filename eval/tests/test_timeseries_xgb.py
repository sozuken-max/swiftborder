import numpy as np
import pandas as pd
import pytest

import timeseries_xgb as tsx
from timeseries_xgb import (
    TimeSeriesConfig,
    XGBTrainConfig,
    apply_slew_rate_limit,
    build_supervised_matrices,
    engineer_features,
    rmse_seconds,
)

STEP = pd.Timedelta(minutes=tsx.STEP_MINUTES)


def _synthetic_frame(n: int = 120, start: str = "2026-09-01 00:00:00", values=None) -> pd.DataFrame:
    """Regular 5-minute export with a known target: 1500 + 2*i seconds."""
    t0 = pd.Timestamp(start)
    times = [t0 + i * STEP + pd.Timedelta(seconds=4) for i in range(n)]
    dur = np.asarray(values, dtype=float) if values is not None else 1500 + np.arange(n) * 2.0
    return pd.DataFrame(
        {
            "observed_at_sgt": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "route_id": ["jb_to_woodlands"] * n,
            "status": ["OK"] * n,
            "duration_sec": dur * 0.9 + 7,
            "duration_in_traffic_sec": dur,
            "error_message": [None] * n,
        }
    )


def _small_config(**kw) -> TimeSeriesConfig:
    base = dict(window_size=10, horizon_steps=3, keep_lags=4, train_fraction=0.8, max_slew_step_sec=0)
    base.update(kw)
    return TimeSeriesConfig(**base)


def _sup(frame: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    raw = tsx.prepare_route_frame(frame, config)
    return tsx.build_supervised_frame(engineer_features(raw, config), config)


# --- existing behaviour -------------------------------------------------------


def test_engineer_and_matrix_shapes():
    config = _small_config()
    features = engineer_features(tsx.prepare_route_frame(_synthetic_frame(80), config), config)
    x_train, x_test, y_train, y_test = build_supervised_matrices(features, config)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert "target_lag_1" in x_train.columns
    assert "hour" in x_train.columns


def test_rmse_seconds():
    assert rmse_seconds([100, 200], [110, 190]) == 10.0


def test_slew_rate_limit_caps_steps():
    raw = np.array([0.0, 500.0, 0.0])
    capped = apply_slew_rate_limit(raw, 300.0)
    assert capped[1] == 300.0
    assert capped[2] == 0.0


def test_slew_rate_limit_restarts_after_nan():
    capped = apply_slew_rate_limit(np.array([0.0, np.nan, 900.0, 950.0]), 300.0)
    assert np.isnan(capped[1])
    assert capped[2] == 900.0  # new segment starts at the observation, not 300
    assert capped[3] == 950.0


def test_default_xgb_train_config_matches_teammate():
    cfg = TimeSeriesConfig().xgb
    assert cfg.n_estimators == 200
    assert cfg.max_depth == 4
    assert cfg.min_child_weight == 50


# --- Task 2: persistence and leakage --------------------------------------------


def test_persistence_is_the_value_h_steps_before_the_label():
    config = _small_config()
    sup = _sup(_synthetic_frame(80), config)
    assert len(sup) > 0
    # Target is 1500 + 2i, so the value h steps earlier is exactly y - 2h.
    np.testing.assert_allclose(sup["persistence"], sup["y"] - 2.0 * config.horizon_steps)
    assert ((sup["target_ts"] - sup["origin_ts"]) == config.horizon_steps * STEP).all()


def test_target_lag_1_is_the_origin_value():
    config = _small_config()
    sup = _sup(_synthetic_frame(80), config)
    np.testing.assert_allclose(sup["target_lag_1"], sup["persistence"])
    np.testing.assert_allclose(sup["target_lag_2"], sup["persistence"] - 2.0)


def test_holdout_persistence_uses_the_persistence_column():
    config = _small_config()
    split = tsx.split_supervised(engineer_features(tsx.prepare_route_frame(_synthetic_frame(80), config), config), config)
    np.testing.assert_allclose(
        tsx.holdout_persistence_predictions(split.test, config), split.test["y"] - 2.0 * config.horizon_steps
    )


def test_features_do_not_change_when_the_future_changes():
    """Causality: perturbing anything after the origin leaves that row's features alone."""
    config = _small_config()
    base = _synthetic_frame(80)
    sup = _sup(base, config)
    row = sup.iloc[len(sup) // 2]
    origin = row["origin_ts"]

    perturbed = base.copy()
    obs = pd.to_datetime(perturbed["observed_at_sgt"])
    future = obs > origin + STEP  # strictly after the origin bin
    perturbed.loc[future, "duration_in_traffic_sec"] += 10_000
    perturbed.loc[future, "duration_sec"] += 10_000
    sup2 = _sup(perturbed, config)
    row2 = sup2.set_index("target_ts").loc[row["target_ts"]]

    cols = tsx.feature_columns(sup)
    pd.testing.assert_series_equal(row[cols].astype(float), row2[cols].astype(float), check_names=False)
    assert row2["y"] != row["y"]  # the label is in the future and does move


def test_duration_sec_is_taken_at_the_origin():
    config = _small_config()
    sup = _sup(_synthetic_frame(80), config)
    # duration_sec = 0.9 * target + 7 on the same observation.
    np.testing.assert_allclose(sup["duration_sec"], sup["persistence"] * 0.9 + 7)


def test_calendar_features_describe_the_target_time():
    config = _small_config()
    sup = _sup(_synthetic_frame(80), config)
    assert (sup["hour"] == sup["target_ts"].dt.hour).all()
    assert (sup["minute"] == sup["target_ts"].dt.minute).all()


def test_d1_is_known_at_the_origin():
    config = _small_config()
    n = 2 * 288 + 50
    sup = _sup(_synthetic_frame(n), config)
    with_d1 = sup.dropna(subset=["d1"])
    assert len(with_d1) > 0
    # d1 = value one day before the target, i.e. y - 2 * 288.
    np.testing.assert_allclose(with_d1["d1"], with_d1["y"] - 2.0 * 288)


def test_horizon_of_a_day_or_more_is_rejected():
    with pytest.raises(ValueError):
        tsx.build_supervised_frame(pd.DataFrame(), _small_config(horizon_steps=288))


# --- Task 3: splits, raw labels, shared test set --------------------------------


def test_labels_are_raw_observations_not_smoothed():
    values = np.full(80, 1500.0)
    values[50] = 3000.0  # spike far above the 300 s slew cap
    config = _small_config(max_slew_step_sec=300)
    sup = _sup(_synthetic_frame(80, values=values), config)
    spike_ts = pd.Timestamp("2026-09-01 00:00:00") + 50 * STEP
    assert sup.set_index("target_ts").loc[spike_ts, "y"] == 3000.0
    # Inputs after the spike are slew-limited (1500 + 300), not the raw 3000.
    after = sup[sup["origin_ts"] == spike_ts]
    assert after["target_lag_1"].iloc[0] == 1800.0


def test_missing_observations_are_not_labels():
    frame = _synthetic_frame(80).drop(index=[60, 61])
    config = _small_config()
    sup = _sup(frame, config)
    missing = {pd.Timestamp("2026-09-01 00:00:00") + i * STEP for i in (60, 61)}
    assert missing.isdisjoint(set(sup["target_ts"]))


def test_inputs_are_filled_causally_without_bfill():
    frame = _synthetic_frame(80).drop(index=[40])
    config = _small_config()
    raw = tsx.prepare_route_frame(frame, config)
    grid = tsx.regular_grid(raw, config)
    gap = pd.Timestamp("2026-09-01 00:00:00") + 40 * STEP
    assert np.isnan(grid.loc[gap, "y_raw"])
    assert grid.loc[gap, "y_ffill"] == grid.loc[gap - STEP, "y_raw"]


def test_long_gaps_are_not_forward_filled_forever():
    frame = _synthetic_frame(80).drop(index=range(30, 45))
    config = _small_config(max_ffill_steps=6)
    grid = tsx.regular_grid(tsx.prepare_route_frame(frame, config), config)
    late_in_gap = pd.Timestamp("2026-09-01 00:00:00") + 40 * STEP
    assert np.isnan(grid.loc[late_in_gap, "y_ffill"])


def test_split_is_chronological_on_target_time():
    config = _small_config()
    features = engineer_features(tsx.prepare_route_frame(_synthetic_frame(200), config), config)
    split = tsx.split_supervised(features, config)
    assert split.train["target_ts"].max() < split.boundary <= split.test["target_ts"].min()


def test_backtest_days_are_strictly_after_the_split():
    config = _small_config()
    n = 288 * 6
    features = engineer_features(tsx.prepare_route_frame(_synthetic_frame(n), config), config)
    split = tsx.split_supervised(features, config, train_fraction=0.5)
    days = tsx.backtest_days(split)
    assert days, "expected at least one full test day"
    for d in days:
        assert pd.Timestamp(d) >= split.boundary


def test_backtest_actual_window_matches_the_supervised_rows():
    xgb = pytest.importorskip("xgboost")  # noqa: F841
    config = _small_config(xgb=XGBTrainConfig(n_estimators=5, min_child_weight=1))
    n = 288 * 4
    features = engineer_features(tsx.prepare_route_frame(_synthetic_frame(n), config), config)
    split = tsx.split_supervised(features, config, train_fraction=0.5)
    model = tsx.train_xgb(split.train[split.feature_cols], split.train["y"].to_numpy(), config=config)
    days = tsx.backtest_days(split)
    scores = tsx.score_forecast_days(model, features, split, days[:1], config)
    day_rows = split.test[split.test["target_ts"].dt.strftime("%Y-%m-%d") == days[0]]
    expected = tsx.mae_minutes(day_rows["y"], model.predict(day_rows[split.feature_cols]))
    got = scores[(scores["method"] == "XGB actual window")]["MAE_min"].iloc[0]
    assert got == pytest.approx(expected)
    persist = scores[scores["method"] == f"Persistence T-{config.horizon_minutes}"]["MAE_min"].iloc[0]
    assert persist == pytest.approx(tsx.mae_minutes(day_rows["y"], day_rows["persistence"]))


# --- Wavelet / trend ablation (Chad, day 2) ------------------------------------


def _wave_config(**kw) -> TimeSeriesConfig:
    """Window long enough for db2 level 2 (and level 3 when asked)."""
    base = dict(
        window_size=32,
        horizon_steps=3,
        keep_lags=4,
        trend_window=12,
        train_fraction=0.7,
        max_slew_step_sec=0,
        include_trends=True,
        use_dwt=True,
        dwt_wavelet="db2",
        dwt_level=2,
    )
    base.update(kw)
    return TimeSeriesConfig(**base)


def _coeff_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("dwt_") and c not in (tsx.DWT_MEAN_COL, tsx.DWT_STD_COL)]


def test_default_wavelet_is_db2_level_2():
    assert tsx.DWT_WAVELET == "db2"
    assert tsx.DWT_LEVEL == 2
    assert tsx.DWT_LEVEL_COARSE == 3
    cfg = TimeSeriesConfig()
    assert cfg.dwt_wavelet == "db2"
    assert cfg.dwt_level == 2
    assert cfg.use_dwt is False
    assert cfg.include_trends is False
    assert cfg.trend_window == 12  # 60 minutes on the 5-minute grid


def test_default_frame_does_not_add_wavelet_or_trend_columns():
    config = _small_config()
    sup = _sup(_synthetic_frame(80), config)
    cols = tsx.feature_columns(sup)
    assert not any(c.startswith("dwt_") or c.startswith("trend_") for c in cols)


def test_zscore_shape_is_invariant_to_shift_and_positive_scale():
    pytest.importorskip("pywt")
    window = np.sin(np.linspace(0.2, 3.0, 36)) + np.array([0.1 * ((i % 5) - 2) for i in range(36)])
    base = tsx.dwt_features_from_windows(window.reshape(1, -1), wavelet="db2", level=2)
    shifted = tsx.dwt_features_from_windows((window + 80.0).reshape(1, -1), wavelet="db2", level=2)
    scaled = tsx.dwt_features_from_windows((window * 4.0).reshape(1, -1), wavelet="db2", level=2)
    coeffs = _coeff_columns(base)
    np.testing.assert_allclose(base[coeffs].to_numpy(), shifted[coeffs].to_numpy(), atol=1e-8)
    np.testing.assert_allclose(base[coeffs].to_numpy(), scaled[coeffs].to_numpy(), atol=1e-8)
    assert base[tsx.DWT_MEAN_COL].iloc[0] == pytest.approx(float(np.mean(window)))
    assert base[tsx.DWT_STD_COL].iloc[0] == pytest.approx(float(np.std(window, ddof=0)))
    assert shifted[tsx.DWT_MEAN_COL].iloc[0] == pytest.approx(float(np.mean(window)) + 80.0)
    assert shifted[tsx.DWT_STD_COL].iloc[0] == pytest.approx(float(np.std(window, ddof=0)))
    assert scaled[tsx.DWT_MEAN_COL].iloc[0] == pytest.approx(float(np.mean(window)) * 4.0)
    assert scaled[tsx.DWT_STD_COL].iloc[0] == pytest.approx(float(np.std(window, ddof=0)) * 4.0)


def test_zero_std_window_is_a_zero_shape_and_keeps_mean():
    pytest.importorskip("pywt")
    constant = np.full(36, 42.0)
    frame = tsx.dwt_features_from_windows(constant.reshape(1, -1), wavelet="db2", level=2)
    row = frame.iloc[0]
    assert row[tsx.DWT_MEAN_COL] == pytest.approx(42.0)
    assert row[tsx.DWT_STD_COL] == pytest.approx(0.0)
    np.testing.assert_allclose(row[_coeff_columns(frame)].to_numpy(dtype=float), 0.0)


def test_db2_level_2_and_level_3_coefficient_counts_differ():
    pytest.importorskip("pywt")
    window = np.sin(np.linspace(0.0, 6.0, 36)).reshape(1, -1)
    level2 = tsx.dwt_features_from_windows(window, wavelet="db2", level=2)
    level3 = tsx.dwt_features_from_windows(window, wavelet="db2", level=3)
    coeffs2 = _coeff_columns(level2)
    coeffs3 = _coeff_columns(level3)
    # periodization on length 36: level 2 is 9+9+18=36; level 3 is 5+5+9+18=37.
    assert len(coeffs2) == 36
    assert len(coeffs3) == 37
    assert len([c for c in coeffs2 if c.startswith("dwt_a_")]) == 9
    assert len([c for c in coeffs3 if c.startswith("dwt_a_")]) == 5
    assert any(c.startswith("dwt_d2_") for c in coeffs2)
    assert not any(c.startswith("dwt_d3_") for c in coeffs2)
    assert any(c.startswith("dwt_d3_") for c in coeffs3)
    assert tsx.DWT_MEAN_COL in level2.columns and tsx.DWT_STD_COL in level3.columns


def test_trend_diff_and_rolling_mean_match_the_origin_window():
    config = _wave_config(use_dwt=False)
    sup = _sup(_synthetic_frame(80), config)
    # Synthetic target is 1500 + 2 per bin, slew off, so the step is 2 seconds.
    np.testing.assert_allclose(sup[tsx.TREND_DIFF_COL], 2.0)
    np.testing.assert_allclose(
        sup[tsx.TREND_ROLL_COL],
        sup["target_lag_1"] - (config.trend_window - 1),
    )


def test_wavelet_and_trends_do_not_change_when_the_future_changes():
    pytest.importorskip("pywt")
    config = _wave_config()
    base = _synthetic_frame(90)
    sup = _sup(base, config)
    row = sup.iloc[len(sup) // 2]
    origin = row["origin_ts"]
    perturbed = base.copy()
    obs = pd.to_datetime(perturbed["observed_at_sgt"])
    future = obs > origin + STEP
    perturbed.loc[future, "duration_in_traffic_sec"] += 10_000
    perturbed.loc[future, "duration_sec"] += 10_000
    sup2 = _sup(perturbed, config)
    row2 = sup2.set_index("target_ts").loc[row["target_ts"]]
    cols = tsx.ablation_feature_columns(sup, "B") + [
        c for c in tsx.ablation_feature_columns(sup, "C") if c not in tsx.ablation_feature_columns(sup, "A")
    ]
    pd.testing.assert_series_equal(row[cols].astype(float), row2[cols].astype(float), check_names=False)
    assert row2["y"] != row["y"]


def test_feature_set_a_is_the_lag_prefix_of_b_and_c():
    pytest.importorskip("pywt")
    config = _wave_config()
    sup = _sup(_synthetic_frame(90), config)
    a = tsx.ablation_feature_columns(sup, "A")
    b = tsx.ablation_feature_columns(sup, "B")
    c = tsx.ablation_feature_columns(sup, "C")
    assert a == [col for col in sup.columns if col.startswith("target_lag_")]
    assert b[: len(a)] == a
    assert c[: len(a)] == a
    assert b == a + [tsx.TREND_DIFF_COL, tsx.TREND_ROLL_COL]
    assert tsx.DWT_MEAN_COL in c and tsx.DWT_STD_COL in c
    assert tsx.TREND_DIFF_COL not in c
    assert not any(col.startswith("dwt_") for col in b)
    assert "hour" not in a and "d1" not in a and "duration_sec" not in a


def test_compare_feature_sets_reports_point_estimates_without_pvalues_on_a_short_holdout():
    pytest.importorskip("pywt")
    pytest.importorskip("xgboost")
    config = _wave_config(xgb=XGBTrainConfig(n_estimators=5, min_child_weight=1))
    features = engineer_features(tsx.prepare_route_frame(_synthetic_frame(120), config), config)
    result = tsx.compare_window_feature_sets(features, config)
    assert result["significance"] == "insufficient data"
    assert result["day_blocks"] < tsx.MIN_BLOCKS
    assert "comparisons" not in result
    assert [row["feature_set"] for row in result["sets"]] == ["A", "B", "C"]
    assert result["wavelet"] == "db2" and result["dwt_level"] == 2
    for row in result["sets"]:
        assert row["mae_min"] >= 0.0 and row["rmse_min"] >= 0.0
    assert set(result["columns"]["A"]) < set(result["columns"]["B"])
    assert set(result["columns"]["A"]) < set(result["columns"]["C"])
