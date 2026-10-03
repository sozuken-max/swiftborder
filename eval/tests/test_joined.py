import numpy as np
import pandas as pd
import pytest

import features as fx
import joined as jx
from run_artifacts import validate_manifest

T0 = pd.Timestamp("2026-09-05 16:00:00", tz="UTC")  # 6 Sep 00:00 SGT


def _frame(days=10, camera_from_day=None, seed=0):
    """Synthetic feature table where rain adds 3 minutes to the future duration."""
    rng = np.random.default_rng(seed)
    rows = []
    n = days * 144
    for direction in ("SG_TO_MY", "MY_TO_SG"):
        base = 25 + 5 * np.sin(np.arange(n + 10) * 2 * np.pi / 144)
        rain = (rng.random(n + 10) < 0.1).astype(float)
        for i in range(n):
            ts = T0 + i * fx.BIN
            rows.append(
                {
                    "direction": direction,
                    "bin_ts": ts,
                    "after_gap": 0,
                    "y_persistence": base[i] + rng.normal(0, 0.3),
                    "maps_typical_min": 12.0,
                    "y_30": base[i + 3] + 3 * rain[i] + rng.normal(0, 0.3),
                    "rain_30": 5.0 * rain[i],
                    "rain_60": 5.0 * rain[i],
                    "fc_rain": rain[i],
                    "fc_heavy": 0.0,
                    "cam_count": (3.0 if camera_from_day is not None and i >= camera_from_day * 144 else np.nan),
                    "cam_extent": np.nan,
                    "cam_age_min": np.nan,
                }
            )
    df = pd.DataFrame(rows)
    for col in fx.MAPS_FEATURES:
        if col not in df:
            df[col] = 0.0
    return df


def test_folds_never_train_on_labels_from_the_test_day():
    data = jx.scorable(_frame(days=6))
    for day in jx.fold_days(data, "2026-09-08", "2026-09-11"):
        train, test = jx.fold_split(data, day)
        day_start = pd.Timestamp(day).tz_localize("Asia/Singapore").tz_convert("UTC")
        assert (train["bin_ts"] + jx.LABEL_LAG <= day_start).all()
        assert (test["date_sgt"] == day).all()
        assert train["bin_ts"].max() < test["bin_ts"].min()


def test_scorable_drops_after_gap_and_unlabelled():
    f = _frame(days=2)
    f.loc[0, "after_gap"] = 1
    f.loc[1, "y_30"] = np.nan
    assert len(jx.scorable(f)) == len(f) - 2


def test_weather_ablation_detects_a_real_effect():
    # 10 shared calendar days: the pooled gate counts days, not days x directions
    data = jx.scorable(_frame(days=14))
    oof = jx.run_folds(data, "2026-09-10", "2026-09-19", min_train_rows=200)
    assert "xgb[maps+weather+camera]" not in oof.columns  # no camera history -> skipped
    mae = lambda c: (oof["y_30"] - oof[c]).abs().mean()  # noqa: E731
    # Rain (10% of bins) adds 3 min, so the attainable MAE gain is about 0.3 min.
    assert mae("xgb[maps+weather]") < mae("xgb[maps]") - 0.15
    assert mae("ridge[maps+weather]") < mae("ridge[maps]") - 0.15
    comps, notes = jx.significance(oof, n_bootstrap=200)
    weather = next(c for c in comps if c.label_challenger == "xgb[maps+weather] (both/all)")
    assert weather.decision == "challenger" and weather.calendar_days == 10
    assert weather.joint_ci_high_min < 0 and weather.day_cluster_pvalue < 0.05
    assert weather.family_size == len(comps)


def test_pooled_gate_counts_shared_days_not_direction_blocks():
    # 6 test days in both directions: 12 per-direction blocks, but only 6 days of traffic
    data = jx.scorable(_frame(days=10))
    oof = jx.run_folds(data, "2026-09-10", "2026-09-15", min_train_rows=200)
    comps, _ = jx.significance(oof, n_bootstrap=200)
    pooled = next(c for c in comps if c.label_challenger == "xgb[maps+weather] (both/all)")
    assert pooled.blocks_per_resample >= 10 and pooled.calendar_days == 6
    assert pooled.decision == "insufficient data"


def test_camera_comparison_is_limited_to_covered_rows_and_flagged(tmp_path):
    data = jx.scorable(_frame(days=10, camera_from_day=7))  # camera from 13 Sep SGT: 3 of 6 test days
    oof = jx.run_folds(data, "2026-09-10", "2026-09-15", min_train_rows=200)
    assert "xgb[maps+weather+camera]" in oof.columns
    comps, notes = jx.significance(oof, n_bootstrap=200)
    cam = [c for c in comps if "camera" in c.label_challenger]
    assert cam and all("camera-covered" in c.label_challenger for c in cam)
    covered = int(oof["cam_count"].notna().sum())
    assert all(c.n <= covered for c in cam)
    assert notes["camera_coverage"]["both"] < jx.MIN_CAMERA_COVERAGE
    assert set(notes["camera_status"].values()) == {"insufficient"}


def test_component_is_valid_v2(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frame = _frame(days=9)
    for col in fx.MAPS_FEATURES:
        frame[col] = frame.get(col, 0.0)
    paths, meta = jx.joined_component(run_dir, frame, start="2026-09-10", end="2026-09-13", n_bootstrap=200)
    manifest = {
        "schema_version": 2, "run_id": "run", "created_at_utc": "x", "components": ["joined"],
        "provenance": {"git_sha": "x", "git_dirty": False, "code_sha256": "y"}, "joined": meta,
    }
    assert validate_manifest(manifest, run_dir) == []
    assert meta["window"]["folds"] == 4
    assert all(p.is_file() for p in paths)


def test_camera_forecast_arm_is_scored_in_its_own_family():
    import camera_forecast as cf

    f = _frame(days=8)
    # a known-in-advance prior that anticipates the sine: the expected change over the next 30 min
    tod = (f["bin_ts"].dt.tz_convert("Asia/Singapore").dt.hour * 6 + f["bin_ts"].dt.tz_convert("Asia/Singapore").dt.minute // 10).to_numpy()
    f["camfc_now"] = 25 + 5 * np.sin(tod * 2 * np.pi / 144)
    f["camfc_30"] = 25 + 5 * np.sin((tod + 3) * 2 * np.pi / 144)
    f["camfc_delta"] = f["camfc_30"] - f["camfc_now"]
    data = jx.scorable(f)
    oof = jx.run_folds(data, "2026-09-10", "2026-09-13", min_train_rows=200, mp_harmonics=2)
    for c in ("xgb[maps+camfc]", "xgb[maps+mpfc]", "xgb[maps+mpfc+camfc]"):
        assert c in oof.columns
    specs = jx.camfc_specs(oof)
    assert ("xgb[maps+camfc]", "xgb[maps]", "all") in specs
    assert ("xgb[maps+mpfc+camfc]", "xgb[maps+mpfc]", "all") in specs
    base = {s[0] for s in jx.comparison_specs(oof)}
    assert not any("camfc" in s for s in base)  # the weather / camera family is unchanged
    reg = jx.regime_metrics(oof, ["xgb[maps]"])
    assert {m["slice"] for m in reg} <= {"both/regime=rising", "both/regime=steady", "both/regime=falling"}
    assert cf.MP_FEATURES[0] in jx.FEATURE_SETS["maps+mpfc"]


def test_feature_sets_with_missing_columns_are_skipped():
    oof = jx.run_folds(jx.scorable(_frame(days=6)), "2026-09-09", "2026-09-10", min_train_rows=200)
    assert "xgb[maps+camfc]" not in oof.columns and "xgb[maps+mpfc]" in oof.columns
