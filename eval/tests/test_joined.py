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
    data = jx.scorable(_frame(days=10))
    oof = jx.run_folds(data, "2026-09-10", "2026-09-15", min_train_rows=200)
    assert "xgb[maps+weather+camera]" not in oof.columns  # no camera history -> skipped
    mae = lambda c: (oof["y_30"] - oof[c]).abs().mean()  # noqa: E731
    # Rain (10% of bins) adds 3 min, so the attainable MAE gain is about 0.3 min.
    assert mae("xgb[maps+weather]") < mae("xgb[maps]") - 0.15
    assert mae("ridge[maps+weather]") < mae("ridge[maps]") - 0.15
    comps, notes = jx.significance(oof, n_bootstrap=200)
    weather = next(c for c in comps if c.label_challenger == "xgb[maps+weather] (both/all)")
    assert weather.decision == "challenger"
    assert weather.family_size == len(comps)


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
