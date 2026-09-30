import numpy as np
import pandas as pd
import pytest

import camera_forecast as cf


def _history(days=12, drop_day=None):
    rows = []
    start = pd.Timestamp("2026-03-16 00:00", tz="Asia/Singapore")
    rng = np.random.default_rng(0)
    for d in range(days):
        for k in range(144):
            ts = start + pd.Timedelta(days=d, minutes=10 * k)
            if drop_day is not None and d == drop_day and k > 10:
                continue
            hour = ts.hour + ts.minute / 60
            weekend = ts.dayofweek >= 5
            base = 60 + 40 * np.sin(2 * np.pi * (hour - 6) / 24) + (15 if weekend else 0)
            rows.append({"direction": "MY_TO_SG", "bin_ts": ts.tz_convert("UTC"), "vis_count": base + rng.normal(0, 3)})
            rows.append({"direction": "SG_TO_MY", "bin_ts": ts.tz_convert("UTC"), "vis_count": 20 + rng.normal(0, 3)})
    return pd.DataFrame(rows)


def test_clean_history_drops_partial_days():
    kept, info = cf.clean_history(_history(days=5, drop_day=2))
    assert info["days_in"] == 5
    assert info["days_kept_per_direction"] == {"SG_TO_MY": 4, "MY_TO_SG": 4}


def test_fourier_profile_recovers_shape_and_weekend_offset():
    kept, _ = cf.clean_history(_history())
    p = cf.FourierProfile(harmonics=2).fit(kept)
    wed = pd.Series([pd.Timestamp("2026-09-16 12:00", tz="Asia/Singapore").tz_convert("UTC") - cf.NOW_OFFSET])
    sat = pd.Series([pd.Timestamp("2026-09-19 12:00", tz="Asia/Singapore").tz_convert("UTC") - cf.NOW_OFFSET])
    expect = 60 + 40 * np.sin(2 * np.pi * 6 / 24)
    assert p.predict(["MY_TO_SG"], wed)[0] == pytest.approx(expect, abs=3)
    assert p.predict(["MY_TO_SG"], sat)[0] - p.predict(["MY_TO_SG"], wed)[0] == pytest.approx(15, abs=4)
    assert p.predict(["SG_TO_MY"], wed)[0] == pytest.approx(20, abs=2)


def test_cross_validation_prefers_a_profile_over_the_mean():
    kept, _ = cf.clean_history(_history())
    cv = {r["candidate"]: r["mae_count"] for r in cf.cross_validate(kept)}
    assert cv["fourier K=2"] < cv["direction mean"]
    profile, info = cf.fit_camera_profile(_history())
    assert info["chosen"].startswith("fourier") and info["harmonics"] in cf.HARMONICS


def test_profile_features_use_only_the_calendar_and_the_label_bin_offset():
    profile, _ = cf.fit_camera_profile(_history())
    t0 = pd.Timestamp("2026-09-16 06:00", tz="Asia/Singapore").tz_convert("UTC")
    frame = pd.DataFrame({"direction": ["MY_TO_SG", "MY_TO_SG"], "bin_ts": [t0, t0 + pd.Timedelta(minutes=30)]})
    out = cf.add_profile_features(frame, profile, "camfc")
    # camfc_30 at origin t equals camfc_now at origin t + 30 min (same calendar point)
    assert out.loc[0, "camfc_30"] == pytest.approx(out.loc[1, "camfc_now"])
    assert out.loc[0, "camfc_delta"] == pytest.approx(out.loc[0, "camfc_30"] - out.loc[0, "camfc_now"])
    assert out.loc[0, "camfc_delta"] > 0  # the synthetic profile rises after 06:00


def test_fetch_history_reads_the_cache(tmp_path):
    path = tmp_path / "h.csv"
    _history(days=2).to_csv(path, index=False)
    df = cf.fetch_history(path)
    assert str(df["bin_ts"].dt.tz) == "UTC" and len(df) == 2 * 2 * 144


def test_fold_maps_profile_fits_on_travel_time():
    t0 = pd.Timestamp("2026-09-10 00:00", tz="UTC")
    train = pd.DataFrame({"direction": ["SG_TO_MY"] * 300, "bin_ts": [t0 + pd.Timedelta(minutes=10 * i) for i in range(300)], "y_persistence": 25.0})
    p = cf.fold_maps_profile(train, 2)
    assert p.predict(["SG_TO_MY"], pd.Series([t0]))[0] == pytest.approx(25.0, abs=1e-6)
