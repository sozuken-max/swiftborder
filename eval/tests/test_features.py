import numpy as np
import pandas as pd
import pytest

import features as fx

T0 = pd.Timestamp("2026-09-20 00:00:00", tz="UTC")  # 08:00 SGT, a Sunday in SGT


def _tt(n=60, directions=(("jb_to_woodlands", "MY_TO_SG"), ("mandai_to_shell_jb", "SG_TO_MY")), drop=()):
    rows = []
    for route, direction in directions:
        for i in range(n):
            if i in drop:
                continue
            for k in (0, 5):  # two observations per 10-min bin
                rows.append(
                    {
                        "observed_at": (T0 + pd.Timedelta(minutes=10 * i + k, seconds=3)).isoformat(),
                        "route_id": route,
                        "direction": direction,
                        "status": "OK",
                        "duration_in_traffic_sec": 60.0 * (20 + i) + (30 if k else -30),
                        "duration_sec": 900,
                        "congestion_ratio": 1.5,
                        "speed_kmh": 30.0,
                    }
                )
    return pd.DataFrame(rows)


def _row(frame, direction, i):
    return frame[(frame["direction"] == direction) & (frame["bin_ts"] == T0 + i * fx.BIN)].iloc[0]


def test_bins_lags_and_labels():
    f = fx.maps_features(fx.maps_bins(_tt()))
    r = _row(f, "MY_TO_SG", 10)
    assert r["y_persistence"] == pytest.approx(30.0)  # mean of 20+i +/- 0.5
    assert r["lag_10"] == pytest.approx(29.0)
    assert r["lag_60"] == pytest.approx(24.0)
    assert r["roll_mean_30"] == pytest.approx(28.0)
    assert r["slope_30"] == pytest.approx(3.0)
    assert r["y_30"] == pytest.approx(33.0)
    assert r["y_60"] == pytest.approx(36.0)
    assert r["asof"] == r["bin_ts"] + fx.BIN


def test_calendar_matches_bigquery_conventions():
    f = fx.maps_features(fx.maps_bins(_tt()))
    r = _row(f, "MY_TO_SG", 0)  # 2026-09-20 08:00 SGT, Sunday
    assert r["dow"] == 1 and r["is_weekend"] == 1
    assert r["is_morning_peak"] == 1 and r["is_evening_peak"] == 0
    assert r["tod_block"] == pytest.approx(8 * 60 / 5)


def test_missing_bin_gives_nan_not_the_previous_row():
    f = fx.maps_features(fx.maps_bins(_tt(drop={13})))
    r = _row(f, "MY_TO_SG", 10)
    assert np.isnan(r["y_30"])  # bin 13 is missing
    r = _row(f, "MY_TO_SG", 14)
    assert np.isnan(r["lag_10"])


def test_after_gap_nulls_lags_like_the_view():
    f = fx.maps_features(fx.maps_bins(_tt(drop=set(range(20, 24)))))  # 50-minute gap
    r = _row(f, "MY_TO_SG", 24)
    assert r["after_gap"] == 1 and np.isnan(r["lag_60"])
    assert _row(f, "MY_TO_SG", 31)["after_gap"] == 0


def test_parity_with_view_rows_when_no_bin_is_missing():
    f = fx.maps_features(fx.maps_bins(_tt()))
    view = f[["direction", "bin_ts", "y_persistence", "lag_10", "y_30"]].copy()
    view["bin_ts"] = view["bin_ts"].astype(str)
    p = fx.parity(f, view, cols=["y_persistence", "lag_10", "y_30"])
    assert p["rows_compared"] == len(f)
    assert p["lag_10"] == 0 and p["y_30"] == 0


def test_causality_future_values_do_not_move_features():
    tt = _tt()
    base = fx.build(tt, _rain(), _forecast(), _camera())
    later = tt.copy()
    obs = pd.to_datetime(later["observed_at"], utc=True)
    later.loc[obs >= T0 + 21 * fx.BIN, "duration_in_traffic_sec"] += 9999
    moved = fx.build(later, _rain(shift_after=T0 + 21 * fx.BIN), _forecast(), _camera(shift_after=T0 + 21 * fx.BIN))
    cols = fx.MAPS_FEATURES + fx.WEATHER_FEATURES + fx.CAMERA_FEATURES
    a = base[base["asof"] <= T0 + 21 * fx.BIN].set_index(["direction", "bin_ts"])[cols]
    b = moved[moved["asof"] <= T0 + 21 * fx.BIN].set_index(["direction", "bin_ts"])[cols]
    pd.testing.assert_frame_equal(a, b)
    # the label for bin 18 (= bin 21) did change
    assert _row(moved, "MY_TO_SG", 18)["y_30"] != _row(base, "MY_TO_SG", 18)["y_30"]


def _rain(shift_after=None):
    ts = [T0 + pd.Timedelta(minutes=5 * i) for i in range(1, 200)]
    vals = [1.0 if 30 <= i < 36 else 0.0 for i in range(1, 200)]
    df = pd.DataFrame({"ts": ts, "value_mm": vals})
    if shift_after is not None:
        df.loc[df["ts"] > shift_after, "value_mm"] += 50
    return df


def test_rain_windows_are_trailing_and_inclusive():
    rain = _rain()
    asof = pd.Series([T0 + pd.Timedelta(minutes=180), T0 + pd.Timedelta(minutes=150)])
    # readings 30..35 (at 150..175 min) are 1.0 mm
    np.testing.assert_allclose(fx.rain_sums(asof, rain, 30), [5.0, 1.0])
    np.testing.assert_allclose(fx.rain_sums(asof, rain, 60), [6.0, 1.0])


def _forecast():
    rows = []
    for k in range(0, 12):
        issue = T0 + pd.Timedelta(minutes=30 * k)
        rows.append({"issue_timestamp": issue, "valid_start": issue, "valid_end": issue + pd.Timedelta(hours=2), "forecast": "Thundery Showers" if k == 4 else "Cloudy"})
    return pd.DataFrame(rows)


def test_forecast_uses_valid_period_and_issue_before_asof():
    fc = _forecast()
    asof = pd.Series([T0 + pd.Timedelta(minutes=125), T0 + pd.Timedelta(minutes=115)])
    target = asof + pd.Timedelta(minutes=20)
    flags = fx.forecast_flags(asof, target, fc)
    # asof 125 min: latest issue is k=4 (120 min, thundery) and it covers the target
    assert flags["fc_rain"].iloc[0] == 1 and flags["fc_heavy"].iloc[0] == 1
    # asof 115 min: k=4 is not yet issued, so k=3 (cloudy) is used
    assert flags["fc_rain"].iloc[1] == 0


def test_forecast_is_selected_by_acquisition_time(tmp_path):
    issue = T0 + pd.Timedelta(minutes=120)
    rows = [
        # issued at 120 min but only acquired by data.gov.sg at 135 min, then revised at 150 min
        {"issue_timestamp": issue.isoformat(), "valid_start": issue.isoformat(), "valid_end": (issue + pd.Timedelta(hours=2)).isoformat(), "area": "Woodlands", "forecast": "Cloudy", "update_timestamp": (issue + pd.Timedelta(minutes=15)).isoformat()},
        {"issue_timestamp": issue.isoformat(), "valid_start": issue.isoformat(), "valid_end": (issue + pd.Timedelta(hours=2)).isoformat(), "area": "Woodlands", "forecast": "Heavy Thundery Showers", "update_timestamp": (issue + pd.Timedelta(minutes=30)).isoformat()},
        # older forecast without acquisition time: available 10 min after issue
        {"issue_timestamp": (issue - pd.Timedelta(minutes=30)).isoformat(), "valid_start": (issue - pd.Timedelta(minutes=30)).isoformat(), "valid_end": (issue + pd.Timedelta(hours=2)).isoformat(), "area": "Woodlands", "forecast": "Fair", "update_timestamp": ""},
    ]
    p = tmp_path / "fc.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    fc = fx.load_forecast(p)
    asof = pd.Series([issue + pd.Timedelta(minutes=10), issue + pd.Timedelta(minutes=20), issue + pd.Timedelta(minutes=40)])
    flags = fx.forecast_flags(asof, asof + pd.Timedelta(minutes=20), fc)
    # +10: nothing from this issue acquired yet -> the older 'Fair'; +20: 'Cloudy'; +40: revision
    assert list(flags["fc_rain"]) == [0.0, 0.0, 1.0]
    assert list(flags["fc_heavy"]) == [0.0, 0.0, 1.0]


def _camera(shift_after=None):
    ts = [T0 + pd.Timedelta(minutes=10 * i + 2) for i in range(0, 40, 2)]  # every 20 min
    df = pd.DataFrame(
        {
            "frame_ts": ts,
            "sg_my": [0 if i == 0 else 3 for i in range(len(ts))],
            "my_sg": [5] * len(ts),
            "sg_my_extent": [0.0] * len(ts),
            "my_sg_extent": [0.4] * len(ts),
        }
    )
    if shift_after is not None:
        df.loc[df["frame_ts"] > shift_after, ["sg_my", "my_sg"]] += 100
    return df


def test_camera_zero_is_zero_and_missing_is_nan():
    f = fx.build(_tt(), camera=_camera())
    first = _row(f, "SG_TO_MY", 0)  # asof 00:10, frame at 00:02 with 0 SG-MY vehicles
    assert first["cam_count"] == 0 and first["cam_age_min"] == pytest.approx(8)
    assert _row(f, "MY_TO_SG", 0)["cam_count"] == 5
    late = _row(f, "SG_TO_MY", 50)  # frames stop at 06:22 -> older than 30 min
    assert np.isnan(late["cam_count"])
    no_cam = fx.build(_tt())
    assert no_cam["cam_count"].isna().all()


def test_load_camera_keeps_only_scored_frames(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame(
        {
            "bin_sgt": ["2026-09-20T08:00:00", "2026-09-20T08:10:00"],
            "frame_ts": ["2026-09-20T07:59:10", ""],
            "status": ["ok", "missing_frame"],
            "sg_my": [2, ""],
            "my_sg": [1, ""],
            "sg_my_extent": [0.1, ""],
            "my_sg_extent": [0.2, ""],
        }
    ).to_csv(p, index=False)
    cam = fx.load_camera(p)
    assert len(cam) == 1
    assert cam["frame_ts"].iloc[0] == pd.Timestamp("2026-09-19 23:59:10", tz="UTC")


def test_coverage_report():
    f = fx.build(_tt(), rain=_rain())
    cov = fx.coverage_report(f)
    rain_cov = cov[(cov["feature"] == "rain_30")]["coverage"]
    assert (rain_cov == 1.0).all()
    assert (cov[cov["feature"] == "cam_count"]["coverage"] == 0).all()
