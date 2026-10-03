"""--data-cutoff: observations after the cutoff are dropped everywhere; BQML window and joined days are checked."""

import json

import numpy as np
import pandas as pd
import pytest

import generate_comparison_plots as gcp
import joined as jx
import timeseries_xgb as tsx

CUT = "2026-10-19 23:59"


def test_parse_data_cutoff_treats_naive_as_sgt():
    ts = tsx.parse_data_cutoff(CUT)
    assert ts == pd.Timestamp("2026-10-19 15:59", tz="UTC")
    assert tsx.parse_data_cutoff(None) is None and tsx.parse_data_cutoff("") is None
    assert tsx.parse_data_cutoff("2026-10-19T15:59:00Z") == ts


def test_apply_data_cutoff_on_utc_and_sgt_columns():
    cut = tsx.parse_data_cutoff(CUT)
    utc = pd.DataFrame({"observed_at": ["2026-10-19T15:58:00Z", "2026-10-19T15:59:00Z", "2026-10-19T16:00:00Z"], "v": [1, 2, 3]})
    assert list(tsx.apply_data_cutoff(utc, cut)["v"]) == [1, 2]
    sgt = pd.DataFrame({"observed_at_sgt": ["2026-10-19T23:59:00", "2026-10-20T00:00:00"], "v": [1, 2]})
    assert list(tsx.apply_data_cutoff(sgt, cut)["v"]) == [1]
    assert tsx.apply_data_cutoff(utc, None) is utc


def test_no_scored_label_after_the_cutoff():
    # 5-minute series running past the cutoff: after the cut, the last 60-min label is at or before it.
    start = pd.Timestamp("2026-10-15 00:00")
    times = [start + pd.Timedelta(minutes=5 * i) for i in range(6 * 288)]
    export = pd.DataFrame(
        {
            "observed_at_sgt": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "route_id": tsx.DEFAULT_ROUTE_ID,
            "status": "OK",
            "duration_sec": 600.0,
            "duration_in_traffic_sec": 1200.0 + 60 * np.sin(np.arange(len(times)) / 20),
            "error_message": None,
        }
    )
    cut = tsx.parse_data_cutoff(CUT)
    cfg = tsx.TimeSeriesConfig()
    feats = tsx.engineer_features(tsx.prepare_route_frame(tsx.apply_data_cutoff(export, cut), cfg), cfg)
    sup = tsx.build_supervised_frame(feats, cfg)
    assert sup["target_ts"].max() <= pd.Timestamp("2026-10-19 23:59")


def test_cutoff_window_end_default_and_validation():
    cut = tsx.parse_data_cutoff(CUT)
    assert gcp.cutoff_window_end(None, "2026-09-30 23:50") == "2026-09-30 23:50"
    # last origin bin whose [t+30, t+40) label ends by 23:59 SGT
    assert gcp.cutoff_window_end(cut, None) == "2026-10-19 23:10"
    assert gcp.cutoff_window_end(cut, "2026-10-19 23:10") == "2026-10-19 23:10"
    with pytest.raises(ValueError):
        gcp.cutoff_window_end(cut, "2026-10-19 23:30")


def test_main_rejects_joined_end_after_cutoff_day():
    with pytest.raises(SystemExit):
        gcp.main(["--bqml-only", "--joined", "--data-cutoff", CUT, "--joined-end", "2026-10-20"])


def test_cut_weather_drops_late_rows():
    cut = tsx.parse_data_cutoff(CUT)
    rain = pd.DataFrame({"ts": pd.to_datetime(["2026-10-19T15:55Z", "2026-10-19T16:05Z"], utc=True), "value_mm": [1.0, 2.0]})
    fc = pd.DataFrame({"available_at": pd.to_datetime(["2026-10-19T15:00Z", "2026-10-19T16:30Z"], utc=True), "forecast": ["a", "b"]})
    r, f = jx.cut_weather(rain, fc, cut)
    assert list(r["value_mm"]) == [1.0] and list(f["forecast"]) == ["a"]
    assert jx.cut_weather(rain, fc, None) == (rain, fc)


def test_promote_target_choice(tmp_path, monkeypatch):
    import promote_report_run as prr

    assert prr.TARGETS == ("report", "report-confirm", "parity-bqml")
    seen = {}
    monkeypatch.setattr(prr, "RUNS_ROOT", tmp_path)
    (tmp_path / "r1").mkdir()
    monkeypatch.setattr(prr, "promote", lambda source, dest, allow_dirty=False: seen.update(dest=dest))
    assert prr.main(["r1", "--target", "report-confirm"]) == 0
    assert seen["dest"] == tmp_path / "report-confirm"


def test_replay_refuses_confirm_snapshot():
    from replay_series import SNAPSHOTS, replay_series_csvs

    with pytest.raises(ValueError):
        replay_series_csvs(SNAPSHOTS[1])


def test_confirmation_run_flags_parse(monkeypatch):
    """Run B: local only (--skip-offline --joined --ensemble), no BigQuery ML."""

    def stop(**kwargs):
        raise RuntimeError(f"reached run dir creation with {kwargs['components']}")

    monkeypatch.setattr(gcp, "create_run_dir", stop)
    with pytest.raises(RuntimeError, match=r"\['joined', 'ensemble'\]"):
        gcp.main(["--data-cutoff", CUT, "--skip-offline", "--joined", "--joined-start", "2026-10-01",
                  "--joined-end", "2026-10-19", "--ensemble"])
    with pytest.raises(SystemExit):
        gcp.main(["--ensemble"])  # ensemble needs the joined rows
