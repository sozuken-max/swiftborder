"""Offline checks for the current-duration camera test. No BigQuery."""

from datetime import datetime, timezone

import pandas as pd

from camera_current import attach_counts, label_vs_calendar, october_rows, september_rows


def test_label_uses_the_half_minute_bar():
    assert label_vs_calendar(-1.2, -0.6) == "improvement"
    assert label_vs_calendar(0.6, 1.4) == "regression"
    assert label_vs_calendar(-0.2, 0.3) == "no meaningful gain"
    assert label_vs_calendar(-0.8, 0.2) == "inconclusive"
    assert label_vs_calendar(None, 0.1) == "inconclusive"


def test_missing_frame_is_not_zero_and_october_requires_a_count():
    maps = pd.DataFrame(
        {
            "bin_ts": pd.to_datetime(["2026-10-01 00:00:00+00:00", "2026-10-01 00:10:00+00:00"]),
            "direction": ["SG_TO_MY", "SG_TO_MY"],
            "y_persistence": [20.0, 21.0],
            "after_gap": [0, 0],
            "date_sgt": ["2026-10-01", "2026-10-01"],
        }
    )
    counts = pd.DataFrame(
        {
            "frame_ts": pd.to_datetime(["2026-10-01 00:02:00+00:00"]),
            "sg_my": [4.0],
            "my_sg": [1.0],
        }
    )
    joined = attach_counts(maps, counts)
    scored = october_rows(joined)
    assert len(scored) == 1
    assert scored.iloc[0]["cam_dir"] == 4.0
    assert joined.iloc[1]["cam_dir"] != joined.iloc[1]["cam_dir"] or pd.isna(joined.iloc[1]["cam_dir"])


def test_september_stops_before_october():
    rows = pd.DataFrame(
        {
            "bin_ts": pd.to_datetime(
                ["2026-09-30 15:00:00+00:00", "2026-09-30 16:00:00+00:00"],
                utc=True,
            ),
            "direction": ["SG_TO_MY", "SG_TO_MY"],
            "y_persistence": [10.0, 11.0],
            "cam_dir": [3.0, 4.0],
            "after_gap": [0, 0],
            "date_sgt": ["2026-09-30", "2026-10-01"],
        }
    )
    kept = september_rows(rows)
    assert list(kept["y_persistence"]) == [10.0]
    assert datetime(2026, 10, 1, tzinfo=timezone.utc)
