import csv

from filter_area_forecast import rows_for_area
from filter_station_history import day_files, rows_for_station


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_rows_for_station_filters_one_station(tmp_path):
    day1 = tmp_path / "rainfall_2026-01-01.csv"
    day2 = tmp_path / "rainfall_2026-01-02.csv"
    _write(day1, ["timestamp", "station_id", "value_mm"], [["2026-01-01T10:00:00", "S210", "1.2"], ["2026-01-01T10:05:00", "S999", "0.0"]])
    _write(day2, ["timestamp", "station_id", "value_mm"], [["2026-01-02T08:00:00", "S210", "0.5"]])

    rows = rows_for_station(sorted([str(day1), str(day2)]), "S210")
    assert len(rows) == 2
    assert all(r[1] == "S210" for r in rows)
    assert rows[0][2] == "1.2"
    assert rows[1][2] == "0.5"


def test_rows_for_station_sorts_and_dedups(tmp_path):
    header = ["timestamp", "station_id", "value_mm"]
    _write(tmp_path / "rainfall_2026-01-02.csv", header, [["2026-01-01T23:55:00", "S210", "0"], ["2026-01-02T00:05:00", "S210", "1"]])
    _write(tmp_path / "rainfall_2026-01-01.csv", header, [["2026-01-01T23:55:00", "S210", "0"]])
    rows = rows_for_station(day_files(str(tmp_path), "rainfall"), "S210")
    assert [r[0] for r in rows] == ["2026-01-01T23:55:00", "2026-01-02T00:05:00"]


def test_day_files_ignore_partial_and_outputs(tmp_path):
    header = ["timestamp", "station_id", "value_mm"]
    for name in ("rainfall_2026-01-01.csv", "rainfall_2026-01-02.partial.csv", "S210_Woodlands_Centre.csv", "rainfall_notes.csv"):
        _write(tmp_path / name, header, [])
    names = [p.split("\\")[-1].split("/")[-1] for p in day_files(str(tmp_path), "rainfall")]
    assert names == ["rainfall_2026-01-01.csv"]
    with_partial = day_files(str(tmp_path), "rainfall", include_partial=True)
    assert len(with_partial) == 2


def test_rows_for_area_dedups_across_midnight_and_ignores_case(tmp_path):
    header = ["issue_timestamp", "valid_start", "valid_end", "area", "forecast", "update_timestamp"]
    late = ["2026-01-01T23:30:00+08:00", "2026-01-01T23:30:00+08:00", "2026-01-02T01:30:00+08:00", "Woodlands", "Fair", "2026-01-01T23:35:00+08:00"]
    _write(tmp_path / "forecast_2026-01-01.csv", header, [late, ["2026-01-01T12:00:00+08:00", "", "", "Yishun", "Rain", ""]])
    _write(tmp_path / "forecast_2026-01-02.csv", header, [["2026-01-02T01:00:00+08:00", "", "", "woodlands", "Showers", ""], late])
    rows = rows_for_area(day_files(str(tmp_path), "forecast"), "WOODLANDS")
    assert [r[0] for r in rows] == ["2026-01-01T23:30:00+08:00", "2026-01-02T01:00:00+08:00"]


def test_rows_for_area_keeps_revisions_and_old_files(tmp_path):
    old_header = ["issue_timestamp", "valid_start", "valid_end", "area", "forecast"]
    _write(tmp_path / "forecast_2026-01-01.csv", old_header, [["2026-01-01T08:00:00+08:00", "", "", "Woodlands", "Fair"]])
    header = old_header + ["update_timestamp"]
    _write(
        tmp_path / "forecast_2026-01-02.csv",
        header,
        [
            ["2026-01-02T08:00:00+08:00", "", "", "Woodlands", "Fair", "2026-01-02T08:05:00+08:00"],
            ["2026-01-02T08:00:00+08:00", "", "", "Woodlands", "Showers", "2026-01-02T08:20:00+08:00"],
        ],
    )
    rows = rows_for_area(day_files(str(tmp_path), "forecast"), "Woodlands")
    assert len(rows) == 3
    assert rows[0][5] == ""  # old file: no acquisition time
    assert [r[4] for r in rows[1:]] == ["Fair", "Showers"]
