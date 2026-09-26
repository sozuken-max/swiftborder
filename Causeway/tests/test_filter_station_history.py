import csv
import os

from filter_station_history import rows_for_station


def test_rows_for_station_filters_one_station(tmp_path):
    day1 = tmp_path / "rainfall_2026-01-01.csv"
    day2 = tmp_path / "rainfall_2026-01-02.csv"
    with open(day1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "station_id", "value_mm"])
        w.writerow(["2026-01-01T10:00:00", "S210", "1.2"])
        w.writerow(["2026-01-01T10:05:00", "S999", "0.0"])
    with open(day2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "station_id", "value_mm"])
        w.writerow(["2026-01-02T08:00:00", "S210", "0.5"])

    rows = rows_for_station(sorted([str(day1), str(day2)]), "S210")
    assert len(rows) == 2
    assert all(r[1] == "S210" for r in rows)
    assert rows[0][2] == "1.2"
    assert rows[1][2] == "0.5"
