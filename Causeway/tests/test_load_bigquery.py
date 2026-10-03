"""load_bigquery with a fake BigQuery client (no network)."""

import csv
import datetime as dt

import pytest

import load_bigquery as lb


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def files(tmp_path):
    stations = tmp_path / "stations.csv"
    _write(stations, ["station_id", "name", "latitude", "longitude"], [["S210", "Woodlands Centre Road", 1.4, 103.7]])
    rain = tmp_path / "rain.csv"
    _write(rain, ["timestamp", "station_id", "value_mm"], [["2026-09-01T00:05:00+08:00", "S210", "0.2"], ["2026-09-01T00:00:00+08:00", "S210", "0"]])
    fc = tmp_path / "fc.csv"
    header = ["issue_timestamp", "valid_start", "valid_end", "area", "forecast", "update_timestamp"]
    _write(
        fc,
        header,
        [
            ["2026-09-01T08:00:00+08:00", "2026-09-01T08:00:00+08:00", "2026-09-01T10:00:00+08:00", "Woodlands", "Fair", "2026-09-01T08:06:00+08:00"],
            ["2026-09-01T08:00:00+08:00", "2026-09-01T08:00:00+08:00", "2026-09-01T10:00:00+08:00", "Woodlands", "Showers", "2026-09-01T08:21:00+08:00"],
            ["2026-09-01T08:30:00+08:00", "2026-09-01T08:30:00+08:00", "2026-09-01T10:30:00+08:00", "Woodlands", "Cloudy", ""],
        ],
    )
    return stations, rain, fc


def test_utc_conversion_matches_table_convention():
    assert lb.to_utc_iso("2026-09-01T00:00:00+08:00") == "2026-08-31 16:00:00"
    with pytest.raises(ValueError):
        lb.to_utc_iso("2026-09-01T00:00:00")


def test_s210_keeps_the_table_name_even_if_data_gov_renames_it(files, tmp_path):
    _, rain, _ = files
    renamed = tmp_path / "renamed.csv"
    _write(renamed, ["station_id", "name", "latitude", "longitude"], [["S210", "Woodlands Centre", 1.4, 103.7]])
    assert {r["station_id"] for r in lb.rainfall_rows(str(rain), str(renamed))} == {"Woodlands Centre Road"}


def test_rainfall_rows_use_station_name_and_sort(files):
    stations, rain, _ = files
    rows = lb.rainfall_rows(str(rain), str(stations))
    assert rows[0] == {"timestamp": "2026-08-31 16:00:00", "station_id": "Woodlands Centre Road", "value_mm": 0.0}
    assert rows[1]["value_mm"] == 0.2


def test_rainfall_duplicates_are_refused(files, tmp_path):
    stations, _, _ = files
    dup = tmp_path / "dup.csv"
    _write(dup, ["timestamp", "station_id", "value_mm"], [["2026-09-01T00:00:00+08:00", "S210", "0"]] * 2)
    with pytest.raises(lb.LoadRefused):
        lb.rainfall_rows(str(dup), str(stations))


def test_forecast_keeps_latest_revision_per_issue(files):
    _, _, fc = files
    rows = lb.forecast_rows(str(fc))
    assert len(rows) == 2
    assert rows[0]["forecast"] == "Showers"
    assert rows[0]["update_timestamp"] == "2026-09-01 00:21:00"
    assert rows[1]["update_timestamp"] is None


class FakeJob:
    def __init__(self, rows=None, output_rows=None):
        self._rows = rows or []
        self.output_rows = output_rows

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self, existing=0):
        self.existing = existing
        self.queries, self.loads = [], []

    def query(self, sql, job_config=None, location=None):
        self.queries.append(sql)
        return FakeJob([{"n": self.existing}])

    def get_table(self, name):
        from google.cloud import bigquery

        class T:
            schema = [bigquery.SchemaField("issue_timestamp", "TIMESTAMP")]

        return T()

    def load_table_from_json(self, rows, table, job_config=None, location=None):
        self.loads.append((table, rows, job_config))
        return FakeJob(output_rows=len(rows))


def _plans(files):
    stations, rain, fc = files
    return lambda which: [
        lb.Plan(lb.RAIN_TABLE, "timestamp", lb.rainfall_rows(str(rain), str(stations))),
        lb.Plan(lb.FORECAST_TABLE, "issue_timestamp", lb.forecast_rows(str(fc)), add_update_timestamp=True),
    ]


def test_dry_run_makes_no_calls(files):
    out = lb.run(("rainfall", "forecast"), execute=False, make_plans=_plans(files))
    assert [r["rows"] for r in out] == [2, 2]
    assert "loaded" not in out[0]


def test_execute_snapshots_then_appends(files):
    client = FakeClient()
    out = lb.run(("rainfall", "forecast"), execute=True, client=client, today=dt.date(2026, 10, 1), make_plans=_plans(files))
    assert any("CREATE SNAPSHOT TABLE IF NOT EXISTS `swiftborder.rainfall.rainfall_snapshot_20261001`" in q for q in client.queries)
    assert [t for t, _, _ in client.loads] == [lb.RAIN_TABLE, lb.FORECAST_TABLE]
    rain_rows = client.loads[0][1]
    assert "update_timestamp" not in rain_rows[0]  # rainfall schema unchanged
    fc_cfg = client.loads[1][2]
    assert "update_timestamp" in {f.name for f in fc_cfg.schema}
    assert fc_cfg.write_disposition == "WRITE_APPEND"
    assert out[1]["loaded"] == 2


def test_overlap_is_refused_before_any_write(files):
    client = FakeClient(existing=5)
    with pytest.raises(lb.LoadRefused):
        lb.run(("rainfall",), execute=True, client=client, make_plans=_plans(files))
    assert client.loads == []
    assert not any("SNAPSHOT" in q for q in client.queries)


def test_both_tables_are_checked_before_the_first_snapshot(files):
    client = FakeClient()
    lb.run(("rainfall", "forecast"), execute=True, client=client, today=dt.date(2026, 10, 1), make_plans=_plans(files))
    kinds = ["snapshot" if "SNAPSHOT" in sql else "count" for sql in client.queries]
    assert kinds[:2] == ["count", "count"]
    assert kinds.index("snapshot") == 2


def test_later_table_overlap_refuses_before_any_snapshot(files):
    client = FakeClient()

    def query(sql, job_config=None, location=None):
        client.queries.append(sql)
        existing = 4 if "COUNT" in sql and "weatherforecast" in sql else 0
        return FakeJob([{"n": existing}])

    client.query = query
    with pytest.raises(lb.LoadRefused, match="weatherforecast"):
        lb.run(("rainfall", "forecast"), execute=True, client=client, make_plans=_plans(files))
    assert client.loads == []
    assert not any("SNAPSHOT" in sql for sql in client.queries)
    assert sum("COUNT" in sql for sql in client.queries) == 2
