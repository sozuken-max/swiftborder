"""Mocked-session tests for the data.gov.sg day fetcher (no network)."""

import csv
import datetime
import os

import pytest
import requests

import datagov
import fetch_forecast_history as ffh
import fetch_rainfall_history as frh

URL = "https://example.test/api"


class Resp:
    def __init__(self, status=200, payload=None, headers=None, bad_json=False):
        self.status_code = status
        self._payload = payload if payload is not None else {"code": 0, "data": {}}
        self.headers = headers or {}
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._payload


class Session:
    """Returns queued responses (or raises queued exceptions) in order; records params."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((dict(params or {}), dict(headers or {})))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def page(data, token=None):
    d = dict(data)
    if token:
        d["paginationToken"] = token
    return Resp(200, {"code": 0, "data": d})


NO_SLEEP = dict(sleep=lambda s: None)


# --- retries ------------------------------------------------------------------------


def test_retry_after_seconds_and_http_date():
    now = datetime.datetime(2026, 9, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)
    assert datagov.parse_retry_after("7", 1.0) == 7.0
    assert datagov.parse_retry_after("Wed, 30 Sep 2026 12:00:10 GMT", 1.0, now=now) == 10.0
    assert datagov.parse_retry_after("garbage", 3.0) == 3.0
    assert datagov.parse_retry_after(None, 2.0) == 2.0


def test_429_then_success_honours_retry_after():
    waits = []
    s = Session(Resp(429, headers={"Retry-After": "4"}), page({"x": 1}))
    resp = datagov.get_with_retries(s, URL, {"date": "d"}, sleep=waits.append)
    assert resp.status_code == 200
    assert waits == [4.0]


def test_5xx_is_retried():
    s = Session(Resp(502), Resp(503), page({}))
    assert datagov.get_with_retries(s, URL, {}, **NO_SLEEP).status_code == 200
    assert len(s.calls) == 3


def test_network_errors_are_retried_then_give_up():
    s = Session(*[requests.ConnectionError("boom")] * 3)
    with pytest.raises(datagov.FetchError, match="gave up after 3"):
        datagov.get_with_retries(s, URL, {}, max_retries=3, **NO_SLEEP)


def test_exhausted_429_raises_instead_of_returning_none():
    s = Session(*[Resp(429)] * 3)
    with pytest.raises(datagov.FetchError):
        datagov.get_with_retries(s, URL, {}, max_retries=3, **NO_SLEEP)


def test_400_is_fatal_without_retry():
    s = Session(Resp(400, {"code": 4, "errorMsg": "Invalid pagination token"}))
    with pytest.raises(datagov.FetchError, match="Invalid pagination token"):
        datagov.get_with_retries(s, URL, {}, **NO_SLEEP)
    assert len(s.calls) == 1


def test_api_key_header_is_optional(monkeypatch):
    monkeypatch.setenv("DATAGOV_API_KEY", "k")
    s = Session(page({}))
    datagov.get_with_retries(s, URL, {}, **NO_SLEEP)
    assert s.calls[0][1] == {"x-api-key": "k"}
    monkeypatch.delenv("DATAGOV_API_KEY")
    s = Session(page({}))
    datagov.get_with_retries(s, URL, {}, **NO_SLEEP)
    assert s.calls[0][1] == {}


# --- pagination ---------------------------------------------------------------------


def test_pagination_follows_tokens():
    s = Session(page({"n": 1}, "t1"), page({"n": 2}, "t2"), page({"n": 3}))
    pages = datagov.fetch_day_pages(s, URL, "2026-09-05", **NO_SLEEP)
    assert [p["n"] for p in pages] == [1, 2, 3]
    assert s.calls[1][0]["paginationToken"] == "t1"
    assert all(c[0]["date"] == "2026-09-05" for c in s.calls)


def test_first_page_404_means_no_data():
    assert datagov.fetch_day_pages(Session(Resp(404)), URL, "d", **NO_SLEEP) == []


def test_later_page_404_is_a_failure():
    with pytest.raises(datagov.FetchError, match="404 on page 2"):
        datagov.fetch_day_pages(Session(page({}, "t1"), Resp(404)), URL, "d", **NO_SLEEP)


def test_repeated_token_is_a_failure():
    with pytest.raises(datagov.FetchError, match="repeated"):
        datagov.fetch_day_pages(Session(page({}, "t"), page({}, "t"), page({}, "t")), URL, "d", **NO_SLEEP)


def test_nonzero_code_and_bad_json_are_failures():
    with pytest.raises(datagov.FetchError, match="API code 17"):
        datagov.fetch_day_pages(Session(Resp(200, {"code": 17, "errorMsg": "x"})), URL, "d", **NO_SLEEP)
    with pytest.raises(datagov.FetchError, match="non-JSON"):
        datagov.fetch_day_pages(Session(Resp(200, bad_json=True)), URL, "d", **NO_SLEEP)


# --- dates ----------------------------------------------------------------------------


def test_resolve_dates():
    today = datetime.date(2026, 9, 30)
    assert datagov.resolve_dates(3, None, None, today=today) == (datetime.date(2026, 9, 28), today)
    assert datagov.resolve_dates(1, "2026-09-05", "2026-09-29", today=today)[0] == datetime.date(2026, 9, 5)
    for bad in [(0, None, None), (1, "2026-09-10", "2026-09-05"), (1, None, "2026-10-01")]:
        with pytest.raises(ValueError):
            datagov.resolve_dates(*bad, today=today)


# --- backfill ------------------------------------------------------------------------


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_failed_day_writes_nothing_and_is_retried_next_run(tmp_path):
    days = [datetime.date(2026, 9, 5), datetime.date(2026, 9, 6)]
    today = datetime.date(2026, 9, 30)

    def flaky(ds):
        if ds == "2026-09-06":
            raise datagov.FetchError("boom")
        return [("t", "S1", "0.2")], {}

    report = datagov.run_backfill(days, output_dir=str(tmp_path), prefix="rainfall", header=frh.HEADER, fetch_rows=flaky, today=today)
    assert [o.status for o in report.outcomes] == ["fetched", "failed"]
    assert not (tmp_path / "rainfall_2026-09-06.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))

    calls = []

    def ok(ds):
        calls.append(ds)
        return [("t", "S1", "0.1")], {}

    report = datagov.run_backfill(days, output_dir=str(tmp_path), prefix="rainfall", header=frh.HEADER, fetch_rows=ok, today=today)
    assert calls == ["2026-09-06"]  # complete day skipped, failed day retried
    assert [o.status for o in report.outcomes] == ["skipped", "fetched"]


def test_today_is_skipped_or_written_as_partial(tmp_path):
    today = datetime.date(2026, 9, 30)
    rows = lambda ds: ([("t", "S1", "1")], {})  # noqa: E731
    r = datagov.run_backfill([today], output_dir=str(tmp_path), prefix="rainfall", header=frh.HEADER, fetch_rows=rows, today=today)
    assert r.outcomes[0].status == "skipped_today"
    assert not list(tmp_path.iterdir())
    r = datagov.run_backfill(
        [today], output_dir=str(tmp_path), prefix="rainfall", header=frh.HEADER, fetch_rows=rows, today=today, allow_partial=True
    )
    assert r.outcomes[0].status == "partial"
    assert (tmp_path / "rainfall_2026-09-30.partial.csv").exists()
    assert not (tmp_path / "rainfall_2026-09-30.csv").exists()
    # Next day the complete file replaces the partial one.
    r = datagov.run_backfill(
        [today], output_dir=str(tmp_path), prefix="rainfall", header=frh.HEADER, fetch_rows=rows, today=today + datetime.timedelta(days=1)
    )
    assert r.outcomes[0].status == "fetched"
    assert (tmp_path / "rainfall_2026-09-30.csv").exists()
    assert not (tmp_path / "rainfall_2026-09-30.partial.csv").exists()


def test_no_data_day_is_marked_and_retried(tmp_path):
    day = [datetime.date(2026, 9, 5)]
    kw = dict(output_dir=str(tmp_path), prefix="forecast", header=ffh.HEADER, today=datetime.date(2026, 9, 30))
    r = datagov.run_backfill(day, fetch_rows=lambda ds: ([], {}), **kw)
    assert r.outcomes[0].status == "no_data"
    assert not (tmp_path / "forecast_2026-09-05.csv").exists()
    assert (tmp_path / "forecast_2026-09-05.nodata").exists()
    # Data arrives later: the next run fetches it and removes the marker.
    r = datagov.run_backfill(day, fetch_rows=lambda ds: ([("t", "v", "v", "Woodlands", "Fair", "u")], {}), **kw)
    assert r.outcomes[0].status == "fetched"
    assert (tmp_path / "forecast_2026-09-05.csv").exists()
    assert not (tmp_path / "forecast_2026-09-05.nodata").exists()


def test_atomic_write_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "x.csv")
    datagov.atomic_write_csv(path, ["a"], [[1], [2]])
    assert _read(path) == [["a"], ["1"], ["2"]]
    assert not os.path.exists(path + ".tmp")


def test_metadata_merge(tmp_path):
    path = str(tmp_path / "stations.csv")
    datagov.merge_metadata_csv(path, frh.STATION_HEADER, "station_id", {"S2": ("B", 1, 2)})
    datagov.merge_metadata_csv(path, frh.STATION_HEADER, "station_id", {"S1": ("A", 3, 4), "S2": ("B2", 1, 2)})
    assert _read(path) == [frh.STATION_HEADER, ["S1", "A", "3", "4"], ["S2", "B2", "1", "2"]]


# --- parsers --------------------------------------------------------------------------


def test_rainfall_parse_pages():
    pages = [
        {
            "stations": [{"id": "S210", "name": "Woodlands Centre", "location": {"latitude": 1.4, "longitude": 103.8}}],
            "readings": [{"timestamp": "2026-09-05T08:05:00+08:00", "data": [{"stationId": "S210", "value": 0.2}]}],
        },
        {"stations": [{"name": "no id"}], "readings": [{"timestamp": "2026-09-05T08:00:00+08:00", "data": [{"stationId": "S210", "value": 0}]}]},
    ]
    readings, stations = frh.parse_pages(pages)
    assert readings == [("2026-09-05T08:00:00+08:00", "S210", 0), ("2026-09-05T08:05:00+08:00", "S210", 0.2)]
    assert stations == {"S210": ("Woodlands Centre", 1.4, 103.8)}


def test_forecast_parse_pages_dedups_and_accepts_dict_text():
    item = {
        "timestamp": "2026-09-05T08:00:00+08:00",
        "valid_period": {"start": "2026-09-05T08:00:00+08:00", "end": "2026-09-05T10:00:00+08:00"},
        "forecasts": [{"area": "Woodlands", "forecast": "Showers"}, {"area": "Yishun", "forecast": {"code": "FW", "text": "Fair"}}],
    }
    item["update_timestamp"] = "2026-09-05T08:05:54+08:00"
    records, areas = ffh.parse_pages([{"items": [item], "area_metadata": [{"name": "Woodlands", "label_location": {"latitude": 1, "longitude": 2}}]}, {"items": [item]}])
    assert len(records) == 2
    assert records[1][4] == "Fair"
    assert records[0][5] == "2026-09-05T08:05:54+08:00"  # acquisition time kept for causal joins
    assert areas == {"Woodlands": (1, 2)}


def test_rainfall_main_end_to_end_with_fake_session(tmp_path, monkeypatch):
    day_payload = {
        "stations": [{"id": "S210", "name": "W", "location": {"latitude": 1, "longitude": 2}}],
        "readings": [{"timestamp": "2026-09-05T08:00:00+08:00", "data": [{"stationId": "S210", "value": 0.4}]}],
    }
    monkeypatch.setattr(frh.requests, "Session", lambda: Session(page(day_payload)))
    code = frh.main(["--start-date", "2026-09-05", "--end-date", "2026-09-05", "--output-dir", str(tmp_path), "--sleep", "0"])
    assert code == 0
    assert _read(tmp_path / "rainfall_2026-09-05.csv")[1] == ["2026-09-05T08:00:00+08:00", "S210", "0.4"]
    assert (tmp_path / "stations.csv").exists()


def test_main_exit_code_is_1_when_a_day_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(frh.requests, "Session", lambda: Session(Resp(400)))
    code = frh.main(["--start-date", "2026-09-05", "--end-date", "2026-09-05", "--output-dir", str(tmp_path), "--sleep", "0"])
    assert code == 1
    assert not (tmp_path / "rainfall_2026-09-05.csv").exists()
