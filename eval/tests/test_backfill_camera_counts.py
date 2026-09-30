import csv
import datetime as dt

import pytest
import requests

import backfill_camera_counts as bcc


def test_peak_first_ordering_and_counts():
    day = dt.date(2026, 9, 20)
    bins = bcc.planned_bins(day, day)
    peak = [b for b in bins if b.hour in bcc.PEAK_HOURS]
    assert len(peak) == 11 * 6
    assert len(bins) == 11 * 6 + 13 * 3
    # all peak bins come before any off-peak bin
    first_off = next(i for i, b in enumerate(bins) if b.hour not in bcc.PEAK_HOURS)
    assert all(b.hour in bcc.PEAK_HOURS for b in bins[:first_off])
    assert all(b.minute % 20 == 0 for b in bins[first_off:])
    assert len(bcc.planned_bins(day, day, "full")) == 144


def _result(total=3):
    return {
        "kept": [{}] * total,
        "summary": {"sg_my": {"count": 1, "extent": 0.2}, "my_sg": {"count": 2, "extent": 0.1}, "unknown": {"count": 0}},
        "image_size": (1920, 1080),
    }


def _scorer(frame_offset_min=1, detect=None, url="u"):
    def score(b):
        return bcc.score_bin(
            b,
            lookup=lambda x: bcc.Frame(url, x + dt.timedelta(minutes=frame_offset_min) if url else None),
            download=lambda u: b"img",
            detect_frame=detect or (lambda data: _result()),
        )

    return score


BIN = dt.datetime(2026, 9, 20, 8, 0)


def test_score_ok_row():
    row = _scorer()(BIN)
    assert row["status"] == "ok"
    assert (row["total"], row["sg_my"], row["my_sg"]) == (3, 1, 2)
    assert row["inference_called"] == 1


def test_missing_and_stale_frames_are_not_zero_and_not_billed():
    row = _scorer(url=None)(BIN)
    assert row["status"] == "missing_frame" and row["inference_called"] == 0 and "total" not in row
    row = _scorer(frame_offset_min=-45)(BIN)
    assert row["status"] == "stale_frame" and row["inference_called"] == 0


def _http_error(status, text=""):
    resp = requests.Response()
    resp.status_code = status
    resp._content = text.encode()
    return requests.HTTPError(response=resp)


@pytest.mark.parametrize("status,text", [(402, ""), (403, ""), (429, "Monthly credit quota exceeded")])
def test_quota_errors_stop(status, text):
    def detect(data):
        raise _http_error(status, text)

    with pytest.raises(bcc.QuotaExhausted):
        _scorer(detect=detect)(BIN)


def test_plain_rate_limit_is_an_error_row_not_a_stop():
    def detect(data):
        raise _http_error(429, "slow down")

    row = _scorer(detect=detect)(BIN)
    assert row["status"] == "error"


def test_run_resumes_without_duplicates_and_respects_budget(tmp_path):
    out = tmp_path / "c.csv"
    bins = bcc.planned_bins(dt.date(2026, 9, 20), dt.date(2026, 9, 20))
    r1 = bcc.run(bins, out, score=_scorer(), max_calls=5, rate_limit_sec=0)
    assert r1["inference_calls"] == 5 and r1["stopped"] == "max_calls"
    r2 = bcc.run(bins, out, score=_scorer(), max_calls=5, rate_limit_sec=0)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert len({r["bin_sgt"] for r in rows}) == 10
    assert r2["already_done"] == 5


def test_quota_stop_keeps_earlier_rows(tmp_path):
    out = tmp_path / "c.csv"
    calls = {"n": 0}

    def detect(data):
        calls["n"] += 1
        if calls["n"] > 3:
            raise _http_error(402)
        return _result()

    bins = bcc.planned_bins(dt.date(2026, 9, 20), dt.date(2026, 9, 20))
    r = bcc.run(bins, out, score=_scorer(detect=detect), rate_limit_sec=0)
    assert r["stopped"].startswith("quota")
    assert r["written"] == 3
    assert bcc.read_done(out) == {b.isoformat() for b in bins[:3]}


def test_error_rows_are_retried(tmp_path):
    out = tmp_path / "c.csv"
    bins = [BIN]

    def bad(data):
        raise requests.ConnectionError("x")

    bcc.run(bins, out, score=_scorer(detect=bad), rate_limit_sec=0)
    assert bcc.read_done(out) == set()
    bcc.run(bins, out, score=_scorer(), rate_limit_sec=0)
    assert bcc.read_done(out) == {BIN.isoformat()}
    assert bcc.coverage(out) == [("2026-09-20", 1, 1)]


def test_dry_run_makes_no_calls(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bcc, "_load_camdetect", lambda: pytest.fail("should not load camdetect"))
    code = bcc.main(["--dry-run", "--start", "2026-09-20", "--end", "2026-09-20", "--out", str(tmp_path / "c.csv")])
    assert code == 0
    assert "105 bins planned" in capsys.readouterr().out


def test_lookup_frame_parses_timestamp():
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": [{"timestamp": "2026-09-20T08:00:30+08:00", "cameras": [{"camera_id": "2701", "image": "u", "timestamp": "2026-09-20T07:59:10+08:00"}]}]}

    frame = bcc.lookup_frame(BIN, get=lambda *a, **k: R())
    assert frame.url == "u"
    assert frame.timestamp == dt.datetime(2026, 9, 20, 7, 59, 10)
