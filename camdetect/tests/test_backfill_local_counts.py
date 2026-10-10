"""backfill_local_counts: name parsing, the SGT date window, resume. No GCS and no model run."""

import csv
import datetime as dt

import backfill_local_counts as bf
import main


def test_parse_timestamp_formats():
    sgt = dt.datetime(2026, 9, 6, 8, 15, 30)
    for text in ("20260906_081530", "2026-09-06T08:15:30", "2701_2026-09-06_08-15-30", "2026-09-06T00:15:30Z"):
        assert bf.frame_time(f"camera_id=2701/month=2026-09/{text}.jpg") == sgt, text
    assert bf.frame_time("x/2026-09-06T08:15:30+08:00.jpg") == sgt
    assert bf.frame_time("x/1788653730.jpg") == sgt  # unix seconds
    assert bf.frame_time("x/2026-09-06T08:15.jpg") == dt.datetime(2026, 9, 6, 8, 15)


def test_naive_names_follow_name_tz_and_metadata_is_the_fallback():
    utc = dt.timezone.utc
    assert bf.frame_time("x/20260905_160000.jpg", name_tz=utc) == dt.datetime(2026, 9, 6, 0, 0)
    assert bf.frame_time("x/frame.jpg") is None
    assert bf.frame_time("x/frame.jpg", {"capture_timestamp": "2026-09-06T08:00:00+08:00"}) == dt.datetime(
        2026, 9, 6, 8, 0
    )


def test_month_prefixes_cover_the_window_with_a_day_of_padding():
    assert bf.month_prefixes("2701", dt.date(2026, 9, 6), dt.date(2026, 10, 4)) == [
        "camera_id=2701/month=2026-09/",
        "camera_id=2701/month=2026-10/",
    ]
    assert bf.month_prefixes("2701", dt.date(2026, 9, 1), dt.date(2026, 9, 30))[0] == "camera_id=2701/month=2026-08/"


def _tree(tmp_path, names):
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    return bf.LocalSource(tmp_path)


def test_plan_keeps_sgt_dates_inclusive_drops_duplicates_and_reports_unparsed(tmp_path):
    source = _tree(
        tmp_path,
        [
            "camera_id=2701/month=2026-09/2026-09-05T15:59:00Z.jpg",  # 23:59 SGT 5 Sep: out
            "camera_id=2701/month=2026-09/2026-09-05T16:00:00Z.jpg",  # 00:00 SGT 6 Sep: in
            "camera_id=2701/month=2026-09/20260906_160000.jpg",  # 6 Sep 16:00 SGT
            "camera_id=2701/month=2026-09/2026-09-06T08:00:00Z.jpg",  # same instant, stored twice
            "camera_id=2701/month=2026-10/20261004_235500.jpg",  # last bin: in
            "camera_id=2701/month=2026-10/20261005_000000.jpg",  # out
            "camera_id=2701/month=2026-10/frame.jpg",  # no time
            "camera_id=2701/month=2026-10/20261001_100000.txt",  # not an image
            "camera_id=2702/month=2026-09/20260910_100000.jpg",  # other camera
        ],
    )
    frames, unparsed, duplicates = bf.plan_frames(source, "2701", dt.date(2026, 9, 6), dt.date(2026, 10, 4))
    assert [stamp for stamp, _ in frames] == [
        dt.datetime(2026, 9, 6, 0, 0),
        dt.datetime(2026, 9, 6, 16, 0),
        dt.datetime(2026, 10, 4, 23, 55),
    ]
    assert duplicates == 1
    assert unparsed == ["camera_id=2701/month=2026-10/frame.jpg"]


def test_main_writes_integer_counts_and_resumes(tmp_path, monkeypatch):
    root = tmp_path / "bucket"
    _tree(root, ["camera_id=2701/month=2026-09/20260906_080000.jpg", "camera_id=2701/month=2026-09/20260906_081000.jpg"])
    calls = []

    def fake_detect(image_bytes, camera_id, workflow_id=None, **_):
        calls.append(workflow_id)
        kept = [{}] * 7
        return {"kept": kept, "summary": {"sg_my": {"count": 2}, "my_sg": {"count": 5}, "unknown": {"count": 0}}}

    monkeypatch.setattr(main, "detect_frame", fake_detect)
    out = tmp_path / "out.csv"
    args = ["--local-dir", str(root), "--out", str(out), "--start", "2026-09-06", "--end", "2026-09-06"]

    assert bf.main(args + ["--limit", "1"]) == 0
    assert bf.main(args) == 0
    assert calls == [main.LOCAL_MODEL_ID, main.LOCAL_MODEL_ID]  # second run only scored the new frame
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["frame_datetime_sgt"], r["sg_my"], r["my_sg"], r["total"]) for r in rows] == [
        ("2026-09-06T08:00:00", "2", "5", "7"),
        ("2026-09-06T08:10:00", "2", "5", "7"),
    ]
