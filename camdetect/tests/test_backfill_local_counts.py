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
            "camera_id=2701/month=2026-09/2026-09-05T15-59-00Z.jpg",  # 23:59 SGT 5 Sep: out
            "camera_id=2701/month=2026-09/2026-09-05T16-00-00Z.jpg",  # 00:00 SGT 6 Sep: in
            "camera_id=2701/month=2026-09/20260906_160000.jpg",  # 6 Sep 16:00 SGT
            "camera_id=2701/month=2026-09/2026-09-06T08-00-00Z.jpg",  # same instant, stored twice
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


def test_task_shard_splits_work_across_cloud_run_tasks():
    todo = list(range(10))
    assert bf.task_shard(todo, {}) == (todo, 0, 1)
    shards = [bf.task_shard(todo, {"CLOUD_RUN_TASK_INDEX": str(i), "CLOUD_RUN_TASK_COUNT": "3"})[0] for i in range(3)]
    assert sorted(sum(shards, [])) == todo and shards[1] == [1, 4, 7]


def test_make_row_matches_the_bigquery_schema():
    row = bf.make_row("2701", dt.datetime(2026, 9, 6, 8, 0), {"sg_my": 2, "my_sg": 5, "unknown": 0, "total": 7},
                      "gs://b/x.jpg", dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc))
    assert set(row) == {name for name, _, _ in bf.BQ_SCHEMA}
    assert row["frame_datetime_sgt"] == "2026-09-06T08:00:00"
    assert row["frame_ts"] == "2026-09-06T00:00:00+00:00"
    assert row["model_id"] == main.LOCAL_MODEL_ID


def test_bigquery_schema_matches_the_checked_in_ddl():
    ddl = (bf.HERE.parent / "sql" / "bigquery" / "cam2701" / "local_counts.sql").read_text()
    for name, kind, mode in bf.BQ_SCHEMA:
        assert f"  {name} {kind}{' NOT NULL' if mode == 'REQUIRED' else ''}" in ddl, name


class FakeBigQuery:
    def __init__(self, existing=()):
        self.created, self.loads, self.existing = [], [], set(existing)

    def create_table(self, table, exists_ok=False):
        self.created.append((table, exists_ok))

    def query(self, sql, job_config=None):
        rows = [{"source": s} for s in self.existing]
        return type("Job", (), {"result": lambda self: rows})()

    def load_table_from_json(self, rows, table_id, job_config=None):
        self.loads.append((list(rows), table_id, job_config.write_disposition))
        return type("Job", (), {"result": lambda self: None})()


def test_bigquery_sink_creates_the_table_reads_done_and_appends():
    pytest = __import__("pytest")
    pytest.importorskip("google.cloud.bigquery")
    client = FakeBigQuery(existing={"gs://b/a.jpg"})
    sink = bf.BigQuerySink("swiftborder.cam2701.local_counts", client=client)
    (table, exists_ok), = client.created
    assert exists_ok and [f.name for f in table.schema] == [name for name, _, _ in bf.BQ_SCHEMA]
    assert sink.done("2701") == {"gs://b/a.jpg"}
    sink.write([{"source": "gs://b/b.jpg"}])
    assert client.loads == [([{"source": "gs://b/b.jpg"}], "swiftborder.cam2701.local_counts", "WRITE_APPEND")]


def test_main_with_bq_table_batches_rows_and_skips_scored_frames(tmp_path, monkeypatch):
    root = tmp_path / "bucket"
    source = _tree(root, [f"camera_id=2701/month=2026-09/20260906_08{m}000.jpg" for m in range(5)])

    class Sink:
        batch_size = 2
        writes = []

        def __init__(self, table_id):
            self.table_id = table_id

        def done(self, camera_id):
            return {source.uri("camera_id=2701/month=2026-09/20260906_080000.jpg")}

        def write(self, rows):
            Sink.writes.append([r["frame_datetime_sgt"] for r in rows])

    monkeypatch.setattr(bf, "BigQuerySink", Sink)
    monkeypatch.setattr(main, "detect_frame", lambda *a, **k: {
        "kept": [{}], "summary": {"sg_my": {"count": 1}, "my_sg": {"count": 0}, "unknown": {"count": 0}}})
    args = ["--local-dir", str(root), "--bq-table", "p.d.t", "--start", "2026-09-06", "--end", "2026-09-06"]
    assert bf.main(args) == 0
    assert Sink.writes == [
        ["2026-09-06T08:10:00", "2026-09-06T08:20:00"],
        ["2026-09-06T08:30:00", "2026-09-06T08:40:00"],
    ]
