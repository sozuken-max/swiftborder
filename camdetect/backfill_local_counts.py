#!/usr/bin/env python3
"""Retro-score stored camera frames with the in-container model (``model=local``), offline.

Lists the frames for one camera under ``gs://sg-lta-traffic-cameras/camera_id=<id>/month=<YYYY-MM>/``
(or a local mirror of that tree), keeps the ones whose capture time falls between ``--start`` and
``--end`` inclusive (SGT dates), and scores each with ``main.detect_frame(..., workflow_id=
main.LOCAL_MODEL_ID)``. That is the same code path as the live ``model=local`` request behind the
front end's "Local model" pane: YOLO26s ONNX, ``LOCAL_DEFAULT_CONFIDENCE``, overlap dedupe
(``LOCAL_MAX_OVERLAP`` / ``LOCAL_SIZE_RATIO``) and the per-camera dividing line that splits SG-MY
from MY-SG. No Roboflow call is made, so nothing is billed.

One row per frame: capture time (SGT) and integer counts per direction, written to a CSV (default)
or, with ``--bq-table``, to a BigQuery table that is created with ``BQ_SCHEMA`` if it is missing
(DDL: ``sql/bigquery/cam2701/local_counts.sql``). Rows are saved as frames are scored, so an
interrupted run resumes where it stopped (frames already in the output are skipped). A frame that
fails to download is reported and left out, so a re-run retries it. A stored JPEG that cannot be
decoded (truncated upload) is logged as BROKEN and skipped: a retry cannot fix it.

With ``--gcs-out gs://BUCKET/FOLDER`` each frame is instead saved as an annotated JPEG, drawn by the
service's own ``format=directional`` code (the image the front end shows: boxes coloured by
direction, the dividing line, and the SG to MY / MY to SG count banner), at
``FOLDER/camera_id=<id>/month=<YYYY-MM>/<source file name>``. The counts ride along as object
metadata. Frames whose image already exists there are skipped.

On a Cloud Run Job with N tasks, task i scores every N-th remaining frame
(``CLOUD_RUN_TASK_INDEX`` / ``CLOUD_RUN_TASK_COUNT``), so the tasks split the work between them.

    pip install -r requirements-backfill.txt
    gcloud auth application-default login          # storage read on the bucket (+ BigQuery write)
    python backfill_local_counts.py --dry-run      # list and parse names, no inference
    python backfill_local_counts.py                # 6 Sep - 4 Oct 2026, camera 2701, CSV
    python backfill_local_counts.py --bq-table swiftborder.cam2701.local_counts
    python backfill_local_counts.py --gcs-out gs://sg-lta-traffic-cameras/processed/local/

Or score a local copy (``gsutil -m cp -r gs://sg-lta-traffic-cameras/camera_id=2701 frames/``)
with ``--local-dir frames``. The object names must carry the capture time; ``--dry-run`` prints a
sample of names next to the parsed time so the parse can be checked before a full run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import io
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import main as camdetect

HERE = Path(__file__).resolve().parent
SGT = ZoneInfo("Asia/Singapore")
DEFAULT_BUCKET = "sg-lta-traffic-cameras"
DEFAULT_START = "2026-09-06"
DEFAULT_END = "2026-10-04"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
FIELDS = ["frame_datetime_sgt", "sg_my", "my_sg", "unknown", "total", "source"]
# (name, type, mode). Keep in step with sql/bigquery/cam2701/local_counts.sql.
BQ_SCHEMA = [
    ("camera_id", "STRING", "REQUIRED"),
    ("frame_datetime_sgt", "DATETIME", "REQUIRED"),
    ("frame_ts", "TIMESTAMP", "REQUIRED"),
    ("sg_my", "INT64", "REQUIRED"),
    ("my_sg", "INT64", "REQUIRED"),
    ("unknown", "INT64", "REQUIRED"),
    ("total", "INT64", "REQUIRED"),
    ("model_id", "STRING", "REQUIRED"),
    ("min_confidence", "FLOAT64", "REQUIRED"),
    ("max_overlap", "FLOAT64", "NULLABLE"),
    ("source", "STRING", "REQUIRED"),
    ("scored_at", "TIMESTAMP", "REQUIRED"),
]

# 2026-09-06T08:15:30, 20260906_081530, 2026-09-06 08-15-30, optional Z / +08:00 suffix.
_STAMP = re.compile(
    r"(?<!\d)(\d{4})-?(\d{2})-?(\d{2})[T_ \-]?(\d{2})[:\-_]?(\d{2})(?:[:\-_]?(\d{2}))?(?:\.\d+)?"
    r"(Z|[+\-]\d{2}:?\d{2})?(?!\d)"
)
# Unix seconds or milliseconds, as a whole path segment or file stem.
_EPOCH = re.compile(r"(?<![\d])(\d{10}|\d{13})(?![\d])")
_METADATA_KEYS = ("capture_timestamp", "timestamp", "date_time", "datetime", "captured_at")


def _default_out(start, end):
    return HERE / "backfill" / f"cam2701_local_counts_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def parse_timestamp(text, name_tz=SGT):
    """Aware datetime from a name or metadata value, or None. A naive stamp is read in ``name_tz``."""
    if not text:
        return None
    for match in _STAMP.finditer(text):
        year, month, day, hour, minute, second, zone = match.groups()
        try:
            stamp = dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second or 0))
        except ValueError:
            continue
        if not 2000 <= stamp.year <= 2100:
            continue
        if zone is None:
            return stamp.replace(tzinfo=name_tz)
        if zone == "Z":
            return stamp.replace(tzinfo=dt.timezone.utc)
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        offset = dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
        return stamp.replace(tzinfo=dt.timezone(sign * offset))
    for match in _EPOCH.finditer(text):
        value = int(match.group(1))
        seconds = value / 1000 if len(match.group(1)) == 13 else value
        stamp = dt.datetime.fromtimestamp(seconds, dt.timezone.utc)
        if 2000 <= stamp.year <= 2100:
            return stamp
    return None


def frame_time(name, metadata=None, name_tz=SGT):
    """Capture time in SGT (naive) from the file name, then the path, then object metadata."""
    base = name.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    candidates = [stem, name.rsplit("/", 1)[0] + "/" + stem if "/" in name else None]
    candidates += [(metadata or {}).get(key) for key in _METADATA_KEYS]
    for text in candidates:
        stamp = parse_timestamp(text, name_tz)
        if stamp is not None:
            return stamp.astimezone(SGT).replace(tzinfo=None)
    return None


def month_prefixes(camera_id, start, end):
    """Hive-style month partitions covering start..end, padded a day each side for UTC names."""
    first = (start - dt.timedelta(days=1)).replace(day=1)
    last = end + dt.timedelta(days=1)
    months, cur = [], first
    while cur <= last:
        months.append(f"camera_id={camera_id}/month={cur:%Y-%m}/")
        cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return months


class GcsSource:
    def __init__(self, bucket):
        from google.cloud import storage  # imported here so --local-dir runs without it

        self.bucket_name = bucket
        self.bucket = storage.Client().bucket(bucket)

    def list(self, prefix):
        for blob in self.bucket.client.list_blobs(self.bucket, prefix=prefix):
            yield blob.name, blob.metadata or {}

    def uri(self, name):
        return f"gs://{self.bucket_name}/{name}"

    def read(self, name):
        return self.bucket.blob(name).download_as_bytes()


class LocalSource:
    """A local copy of the bucket tree: ``<root>/camera_id=2701/month=2026-09/...``."""

    def __init__(self, root):
        self.root = Path(root)

    def list(self, prefix):
        base = self.root / prefix
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file():
                yield path.relative_to(self.root).as_posix(), {}

    def uri(self, name):
        return str(self.root / name)

    def read(self, name):
        return (self.root / name).read_bytes()


def plan_frames(source, camera_id, start, end, name_tz=SGT):
    """Frames to score in time order, plus names that were images but had no readable time.

    Two objects with the same capture time are one frame stored twice; the first name wins.
    """
    lo = dt.datetime.combine(start, dt.time())
    hi = dt.datetime.combine(end + dt.timedelta(days=1), dt.time())
    frames, unparsed, duplicates = {}, [], 0
    for prefix in month_prefixes(camera_id, start, end):
        for name, metadata in source.list(prefix):
            if not name.lower().endswith(IMAGE_SUFFIXES):
                continue
            stamp = frame_time(name, metadata, name_tz)
            if stamp is None:
                unparsed.append(name)
            elif lo <= stamp < hi:
                if stamp in frames:
                    duplicates += 1
                else:
                    frames[stamp] = name
    return sorted(frames.items()), unparsed, duplicates


class CsvSink:
    batch_size = 1  # one row per write: a killed run loses at most the frame in flight
    wants_image = False

    def __init__(self, path):
        self.path = Path(path)

    def __str__(self):
        return str(self.path)

    def key(self, uri, name):
        return uri

    def done(self, camera_id):
        if not self.path.exists():
            return set()
        with open(self.path, newline="", encoding="utf-8") as f:
            return {row["source"] for row in csv.DictReader(f)}

    def write(self, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerows(rows)


class BigQuerySink:
    """Appends rows with load jobs (free, no streaming buffer), creating the table if missing."""

    wants_image = False

    def __init__(self, table_id, batch_size=100, client=None):
        from google.cloud import bigquery  # imported here so CSV runs need no BigQuery package

        self.bigquery = bigquery
        self.client = client or bigquery.Client(project=table_id.split(".")[0])
        self.table_id = table_id
        self.batch_size = batch_size
        self.schema = [bigquery.SchemaField(name, kind, mode=mode) for name, kind, mode in BQ_SCHEMA]
        table = bigquery.Table(table_id, schema=self.schema)
        table.clustering_fields = ["camera_id"]
        table.description = "Per-frame model=local vehicle counts by direction (camdetect/backfill_local_counts.py)."
        self.client.create_table(table, exists_ok=True)

    def __str__(self):
        return self.table_id

    def key(self, uri, name):
        return uri

    def done(self, camera_id):
        job = self.client.query(
            f"SELECT DISTINCT source FROM `{self.table_id}` WHERE camera_id = @camera_id",
            job_config=self.bigquery.QueryJobConfig(
                query_parameters=[self.bigquery.ScalarQueryParameter("camera_id", "STRING", camera_id)]
            ),
        )
        return {row["source"] for row in job.result()}

    def write(self, rows):
        config = self.bigquery.LoadJobConfig(
            schema=self.schema,
            write_disposition="WRITE_APPEND",
            source_format="NEWLINE_DELIMITED_JSON",
        )
        self.client.load_table_from_json(rows, self.table_id, job_config=config).result()


class GcsImageSink:
    """Uploads one annotated JPEG per frame, at the source frame's path under ``gs://BUCKET/FOLDER``."""

    batch_size = 1
    wants_image = True
    METADATA = ("frame_datetime_sgt", "sg_my", "my_sg", "unknown", "total", "model_id",
                "min_confidence", "max_overlap", "source")

    def __init__(self, uri, client=None):
        match = re.fullmatch(r"gs://([^/]+)/?(.*)", uri)
        if not match:
            raise ValueError(f"--gcs-out must look like gs://BUCKET/FOLDER, got {uri!r}")
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        self.client = client
        self.bucket = client.bucket(match.group(1))
        folder = match.group(2).strip("/")
        self.prefix = folder + "/" if folder else ""

    def __str__(self):
        return f"gs://{self.bucket.name}/{self.prefix}"

    def key(self, uri, name):
        return name  # the source object name, e.g. camera_id=2701/month=2026-09/2701_...jpg

    def done(self, camera_id):
        start = len(self.prefix)
        blobs = self.client.list_blobs(self.bucket, prefix=f"{self.prefix}camera_id={camera_id}/")
        return {blob.name[start:] for blob in blobs}

    def write(self, rows):
        for row in rows:
            blob = self.bucket.blob(self.prefix + row["object_name"])
            blob.metadata = {k: "" if row[k] is None else str(row[k]) for k in self.METADATA}
            blob.upload_from_string(row["annotated_jpeg"], content_type="image/jpeg")


def render_directional(image_bytes, result):
    """The service's ``format=directional`` JPEG for this frame (same drawing code and quality)."""
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    camdetect._draw_directional(image, result["kept"], result["points"], result["summary"])
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def task_shard(todo, environ=None):
    """This Cloud Run task's share of ``todo`` (all of it outside a multi-task job)."""
    environ = os.environ if environ is None else environ
    count = int(environ.get("CLOUD_RUN_TASK_COUNT") or 1)
    index = int(environ.get("CLOUD_RUN_TASK_INDEX") or 0)
    return todo[index::count] if count > 1 else todo, index, count


def make_row(camera_id, stamp, counts, source_uri, scored_at):
    return {
        "camera_id": str(camera_id),
        "frame_datetime_sgt": stamp.isoformat(),
        "frame_ts": stamp.replace(tzinfo=SGT).astimezone(dt.timezone.utc).isoformat(),
        **counts,
        "model_id": camdetect.LOCAL_MODEL_ID,
        "min_confidence": camdetect.LOCAL_DEFAULT_CONFIDENCE,
        "max_overlap": camdetect.LOCAL_MAX_OVERLAP if camdetect.LOCAL_MAX_OVERLAP < 1 else None,
        "source": source_uri,
        "scored_at": scored_at.isoformat(),
    }


def score(image_bytes, camera_id):
    """(counts, detect_frame result) for one frame."""
    result = camdetect.detect_frame(image_bytes, camera_id, workflow_id=camdetect.LOCAL_MODEL_ID)
    summary = result["summary"]
    counts = {
        "sg_my": int(summary["sg_my"]["count"]),
        "my_sg": int(summary["my_sg"]["count"]),
        "unknown": int(summary["unknown"]["count"]),
        "total": len(result["kept"]),
    }
    return counts, result


def report_plan(frames, unparsed, duplicates, start, end):
    per_day = {}
    for stamp, _ in frames:
        per_day[stamp.date()] = per_day.get(stamp.date(), 0) + 1
    day = start
    print("frames per SGT day:")
    while day <= end:
        print(f"  {day}  {per_day.get(day, 0)}")
        day += dt.timedelta(days=1)
    print(f"total {len(frames)} frames, {duplicates} duplicate timestamps dropped, {len(unparsed)} unparsed names")
    for stamp, name in frames[:3] + frames[max(3, len(frames) - 3):]:
        print(f"  {stamp:%Y-%m-%d %H:%M:%S} SGT  <-  {name}")
    for name in unparsed[:5]:
        print(f"  no time in: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", default="2701")
    parser.add_argument("--start", default=DEFAULT_START, help="first SGT date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="last SGT date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--local-dir", help="score a local copy of the bucket tree instead of GCS")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out", type=Path, help="CSV path (default camdetect/backfill/...)")
    output.add_argument("--bq-table", help="write to BigQuery PROJECT.DATASET.TABLE instead of a CSV")
    output.add_argument("--gcs-out", help="save annotated JPEGs under gs://BUCKET/FOLDER instead of a CSV")
    parser.add_argument(
        "--name-tz",
        default="Asia/Singapore",
        help="zone of capture times in object names that carry no offset (default Asia/Singapore)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list and parse frames; no inference")
    parser.add_argument("--limit", type=int, help="score at most N new frames this run")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (inference is serial)")
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    source = LocalSource(args.local_dir) if args.local_dir else GcsSource(args.bucket)

    frames, unparsed, duplicates = plan_frames(source, args.camera, start, end, ZoneInfo(args.name_tz))
    report_plan(frames, unparsed, duplicates, start, end)
    if args.dry_run:
        return 0
    if not frames:
        print("nothing to score", file=sys.stderr)
        return 1

    if args.gcs_out:
        sink = GcsImageSink(args.gcs_out)
    elif args.bq_table:
        sink = BigQuerySink(args.bq_table)
    else:
        sink = CsvSink(args.out or _default_out(start, end))
    key = getattr(sink, "key", lambda uri, name: uri)
    wants_image = getattr(sink, "wants_image", False)
    done = sink.done(args.camera)
    todo = [(stamp, name) for stamp, name in frames if key(source.uri(name), name) not in done]
    print(f"{len(frames) - len(todo)} already in {sink}; {len(todo)} to score")
    todo, task_index, task_count = task_shard(todo)
    if task_count > 1:
        print(f"task {task_index} of {task_count}: {len(todo)} frames")
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"model {camdetect.LOCAL_MODEL_ID}, confidence {camdetect.LOCAL_DEFAULT_CONFIDENCE}, "
        f"overlap {camdetect.LOCAL_MAX_OVERLAP}, size ratio {camdetect.LOCAL_SIZE_RATIO}"
    )

    failures = broken = 0
    pending = []
    began = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            # Downloads run ahead of inference; map keeps time order so the CSV stays sorted.
            downloads = pool.map(lambda item: _fetch(source, item[1]), todo)
            for i, ((stamp, name), (image_bytes, error)) in enumerate(zip(todo, downloads), 1):
                if error is None:
                    try:
                        counts, result = score(image_bytes, args.camera)
                        annotated = render_directional(image_bytes, result) if wants_image else None
                    except camdetect.InvalidImage as exc:
                        # The stored JPEG itself is damaged (e.g. a truncated upload); retrying
                        # cannot fix it, so it is logged and does not fail the task.
                        broken += 1
                        print(f"BROKEN {name}: {exc} (skipped; the stored image is damaged)", file=sys.stderr)
                        continue
                    except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the month
                        error = exc
                if error is not None:
                    failures += 1
                    print(f"FAILED {name}: {type(error).__name__}: {error}", file=sys.stderr)
                    continue
                now = dt.datetime.now(dt.timezone.utc)
                row = make_row(args.camera, stamp, counts, source.uri(name), now)
                if wants_image:
                    row.update(object_name=name, annotated_jpeg=annotated)
                pending.append(row)
                if len(pending) >= sink.batch_size:
                    sink.write(pending)
                    pending = []
                if i % 100 == 0 or i == len(todo):
                    elapsed = time.monotonic() - began
                    rate = i / elapsed if elapsed > 0 else 0.0
                    print(f"  {i}/{len(todo)}  {stamp:%Y-%m-%d %H:%M}  {rate:.1f} frames/s", flush=True)
    finally:
        if pending:
            sink.write(pending)  # keep what was scored even if the run is stopping on an error

    scored = len(todo) - failures - broken
    print(f"done: {scored} scored, {failures} failed, {broken} broken source images skipped, output {sink}")
    return 1 if failures else 0


def _fetch(source, name):
    try:
        return source.read(name), None
    except Exception as exc:  # noqa: BLE001 - reported per frame by the caller
        return None, exc


if __name__ == "__main__":
    sys.exit(main())
