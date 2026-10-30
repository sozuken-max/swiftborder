#!/usr/bin/env python3
"""Download a stratified random sample of frames, N per camera, from every LTA camera except 2701.

Reads ``gs://sg-lta-traffic-cameras/camera_id=<id>/month=<YYYY-MM>/<id>_<YYYYMMDD>T<HHMMSS>.jpg``
(capture time in SGT, as in the file name) and writes:

    <out>/images/<camera>/<file name>.jpg   the sampled frames
    <out>/manifest.csv                      one row per frame: object path, camera, month, hour, hour bucket
    <out>_<camera>.zip                      one zip per camera (its images and its manifest rows), unless --no-zip

Sampling, repeated independently for each camera (--n frames each, default 1000):
  * Hour buckets are 0, 1, 2-5 (one bucket: almost no traffic), then 6, 7, ... 23. That is 21
    buckets, each given the same quota (N // 21, with the remainder spread over random buckets).
  * Inside each bucket the picks cycle through every month that camera has data for, in shuffled
    order, so its months are equally represented too. The day and the hour inside the bucket are
    random; the minute comes from the frames that exist in that hour.
  * A pick that lands on an hour with no frames is redrawn, so an unlucky month can end up a frame
    short. The summary printed at the end shows the final spread per camera.

Setup (Anaconda prompt):

    pip install google-cloud-storage
    gcloud auth application-default login      # an account with Storage Object Viewer on the bucket
    python sample_other_cameras.py --out other_cameras_1000

Re-running with the same --seed and --out resumes: files already on disk are not downloaded again.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import random
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

BUCKET = "sg-lta-traffic-cameras"
PROJECT = "swiftborder"
HOUR_BUCKETS = [(0,), (1,), (2, 3, 4, 5)] + [(h,) for h in range(6, 24)]
NAME_RE = re.compile(r"^(?P<cam>\d+)_(?P<date>\d{8})T(?P<time>\d{6})\.jpg$")


def bucket_label(hours):
    return f"{hours[0]:02d}" if len(hours) == 1 else f"{hours[0]:02d}-{hours[-1]:02d}"


def list_prefixes(client, bucket, prefix=""):
    """Immediate 'sub-folders' under prefix, as full prefixes ending in '/'."""
    it = client.list_blobs(bucket, prefix=prefix, delimiter="/")
    for _ in it:  # the prefixes are only filled in once the pages have been consumed
        pass
    return sorted(it.prefixes)


def discover(client, bucket, exclude):
    """{camera_id: [YYYY-MM, ...]} for every camera folder not in exclude."""
    cams = {}
    for p in list_prefixes(client, bucket):
        m = re.fullmatch(r"camera_id=(\d+)/", p)
        if not m or m.group(1) in exclude:
            continue
        months = [q.split("month=")[1].rstrip("/") for q in list_prefixes(client, bucket, p)]
        if months:
            cams[m.group(1)] = months
    return cams


def quotas(n, rng):
    base, extra = divmod(n, len(HOUR_BUCKETS))
    plus = set(rng.sample(range(len(HOUR_BUCKETS)), extra))
    return [base + (i in plus) for i in range(len(HOUR_BUCKETS))]


def plan_slots(n, months, rng):
    """[(bucket_index, month)] with n entries: equal per hour bucket, months cycled within each."""
    months = sorted(months)
    slots = []
    for bi, q in enumerate(quotas(n, rng)):
        order = []
        while len(order) < q:
            cycle = months[:]
            rng.shuffle(cycle)
            order += cycle
        slots += [(bi, m) for m in order[:q]]
    return slots


def draw_frame(client, bucket, cam, bi, month, rng, taken, lock, tries=60):
    """One random frame from camera cam in hour bucket bi and the given month, or None."""
    year, mon = map(int, month.split("-"))
    ndays = calendar.monthrange(year, mon)[1]
    for _ in range(tries):
        hour = rng.choice(HOUR_BUCKETS[bi])
        day = rng.randint(1, ndays)
        prefix = f"camera_id={cam}/month={month}/{cam}_{year:04d}{mon:02d}{day:02d}T{hour:02d}"
        blobs = [b for b in client.list_blobs(bucket, prefix=prefix, max_results=200)
                 if NAME_RE.match(b.name.rsplit("/", 1)[-1])]
        if not blobs:
            continue
        blob = rng.choice(blobs)
        with lock:
            if blob.name in taken:
                continue
            taken.add(blob.name)
        return dict(path=blob.name, camera=cam, month=month, day=day, hour=hour,
                    hour_bucket=bucket_label(HOUR_BUCKETS[bi]), size=blob.size)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="frames PER CAMERA (default 1000)")
    ap.add_argument("--out", default="other_cameras_1000", help="output folder (default other_cameras_1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--exclude", nargs="*", default=["2701"], help="camera ids to skip (default 2701)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("pip install google-cloud-storage")
    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)

    cams = discover(client, args.bucket, set(args.exclude))
    if not cams:
        sys.exit("no cameras found (check the bucket name, your login, and --exclude)")
    print("cameras:", ", ".join(f"{c} ({len(m)} months)" for c, m in sorted(cams.items())))

    rng = random.Random(args.seed)
    jobs = []  # (camera, bucket_index, month, rng) for every pick, 1000 per camera
    for cam, months in sorted(cams.items()):
        for (bi, month) in plan_slots(args.n, months, rng):
            jobs.append((cam, bi, month, random.Random(rng.random())))  # one rng per pick: reproducible threads
    taken, lock = set(), Lock()
    with ThreadPoolExecutor(args.workers) as ex:
        picks = list(ex.map(lambda j: draw_frame(client, args.bucket, j[0], j[1], j[2], j[3], taken, lock), jobs))
    rows = [p for p in picks if p]
    print(f"picked {len(rows)} of {len(jobs)} frames ({len(jobs) - len(rows)} slots found no frame)")

    out = Path(args.out)

    def dest_of(row):
        return out / "images" / row["camera"] / row["path"].rsplit("/", 1)[-1]

    def fetch(row):
        dest = dest_of(row)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_suffix(".part")
            bucket.blob(row["path"]).download_to_filename(tmp)
            tmp.replace(dest)
        return dest

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(fetch, rows), 1):
            if i % 250 == 0 or i == len(rows):
                print(f"downloaded {i}/{len(rows)}")

    rows.sort(key=lambda r: (r["camera"], r["path"].rsplit("/", 1)[-1]))
    fields = ["file", "camera", "month", "day", "hour", "hour_bucket", "size", "gcs_path"]

    def manifest_row(r):
        return dict(file=r["path"].rsplit("/", 1)[-1], camera=r["camera"], month=r["month"], day=r["day"],
                    hour=r["hour"], hour_bucket=r["hour_bucket"], size=r["size"],
                    gcs_path=f"gs://{args.bucket}/{r['path']}")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_row(r) for r in rows)

    for cam in sorted(cams):
        mine = [r for r in rows if r["camera"] == cam]
        print(f"camera {cam}: {len(mine)} frames")
        print("  by month:", dict(sorted(Counter(r["month"] for r in mine).items())))
        print("  by hour bucket:", dict(sorted(Counter(r["hour_bucket"] for r in mine).items())))

    if not args.no_zip:
        for cam in sorted(cams):
            mine = [r for r in rows if r["camera"] == cam]
            zpath = Path(f"{out}_{cam}.zip")
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
                z.writestr("manifest.csv", ",".join(fields) + "\n" + "".join(
                    ",".join(str(manifest_row(r)[k]) for k in fields) + "\n" for r in mine))
                for r in mine:
                    z.write(dest_of(r), f"{cam}/{dest_of(r).name}")
            print("zip:", zpath.resolve())
    print("folder:", out.resolve())


if __name__ == "__main__":
    main()
