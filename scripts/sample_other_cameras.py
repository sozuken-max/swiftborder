#!/usr/bin/env python3
"""Download a stratified random sample of frames, N per camera, from every LTA camera except 2701.

Reads ``gs://sg-lta-traffic-cameras/camera_id=<id>/month=<YYYY-MM>/<id>_<YYYYMMDD>T<HHMMSS>.jpg``
(capture time in SGT, as in the file name) and writes:

    lta_samples/<camera>/<file name>.jpg    the sampled frames, one folder per camera
    lta_samples/manifest.csv                one row per frame: file, camera, month, hour, hour bucket, gs:// path
    lta_samples_<camera>.zip                one zip per camera, unless --no-zip. Each holds
                                            lta_samples/<camera>/... and lta_samples/manifest_<camera>.csv,
                                            so unzipping them all into one place gives a single lta_samples/ tree

Sampling, repeated independently for each camera (--n frames each, default 1000):
  * Hour buckets are 0, 1, 2-5 (one bucket: almost no traffic), then 6, 7, ... 23. That is 21
    buckets, each given the same quota (N // 21, with the remainder spread over random buckets).
  * Inside each bucket the picks cycle through every month that camera has data for, in shuffled
    order, so its months are equally represented too. The day and the hour inside the bucket are
    random; the minute comes from the frames that exist in that hour.
  * A pick that lands on an hour with no frames is redrawn, so an unlucky month can end up a frame
    short. The summary printed at the end shows the final spread per camera.

In Google Colab: upload this file and run `!python sample_other_cameras.py`. It signs you in (a pop-up
asks for the Google account), writes to /content and downloads the zips to your computer at the end.
Nothing else to set up; do not close the tab until the downloads start.

Setup (Anaconda prompt):

    pip install google-cloud-storage
    gcloud auth application-default login      # an account with Storage Object Viewer on the bucket
    python sample_other_cameras.py

The sample is fixed by --seed (default 0): the same seed picks the same frames every run, as long as the
bucket still holds the same files. Use --seed 1, 2, ... for a different sample. Re-running with the same
--out resumes: files already on disk are not downloaded again.
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


def draw_frame(client, bucket, cam, bi, month, rng, avoid=frozenset(), tries=60):
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
        if blob.name in avoid:
            continue
        return dict(path=blob.name, camera=cam, month=month, day=day, hour=hour,
                    hour_bucket=bucket_label(HOUR_BUCKETS[bi]), size=blob.size)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="frames PER CAMERA (default 1000)")
    ap.add_argument("--out", default="lta_samples",
                    help="output folder (default lta_samples; /content/lta_samples in Colab)")
    ap.add_argument("--seed", type=int, default=0, help="same seed = same sample (default 0)")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--exclude", nargs="*", default=["2701"], help="camera ids to skip (default 2701)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-zip", action="store_true")
    args, _ = ap.parse_known_args()  # notebooks add their own flags (-f kernel.json)

    try:
        from google.colab import auth as colab_auth  # only exists inside Colab
        colab_auth.authenticate_user()
        in_colab = True
        if args.out == "lta_samples":
            args.out = "/content/lta_samples"
    except ImportError:
        in_colab = False

    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("pip install google-cloud-storage")
    client = storage.Client(project=args.project)
    try:  # the default pool is 10 connections, which throttles the worker threads
        from requests.adapters import HTTPAdapter
        client._http.mount("https://", HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers))
    except Exception:
        pass
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
    print(f"choosing {len(jobs)} frames (one bucket lookup each; progress every 250)...")
    picks = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, pick in enumerate(ex.map(lambda j: draw_frame(client, args.bucket, j[0], j[1], j[2], j[3]),
                                        jobs), 1):
            picks.append(pick)
            if i % 250 == 0 or i == len(jobs):
                print(f"chosen {i}/{len(jobs)}", flush=True)
    # Two picks can land on the same file. Keep the first (job order) and redraw the rest one by one
    # with a seeded rng, so the final sample does not depend on thread timing.
    seen, redraw_rng = set(), random.Random(args.seed + 1)
    for k, pick in enumerate(picks):
        if pick and pick["path"] in seen:
            cam, bi, month, _ = jobs[k]
            pick = picks[k] = draw_frame(client, args.bucket, cam, bi, month, redraw_rng, avoid=seen)
        if pick:
            seen.add(pick["path"])
    rows = [p for p in picks if p]
    print(f"picked {len(rows)} of {len(jobs)} frames ({len(jobs) - len(rows)} slots found no frame)")

    out = Path(args.out)

    def dest_of(row):
        return out / row["camera"] / row["path"].rsplit("/", 1)[-1]

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
        zips = []
        for cam in sorted(cams):
            mine = [r for r in rows if r["camera"] == cam]
            zpath = out.parent / f"{out.name}_{cam}.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
                z.writestr(f"{out.name}/manifest_{cam}.csv", ",".join(fields) + "\n" + "".join(
                    ",".join(str(manifest_row(r)[k]) for k in fields) + "\n" for r in mine))
                for r in mine:
                    z.write(dest_of(r), f"{out.name}/{cam}/{dest_of(r).name}")
            print("zip:", zpath.resolve())
            zips.append(zpath)
    print("folder:", out.resolve())
    if in_colab and not args.no_zip:
        from google.colab import files
        for z in zips:  # the browser may ask to allow multiple downloads
            files.download(str(z))


if __name__ == "__main__":
    main()
