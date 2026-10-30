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
  * The script first lists the frames that exist in every month of the camera (names only), then
    samples from that list, so it never aims at an hour that has no frame.
  * Hour buckets are 0, 1, 2-5 (one bucket: almost no traffic), then 6, 7, ... 23. That is 21
    buckets, each given the same quota (N // 21, with the remainder spread over random buckets).
  * Inside each bucket the months take turns in shuffled order, each turn drawing a random unused
    frame from that month. A month that has run out (a sparse or partial month) drops out of the
    rotation and the other months carry on, so you still get N frames unless the camera has fewer
    than a bucket's quota of frames in total. The final summary lists any bucket that fell short.

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


def index_month(client, bucket, cam, month):
    """{hour: [(object name, size), ...]} for every frame that exists in one camera-month."""
    out = {}
    it = client.list_blobs(bucket, prefix=f"camera_id={cam}/month={month}/",
                           fields="items(name,size),nextPageToken")
    for b in it:
        m = NAME_RE.match(b.name.rsplit("/", 1)[-1])
        if m:
            out.setdefault(int(m["time"][:2]), []).append((b.name, b.size))
    return out


def sample_camera(n, index, rng):
    """Pick n frames for one camera from index {month: {hour: [(name, size)]}}.

    Every hour bucket gets the same quota. Inside a bucket the months take turns (in shuffled order)
    and each turn draws an unused frame at random from that month's frames in the bucket. A month
    with no frames left in the bucket drops out of the rotation, and the others carry on, so the
    quota is met whenever the camera has enough frames in that bucket at all.
    Returns (rows, shortfall_by_bucket_label).
    """
    rows, short = [], {}
    for bi, q in enumerate(quotas(n, rng)):
        label = bucket_label(HOUR_BUCKETS[bi])
        pools = {}
        for month in sorted(index):
            frames = sorted(f for h in HOUR_BUCKETS[bi] for f in index[month].get(h, []))
            if frames:
                rng.shuffle(frames)
                pools[month] = frames
        got = 0
        while got < q and pools:
            order = sorted(pools)
            rng.shuffle(order)
            for month in order:
                if got == q:
                    break
                name, size = pools[month].pop()
                if not pools[month]:
                    del pools[month]
                fname = name.rsplit("/", 1)[-1]
                m = NAME_RE.match(fname)
                rows.append(dict(path=name, month=month, day=int(m["date"][6:]), hour=int(m["time"][:2]),
                                 hour_bucket=label, size=size))
                got += 1
        if got < q:
            short[label] = q - got
    return rows, short


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

    # 1) list the frames that actually exist, one listing per camera-month (names only)
    todo = [(c, m) for c, ms in sorted(cams.items()) for m in sorted(ms)]
    print(f"listing {len(todo)} camera-months (progress every 10)...", flush=True)
    indexes = {c: {} for c in cams}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, ((c, m), idx) in enumerate(zip(todo, ex.map(lambda t: index_month(client, args.bucket, *t), todo)), 1):
            indexes[c][m] = idx
            if i % 10 == 0 or i == len(todo):
                print(f"listed {i}/{len(todo)}", flush=True)
    for c in sorted(cams):
        counts = {m: sum(len(v) for v in indexes[c][m].values()) for m in sorted(indexes[c])}
        print(f"camera {c} frames available by month:", counts)
    cams = {c: [m for m in ms if indexes[c][m]] for c, ms in cams.items()}  # months with no frames drop out

    # 2) sample from them, camera by camera, with one seeded rng
    rng = random.Random(args.seed)
    rows = []
    for c in sorted(cams):
        picked, short = sample_camera(args.n, indexes[c], rng)
        rows += [dict(r, camera=c) for r in picked]
        msg = f"camera {c}: picked {len(picked)} of {args.n}"
        print(msg + (f"; no frames left in hour buckets {short}" if short else ""))

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
