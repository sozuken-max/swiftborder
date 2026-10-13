#!/usr/bin/env python3
"""Time the same detection requests against two swiftbackend URLs (e.g. CPU vs GPU service).

Each frame time is sent to both services, so both score the same frame. The first request per
service is reported separately as the cold start. Counts come back on the response headers and are
compared so a GPU that answers differently shows up.

    python bench_backends.py --cpu URL --gpu URL [--token TOKEN] --n 30 --out results.csv
"""

import argparse
import csv
import datetime as dt
import statistics
import time

import requests


def call(url, camera, model, date_time, token):
    params = {"format": "directional", "camera_id": camera, "model": model, "date_time": date_time}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    t0 = time.perf_counter()
    res = requests.get(url, params=params, headers=headers, timeout=300)
    ms = (time.perf_counter() - t0) * 1000
    counts = {k: v for k, v in res.headers.items() if "count" in k.lower()}
    return res.status_code, ms, counts


def summary(name, rows):
    warm = [r["ms"] for r in rows[1:] if r["status"] == 200]
    if not warm:
        return f"{name}: no successful warm requests"
    p95 = sorted(warm)[max(0, round(0.95 * len(warm)) - 1)]
    return (f"{name}: cold {rows[0]['ms']:.0f} ms | warm n={len(warm)} "
            f"p50 {statistics.median(warm):.0f} ms, p95 {p95:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--token", default="", help="ID token, only for a private service")
    ap.add_argument("--camera", default="2701")
    ap.add_argument("--model", default="local")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="bench_results.csv")
    args = ap.parse_args()

    # Frames 5 minutes apart, starting a day ago, so they are in the camera archive.
    start = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))) - dt.timedelta(days=1)
    times = [(start + dt.timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(args.n)]

    rows = {"cpu": [], "gpu": []}
    for i, t in enumerate(times):
        for name, url in (("cpu", args.cpu), ("gpu", args.gpu)):
            token = args.token if name == "gpu" else ""
            status, ms, counts = call(url, args.camera, args.model, t, token)
            rows[name].append({"service": name, "i": i, "date_time": t, "status": status,
                               "ms": round(ms, 1), "counts": counts})
        same = rows["cpu"][-1]["counts"] == rows["gpu"][-1]["counts"]
        print(f"{i:3d} {t} cpu {rows['cpu'][-1]['ms']:7.0f} ms  gpu {rows['gpu'][-1]['ms']:7.0f} ms"
              f"  counts {'same' if same else 'DIFFER'}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["service", "i", "date_time", "status", "ms", "counts"])
        w.writeheader()
        for name in rows:
            w.writerows(rows[name])
    print(summary("CPU", rows["cpu"]))
    print(summary("GPU", rows["gpu"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
