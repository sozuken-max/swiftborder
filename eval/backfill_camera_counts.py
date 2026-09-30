#!/usr/bin/env python3
"""Backfill camera 2701 directional vehicle counts for the joined-feature experiment (offline).

For each sampled 10-minute bin (SGT) the script asks data.gov.sg for the camera 2701 frame at that
time, downloads it, and runs ``camdetect.detect_frame`` (Roboflow serverless workflow; **billed
against your Roboflow credits**). Results append to ``eval/data/camera/cam2701_counts.csv``
(gitignored).

Budget rules (Roboflow free tier has no overage billing: when credits run out, calls fail):

- ``--dry-run`` prints how many bins and inference calls a run would make, and makes no calls.
- ``--max-calls N`` caps inference calls for this invocation.
- A quota or payment error (HTTP 402, 403, or 429 mentioning quota/credit/limit) stops the run at
  once. Everything scored so far is already saved; re-run after the credit reset to resume.
- Sampling (default ``peak-first``): peak hours 06:00-10:59 and 16:00-21:59 SGT at 10-minute bins
  first, then off-peak at 20-minute bins. ``--mode full`` scores every 10-minute bin.

Resume is automatic: a bin already recorded as ``ok``, ``missing_frame``, ``stale_frame`` or
``invalid_image`` is skipped; ``error`` rows are retried. A bin with no usable frame is recorded as
missing, never as zero vehicles.

    python backfill_camera_counts.py --dry-run
    python backfill_camera_counts.py --max-calls 50 --start 2026-09-20 --end 2026-09-20   # pilot
    python backfill_camera_counts.py --coverage
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
DEFAULT_OUT = EVAL_ROOT / "data" / "camera" / "cam2701_counts.csv"
CAMERA_ID = "2701"
TRAFFIC_IMAGES_API = "https://api.data.gov.sg/v1/transport/traffic-images"
PEAK_HOURS = set(range(6, 11)) | set(range(16, 22))
MAX_FRAME_AGE = dt.timedelta(minutes=10)
FIELDS = [
    "bin_sgt",
    "frame_ts",
    "status",
    "total",
    "sg_my",
    "my_sg",
    "unknown",
    "sg_my_extent",
    "my_sg_extent",
    "width",
    "height",
    "inference_called",
    "recorded_at_utc",
    "detail",
]
DONE_STATUSES = {"ok", "missing_frame", "stale_frame", "invalid_image"}


class QuotaExhausted(RuntimeError):
    """Roboflow refused the call for quota / payment reasons. Stop and resume after the reset."""


def planned_bins(start: dt.date, end: dt.date, mode: str = "peak-first") -> List[dt.datetime]:
    """Naive SGT bin starts in scoring order."""
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    all_bins = [dt.datetime.combine(d, dt.time()) + dt.timedelta(minutes=10 * k) for d in days for k in range(144)]
    if mode == "full":
        return all_bins
    if mode != "peak-first":
        raise ValueError(f"unknown mode {mode!r}")
    peak = [b for b in all_bins if b.hour in PEAK_HOURS]
    off = [b for b in all_bins if b.hour not in PEAK_HOURS and b.minute % 20 == 0]
    return peak + off


def read_done(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["bin_sgt"] for r in csv.DictReader(f) if r.get("status") in DONE_STATUSES}


def append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def is_quota_error(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if resp is None:
        return False
    status = getattr(resp, "status_code", None)
    if status in (402, 403):
        return True
    if status == 429:
        text = (getattr(resp, "text", "") or "").lower()
        return any(w in text for w in ("quota", "credit", "limit", "billing", "plan"))
    return False


@dataclass
class Frame:
    url: Optional[str]
    timestamp: Optional[dt.datetime]  # naive SGT


def lookup_frame(bin_sgt: dt.datetime, get: Callable = None) -> Frame:
    """data.gov.sg traffic-images snapshot for camera 2701 at ``bin_sgt``."""
    import requests

    get = get or requests.get
    resp = get(TRAFFIC_IMAGES_API, params={"date_time": bin_sgt.strftime("%Y-%m-%dT%H:%M:%S")}, timeout=30)
    resp.raise_for_status()
    for item in resp.json().get("items", []) or []:
        for cam in item.get("cameras", []) or []:
            if str(cam.get("camera_id")) == CAMERA_ID:
                ts = cam.get("timestamp") or item.get("timestamp")
                parsed = None
                if ts:
                    parsed = dt.datetime.fromisoformat(ts)
                    if parsed.tzinfo is not None:
                        parsed = parsed.astimezone(dt.timezone(dt.timedelta(hours=8))).replace(tzinfo=None)
                return Frame(cam.get("image"), parsed)
    return Frame(None, None)


def _load_camdetect():
    sys.path.insert(0, str(REPO_ROOT / "camdetect"))
    import main as camdetect  # noqa: E402

    return camdetect


def score_bin(
    bin_sgt: dt.datetime,
    *,
    lookup: Callable[[dt.datetime], Frame],
    download: Callable[[str], bytes],
    detect_frame: Callable[[bytes], dict],
) -> Dict[str, object]:
    """One CSV row for one bin. Raises QuotaExhausted on quota/payment errors."""
    import requests

    row: Dict[str, object] = {"bin_sgt": bin_sgt.isoformat(), "inference_called": 0}
    try:
        frame = lookup(bin_sgt)
    except (requests.RequestException, ValueError) as exc:
        return {**row, "status": "error", "detail": f"lookup: {exc.__class__.__name__}"}
    if not frame.url:
        return {**row, "status": "missing_frame"}
    row["frame_ts"] = frame.timestamp.isoformat() if frame.timestamp else ""
    if frame.timestamp is None or abs(bin_sgt - frame.timestamp) > MAX_FRAME_AGE:
        return {**row, "status": "stale_frame"}
    try:
        image_bytes = download(frame.url)
    except requests.RequestException as exc:
        return {**row, "status": "error", "detail": f"download: {exc.__class__.__name__}"}
    except ValueError:
        return {**row, "status": "invalid_image"}
    try:
        row["inference_called"] = 1
        result = detect_frame(image_bytes)
    except requests.RequestException as exc:
        if is_quota_error(exc):
            raise QuotaExhausted(str(getattr(exc.response, "status_code", ""))) from exc
        return {**row, "status": "error", "detail": f"inference: {exc.__class__.__name__}"}
    except ValueError as exc:
        return {**row, "status": "invalid_image" if exc.__class__.__name__ == "InvalidImage" else "error", "detail": "value"}
    s = result["summary"]
    w, h = result["image_size"]
    return {
        **row,
        "status": "ok",
        "total": len(result["kept"]),
        "sg_my": s["sg_my"]["count"],
        "my_sg": s["my_sg"]["count"],
        "unknown": s["unknown"]["count"],
        "sg_my_extent": s["sg_my"].get("extent", ""),
        "my_sg_extent": s["my_sg"].get("extent", ""),
        "width": w,
        "height": h,
    }


def run(
    bins: Sequence[dt.datetime],
    out: Path,
    *,
    score: Callable[[dt.datetime], Dict[str, object]],
    max_calls: Optional[int] = None,
    rate_limit_sec: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    log=sys.stdout,
) -> Dict[str, object]:
    done = read_done(out)
    todo = [b for b in bins if b.isoformat() not in done]
    calls = 0
    written = 0
    stopped = None
    for b in todo:
        if max_calls is not None and calls >= max_calls:
            stopped = "max_calls"
            break
        try:
            row = score(b)
        except QuotaExhausted as exc:
            stopped = f"quota ({exc})"
            print(f"Roboflow refused the call ({exc}). Stopping; resume after the credit reset.", file=log)
            break
        row["recorded_at_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        append_row(out, row)
        written += 1
        calls += int(row.get("inference_called") or 0)
        if row.get("inference_called") and rate_limit_sec:
            sleep(rate_limit_sec)
    return {"planned": len(bins), "already_done": len(bins) - len(todo), "written": written, "inference_calls": calls, "stopped": stopped}


def coverage(out: Path) -> List[Tuple[str, int, int]]:
    """(date, ok bins, recorded bins) per SGT day, latest status per bin."""
    if not out.exists():
        return []
    latest: Dict[str, str] = {}
    with open(out, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            latest[r["bin_sgt"]] = r["status"]
    per_day: Dict[str, List[int]] = {}
    for b, status in latest.items():
        d = b[:10]
        per_day.setdefault(d, [0, 0])
        per_day[d][1] += 1
        per_day[d][0] += status == "ok"
    return [(d, v[0], v[1]) for d, v in sorted(per_day.items())]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2026-09-05", help="first SGT date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-30", help="last SGT date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=("peak-first", "full"), default="peak-first")
    parser.add_argument("--max-calls", type=int, default=None, help="cap on Roboflow inference calls this run")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="seconds between inference calls")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="count bins and calls; make no requests")
    parser.add_argument("--coverage", action="store_true", help="print per-day coverage of the output CSV")
    args = parser.parse_args(argv)

    if args.coverage:
        rows = coverage(args.out)
        for d, ok, rec in rows:
            print(f"{d}  ok={ok:4d}  recorded={rec:4d}")
        print(f"total ok={sum(r[1] for r in rows)} recorded={sum(r[2] for r in rows)}")
        return 0

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if start > end:
        parser.error("--start is after --end")
    bins = planned_bins(start, end, args.mode)
    now_sgt = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).replace(tzinfo=None)
    bins = [b for b in bins if b + dt.timedelta(minutes=10) <= now_sgt]
    done = read_done(args.out)
    todo = [b for b in bins if b.isoformat() not in done]
    print(f"{args.mode}: {len(bins)} bins planned, {len(bins) - len(todo)} already recorded, {len(todo)} to score")
    print(f"Upper bound on Roboflow inference calls: {len(todo) if args.max_calls is None else min(len(todo), args.max_calls)}")
    if args.dry_run:
        return 0

    camdetect = _load_camdetect()
    if not camdetect.ROBOFLOW_API_KEY:
        print("ROBOFLOW_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    def download(url: str) -> bytes:
        try:
            return camdetect._download_image(url)
        except camdetect.InvalidImage as exc:
            raise ValueError(str(exc)) from exc

    result = run(
        bins,
        args.out,
        score=lambda b: score_bin(
            b,
            lookup=lookup_frame,
            download=download,
            detect_frame=lambda data: camdetect.detect_frame(data, CAMERA_ID),
        ),
        max_calls=args.max_calls,
        rate_limit_sec=args.rate_limit,
    )
    print(result)
    return 3 if result["stopped"] and result["stopped"].startswith("quota") else 0


if __name__ == "__main__":
    raise SystemExit(main())
