#!/usr/bin/env python3
"""Time the SwiftBorder YOLO26s ONNX model on 20 LTA camera frames, on CPU or GPU.

In Colab:
    !pip install -q onnxruntime-gpu pillow numpy requests   # on a CPU runtime: onnxruntime
    # upload camdetect/models/yolo26s_v6_boxfix.onnx to /content, then:
    !python bench_colab.py --model /content/yolo26s_v6_boxfix.onnx
    # or paste this whole file into a cell and run it (model read from /content)

Run it once on a CPU runtime and once on a GPU runtime (Runtime -> Change runtime type).
Frames are downloaded before timing, so only preprocessing + inference is measured.

--frames DIR times the JPEGs in DIR instead of downloading live frames, so a rerun
uses the same images (e.g. camdetect/test-reports/frames). The recorded GPU run in
camdetect/test-reports/README.md used live frames, which were not kept.
"""

import argparse
import datetime as dt
import glob
import io
import os
import statistics
import time

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

API = "https://api.data.gov.sg/v1/transport/traffic-images"
CAMERAS = ("2701", "2702")  # Woodlands causeway
INPUT_H, INPUT_W = 736, 1280


def fetch_frames(n):
    """n JPEGs: both Woodlands cameras, every 10 minutes going back from 1 hour ago."""
    sgt = dt.timezone(dt.timedelta(hours=8))
    start = dt.datetime.now(sgt) - dt.timedelta(hours=1)
    frames, step = [], 0
    while len(frames) < n:
        stamp = (start - dt.timedelta(minutes=10 * step)).strftime("%Y-%m-%dT%H:%M:%S")
        step += 1
        items = requests.get(API, params={"date_time": stamp}, timeout=30).json().get("items", [])
        for cam in (items[0].get("cameras", []) if items else []):
            if cam["camera_id"] in CAMERAS and len(frames) < n:
                frames.append(requests.get(cam["image"], timeout=30).content)
    return frames


def preprocess(jpeg):
    """Letterbox to 736x1280, RGB float 0-1, NCHW."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    scale = min(INPUT_W / img.width, INPUT_H / img.height)
    w, h = round(img.width * scale), round(img.height * scale)
    canvas = Image.new("RGB", (INPUT_W, INPUT_H), (114, 114, 114))
    canvas.paste(img.resize((w, h), Image.BILINEAR), ((INPUT_W - w) // 2, (INPUT_H - h) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/content/yolo26s_v6_boxfix.onnx")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is present")
    ap.add_argument("--frames", help="folder of JPEGs to time instead of live frames")
    # parse_known_args: pasted into a notebook cell, Jupyter adds its own "-f kernel.json" argument.
    args, _ = ap.parse_known_args()

    providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    session = ort.InferenceSession(args.model, providers=providers)
    print("onnxruntime", ort.__version__, "| running on:", session.get_providers()[0])

    if args.frames:
        paths = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))[: args.n]
        print(f"reading {len(paths)} frames from {args.frames}...")
        frames = [open(p, "rb").read() for p in paths]
    else:
        print(f"downloading {args.n} frames...")
        frames = fetch_frames(args.n)
    name = session.get_inputs()[0].name

    for _ in range(args.warmup):  # first runs include CUDA/graph setup
        session.run(None, {name: preprocess(frames[0])})

    pre_ms, inf_ms = [], []
    for i, jpeg in enumerate(frames, 1):
        t0 = time.perf_counter()
        x = preprocess(jpeg)
        t1 = time.perf_counter()
        session.run(None, {name: x})
        t2 = time.perf_counter()
        pre_ms.append((t1 - t0) * 1000)
        inf_ms.append((t2 - t1) * 1000)
        print(f"image {i:2d}: preprocess {pre_ms[-1]:7.1f} ms  inference {inf_ms[-1]:7.1f} ms"
              f"  total {pre_ms[-1] + inf_ms[-1]:7.1f} ms")

    total = [a + b for a, b in zip(pre_ms, inf_ms)]
    print(f"\n{session.get_providers()[0]} over {len(frames)} images:")
    print(f"  preprocess avg {statistics.mean(pre_ms):.1f} ms")
    print(f"  inference  avg {statistics.mean(inf_ms):.1f} ms  (median {statistics.median(inf_ms):.1f})")
    print(f"  total      avg {statistics.mean(total):.1f} ms  -> {1000 / statistics.mean(total):.1f} images/s")


if __name__ == "__main__":
    main()
