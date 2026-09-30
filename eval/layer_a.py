#!/usr/bin/env python3
"""Layer A (vision) scorer: detection mAP, precision/recall, count error, split by day and night.

Ground truth: a Roboflow export in **COCO** (``_annotations.coco.json``) or **YOLO** (``labels/*.txt``
+ ``images/`` for sizes, or ``--image-size``) format. Predictions: a JSON file mapping image file
name to a list of Roboflow-style boxes ``{"x", "y", "width", "height", "class", "confidence"}``
(centre coordinates, pixels). ``--predict-with-roboflow`` generates that file by running
``camdetect.detect_frame`` on each image (billed against Roboflow credits).

Metrics (IoU-matched, class-aware, greedy by confidence as in COCO):

- ``map50`` and ``map50_95``: mean over classes of 101-point interpolated AP at IoU 0.5 and at
  IoU 0.50:0.05:0.95.
- ``precision`` / ``recall`` at IoU 0.5 for predictions with confidence >= ``--confidence``
  (default 0.1, the service default).
- ``count_mae`` / ``count_bias``: per-image |predicted count - true count| and signed mean, at
  the same confidence. With ``--camera 2701`` also per direction (SG-MY / MY-SG) using the
  service's dividing line, so the numbers describe what ``swiftbackend`` reports.
- Slices: ``all``, ``day``, ``night``. Day is 07:00-18:59 SGT from ``--frame-times`` (CSV
  ``file_name,timestamp``); without a time the mean luminance decides (< 60/255 = night).

Hold-out discipline: score only a split that was not used for training (Roboflow ``test`` or
``valid``). Result cells in docs/evaluation.md stay ``pending`` until this runs on a real export.

    python layer_a.py --coco tests/fixtures/layer_a/_annotations.coco.json \\
        --predictions tests/fixtures/layer_a/predictions.json \\
        --frame-times tests/fixtures/layer_a/frame_times.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
NIGHT_LUMA = 60.0
DAY_HOURS = (7, 19)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str
    confidence: float = 1.0

    @classmethod
    def from_center(cls, x, y, w, h, label, confidence=1.0) -> "Box":
        return cls(x - w / 2, y - h / 2, x + w / 2, y + h / 2, str(label), float(confidence))

    def as_prediction(self) -> dict:
        return {
            "x": (self.x1 + self.x2) / 2,
            "y": (self.y1 + self.y2) / 2,
            "width": self.x2 - self.x1,
            "height": self.y2 - self.y1,
            "class": self.cls,
            "confidence": self.confidence,
        }


@dataclass
class ImageRecord:
    file_name: str
    width: int
    height: int
    boxes: List[Box] = field(default_factory=list)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


# --- loaders ------------------------------------------------------------------------


def load_coco(path: Path) -> Dict[str, ImageRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {im["id"]: ImageRecord(im["file_name"], int(im["width"]), int(im["height"])) for im in data["images"]}
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        images[ann["image_id"]].boxes.append(Box(x, y, x + w, y + h, cats.get(ann["category_id"], str(ann["category_id"]))))
    return {r.file_name: r for r in images.values()}


def load_yolo(
    labels_dir: Path,
    class_names: Sequence[str],
    *,
    images_dir: Optional[Path] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, ImageRecord]:
    """YOLO txt labels (``cls cx cy w h`` normalised). Sizes from images_dir or image_size."""
    out: Dict[str, ImageRecord] = {}
    for txt in sorted(Path(labels_dir).glob("*.txt")):
        file_name, (w, h) = _image_for_label(txt, images_dir, image_size)
        rec = ImageRecord(file_name, w, h)
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            k, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            rec.boxes.append(Box.from_center(cx * w, cy * h, bw * w, bh * h, class_names[k]))
        out[file_name] = rec
    return out


def _image_for_label(txt: Path, images_dir: Optional[Path], image_size: Optional[Tuple[int, int]]):
    if images_dir is not None:
        for ext in (".jpg", ".jpeg", ".png"):
            img = Path(images_dir) / (txt.stem + ext)
            if img.exists():
                from PIL import Image

                with Image.open(img) as im:
                    return img.name, im.size
    if image_size is None:
        raise ValueError(f"no image for {txt.name}; pass --images or --image-size")
    return txt.stem + ".jpg", image_size


def load_predictions(path: Path) -> Dict[str, List[Box]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, List[Box]] = {}
    for name, preds in data.items():
        boxes = []
        for p in preds:
            try:
                boxes.append(Box.from_center(float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"]), p.get("class", "vehicle"), float(p.get("confidence", 0))))
            except (KeyError, TypeError, ValueError):
                continue
        out[name] = boxes
    return out


def load_frame_times(path: Optional[Path]) -> Dict[str, dt.datetime]:
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = dt.datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is not None:
                ts = ts.astimezone(dt.timezone(dt.timedelta(hours=8))).replace(tzinfo=None)
            out[row["file_name"]] = ts
    return out


# --- metrics ------------------------------------------------------------------------


def _match(gt: List[Box], preds: List[Box], thr: float) -> Tuple[List[Tuple[float, bool]], int]:
    """Greedy match within one image and class. Returns [(confidence, is_tp)] and n_gt."""
    used = set()
    out = []
    for p in sorted(preds, key=lambda b: -b.confidence):
        best, best_iou = None, thr
        for j, g in enumerate(gt):
            if j in used:
                continue
            v = iou(p, g)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
        out.append((p.confidence, best is not None))
    return out, len(gt)


def average_precision(scored: List[Tuple[float, bool]], n_gt: int) -> float:
    """101-point interpolated AP (COCO)."""
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    tp = np.cumsum([1 if t else 0 for _, t in scored])
    fp = np.cumsum([0 if t else 1 for _, t in scored])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in RECALL_POINTS:
        idx = np.searchsorted(recall, r, side="left")
        ap += envelope[idx] if idx < len(envelope) else 0.0
    return float(ap / len(RECALL_POINTS))


def detection_metrics(gt: Dict[str, ImageRecord], preds: Dict[str, List[Box]], names: Iterable[str], *, confidence: float = 0.1) -> Dict[str, float]:
    names = list(names)
    classes = sorted({b.cls for n in names for b in gt[n].boxes} | {b.cls for n in names for b in preds.get(n, [])})
    aps: Dict[float, List[float]] = {t: [] for t in IOU_THRESHOLDS}
    tp = fp = n_gt_total = 0
    for c in classes:
        for t in IOU_THRESHOLDS:
            scored, n_gt = [], 0
            for n in names:
                s, k = _match([b for b in gt[n].boxes if b.cls == c], [b for b in preds.get(n, []) if b.cls == c], t)
                scored += s
                n_gt += k
            ap = average_precision(scored, n_gt)
            if not np.isnan(ap):
                aps[t].append(ap)
        for n in names:
            s, k = _match([b for b in gt[n].boxes if b.cls == c], [b for b in preds.get(n, []) if b.cls == c and b.confidence >= confidence], 0.5)
            tp += sum(1 for _, ok in s if ok)
            fp += sum(1 for _, ok in s if not ok)
    n_gt_total = sum(len(gt[n].boxes) for n in names)
    per_t = {t: float(np.mean(v)) if v else float("nan") for t, v in aps.items()}
    return {
        "map50": per_t[0.5],
        "map50_95": float(np.nanmean(list(per_t.values()))) if any(not np.isnan(v) for v in per_t.values()) else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / n_gt_total if n_gt_total else float("nan"),
        "n_gt_boxes": n_gt_total,
    }


def count_errors(gt, preds, names, *, confidence: float = 0.1, camera: Optional[str] = None) -> Dict[str, float]:
    diffs = []
    per_dir: Dict[str, List[int]] = {"SG-MY": [], "MY-SG": []}
    line_fn = _direction_fn(camera) if camera else None
    for n in names:
        p = [b for b in preds.get(n, []) if b.confidence >= confidence]
        g = gt[n].boxes
        diffs.append(len(p) - len(g))
        if line_fn:
            size = (gt[n].width, gt[n].height)
            for d in per_dir:
                per_dir[d].append(sum(line_fn(b, size) == d for b in p) - sum(line_fn(b, size) == d for b in g))
    out = {
        "count_mae": float(np.mean(np.abs(diffs))) if diffs else float("nan"),
        "count_bias": float(np.mean(diffs)) if diffs else float("nan"),
    }
    for d, v in per_dir.items():
        if line_fn and v:
            out[f"count_mae_{d}"] = float(np.mean(np.abs(v)))
    return out


def _direction_fn(camera: str):
    sys.path.insert(0, str(REPO_ROOT / "camdetect"))
    import main as camdetect  # noqa: E402

    def fn(box: Box, size):
        points = camdetect._dividing_line(camera, size)
        return camdetect._classify_direction(box.as_prediction(), points)

    return fn


def light_of(name: str, rec: ImageRecord, times: Dict[str, dt.datetime], images_dir: Optional[Path]) -> str:
    ts = times.get(name)
    if ts is not None:
        return "day" if DAY_HOURS[0] <= ts.hour < DAY_HOURS[1] else "night"
    if images_dir is not None and (Path(images_dir) / name).exists():
        from PIL import Image

        with Image.open(Path(images_dir) / name) as im:
            luma = float(np.asarray(im.convert("L"), dtype=float).mean())
        return "night" if luma < NIGHT_LUMA else "day"
    return "unknown"


def score(
    gt: Dict[str, ImageRecord],
    preds: Dict[str, List[Box]],
    *,
    confidence: float = 0.1,
    camera: Optional[str] = None,
    times: Optional[Dict[str, dt.datetime]] = None,
    images_dir: Optional[Path] = None,
    candidate: str = "detector",
) -> List[dict]:
    times = times or {}
    light = {n: light_of(n, r, times, images_dir) for n, r in gt.items()}
    rows = []
    for slice_name in ("all", "day", "night"):
        names = [n for n in gt if slice_name == "all" or light[n] == slice_name]
        if not names:
            continue
        row = {"candidate": candidate, "slice": slice_name, "n": len(names)}
        row.update(detection_metrics(gt, preds, names, confidence=confidence))
        row.update(count_errors(gt, preds, names, confidence=confidence, camera=camera))
        rows.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()})
    return rows


def predict_with_roboflow(gt: Dict[str, ImageRecord], images_dir: Path, camera: str) -> Dict[str, list]:
    """Billed: one Roboflow call per image via camdetect.detect_frame."""
    sys.path.insert(0, str(REPO_ROOT / "camdetect"))
    import main as camdetect  # noqa: E402

    out = {}
    for name in gt:
        data = (Path(images_dir) / name).read_bytes()
        out[name] = camdetect.detect_frame(data, camera, 0.0)["kept"]
    return out


def component(rows: List[dict], gt_path: Path, pred_path: Path, times: Dict[str, dt.datetime], confidence: float, split: str) -> dict:
    from run_artifacts import file_fingerprint

    def fp(p):
        try:
            return file_fingerprint(p)
        except ValueError:
            return {"path": Path(p).name}

    ts = sorted(times.values())
    return {
        "dataset": {"ground_truth": fp(gt_path), "predictions": fp(pred_path), "split": split, "images": rows[0]["n"] if rows else 0},
        "window": {"basis": "frame times of the scored split", "start": ts[0].isoformat() if ts else "unknown", "end": ts[-1].isoformat() if ts else "unknown"},
        "models": sorted({r["candidate"] for r in rows}),
        "horizon_minutes": None,
        "confidence": confidence,
        "metrics": rows,
        "significance": [],
        "artifacts": [],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco", type=Path, help="COCO annotations JSON")
    src.add_argument("--yolo-labels", type=Path, help="YOLO labels directory")
    parser.add_argument("--classes", nargs="*", default=None, help="YOLO class names in index order")
    parser.add_argument("--images", type=Path, default=None, help="image directory (sizes, luminance, Roboflow predictions)")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--predictions", type=Path, required=True, help="predictions JSON (read, or written with --predict-with-roboflow)")
    parser.add_argument("--predict-with-roboflow", action="store_true", help="BILLED: run camdetect.detect_frame per image")
    parser.add_argument("--candidate", default="detector")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--camera", default="2701", help="camera id for per-direction count error ('' to skip)")
    parser.add_argument("--frame-times", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--json-out", type=Path, default=None, help="write the layer_a manifest component here")
    args = parser.parse_args(argv)

    if args.coco:
        gt = load_coco(args.coco)
        gt_path = args.coco
    else:
        if not args.classes:
            parser.error("--classes is required with --yolo-labels")
        gt = load_yolo(args.yolo_labels, args.classes, images_dir=args.images, image_size=tuple(args.image_size) if args.image_size else None)
        gt_path = args.yolo_labels
    if args.predict_with_roboflow:
        if not args.images:
            parser.error("--predict-with-roboflow needs --images")
        args.predictions.write_text(json.dumps(predict_with_roboflow(gt, args.images, args.camera or "2701"), indent=1), encoding="utf-8")
    preds = load_predictions(args.predictions)
    times = load_frame_times(args.frame_times)
    rows = score(gt, preds, confidence=args.confidence, camera=args.camera or None, times=times, images_dir=args.images, candidate=args.candidate)

    keys = ["slice", "n", "map50", "map50_95", "precision", "recall", "count_mae", "count_bias", "count_mae_SG-MY", "count_mae_MY-SG"]
    print("| " + " | ".join(keys) + " |")
    print("|" + " --- |" * len(keys))
    for r in rows:
        print("| " + " | ".join("-" if r.get(k) is None else (f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r.get(k))) for k in keys) + " |")
    if args.json_out:
        args.json_out.write_text(json.dumps(component(rows, gt_path, args.predictions, times, args.confidence, args.split), indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
