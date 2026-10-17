#!/usr/bin/env python3
"""Rescore the 15-image Layer A count set with the in-container YOLO26s.

Ground truth is the caption totals in ``runs/layer_a_count/artifact_counts.json``
(the Claude artifact comparison). This script does not compute mAP: the export
has no boxes in git. It downloads the original 1920x1080 frames from
``traffic_images.metadata`` paths and runs ``camdetect.detect_frame`` at the
live ``model=local`` point (confidence 0.2, overlap 0.6) and at confidence 0.30
with overlap off, which is the artifact's published threshold for
"YOLO-S trained (freeze)".

No Roboflow call. Frames are read from ``--frames-dir`` (file name is the GCS
object name). Download them first; this script does not talk to Cloud Storage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
CAMDETECT = REPO_ROOT / "camdetect"
sys.path.insert(0, str(CAMDETECT))

import main as camdetect  # noqa: E402

ART = EVAL_ROOT / "runs" / "layer_a_count" / "artifact_counts.json"
OUT = EVAL_ROOT / "runs" / "layer_a_count" / "local_rescore.json"

# ``artifact_conf_overlap_off`` uses OVERLAP_SUPPRESSION_OFF (1.0), not None: detect_frame treats
# None as the live default (LOCAL_MAX_OVERLAP 0.6) and normalizes values >= 1 to off before NMS.
SETTINGS = (
    ("live", 0.2, 0.6),
    ("artifact_conf_overlap_off", 0.30, camdetect.OVERLAP_SUPPRESSION_OFF),
    ("artifact_conf_overlap_on", 0.30, 0.6),
)


def _counts(kept: list) -> dict:
    by_class: dict[str, int] = {}
    by_dir: dict[str, int] = {}
    for pred in kept:
        by_class[pred["class"]] = by_class.get(pred["class"], 0) + 1
        direction = pred.get("direction") or "unknown"
        by_dir[direction] = by_dir.get(direction, 0) + 1
    return {
        "cars": by_class.get("car", 0),
        "veh": len(kept),
        "by_class": by_class,
        "by_direction": by_dir,
    }


def _slice(rows: list, key: str, metric: str) -> dict:
    def keep(row, label):
        hour = int(row["t"][11:13])
        day = 7 <= hour <= 18
        return day if label == "day" else not day

    return {label: _mae([row for row in rows if keep(row, label)], key, metric) for label in ("day", "night")}


def _mae(rows: list, key: str, metric: str) -> dict:
    errors = [row["pred"][key][metric] - row["gt"][metric] for row in rows]
    n = len(errors)
    return {
        "mae": round(sum(abs(e) for e in errors) / n, 2),
        "bias": round(sum(errors) / n, 2),
        "exact": sum(e == 0 for e in errors),
        "n": n,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: score_layer_a_local.py FRAMES_DIR")
    frames_dir = Path(sys.argv[1])
    artifact = json.loads(ART.read_text(encoding="utf-8"))
    frames = []
    for image in artifact["images"]:
        object_name = image["gcs_path"].rsplit("/", 1)[-1]
        raw = (frames_dir / object_name).read_bytes()
        row = {
            "t": image["t"],
            "gcs_path": image["gcs_path"],
            "gt": image["gt"],
            "artifact_trained_s": image["pred"]["trained_s"],
            "pred": {},
        }
        for name, conf, overlap in SETTINGS:
            result = camdetect.detect_frame(
                raw,
                camera_id="2701",
                min_confidence=conf,
                workflow_id=camdetect.LOCAL_MODEL_ID,
                max_overlap=overlap,
            )
            row["pred"][name] = _counts(result["kept"])
            row["pred"][name]["max_overlap"] = result["max_overlap"]
            row["pred"][name]["overlap_suppressed"] = result["overlap_suppressed"]
        frames.append(row)
        print(image["t"], {k: row["pred"][k]["cars"] for k, _, _ in SETTINGS}, "gt", image["gt"]["cars"])

    summary = {}
    for name, conf, overlap in SETTINGS:
        applied_overlap = frames[0]["pred"][name].get("max_overlap")
        summary[name] = {
            "min_confidence": conf,
            "max_overlap": applied_overlap,
            "model_id": camdetect.LOCAL_MODEL_ID,
            "cars": _mae(frames, name, "cars"),
            "veh": _mae(frames, name, "veh"),
            "cars_day_night": _slice(frames, name, "cars"),
            "cars_vs_artifact_trained_s": {
                "exact": sum(
                    row["pred"][name]["cars"] == row["artifact_trained_s"]["cars"] for row in frames
                ),
                "n": len(frames),
            },
        }
    payload = {"summary": summary, "frames": frames}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
