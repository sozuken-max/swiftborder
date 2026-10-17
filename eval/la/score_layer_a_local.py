#!/usr/bin/env python3
"""Rescore the 15-image Layer A count set with the in-container YOLO26s.

Ground truth is the caption totals in ``runs/layer_a_count/artifact_counts.json``
(the Claude artifact comparison). This script does not compute mAP: the export
has no boxes in git. It runs ``camdetect.detect_frame`` on 1920x1080 originals
from a local directory at the live ``model=local`` point (confidence 0.2,
overlap 0.6) and at confidence 0.30 with overlap off and on.

No Roboflow call. Frames are read from ``--frames-dir`` (file name is the GCS
object name). Download with ``download_layer_a_frames.py`` first.

Writes ``runs/layer_a_count/local_rescore.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
CAMDETECT = REPO_ROOT / "services" / "camdetect"
sys.path.insert(0, str(CAMDETECT))

import main as camdetect  # noqa: E402

from layer_a_count_lib import (  # noqa: E402
    RUN_DIR,
    day_night_slice,
    gcs_object_name,
    load_artifact,
    mae_rows,
    run_provenance,
    sha256_file,
)

OUT = RUN_DIR / "local_rescore.json"

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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: score_layer_a_local.py FRAMES_DIR")
    frames_dir = Path(sys.argv[1])
    onnx_path = Path(camdetect.LOCAL_MODEL_PATH)
    artifact = load_artifact()
    frames = []
    for image in artifact["images"]:
        object_name = gcs_object_name(image["gcs_path"])
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
            "cars": mae_rows(frames, name, "cars"),
            "veh": mae_rows(frames, name, "veh"),
            "cars_day_night": day_night_slice(frames, name, "cars"),
            "cars_vs_artifact_trained_s": {
                "exact": sum(
                    row["pred"][name]["cars"] == row["artifact_trained_s"]["cars"] for row in frames
                ),
                "n": len(frames),
            },
        }
    try:
        import onnxruntime as ort  # noqa: PLC0415

        ort_version = ort.__version__
    except ImportError:
        ort_version = None

    payload = {
        "run": run_provenance(
            {
                "scorer": "score_layer_a_local.py",
                "frames_dir": str(frames_dir),
                "onnx_path": str(onnx_path),
                "onnx_sha256": sha256_file(onnx_path) if onnx_path.is_file() else None,
                "onnxruntime_version": ort_version,
                "local_default_confidence": camdetect.LOCAL_DEFAULT_CONFIDENCE,
                "local_max_overlap": camdetect.LOCAL_MAX_OVERLAP,
            }
        ),
        "summary": summary,
        "frames": frames,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
