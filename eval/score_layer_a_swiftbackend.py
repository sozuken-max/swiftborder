#!/usr/bin/env python3
"""Rescore the 15-frame set with live ``swiftbackend`` ``model=v6`` (Roboflow workflow).

Uses the public detect endpoint with ``camera_id``, ``date_time``, ``model=v6``,
``confidence=0.2``, and default overlap off (Roboflow path). Fifteen billed calls.

Writes ``runs/layer_a_count/roboflow_v6_conf02.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from layer_a_count_lib import RUN_DIR, day_night_slice, load_artifact, mae_rows, run_provenance

OUT = RUN_DIR / "roboflow_v6_conf02.json"
DEFAULT_BASE = os.environ.get(
    "SWIFTBACKEND_URL",
    "https://swiftbackend-vyenkfblbq-ew.a.run.app/",
)
MIN_CONFIDENCE = 0.2


def _swiftbackend_revision() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                "swiftbackend",
                "--project=swiftborder",
                "--region=europe-west1",
                "--format=value(status.latestReadyRevisionName)",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _detect(base: str, t: str) -> dict:
    query = urllib.parse.urlencode(
        {
            "camera_id": "2701",
            "date_time": t,
            "model": "v6",
            "confidence": str(MIN_CONFIDENCE),
            "format": "json",
        }
    )
    url = base.rstrip("/") + "/?" + query
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    artifact = load_artifact()
    frames = []
    workflow_id = None
    for image in artifact["images"]:
        payload = _detect(base, image["t"])
        preds = payload.get("predictions") or []
        workflow_id = payload.get("workflow_id") or workflow_id
        by_class: dict[str, int] = {}
        for pred in preds:
            cls = pred.get("class") or "unknown"
            by_class[cls] = by_class.get(cls, 0) + 1
        cars = by_class.get("car", 0)
        frames.append(
            {
                "t": image["t"],
                "gcs_path": image["gcs_path"],
                "gt": image["gt"],
                "artifact_v6": image["pred"]["roboflow_v6"],
                "cars": cars,
                "veh": len(preds),
                "by_class": by_class,
                "min_confidence": payload.get("min_confidence", MIN_CONFIDENCE),
                "max_overlap": payload.get("max_overlap"),
                "workflow_id": payload.get("workflow_id"),
                "overlap_suppressed": payload.get("overlap_suppressed"),
                "source_image": payload.get("source_image"),
            }
        )
        print(image["t"], "cars", cars, "gt", image["gt"]["cars"])

    summary = {
        "cars": mae_rows(frames, "cars", nested=False),
        "veh": mae_rows(frames, "veh", nested=False),
        "cars_day_night": day_night_slice(frames, "cars", nested=False),
    }
    payload = {
        "run": run_provenance(
            {
                "scorer": "score_layer_a_swiftbackend.py",
                "swiftbackend_url": base,
                "swiftbackend_revision": _swiftbackend_revision(),
                "model": "v6",
                "min_confidence": MIN_CONFIDENCE,
                "max_overlap": None,
                "workflow_id": workflow_id,
            }
        ),
        "model": "roboflow_v6",
        "min_confidence": MIN_CONFIDENCE,
        "max_overlap": None,
        "n": len(frames),
        "summary": summary,
        "frames": frames,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
