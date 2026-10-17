#!/usr/bin/env python3
"""Compute artifact detector MAE from ``artifact_counts.json`` (report table)."""

from __future__ import annotations

import json
from pathlib import Path

from layer_a_count_lib import RUN_DIR, day_night_slice, load_artifact, mae_rows

OUT = RUN_DIR / "artifact_summary.json"


def main() -> None:
    artifact = load_artifact()
    images = artifact["images"]
    summary = {}
    for model in artifact["models"]:
        key = model["key"]
        rows = []
        for image in images:
            pred = image["pred"][key]
            rows.append(
                {
                    "t": image["t"],
                    "gt": image["gt"],
                    "cars": pred["cars"],
                    "veh": pred["veh"],
                }
            )
        summary[key] = {
            "name": model["name"],
            "conf": model["conf"],
            "cars": mae_rows(rows, "cars", "cars", nested=False),
            "veh": mae_rows(rows, "veh", "veh", nested=False),
            "cars_day_night": day_night_slice(rows, "cars", "cars", nested=False),
        }
    OUT.write_text(json.dumps({"summary": summary, "n": len(images)}, indent=2), encoding="utf-8")
    print(json.dumps({k: v["cars"]["mae"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
