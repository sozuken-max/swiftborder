"""Shared helpers for the 15-frame Layer A count comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = EVAL_ROOT / "runs" / "layer_a_count"
ARTIFACT_PATH = RUN_DIR / "artifact_counts.json"


def load_artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def gcs_object_name(gcs_path: str) -> str:
    return gcs_path.rsplit("/", 1)[-1]


def mae_rows(rows: list, pred_cars: str, metric: str = "cars", nested: bool = True) -> dict[str, Any]:
    if nested:
        errors = [row["pred"][pred_cars][metric] - row["gt"][metric] for row in rows]
    else:
        errors = [row[pred_cars] - row["gt"][metric] for row in rows]
    n = len(errors)
    return {
        "mae": round(sum(abs(e) for e in errors) / n, 2),
        "bias": round(sum(errors) / n, 2),
        "exact": sum(e == 0 for e in errors),
        "within_2": sum(abs(e) <= 2 for e in errors),
        "n": n,
    }


def day_night_slice(rows: list, pred_key: str, metric: str = "cars", nested: bool = True) -> dict[str, dict]:
    def keep(row: dict, label: str) -> bool:
        hour = int(row["t"][11:13])
        day = 7 <= hour <= 18
        return day if label == "day" else not day

    return {
        label: mae_rows([row for row in rows if keep(row, label)], pred_key, metric, nested=nested)
        for label in ("day", "night")
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_head(EVAL_ROOT.parent),
        "artifact_path": str(ARTIFACT_PATH.relative_to(EVAL_ROOT.parent)),
        "n_images": len(load_artifact()["images"]),
    }
    if extra:
        block.update(extra)
    return block
