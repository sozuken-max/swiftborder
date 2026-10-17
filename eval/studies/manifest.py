"""Schema-v2 ``run.json`` helpers for exploratory study scripts."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.run_artifacts import code_fingerprint, provenance as harness_provenance


def git_head() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except FileNotFoundError:
        return None


def base_manifest(run_id: str, components: List[str], *, created_at_utc: Optional[str] = None) -> Dict[str, Any]:
    prov = harness_provenance()
    prov["code_sha256"] = code_fingerprint()
    return {
        "schema_version": 2,
        "run_id": run_id,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "components": components,
        "provenance": prov,
    }


def write_run_json(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
