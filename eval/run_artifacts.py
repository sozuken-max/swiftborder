"""Per-run output directories under eval/runs/ with manifest metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
LATEST_POINTER = RUNS_ROOT / "LATEST.json"


def make_run_id(components: Sequence[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "-".join(c for c in components if c) or "eval"
    return f"{ts}_{slug}"


def create_run_dir(
    *,
    components: Sequence[str],
    run_id: Optional[str] = None,
    runs_root: Optional[Path] = None,
    exist_ok: bool = False,
) -> Path:
    root = runs_root or RUNS_ROOT
    run_id = run_id or make_run_id(components)
    run_dir = root / run_id
    if run_dir.exists() and not exist_ok:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    return run_dir


def subdir(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    manifest.setdefault("run_id", run_dir.name)
    manifest.setdefault(
        "created_at_utc",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    path = run_dir / "run.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def write_run_readme(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    lines = [
        f"# Eval run `{manifest.get('run_id', run_dir.name)}`",
        "",
        f"Created (UTC): {manifest.get('created_at_utc', '')}",
        "",
    ]
    if manifest.get("offline"):
        off = manifest["offline"]
        ds = off.get("dataset", {})
        lines.extend(
            [
                "## Offline (sklearn XGB, 60 min)",
                "",
                f"- **Table:** `{ds.get('source', '')}`",
                f"- **Cache:** `{ds.get('cache_path', '')}` (refreshed: {ds.get('refreshed_from_bq', False)})",
                f"- **Rows:** {ds.get('canonical_rows', '?')} canonical; {ds.get('route_rows', '?')} after route `{ds.get('route_id', '')}`",
                f"- **Models:** {', '.join(off.get('models', []))}",
                f"- **Horizon:** {off.get('horizon_minutes', 60)} min",
                "",
                "### Figures (`offline/`)",
                "",
            ]
        )
        for art in off.get("artifacts", []):
            lines.append(f"- `{art}`")
        lines.append("")

    if manifest.get("bqml"):
        bq = manifest["bqml"]
        ds = bq.get("dataset", {})
        lines.extend(
            [
                "## BQML serve audit (30 min)",
                "",
                f"- **Project:** `{ds.get('project', '')}`",
                f"- **View:** `{ds.get('view', '')}`",
                f"- **Hold-out:** {ds.get('holdout_days', '?')} days",
                f"- **Window:** {ds.get('window_start', '')} .. {ds.get('window_end', '')}",
                f"- **Models:** {', '.join(bq.get('models', []))}",
                "",
                "### Figures (`bqml/`)",
                "",
            ]
        )
        for art in bq.get("artifacts", []):
            lines.append(f"- `{art}`")
        lines.append("")

    lines.append("Machine-readable metadata: [`run.json`](run.json).")
    path = run_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_latest_pointer(
    run_dir: Path,
    manifest: Dict[str, Any],
    *,
    runs_root: Optional[Path] = None,
) -> Path:
    root = runs_root or RUNS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        rel = Path(run_dir.name)
    payload = {
        "run_id": manifest.get("run_id", run_dir.name),
        "path": rel.as_posix(),
        "created_at_utc": manifest.get("created_at_utc"),
        "components": manifest.get("components", []),
    }
    pointer = root / "LATEST.json"
    pointer.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return pointer


def artifact_relpath(run_dir: Path, path: Path) -> str:
    return path.relative_to(run_dir).as_posix()
