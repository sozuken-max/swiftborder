"""Per-run output directories under eval/runs/ with a versioned manifest (``run.json``).

Schema v2 (``SCHEMA_VERSION``) records enough provenance to reproduce or audit a run:

- top level: ``schema_version``, ``run_id``, ``created_at_utc``, ``components``, ``provenance``
  (git SHA, dirty flag, SHA-256 of the eval/sql source files, Python and package versions);
- one block per component (``offline``, ``bqml``, ``joined``, ``layer_a``) with ``dataset``,
  ``window``, ``models``, ``horizon_minutes``, ``metrics`` (rows of candidate x slice),
  ``significance`` (``ComparisonResult.to_dict()`` rows) and ``artifacts`` (paths relative to the
  run directory).

``validate_manifest`` lists every problem; ``promote_report_run.py`` refuses a run that has any.
Paths are repo-relative; absolute paths are rejected so a manifest is portable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
RUNS_ROOT = EVAL_ROOT / "runs"
LATEST_POINTER = RUNS_ROOT / "LATEST.json"
SCHEMA_VERSION = 2
KNOWN_COMPONENTS = ("offline", "bqml", "joined", "deep", "fuzzy", "ensemble", "parity", "layer_a")
COMPONENT_KEYS = ("dataset", "window", "models", "horizon_minutes", "metrics", "significance", "artifacts")
METRIC_KEYS = ("candidate", "slice", "n", "mae_min", "rmse_min")
LAYER_A_METRIC_KEYS = ("candidate", "slice", "n", "map50", "map50_95", "precision", "recall", "count_mae")
FUZZY_METRIC_KEYS = ("candidate", "slice", "n", "accuracy", "macro_f1", "severe_error_rate")
METRIC_KEYS_BY_COMPONENT = {"layer_a": LAYER_A_METRIC_KEYS, "fuzzy": FUZZY_METRIC_KEYS}
SIGNIFICANCE_KEYS = (
    "label_challenger",
    "label_reference",
    "n",
    "mean_ae_diff_min",
    "bootstrap_ci_low_min",
    "bootstrap_ci_high_min",
    "dm_pvalue",
    "decision",
)
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "google-cloud-bigquery", "tensorflow")
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?:Users|home|tmp|var)/|\\\\)")


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


# --- provenance -------------------------------------------------------------------


def repo_relpath(path: Path, root: Optional[Path] = None) -> str:
    """Repo-relative POSIX path (raises if the path is outside the repo)."""
    return Path(path).resolve().relative_to((root or REPO_ROOT).resolve()).as_posix()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    return {"path": repo_relpath(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def code_fingerprint(paths: Optional[Iterable[Path]] = None) -> str:
    """SHA-256 over eval/*.py and sql/**/*.sql (names + contents): identifies code even when dirty."""
    if paths is None:
        paths = sorted(EVAL_ROOT.glob("*.py")) + sorted((REPO_ROOT / "sql").rglob("*.sql"))
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(repo_relpath(p).encode("utf-8") + b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def provenance() -> Dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    env = installed_distributions()
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "code_sha256": code_fingerprint(),
        "python": platform.python_version(),
        "packages": packages,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "environment": {
            "distributions": env,
            "sha256": hashlib.sha256("\n".join(env).encode("utf-8")).hexdigest(),
            "lock_file": "requirements-lock-py311.txt",
        },
    }


def installed_distributions() -> List[str]:
    """Every installed distribution as ``name==version``, sorted (the full resolved environment)."""
    seen = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if name:
            seen[name.lower()] = f"{name}=={dist.version}"
    return [seen[k] for k in sorted(seen)]


# --- multiplicity -----------------------------------------------------------------


def multiplicity_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Sensitivity check: Holm over **every** significance row in the run, not per family.

    The reported decisions use one Holm family per harness question. This recomputes each decision
    with a single run-wide family (two-sided DM p-values, same CI and block rules) and lists the rows
    whose decision would change.
    """
    from significance import holm_adjust

    rows = []
    for name in manifest.get("components", []):
        for i, s in enumerate((manifest.get(name) or {}).get("significance") or []):
            p = s.get("dm_pvalue_two_sided")
            if p is None or p != p:  # missing or NaN
                continue
            rows.append((name, i, s))
    adj = holm_adjust([s["dm_pvalue_two_sided"] for _, _, s in rows])
    changed = []
    counts: Dict[str, int] = {}
    for (name, i, s), p in zip(rows, adj):
        decision = s["decision"]
        if decision != "insufficient data":
            alpha = float(s.get("alpha", 0.05))
            diff, lo, hi = s["mean_ae_diff_min"], s["bootstrap_ci_low_min"], s["bootstrap_ci_high_min"]
            if p < alpha and diff < 0 and hi < 0:
                decision = "challenger"
            elif p < alpha and diff > 0 and lo > 0:
                decision = "reference"
            else:
                decision = "not significant"
        key = f"{s['decision']} -> {decision}"
        counts[key] = counts.get(key, 0) + 1
        if decision != s["decision"]:
            changed.append(
                {
                    "component": name,
                    "family": s.get("family", name),
                    "challenger": s["label_challenger"],
                    "reference": s["label_reference"],
                    "family_holm_p": s.get("holm_adjusted_p"),
                    "global_holm_p": p,
                    "decision": s["decision"],
                    "decision_global": decision,
                }
            )
    return {
        "method": "Holm over all significance rows in the run (two-sided DM p), same CI and block rules",
        "n_comparisons": len(rows),
        "transitions": counts,
        "changed": changed,
    }


# --- manifest I/O -----------------------------------------------------------------


def new_manifest(components: Sequence[str]) -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "components": list(components), "provenance": provenance()}


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    manifest.setdefault("schema_version", SCHEMA_VERSION)
    manifest.setdefault("provenance", provenance())
    manifest.setdefault("run_id", run_dir.name)
    manifest.setdefault("created_at_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = run_dir / "run.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def _walk_strings(value: Any, prefix: str = "") -> Iterable[tuple]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_strings(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk_strings(v, f"{prefix}[{i}]")
    elif isinstance(value, str):
        yield prefix, value


def _artifact_problem(run_dir: Path, art: str) -> Optional[str]:
    """Return why ``art`` is not a file inside ``run_dir``, or None when it is.

    ``resolve`` follows ``..`` and symlinks. A path that lands outside the run
    directory is rejected even when that outside file exists, so promotion cannot
    record an artifact ``copytree`` will not copy.
    """
    root = Path(run_dir).resolve()
    try:
        resolved = (root / str(art)).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError):
        return "escapes the run directory"
    if not resolved.is_file():
        return "not found in run directory"
    return None


def validate_manifest(manifest: Dict[str, Any], run_dir: Optional[Path] = None) -> List[str]:
    """Return a list of problems (empty list = valid schema v2 manifest)."""
    errors: List[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("run_id", "created_at_utc"):
        if not manifest.get(key):
            errors.append(f"missing {key}")
    prov = manifest.get("provenance") or {}
    for key in ("git_sha", "git_dirty", "code_sha256"):
        if key not in prov:
            errors.append(f"provenance.{key} missing")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        errors.append("components must be a non-empty list")
        components = []
    for name in components:
        if name not in KNOWN_COMPONENTS:
            errors.append(f"unknown component {name!r}")
            continue
        block = manifest.get(name)
        if not isinstance(block, dict):
            errors.append(f"component {name!r} listed but missing")
            continue
        for key in COMPONENT_KEYS:
            if key not in block:
                errors.append(f"{name}.{key} missing")
        window = block.get("window") or {}
        if not (window.get("start") and window.get("end")):
            errors.append(f"{name}.window needs start and end")
        metrics = block.get("metrics") or []
        if not metrics:
            errors.append(f"{name}.metrics is empty")
        required = METRIC_KEYS_BY_COMPONENT.get(name, METRIC_KEYS)
        for i, row in enumerate(metrics):
            missing = [k for k in required if k not in row]
            if missing:
                errors.append(f"{name}.metrics[{i}] missing {missing}")
        for i, row in enumerate(block.get("significance") or []):
            missing = [k for k in SIGNIFICANCE_KEYS if k not in row]
            if missing:
                errors.append(f"{name}.significance[{i}] missing {missing}")
        if run_dir is not None:
            for art in block.get("artifacts") or []:
                problem = _artifact_problem(run_dir, art)
                if problem:
                    errors.append(f"{name}.artifacts: {art} {problem}")
    for where, text in _walk_strings(manifest):
        if _ABSOLUTE.match(text):
            errors.append(f"absolute path at {where}: {text}")
    return errors


def write_run_readme(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    prov = manifest.get("provenance") or {}
    lines = [
        f"# Eval run `{manifest.get('run_id', run_dir.name)}`",
        "",
        f"Created (UTC): {manifest.get('created_at_utc', '')}. Schema v{manifest.get('schema_version', '?')}.",
        f"Git `{(prov.get('git_sha') or '?')[:12]}` (dirty: {prov.get('git_dirty')}); code SHA-256 `{(prov.get('code_sha256') or '?')[:12]}`.",
        "",
    ]
    for name in manifest.get("components", []):
        block = manifest.get(name) or {}
        window = block.get("window") or {}
        lines.extend(
            [
                f"## {name}",
                "",
                f"- **Horizon:** {block.get('horizon_minutes', '?')} min",
                f"- **Window:** {window.get('start', '?')} .. {window.get('end', '?')}",
                f"- **Models:** {', '.join(block.get('models', []))}",
                "",
            ]
        )
        metrics = block.get("metrics") or []
        headline = [m for m in metrics if m.get("slice") in ("holdout", "both/all", "all") and "mae_min" in m]
        if headline:
            lines.extend(["| Candidate | Slice | n | MAE (min) | RMSE (min) |", "| --- | --- | --- | --- | --- |"])
            for m in headline:
                lines.append(
                    f"| {m['candidate']} | {m['slice']} | {m['n']} | {m['mae_min']:.3f} | {m['rmse_min']:.3f} |"
                )
            lines.append("")
        cls = [m for m in metrics if m.get("slice") == "holdout" and "accuracy" in m]
        if cls:
            lines.extend(
                [
                    "| Candidate | Slice | n | Accuracy | Macro-F1 | Severe errors | RPS |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for m in cls:
                rps = f"{m['rps']:.3f}" if m.get("rps") is not None else "-"
                lines.append(
                    f"| {m['candidate']} | {m['slice']} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                    f"{m['severe_error_rate']:.3f} | {rps} |"
                )
            lines.append("")
        sig = block.get("significance") or []
        if sig and block.get("loss_for_significance"):
            lines.extend([f"Significance loss: {block['loss_for_significance']}.", ""])
        if sig:
            lines.extend(
                [
                    "| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for s in sig:
                holm = s.get("holm_adjusted_p")
                lines.append(
                    f"| {s['label_challenger']} | {s['label_reference']} | {s['n']} | {s['mean_ae_diff_min']:.3f} | "
                    f"[{s['bootstrap_ci_low_min']:.3f}, {s['bootstrap_ci_high_min']:.3f}] | "
                    f"{_p(s.get('dm_pvalue_two_sided', s.get('dm_pvalue')))} | {_p(holm)} | {s['decision']} |"
                )
            lines.append("")
        arts = block.get("artifacts") or []
        if arts:
            lines.append("Figures (each PNG has a same-stem CSV of the plotted series):")
            lines.extend(f"- `{a}`" for a in arts)
            lines.append("")
    mult = manifest.get("multiplicity")
    if mult:
        lines.extend(
            [
                "## Multiplicity check",
                "",
                f"Holm over all {mult['n_comparisons']} comparisons in the run (instead of per family) changes "
                f"{len(mult['changed'])} decision(s):",
                "",
            ]
        )
        for c in mult["changed"]:
            lines.append(f"- {c['challenger']} vs {c['reference']}: {c['decision']} -> {c['decision_global']} (global Holm p {_p(c['global_holm_p'])})")
        lines.append("")
    lines.append("Machine-readable metadata: [`run.json`](run.json).")
    path = run_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _p(p: Any) -> str:
    if p is None:
        return "-"
    return f"{p:.2g}" if p < 1e-3 else f"{p:.4f}"


def update_latest_pointer(
    run_dir: Path,
    manifest: Dict[str, Any],
    *,
    runs_root: Optional[Path] = None,
) -> Path:
    """Write runs/LATEST.json (gitignored local convenience pointer)."""
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
    return Path(path).relative_to(run_dir).as_posix()


def metric_row(candidate: str, slice_name: str, n: int, mae_min: float, rmse_min: float, **extra: Any) -> Dict[str, Any]:
    row = {"candidate": candidate, "slice": slice_name, "n": int(n), "mae_min": float(mae_min), "rmse_min": float(rmse_min)}
    row.update(extra)
    return row
