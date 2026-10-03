#!/usr/bin/env python3
"""Copy a validated eval run into eval/runs/report/ (the committed citation target).

Refuses a run whose ``run.json`` fails ``run_artifacts.validate_manifest`` (schema v2, windows,
metrics, significance rows, artifact files, no absolute paths), and a run whose
``provenance.code_sha256`` differs from the current eval/sql sources. Refuses a run made from a dirty
working tree unless ``--allow-dirty`` is given; the override is recorded in ``SOURCE_RUN.json``
and the README banner, and ``provenance.code_sha256`` still identifies the exact code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from run_artifacts import RUNS_ROOT, code_fingerprint, load_manifest, validate_manifest

# Committed snapshots: ``report`` is the headline citation target; ``report-confirm`` holds the
# confirmation run on the period no model choice was tuned on (docs/roadmap.md, frozen-window plan);
# ``parity-bqml`` holds the one-off BQML vs local-replica check (bqml_parity.py).
TARGETS = ("report", "report-confirm", "parity-bqml")


class PromotionError(RuntimeError):
    pass


def check_promotable(source_run_dir: Path, *, allow_dirty: bool = False) -> List[str]:
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    if not (source_run_dir / "run.json").is_file():
        raise FileNotFoundError(f"missing run.json in {source_run_dir}")
    manifest = load_manifest(source_run_dir)
    problems = validate_manifest(manifest, source_run_dir)
    prov = manifest.get("provenance") or {}
    if prov.get("git_dirty") and not allow_dirty:
        problems.append("run was made from a dirty working tree (commit first, or pass --allow-dirty)")
    current = code_fingerprint()
    if prov.get("code_sha256") and prov["code_sha256"] != current:
        problems.append(
            f"code changed since the run (run {prov['code_sha256'][:12]}, tree {current[:12]}); re-run before promoting"
        )
    return problems


def promote(
    source_run_dir: Path,
    report_dir: Path,
    *,
    allow_dirty: bool = False,
    runs_root: Optional[Path] = None,
) -> None:
    problems = check_promotable(source_run_dir, allow_dirty=allow_dirty)
    if problems:
        raise PromotionError("refusing to promote:\n- " + "\n- ".join(problems))
    manifest = load_manifest(source_run_dir)
    dirty = bool((manifest.get("provenance") or {}).get("git_dirty"))

    if report_dir.exists():
        shutil.rmtree(report_dir)
    shutil.copytree(source_run_dir, report_dir)

    root = runs_root or RUNS_ROOT
    try:
        source_path = source_run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        source_path = source_run_dir.name
    (report_dir / "SOURCE_RUN.json").write_text(
        json.dumps(
            {
                "promoted_from": source_run_dir.name,
                "source_path": source_path,
                "git_sha": manifest["provenance"].get("git_sha"),
                "git_dirty": dirty,
                "code_sha256": manifest["provenance"].get("code_sha256"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    readme = report_dir / "README.md"
    if readme.is_file():
        banner = (
            "> **Committed report snapshot.** Ephemeral runs stay gitignored; this folder is the citation "
            f"target for the final report. Promoted from `{source_run_dir.name}`."
        )
        if dirty:
            banner += (
                " **Made from uncommitted code** (`--allow-dirty`): `provenance.code_sha256` in `run.json` "
                "identifies the sources. Re-run and re-promote after the code is committed."
            )
        readme.write_text(banner + "\n\n" + readme.read_text(encoding="utf-8"), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run folder name under eval/runs (default: read eval/runs/LATEST.json)",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="Promote a run made from uncommitted code")
    parser.add_argument("--check", action="store_true", help="Validate only; do not copy")
    parser.add_argument("--target", choices=TARGETS, default="report", help="Snapshot folder under eval/runs (default report)")
    args = parser.parse_args(argv)

    if args.run_id:
        source = RUNS_ROOT / args.run_id
    else:
        latest = RUNS_ROOT / "LATEST.json"
        if not latest.is_file():
            print("No LATEST.json; pass run_id explicitly.", file=sys.stderr)
            return 2
        source = RUNS_ROOT / json.loads(latest.read_text(encoding="utf-8"))["path"]

    if args.check:
        problems = check_promotable(source, allow_dirty=args.allow_dirty)
        for p in problems:
            print(f"- {p}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 1

    try:
        promote(source, RUNS_ROOT / args.target, allow_dirty=args.allow_dirty)
    except PromotionError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Promoted {source.name} -> {RUNS_ROOT / args.target}")
    print(f"Commit eval/runs/{args.target}/ when docs/evaluation.md cites the same numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
