#!/usr/bin/env python3
"""Copy a completed eval run into eval/runs/report/ for git commit (final report sources)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from run_artifacts import RUNS_ROOT


def _relative_to_runs(path: Path) -> str:
    try:
        return path.relative_to(RUNS_ROOT).as_posix()
    except ValueError:
        return path.name


def promote(source_run_dir: Path, report_dir: Path) -> None:
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    if not (source_run_dir / "run.json").is_file():
        raise FileNotFoundError(f"missing run.json in {source_run_dir}")

    if report_dir.exists():
        shutil.rmtree(report_dir)
    shutil.copytree(source_run_dir, report_dir)

    source_id = source_run_dir.name
    (report_dir / "SOURCE_RUN.json").write_text(
        json.dumps(
            {
                "promoted_from": source_id,
                "source_path": _relative_to_runs(source_run_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    readme = report_dir / "README.md"
    if readme.is_file():
        prefix = (
            "> **Committed report snapshot.** Ephemeral runs stay gitignored; "
            "this folder is the citation target for the final report. "
            f"Promoted from `{source_id}`.\n\n"
        )
        text = readme.read_text(encoding="utf-8")
        if not text.startswith("> **Committed report snapshot.**"):
            readme.write_text(prefix + text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run folder name under eval/runs (default: read eval/runs/LATEST.json)",
    )
    args = parser.parse_args(argv)

    if args.run_id:
        source = RUNS_ROOT / args.run_id
    else:
        latest = RUNS_ROOT / "LATEST.json"
        if not latest.is_file():
            print("No LATEST.json; pass run_id explicitly.", file=sys.stderr)
            return 2
        payload = json.loads(latest.read_text(encoding="utf-8"))
        source = RUNS_ROOT / payload["path"]

    report_dir = RUNS_ROOT / "report"
    promote(source, report_dir)
    print(f"Promoted {source.name} -> {report_dir}")
    print("Commit eval/runs/report/ when numbers and figures match evaluation.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
