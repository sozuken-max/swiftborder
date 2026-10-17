#!/usr/bin/env python3
"""Run an eval harness by short name: ``python run.py joined --start ...`` (same as ``python -m lb.joined``)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _flat_imports import MODULE_ALIASES, cli_module_names, register_flat_aliases


def _usage() -> str:
    names = ", ".join(cli_module_names())
    return (
        "usage: python run.py <module> [args...]\n"
        f"modules: {names}\n"
        "example: python run.py generate_comparison_plots --joined --ensemble\n"
        "alt:      python -m lb.joined --start 2026-09-13 --end 2026-09-30"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    name = argv[0]
    qualified = MODULE_ALIASES.get(name)
    if qualified is None:
        print(f"unknown module: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    register_flat_aliases()
    sys.argv = [f"{qualified}.py"] + argv[1:]
    runpy.run_module(qualified, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
