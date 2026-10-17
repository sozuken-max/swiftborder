#!/usr/bin/env python3
"""Copy eval mix shims and dependencies into services/forecastapi/shipped_eval/."""

from __future__ import annotations

import shutil
from pathlib import Path


def vendor_forecast_mix_eval(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    dest = root / "services" / "forecastapi" / "shipped_eval"
    warehouse = dest / "warehouse"
    harness = dest / "harness"
    dest.mkdir(parents=True, exist_ok=True)
    warehouse.mkdir(parents=True, exist_ok=True)
    harness.mkdir(parents=True, exist_ok=True)

    eval_root = root / "eval"
    for name in ("layer_b_fcm_mlp.py", "layer_b_fcm_xgb_mix.py"):
        shutil.copy2(eval_root / name, dest / name)
    for path in (eval_root / "warehouse").glob("*.py"):
        shutil.copy2(path, warehouse / path.name)
    for name in ("__init__.py", "significance.py", "metrics.py"):
        shutil.copy2(eval_root / "harness" / name, harness / name)

    mix = dest / "layer_b_fcm_xgb_mix.py"
    if not mix.is_file() or mix.stat().st_size == 0:
        raise SystemExit(f"missing vendored mix module: {mix}")
    return dest


def main() -> None:
    vendor_forecast_mix_eval()


if __name__ == "__main__":
    main()
