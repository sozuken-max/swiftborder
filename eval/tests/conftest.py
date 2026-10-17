"""Register legacy flat imports (``import joined``) for the eval test suite."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from _flat_imports import register_flat_aliases

register_flat_aliases()
