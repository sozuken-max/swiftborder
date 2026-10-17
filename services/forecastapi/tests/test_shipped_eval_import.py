"""Mix modules must import from shipped_eval alone (same layout as the Cloud Run image)."""

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def test_shipped_eval_imports_without_repo_eval():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vendor_forecast_mix_eval",
        _REPO / "scripts" / "vendor_forecast_mix_eval.py",
    )
    vendor_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(vendor_mod)
    shipped = vendor_mod.vendor_forecast_mix_eval(_REPO)
    code = """
import sys
from pathlib import Path
shipped = Path(r\"%s\")
sys.path.insert(0, str(shipped))
import layer_b_fcm_xgb_mix as mix
import layer_b_fcm_mlp as fcm
assert mix.MIX_WEIGHTS
assert fcm.Config
""" % (
        shipped,
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=_REPO)
