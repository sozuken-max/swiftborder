"""Top-level shim; implementation in warehouse/layer_b_fcm_mlp.py."""
import importlib
import runpy
import sys

_mod = importlib.import_module("warehouse.layer_b_fcm_mlp")
for _attr in dir(_mod):
    if _attr.startswith("__") and _attr not in ("__doc__", "__all__"):
        continue
    globals()[_attr] = getattr(_mod, _attr)
__doc__ = _mod.__doc__

if __name__ == "__main__":
    runpy.run_module("warehouse.layer_b_fcm_mlp", run_name="__main__")
