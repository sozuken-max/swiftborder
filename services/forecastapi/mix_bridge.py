"""Fit and score the equal FCM+MLP / xgb[maps] mix with the eval module's own functions.

Cloud Build copies ``eval/layer_b_fcm_mlp.py`` and ``eval/layer_b_fcm_xgb_mix.py`` into
``shipped_eval/`` before the image is packed (the image context is this directory).
A checkout uses ``eval/`` at the repo root. Both paths import the same module, so
``mix_of``, ``fit_day`` and ``predict_direction`` are not reimplemented here.
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import local_models as lm

_HERE = Path(__file__).resolve().parent


def _eval_dir():
    shipped = _HERE / "shipped_eval"
    if (shipped / "layer_b_fcm_xgb_mix.py").is_file() and (shipped / "layer_b_fcm_mlp.py").is_file():
        return shipped
    repo_eval = _HERE.parents[1] / "eval"
    if (repo_eval / "layer_b_fcm_xgb_mix.py").is_file():
        return repo_eval
    raise RuntimeError(
        "mix sources missing: expected services/forecastapi/shipped_eval "
        "or eval/layer_b_fcm_xgb_mix.py"
    )


_EVAL = str(_eval_dir())
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import layer_b_fcm_xgb_mix as mixmod  # noqa: E402  (path is chosen before this import)

MIX_ID = "mix[fcm_mlp+xgb]"
# HTTP component id for the maps member. The module's weight key is ``xgb_maps``.
_XGB_COMPONENT = "xgb[maps]"
WEIGHTS = {"fcm_mlp": float(mixmod.MIX_WEIGHTS["fcm_mlp"]), _XGB_COMPONENT: float(mixmod.MIX_WEIGHTS["xgb_maps"])}
COMPONENTS = [{"model": name, "weight": weight} for name, weight in WEIGHTS.items()]


def combine(fcm_values, xgb_values):
    """The module's ``mix_of`` (0.5 / 0.5, NaN when either input is missing)."""
    return mixmod.mix_of(np.asarray(fcm_values, dtype=float), np.asarray(xgb_values, dtype=float))


def version_for(now):
    """Same daily cutoff string the other local models use (labels closed before 00:00 SGT)."""
    day_start = lm.serving_day_start(now)
    return "labels_before=" + day_start.tz_convert(lm.SGT).isoformat()


def _aware(ts):
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.to_pydatetime()


def _records(frame):
    if frame is None or len(frame) == 0:
        return []
    records = []
    for row in frame.itertuples(index=False):
        dur = getattr(row, "dur_min", None)
        ratio = getattr(row, "congestion_ratio", None)
        speed = getattr(row, "speed_kmh", None)
        records.append(
            {
                "direction": row.direction,
                "bin_ts": _aware(row.bin_ts),
                "dur_min": None if pd.isna(dur) else float(dur),
                "congestion_ratio": None if pd.isna(ratio) else float(ratio),
                "speed_kmh": None if pd.isna(speed) else float(speed),
            }
        )
    return records


def _bins(frame, cfg):
    records = _records(frame)
    if not records:
        raise mixmod.ConfigError("no bins for the mix")
    times = [row["bin_ts"] for row in records]
    grid = mixmod.fcm.Grid.spanning(min(times), max(times))
    history_first, _ = mixmod.fcm.sgt_day_bounds(
        mixmod.fcm.parse_sgt_date(cfg.history_start), mixmod.fcm.parse_sgt_date(cfg.history_start)
    )
    cut = grid.start_unix > mixmod.fcm.unix(history_first)
    return mixmod.bins_from_rows(records, grid, cut=cut), grid


def _config(cfg):
    chosen = mixmod.MixConfig() if cfg is None else cfg
    chosen.validate()
    return chosen


def fit_models(frame, day_start, cfg=None, protect_labels=True):
    """Daily refit. ``protect_labels`` matches the module's live train (the October window stays masked)."""
    chosen = _config(cfg)
    bins, grid = _bins(frame, chosen)
    cutoff = _aware(day_start)
    data = mixmod.build_data(
        chosen, bins, np.arange(grid.n), label_cutoff=cutoff, layer_a={}, protect_labels=protect_labels
    )
    serve_day = mixmod.fcm.date_to_epoch_day(mixmod.fcm.sgt_date(cutoff))
    models, _, diagnostics = mixmod.fit_day(chosen, data, serve_day)
    return {"models": models, "diagnostics": diagnostics, "cfg": chosen}


def predict_value(fitted, history, latest, direction, origin_ts, cfg=None):
    """One origin. Returns ``(minutes, parts)`` or ``(None, parts)`` when the mix is undefined (after a gap)."""
    chosen = fitted["cfg"] if cfg is None else _config(cfg)
    combined = history if latest is None or len(latest) == 0 else pd.concat([history, latest], ignore_index=True)
    bins, grid = _bins(combined, chosen)
    origin = _aware(origin_ts)
    index = grid.index(mixmod.fcm.floor_bin(origin))
    # Labels are not used at serve time. The cutoff only has to be a real timestamp.
    data = mixmod.build_data(
        chosen, bins, np.asarray([index]), label_cutoff=origin, layer_a={}, protect_labels=False
    )
    pred = mixmod.predict_direction(fitted["models"], data[direction], np.asarray([0]))
    mix_value = float(pred["mix"][0])
    rows = fitted["diagnostics"].get("xgb_rows", {}).get("maps")
    parts = {
        "fcm_mlp": float(pred["fcm_mlp"][0]),
        "xgb_maps": float(pred["xgb_maps"][0]),
        "mix": mix_value,
        "training_rows": int(rows) if rows is not None else None,
        "eval_id": mixmod.MODEL_NAME,
    }
    if not np.isfinite(mix_value):
        return None, parts
    return mix_value, parts
