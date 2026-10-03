"""One-off check: do the local replicas (``bq_replica``) forecast like the BigQuery ML models?

Scores BQML ``lin_h30`` / ``xgb_h30`` (``ML.PREDICT`` rows from ``layer_b``) and their frozen local
replicas on the same out-of-sample rows. Reports MAE per model and direction, how far the two
forecasts are apart row by row, and a paired comparison (Diebold-Mariano, day-block CI, Holm over the
two pairs).

**Equivalence rule:** a replica counts as equivalent when the 95% day-block CI of its MAE difference
from BQML lies entirely inside +/- ``EQUIVALENCE_MARGIN_MIN`` (the practical threshold for serving
decisions in docs/roadmap.md). If both are equivalent, the evaluation can use the replicas and drop
BigQuery ML without losing the September models.

    python generate_comparison_plots.py --bqml-only --window-end "2026-09-30 23:50" --bqml-parity
    python promote_report_run.py <run_id> --target parity-bqml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import bq_replica as bq

EQUIVALENCE_MARGIN_MIN = 0.5
PAIRS = (("lin_h30", "lin"), ("xgb_h30", "xgb"))


def parity_frame(bqml_rows, frame: pd.DataFrame) -> pd.DataFrame:
    """BQML scored rows joined to the local feature table on (direction, bin_ts); labels must match."""
    b = pd.DataFrame(bqml_rows)
    b["bin_ts"] = pd.to_datetime(b["bin_ts"], utc=True)
    f = frame.copy()
    f["bin_ts"] = pd.to_datetime(f["bin_ts"], utc=True)
    cols = sorted(set(bq.XGB_FEATURES) - {"is_my_to_sg"})
    m = b.merge(f[["direction", "bin_ts", "y_30", *cols]], on=["direction", "bin_ts"], suffixes=("", "_local"), validate="one_to_one")
    diff = float((m["y_30"] - m["y_30_local"]).abs().max()) if len(m) else float("nan")
    if not diff <= 1e-6:
        raise RuntimeError(f"BQML and local labels disagree (max abs diff {diff})")
    return m.drop(columns=["y_30_local"]).sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def add_replica_predictions(test: pd.DataFrame, frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    train = bq.frozen_training_rows(frame)
    lin, xgb = bq.fit_lin(train), bq.fit_xgb(train)
    out = test.copy()
    out["lin_h30_local"] = bq.predict_lin(lin, out)
    out["xgb_h30_local"] = bq.predict_xgb(xgb, out)
    return out, int(len(train))


def agreement(local: np.ndarray, bqml: np.ndarray) -> Dict[str, float]:
    d = np.asarray(local, float) - np.asarray(bqml, float)
    return {
        "mean_abs_diff_min": float(np.mean(np.abs(d))),
        "p95_abs_diff_min": float(np.quantile(np.abs(d), 0.95)),
        "signed_mean_diff_min": float(np.mean(d)),
        "correlation": float(np.corrcoef(local, bqml)[0, 1]),
    }


def parity_component(run_dir: Path, *, bqml_result: Dict[str, Any], frame: pd.DataFrame, alpha: float = 0.05) -> Tuple[List[Path], Dict[str, Any]]:
    from plots import plot_mae_diff_forest
    from run_artifacts import artifact_relpath, metric_row, subdir
    from significance import apply_holm, compare_absolute_errors, format_comparison_table

    test = parity_frame(bqml_result["holdout"].rows, frame)
    test, n_train = add_replica_predictions(test, frame)
    y = test["y_30"].to_numpy()
    candidates = {
        "Persistence": test["Persistence"].to_numpy(),
        "lin_h30 (BQML)": test["lin_h30"].to_numpy(),
        "lin_h30 (local replica)": test["lin_h30_local"].to_numpy(),
        "xgb_h30 (BQML)": test["xgb_h30"].to_numpy(),
        "xgb_h30 (local replica)": test["xgb_h30_local"].to_numpy(),
    }
    metrics = []
    for direction in ("both", "SG_TO_MY", "MY_TO_SG"):
        k = np.ones(len(test), bool) if direction == "both" else (test["direction"] == direction).to_numpy()
        for name, p in candidates.items():
            e = y[k] - p[k]
            metrics.append(metric_row(name, f"{direction}/all", int(k.sum()), float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), direction=direction))
    comps = apply_holm(
        [
            compare_absolute_errors(
                y, test[f"{m}_local"].to_numpy(), test[m].to_numpy(),
                label_challenger=f"{m} (local replica)", label_reference=f"{m} (BQML)",
                timestamps=list(test["bin_ts"]), groups=list(test["direction"]), horizon_steps=3, alpha=alpha,
                day_utc_offset_hours=8.0,
            )
            for m, _ in PAIRS
        ]
    )
    print("BQML parity (Holm over the two pairs):")
    print(format_comparison_table(comps))
    verdict = {}
    for (m, _), c in zip(PAIRS, comps):
        inside = -EQUIVALENCE_MARGIN_MIN < c.bootstrap_ci_low_min and c.bootstrap_ci_high_min < EQUIVALENCE_MARGIN_MIN
        verdict[m] = {
            "agreement": agreement(test[f"{m}_local"].to_numpy(), test[m].to_numpy()),
            "mae_diff_min": c.mean_ae_diff_min,
            "ci_min": [c.bootstrap_ci_low_min, c.bootstrap_ci_high_min],
            "equivalent_within_margin": bool(inside),
        }
    out = subdir(run_dir, "parity")
    written = [*plot_mae_diff_forest(comps, out / "parity-mae-diff.png", title="Local replica minus BQML, 30 min (day-block CI; margin +/- 0.5 min)")]
    days = sorted(test["bin_ts"].dt.tz_convert("Asia/Singapore").dt.strftime("%Y-%m-%d").unique())
    meta = {
        "dataset": {"rows_scored": int(len(test)), "frozen_training_rows": n_train, "days_sgt": days, "source": "bqml.holdout rows joined to features.build"},
        "window": {"timezone": "Asia/Singapore", "basis": "same rows as the bqml component", "start": days[0], "end": days[-1]},
        "models": list(candidates),
        "horizon_minutes": 30,
        "config": {"replica": bq.settings(), "equivalence_margin_min": EQUIVALENCE_MARGIN_MIN,
                   "equivalence_rule": "95% day-block CI of the MAE difference inside +/- margin"},
        "verdict": verdict,
        "metrics": metrics,
        "significance": [dict(c.to_dict(), family="parity") for c in comps],
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta
