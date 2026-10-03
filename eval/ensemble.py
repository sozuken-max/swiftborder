"""Does ensembling or hybridising the Layer B forecasters help?

Two pools, each scored on rows that every member predicts out-of-sample:

**30 min, both directions, 13-30 Sep** (the headline pool). Members are the served BQML models
(``layer_b``: ``lin_h30``, ``xgb_h30``, trained 12 Sep) and the daily-refit Maps-only models
(``joined``: ``ridge[maps]``, ``xgb[maps]``), joined on (direction, bin_ts), plus persistence.
Candidates:

- singles, **served** (what ``v_forecast_recent`` publishes: the ``model_registry`` choice per
  direction, ``lin_h30`` for SG_TO_MY and persistence for MY_TO_SG) and the existing fixed
  ``ensemble_mean`` of ``lin_h30`` and ``xgb_h30``;
- ``mean[models]``: equal-weight mean of the four models (no fitting);
- ``stack``: per-direction convex weights over the pool (persistence included, so the stack can
  shrink toward it), fitted by least absolute deviation on all earlier days of the window
  (expanding, rolling daily origin; a row is usable once its label is observed, bin + 40 min, before
  the test day starts). Days with fewer than ``MIN_FIT_DAYS`` earlier days use equal weights;
- ``select``: per direction, the single member with the lowest MAE over the previous
  ``SELECT_DAYS`` days (a rolling version of the registry);
- ``fuzzy stack`` (hybrid): the fuzzy light / moderate / heavy partition of the current travel
  time (``fuzzy_traffic.LevelPartition``) gates three sets of convex weights, one per level, each
  fitted by membership-weighted LAD on earlier days. The forecast is the membership-weighted sum
  of the three stacked forecasts, so it can trust persistence in free flow and the models in queues.

**60 min, JB -> SG only, offline split** (small pool): XGBoost and the seed-mean LSTM / GRU /
patch Transformer from ``deep_forecast`` on the same five test days; equal means and a rolling
daily stack. With five day-blocks every decision is "insufficient data".

Significance: ``significance.compare_absolute_errors`` (Diebold-Mariano, day-block bootstrap,
directions as separate groups) with Holm over each pool's family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SGT = "Asia/Singapore"
LABEL_LAG_30 = pd.Timedelta(minutes=40)  # y_30 is the bin [t+30, t+40)
MIN_FIT_DAYS = 2
SELECT_DAYS = 3
REGISTRY = {"SG_TO_MY": "lin_h30", "MY_TO_SG": "Persistence"}  # live model_registry, decided 2026-09-12
POOL_30 = ("Persistence", "lin_h30", "xgb_h30", "ridge[maps]", "xgb[maps]")
MODELS_30 = ("lin_h30", "xgb_h30", "ridge[maps]", "xgb[maps]")
BEST_SINGLE_30 = "xgb[maps]"


# --- weight fitting ----------------------------------------------------------------


def lad_simplex_weights(P: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Convex weights (w >= 0, sum 1) minimising sum_i s_i |y_i - P_i w| (a linear programme)."""
    from scipy.optimize import linprog

    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = P.shape
    if n == 0:
        return np.full(k, 1.0 / k)
    s = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    keep = s > 1e-9
    P, y, s = P[keep], y[keep], s[keep]
    n = len(y)
    if n == 0:
        return np.full(k, 1.0 / k)
    from scipy.sparse import csr_matrix, hstack, identity, vstack

    # variables: w (k), e_plus (n), e_minus (n);  P w - e_plus + e_minus = y;  sum w = 1
    c = np.concatenate([np.zeros(k), s, s])
    rows = hstack([csr_matrix(P), -identity(n, format="csr"), identity(n, format="csr")])
    simplex = csr_matrix(np.concatenate([np.ones(k), np.zeros(2 * n)])[None, :])
    A_eq = vstack([rows, simplex]).tocsr()
    b_eq = np.concatenate([y, [1.0]])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (k + 2 * n), method="highs")
    if not res.success:
        return np.full(k, 1.0 / k)
    w = np.clip(res.x[:k], 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)


@dataclass
class RollingResult:
    prediction: np.ndarray
    weights: List[Dict[str, Any]]


def _day_start_utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(SGT).tz_convert("UTC")


def rolling_stack(
    frame: pd.DataFrame,
    members: Sequence[str],
    *,
    time_col: str = "bin_ts",
    label_lag: pd.Timedelta = LABEL_LAG_30,
    gate: Optional[np.ndarray] = None,
    min_fit_days: int = MIN_FIT_DAYS,
) -> RollingResult:
    """Per direction and test day: fit convex LAD weights on earlier days, predict the day.

    ``gate``: optional (n, g) memberships (rows sum to 1). With a gate, one weight vector is fitted
    per gate column (membership-weighted) and the forecast is sum_g gate_g * P w_g.
    """
    P_all = frame[list(members)].to_numpy(dtype=float)
    y_all = frame["y"].to_numpy(dtype=float)
    pred = np.full(len(frame), np.nan)
    log: List[Dict[str, Any]] = []
    k = len(members)
    for direction, g in frame.groupby("direction", sort=False):
        days = sorted(g["date_sgt"].unique())
        for i, day in enumerate(days):
            test_idx = g.index[g["date_sgt"] == day].to_numpy()
            start = _day_start_utc(day)
            fit_idx = g.index[(g[time_col] + label_lag <= start)].to_numpy()
            n_fit_days = len(set(g.loc[fit_idx, "date_sgt"])) if len(fit_idx) else 0
            P_test = P_all[test_idx]
            if n_fit_days < min_fit_days:
                w = np.full((1 if gate is None else gate.shape[1], k), 1.0 / k)
            elif gate is None:
                w = lad_simplex_weights(P_all[fit_idx], y_all[fit_idx])[None, :]
            else:
                w = np.stack([lad_simplex_weights(P_all[fit_idx], y_all[fit_idx], gate[fit_idx, j]) for j in range(gate.shape[1])])
            if gate is None:
                pred[test_idx] = P_test @ w[0]
            else:
                pred[test_idx] = np.sum(gate[test_idx] * (P_test @ w.T), axis=1)
            log.append({"direction": direction, "day": day, "fit_days": n_fit_days, "weights": np.round(w, 3).tolist()})
    return RollingResult(pred, log)


def rolling_select(frame: pd.DataFrame, members: Sequence[str], *, window_days: int = SELECT_DAYS, min_fit_days: int = 1) -> RollingResult:
    """Per direction and day: use the member with the lowest MAE over the previous ``window_days`` days."""
    pred = np.full(len(frame), np.nan)
    log: List[Dict[str, Any]] = []
    for direction, g in frame.groupby("direction", sort=False):
        days = sorted(g["date_sgt"].unique())
        for i, day in enumerate(days):
            test_idx = g.index[g["date_sgt"] == day].to_numpy()
            prev = days[max(0, i - window_days) : i]
            # the previous day's last 40 min of labels are not observed at midnight; drop them
            start = _day_start_utc(day)
            hist = g[g["date_sgt"].isin(prev) & (g["bin_ts"] + LABEL_LAG_30 <= start)] if "bin_ts" in g else g[g["date_sgt"].isin(prev)]
            if len(prev) < min_fit_days or hist.empty:
                choice = members[0]
            else:
                maes = {m: float(np.mean(np.abs(hist["y"] - hist[m]))) for m in members}
                choice = min(maes, key=maes.get)
            pred[test_idx] = frame.loc[test_idx, choice].to_numpy()
            log.append({"direction": direction, "day": day, "choice": choice})
    return RollingResult(pred, log)


# --- 30-minute pool ----------------------------------------------------------------


def pool_30(bqml_rows: Sequence[dict], oof: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Join BQML scored rows and joined out-of-fold rows on (direction, bin_ts)."""
    b = pd.DataFrame(bqml_rows)
    b["bin_ts"] = pd.to_datetime(b["bin_ts"], utc=True)
    o = oof[["direction", "bin_ts", "y_30", "ridge[maps]", "xgb[maps]"]].copy()
    o["bin_ts"] = pd.to_datetime(o["bin_ts"], utc=True)
    m = b.merge(o, on=["direction", "bin_ts"], suffixes=("", "_joined"), how="inner", validate="one_to_one")
    label_diff = float((m["y_30"] - m["y_30_joined"]).abs().max()) if len(m) else float("nan")
    if not (label_diff <= 1e-6):
        raise RuntimeError(f"BQML and joined labels disagree (max abs diff {label_diff})")
    m = m.rename(columns={"y_30": "y"}).drop(columns=["y_30_joined"])
    m["date_sgt"] = m["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    m["served"] = np.where(m["direction"] == "SG_TO_MY", m[REGISTRY["SG_TO_MY"]], m[REGISTRY["MY_TO_SG"]])
    m = m.sort_values(["direction", "bin_ts"]).reset_index(drop=True)
    info = {
        "bqml_rows": len(b),
        "joined_rows": len(o),
        "matched_rows": len(m),
        "label_max_abs_diff": label_diff,
        "days": sorted(m["date_sgt"].unique()),
    }
    return m, info


def candidates_30(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from fuzzy_traffic import LevelPartition

    out = frame.copy()
    out["mean[models]"] = out[list(MODELS_30)].mean(axis=1)
    stack = rolling_stack(out, POOL_30)
    out["stack"] = stack.prediction
    sel = rolling_select(out, POOL_30)
    out["select"] = sel.prediction
    gate = LevelPartition().memberships(out["Persistence"].to_numpy())
    fz = rolling_stack(out, POOL_30, gate=gate)
    out["fuzzy stack"] = fz.prediction
    logs = {"stack": stack.weights, "select": sel.weights, "fuzzy stack": fz.weights, "pool": list(POOL_30), "gate_levels": ["light", "moderate", "heavy"]}
    return out, logs


LABELS_30 = {
    "Persistence": "Persistence",
    "served": "Served (registry)",
    "lin_h30": "lin_h30",
    "xgb_h30": "xgb_h30",
    "ensemble_mean": "ensemble_mean (lin_h30 + xgb_h30)",
    "ridge[maps]": "ridge[maps] (daily refit)",
    "xgb[maps]": "xgb[maps] (daily refit)",
    "mean[models]": "Equal mean of 4 models",
    "stack": "Rolling LAD stack",
    "select": "Rolling best-model selection",
    "fuzzy stack": "Fuzzy-gated stack (hybrid)",
}
FAMILY_30 = [
    ("served", "Persistence"),
    ("mean[models]", BEST_SINGLE_30),
    ("stack", BEST_SINGLE_30),
    ("select", BEST_SINGLE_30),
    ("fuzzy stack", BEST_SINGLE_30),
    ("fuzzy stack", "stack"),
    ("stack", "served"),
    ("fuzzy stack", "served"),
    (BEST_SINGLE_30, "served"),
]


REGIME_THRESHOLD_MIN = 5.0


def regime(frame: pd.DataFrame, threshold: float = REGIME_THRESHOLD_MIN) -> pd.Series:
    """Observed 30-min change of travel time: 'rising' (> +threshold), 'falling' (< -threshold), 'steady'.

    Uses the label, so it is a diagnostic for slicing errors, never a model input.
    """
    change = frame["y"] - frame["Persistence"]
    return pd.Series(np.where(change > threshold, "rising", np.where(change < -threshold, "falling", "steady")), index=frame.index)


def error_regimes(frame: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Any]:
    """MAE and share of total absolute error per regime: where is the error left to remove?"""
    from run_artifacts import metric_row

    r = regime(frame)
    metrics, summary = [], {"threshold_min": REGIME_THRESHOLD_MIN, "row_share": r.value_counts(normalize=True).round(4).to_dict(), "error_share": {}}
    for c in cols:
        ae = (frame["y"] - frame[c]).abs()
        summary["error_share"][LABELS_30.get(c, c)] = (ae.groupby(r).sum() / ae.sum()).round(4).to_dict()
        for name in ("rising", "steady", "falling"):
            e = (frame["y"] - frame[c])[r == name]
            if len(e):
                metrics.append(metric_row(LABELS_30.get(c, c), f"30min both/regime={name}", int(len(e)), float(e.abs().mean()), float(np.sqrt((e**2).mean()))))
    return {"metrics": metrics, "summary": summary}


# --- 60-minute pool ----------------------------------------------------------------


def pool_60(deep_keep: Dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame({"y": deep_keep["y"], "bin_ts": pd.to_datetime(deep_keep["timestamps"]).tz_localize(SGT).tz_convert("UTC")})
    for label, pred in deep_keep["preds"].items():
        frame[label] = pred
    frame["direction"] = "MY_TO_SG"
    frame["date_sgt"] = frame["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    return frame


def candidates_60(frame: pd.DataFrame, deep_members: Sequence[str], xgb: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = frame.copy()
    out["mean[XGB+deep]"] = out[[xgb, *deep_members]].mean(axis=1)
    out["mean[deep]"] = out[list(deep_members)].mean(axis=1)
    stack = rolling_stack(out, [xgb, *deep_members], label_lag=pd.Timedelta(minutes=5), min_fit_days=1)
    out["stack[XGB+deep]"] = stack.prediction
    return out, {"stack": stack.weights}


# --- component ---------------------------------------------------------------------


def _compare(frame: pd.DataFrame, ch: str, ref: str, labels: Dict[str, str], horizon_steps: int, alpha: float):
    from significance import compare_absolute_errors

    s = frame[frame[ch].notna() & frame[ref].notna()]
    return compare_absolute_errors(
        s["y"], s[ch], s[ref], label_challenger=labels.get(ch, ch), label_reference=labels.get(ref, ref),
        timestamps=list(s["bin_ts"]), groups=list(s["direction"]), horizon_steps=horizon_steps, alpha=alpha,
        day_utc_offset_hours=8.0,
    )


def _metrics(frame: pd.DataFrame, cols: Sequence[str], labels: Dict[str, str], prefix: str) -> List[dict]:
    from run_artifacts import metric_row

    rows = []
    for direction in ["both"] + sorted(frame["direction"].unique()):
        sub = frame if direction == "both" else frame[frame["direction"] == direction]
        if direction != "both" and frame["direction"].nunique() == 1:
            continue
        for c in cols:
            e = (sub["y"] - sub[c]).dropna()
            rows.append(metric_row(labels.get(c, c), f"{prefix}{direction}/all", int(len(e)), float(e.abs().mean()), float(np.sqrt((e**2).mean()))))
    return rows


def ensemble_component(
    run_dir: Path,
    *,
    bqml_result: Dict[str, Any],
    joined_oof: pd.DataFrame,
    deep_keep: Optional[Dict[str, Any]] = None,
    alpha: float = 0.05,
) -> Tuple[List[Path], Dict[str, Any]]:
    from plots import plot_mae_diff_forest
    from run_artifacts import artifact_relpath, subdir
    from significance import apply_holm, format_comparison_table

    out_dir = subdir(run_dir, "ensemble")
    base, info = pool_30(bqml_result["holdout"].rows, joined_oof)
    frame, logs = candidates_30(base)
    cols = list(LABELS_30)
    metrics = _metrics(frame, cols, LABELS_30, "30min ")
    regimes = error_regimes(frame, ["Persistence", "served", BEST_SINGLE_30, "stack"])
    metrics += regimes["metrics"]
    comps = apply_holm([_compare(frame, ch, ref, LABELS_30, 3, alpha) for ch, ref in FAMILY_30])
    print("Ensemble / hybrid, 30 min (Holm over this family):")
    print(format_comparison_table(comps))
    sig = [dict(c.to_dict(), family="30min") for c in comps]
    written = [*plot_mae_diff_forest(comps, out_dir / "ensemble-30min-mae-diff.png", title="30 min, both directions: ensembles and hybrids (day-block CI; Holm)")]
    meta_60: Dict[str, Any] = {}
    if deep_keep is not None:
        f60 = pool_60(deep_keep)
        members = [m for m in deep_keep["preds"] if m != deep_keep["xgb_label"] and "raw target" not in m]
        f60, logs60 = candidates_60(f60, members, deep_keep["xgb_label"])
        labels60 = {c: c for c in f60.columns}
        cols60 = [deep_keep["xgb_label"], *members, "mean[XGB+deep]", "mean[deep]", "stack[XGB+deep]"]
        metrics += _metrics(f60, cols60, labels60, "60min ")
        fam60 = [("mean[XGB+deep]", deep_keep["xgb_label"]), ("stack[XGB+deep]", deep_keep["xgb_label"]), ("mean[deep]", deep_keep["xgb_label"])]
        comps60 = apply_holm([_compare(f60, ch, ref, labels60, 12, alpha) for ch, ref in fam60])
        print("Ensemble, 60 min offline (Holm over this family):")
        print(format_comparison_table(comps60))
        sig += [dict(c.to_dict(), family="60min") for c in comps60]
        written.extend(plot_mae_diff_forest(comps60, out_dir / "ensemble-60min-mae-diff.png", title="60 min offline, JB -> SG: XGB + deep ensembles (5 day-blocks)"))
        meta_60 = {"rows": int(len(f60)), "members": [deep_keep["xgb_label"], *members], "stack_weights": logs60["stack"], "days": sorted(f60["date_sgt"].unique())}
    meta = {
        "dataset": {
            "pool_30min": {**info, "members": list(POOL_30), "registry": REGISTRY},
            "pool_60min": meta_60 or None,
        },
        "window": {"timezone": SGT, "basis": "rolling daily origin inside the scored window; weights use earlier days only", "start": info["days"][0], "end": info["days"][-1]},
        "models": [LABELS_30[c] for c in cols],
        "horizon_minutes": 30,
        "config": {"min_fit_days": MIN_FIT_DAYS, "select_days": SELECT_DAYS, "fit": "convex LAD (scipy linprog / HiGHS)", "gate": "fuzzy_traffic.LevelPartition on current travel time"},
        "weights_30min": logs,
        "error_regimes_30min": regimes["summary"],
        "metrics": metrics,
        "significance": sig,
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta
