#!/usr/bin/env python3
"""
Layer A → Layer B comprehensive evaluation.

Two questions:
  Q1. CORRELATION: How well do cam2701 directional counts correlate with
      the Google Maps travel time series (y_persistence, y_30)?
  Q2. ABLATION: Does adding Layer A camera counts as features reduce
      30-minute MAE vs maps-only XGBoost on the same rolling-origin folds
      used in joined.py (13–30 Sep, daily refit)?

Output: eval/runs/layer_a_eval/ with
  - correlation.json   – Pearson / Spearman / Kendall + scatter stats
  - ablation.json      – per-fold and aggregate MAE for maps vs maps+camera
  - report.md          – human-readable summary

Usage:
    python eval/layer_a_eval.py
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
DATA_DIR = EVAL_ROOT / "data"
JOINED_CSV = DATA_DIR / "layer_a_layer_b_joined.csv"
OUT_DIR = EVAL_ROOT / "runs" / "layer_a_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SGT = timezone(timedelta(hours=8))
TEST_START = "2026-09-13"   # same as joined.py default
TEST_END = "2026-09-30"
MIN_CAMERA_COVERAGE = 0.60
SEED = 42

# ── load ──────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(JOINED_CSV, parse_dates=["bin_ts"])
    df["bin_ts"] = pd.to_datetime(df["bin_ts"], utc=True)
    # SGT date for fold splitting
    df["date_sgt"] = (
        df["bin_ts"].dt.tz_convert("Asia/Singapore").dt.date
    )
    # per-direction camera count (directional, causal)
    df["cam_dir"] = np.where(
        df["direction"] == "SG_TO_MY", df["cam_sg_my"], df["cam_my_sg"]
    )
    df["cam_opposite"] = np.where(
        df["direction"] == "SG_TO_MY", df["cam_my_sg"], df["cam_sg_my"]
    )
    return df


# ── Q1: correlation ───────────────────────────────────────────────────────────

def correlations(df: pd.DataFrame) -> dict:
    """Pearson / Spearman / Kendall between camera counts and Maps metrics."""
    results = {}
    cam_cols = ["cam_sg_my", "cam_my_sg", "cam_total", "cam_dir"]
    maps_cols = ["y_persistence", "y_30"]
    full_df = df.dropna(subset=cam_cols + maps_cols)

    for direction in ("SG_TO_MY", "MY_TO_SG", "BOTH"):
        if direction == "BOTH":
            sub = full_df
        else:
            sub = full_df[full_df["direction"] == direction]

        results[direction] = {}
        for cc in cam_cols:
            results[direction][cc] = {}
            for mc in maps_cols:
                x = sub[cc].values
                y = sub[mc].values
                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                kr, kp = stats.kendalltau(x, y)
                results[direction][cc][mc] = {
                    "n": int(len(x)),
                    "pearson_r": round(float(pr), 4),
                    "pearson_p": round(float(pp), 6),
                    "spearman_r": round(float(sr), 4),
                    "spearman_p": round(float(sp), 6),
                    "kendall_tau": round(float(kr), 4),
                    "kendall_p": round(float(kp), 6),
                    "cam_mean": round(float(np.mean(x)), 3),
                    "maps_mean": round(float(np.mean(y)), 3),
                }
    return results


# ── Q2: ablation – rolling-origin daily folds ─────────────────────────────────

def _fit_xgb(X_tr, y_tr, X_te):
    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=1, verbosity=0)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    except ImportError:
        # fall back to ridge if xgboost not available
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        m = Ridge()
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict(sc.transform(X_te))


MAPS_FEATS = ["y_persistence", "congestion_ratio", "speed_kmh", "is_my_to_sg"]
CAM_FEATS  = ["cam_dir", "cam_opposite", "cam_total", "cam_frames"]


def ablation(df: pd.DataFrame) -> dict:
    """Rolling-origin daily folds over TEST_START…TEST_END."""
    df = df.copy()
    df["is_my_to_sg"] = (df["direction"] == "MY_TO_SG").astype(float)

    test_dates = sorted(
        d for d in df["date_sgt"].unique()
        if TEST_START <= str(d) <= TEST_END
    )
    if not test_dates:
        return {"error": "no test dates in joined data"}

    folds = []
    for test_day in test_dates:
        # training: rows whose label is fully before midnight SGT of test_day
        # (label bin_ts + 40min <= test_day 00:00 SGT  →  bin_ts <= test_day - 40min)
        cutoff_utc = pd.Timestamp(str(test_day), tz="Asia/Singapore") - pd.Timedelta(minutes=40)
        cutoff_utc = cutoff_utc.tz_convert("UTC")
        train = df[df["bin_ts"] < cutoff_utc].dropna(subset=MAPS_FEATS + ["y_30"])
        test  = df[df["date_sgt"] == test_day].dropna(subset=MAPS_FEATS + ["y_30"])
        test_cam = test.dropna(subset=CAM_FEATS)

        if len(train) < 50 or len(test) == 0:
            continue

        y_tr = train["y_30"].values
        y_te = test["y_30"].values

        X_tr_maps = train[MAPS_FEATS].fillna(0).values
        X_te_maps = test[MAPS_FEATS].fillna(0).values
        pred_maps = _fit_xgb(X_tr_maps, y_tr, X_te_maps)
        mae_maps = float(np.mean(np.abs(y_te - pred_maps)))

        persistence_mae = float(np.mean(np.abs(y_te - test["y_persistence"].values)))

        cam_coverage = len(test_cam) / len(test) if len(test) > 0 else 0.0
        sufficient_cam = cam_coverage >= MIN_CAMERA_COVERAGE

        mae_cam = None
        if sufficient_cam and len(test_cam) >= 10:
            train_cam = train.dropna(subset=CAM_FEATS)
            if len(train_cam) >= 50:
                X_tr_cam = train_cam[MAPS_FEATS + CAM_FEATS].fillna(0).values
                y_tr_cam = train_cam["y_30"].values
                X_te_cam = test_cam[MAPS_FEATS + CAM_FEATS].fillna(0).values
                y_te_cam = test_cam["y_30"].values
                pred_cam = _fit_xgb(X_tr_cam, y_tr_cam, X_te_cam)
                mae_cam = float(np.mean(np.abs(y_te_cam - pred_cam)))

        folds.append({
            "test_day": str(test_day),
            "n_test": int(len(test)),
            "n_train": int(len(train)),
            "cam_coverage": round(cam_coverage, 4),
            "sufficient_cam": sufficient_cam,
            "persistence_mae": round(persistence_mae, 4),
            "maps_mae": round(mae_maps, 4),
            "maps_camera_mae": round(mae_cam, 4) if mae_cam is not None else None,
        })

    # aggregate
    fold_df = pd.DataFrame(folds)
    agg_maps = float(fold_df["maps_mae"].mean())
    agg_persistence = float(fold_df["persistence_mae"].mean())

    cam_folds = fold_df.dropna(subset=["maps_camera_mae"])
    agg_cam = float(cam_folds["maps_camera_mae"].mean()) if len(cam_folds) else None
    agg_maps_on_cam_folds = float(cam_folds["maps_mae"].mean()) if len(cam_folds) else None

    delta = None
    if agg_cam is not None and agg_maps_on_cam_folds is not None:
        delta = round(agg_cam - agg_maps_on_cam_folds, 4)

    # significance on cam folds
    sig = {}
    if len(cam_folds) >= 5:
        d = cam_folds["maps_camera_mae"] - cam_folds["maps_mae"]
        t, p = stats.ttest_1samp(d, 0)
        sig = {"paired_t": round(float(t), 4), "p_value": round(float(p), 6),
               "n_folds": int(len(cam_folds)),
               "mean_delta_min": round(float(d.mean()), 4)}

    return {
        "test_window": f"{TEST_START} to {TEST_END}",
        "n_folds": int(len(folds)),
        "n_cam_sufficient_folds": int(len(cam_folds)),
        "min_camera_coverage_threshold": MIN_CAMERA_COVERAGE,
        "aggregate": {
            "persistence_mae_min": round(agg_persistence, 4),
            "maps_only_mae_min": round(agg_maps, 4),
            "maps_camera_mae_min": round(agg_cam, 4) if agg_cam else None,
            "maps_only_mae_on_cam_folds_min": round(agg_maps_on_cam_folds, 4) if agg_maps_on_cam_folds else None,
            "delta_cam_vs_maps_min": delta,
        },
        "significance": sig,
        "folds": folds,
    }


# ── report ────────────────────────────────────────────────────────────────────

def fmt_r(corr_block: dict) -> str:
    lines = []
    for cc, mv in corr_block.items():
        for mc, v in mv.items():
            sig = "**" if v["pearson_p"] < 0.05 else ""
            lines.append(
                f"  `{cc}` → `{mc}`: "
                f"Pearson r={v['pearson_r']:+.3f} (p={v['pearson_p']:.4f}) "
                f"Spearman ρ={v['spearman_r']:+.3f} {sig}"
            )
    return "\n".join(lines)


def write_report(corr: dict, abl: dict):
    agg = abl.get("aggregate", {})
    sig = abl.get("significance", {})
    cov_folds = abl.get("n_cam_sufficient_folds", 0)
    total_folds = abl.get("n_folds", 0)

    lines = [
        "# Layer A → Layer B: Comprehensive Evaluation",
        "",
        f"**Data:** `cam2701.local_counts` joined with `traffic_prediction.v_training_set`  ",
        f"**Joined rows:** {abl.get('n_folds','?')} folds over {abl.get('test_window','')}  ",
        f"**Camera coverage threshold:** {abl.get('min_camera_coverage_threshold', 0.6):.0%}  ",
        f"**Folds with sufficient camera coverage:** {cov_folds}/{total_folds}  ",
        "",
        "---",
        "",
        "## Q1: Correlation — Layer A counts vs Google Maps travel time",
        "",
        "> Does the camera queue depth track Maps congestion?",
        "",
    ]

    for direction in ("SG_TO_MY", "MY_TO_SG", "BOTH"):
        lines.append(f"### Direction: {direction}")
        lines.append(fmt_r(corr.get(direction, {})))
        lines.append("")

    lines += [
        "---",
        "",
        "## Q2: Ablation — Does Layer A improve Layer B (XGBoost) MAE?",
        "",
        "Rolling-origin daily folds, same window as `eval/joined.py` (13–30 Sep 2026).",
        "Camera features: `cam_dir`, `cam_opposite`, `cam_total`, `cam_frames`.",
        "Maps-only features: `y_persistence`, `congestion_ratio`, `speed_kmh`, direction indicator.",
        "",
        "### Aggregate results (across all folds)",
        "",
        f"| Model | Mean MAE (min) |",
        f"|---|---|",
        f"| Persistence | {agg.get('persistence_mae_min', '-')} |",
        f"| XGBoost maps-only (all folds) | {agg.get('maps_only_mae_min', '-')} |",
    ]

    if agg.get("maps_only_mae_on_cam_folds_min") is not None:
        lines += [
            f"| XGBoost maps-only (cam-covered folds only) | {agg.get('maps_only_mae_on_cam_folds_min', '-')} |",
            f"| XGBoost maps+camera (cam-covered folds only) | {agg.get('maps_camera_mae_min', '-')} |",
        ]
        delta = agg.get("delta_cam_vs_maps_min")
        if delta is not None:
            sign = "↓" if delta < 0 else "↑"
            lines.append(f"| **Δ (camera − maps-only)** | **{delta:+.4f} min {sign}** |")
    else:
        lines.append(f"| XGBoost maps+camera | insufficient camera coverage in all folds |")

    lines += [
        "",
        "### Significance (cam-covered folds only)",
        "",
    ]
    if sig:
        lines += [
            f"Paired fold t-test (camera MAE − maps MAE): t={sig.get('paired_t','?')}, "
            f"p={sig.get('p_value','?')}, n={sig.get('n_folds','?')} folds  ",
            f"Mean delta: {sig.get('mean_delta_min','?')} min",
            "",
            "> [!NOTE]",
            "> This is a fold-level paired t-test (not DM), reported for orientation only.",
            "> Formal DM significance requires row-level paired errors.",
        ]
    else:
        lines.append("Insufficient cam-covered folds for significance test.")

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Correlation findings",
    ]

    # auto-interpret correlation
    for direction in ("SG_TO_MY", "MY_TO_SG"):
        d = corr.get(direction, {})
        for cc in ("cam_sg_my", "cam_my_sg"):
            for mc in ("y_persistence",):
                v = d.get(cc, {}).get(mc, {})
                if v:
                    r = v["pearson_r"]
                    interp = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
                    sig_str = "statistically significant" if v["pearson_p"] < 0.05 else "not significant"
                    lines.append(
                        f"- `{cc}` vs `{mc}` ({direction}): "
                        f"Pearson r={r:+.3f} — {interp}, {sig_str}"
                    )

    lines += [
        "",
        "### Ablation findings",
        "",
    ]

    delta = agg.get("delta_cam_vs_maps_min")
    if delta is not None:
        direction_str = "improvement" if delta < 0 else "degradation"
        lines.append(
            f"Camera features produced a **{abs(delta):.4f} min {direction_str}** "
            f"in mean MAE on {cov_folds} folds with ≥{abl.get('min_camera_coverage_threshold',0.6):.0%} camera coverage."
        )
        if sig:
            p = sig.get("p_value", 1.0)
            if p < 0.05:
                lines.append(
                    f"The improvement is statistically significant at α=0.05 (paired-t p={p:.4f})."
                )
            else:
                lines.append(
                    f"The improvement is **not statistically significant** at α=0.05 (paired-t p={p:.4f}). "
                    f"This does not demonstrate Layer A inputs reliably reduce Layer B error on this window."
                )
    else:
        lines.append(
            "Camera coverage was below the 60% threshold on all folds — "
            "no ablation comparison is possible with current `local_counts` data."
        )

    lines += [
        "",
        "### Important caveats",
        "",
        "- Camera frames cover only ~1 frame/10 min. Finer coverage would sharpen the ablation.",
        "- The label (`y_30`) is a future Maps estimate, not a ground-truth crossing time.",
        "- A fold-level t-test on 18 folds is underpowered; row-level DM is needed for formal confirmation.",
        "- `cam_dir` is the direction-matched count (SG_TO_MY → `cam_sg_my`). "
          "The opposite direction is included as a control feature.",
        "- No Layer A frames were available after 4 Oct 2026; the 1–19 Oct protected window "
          "has no camera data and cannot be scored.",
    ]

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {OUT_DIR / 'report.md'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not JOINED_CSV.exists():
        print(f"ERROR: {JOINED_CSV} not found. Run fetch_join.py first.", file=sys.stderr)
        sys.exit(1)

    df = load()
    print(f"Loaded {len(df)} joined rows, {df['date_sgt'].nunique()} dates")
    print(f"  directions: {df['direction'].value_counts().to_dict()}")
    print(f"  date range: {df['date_sgt'].min()} to {df['date_sgt'].max()}")

    print("\n-- Q1: Correlation --")
    corr = correlations(df)
    (OUT_DIR / "correlation.json").write_text(
        json.dumps(corr, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Saved correlation.json")

    # Print summary
    for direction in ("SG_TO_MY", "MY_TO_SG"):
        d = corr[direction]
        print(f"\n  {direction}:")
        for cc in ("cam_sg_my", "cam_my_sg", "cam_total", "cam_dir"):
            for mc in ("y_persistence", "y_30"):
                v = d[cc][mc]
                print(f"    {cc} → {mc}: Pearson r={v['pearson_r']:+.4f} (p={v['pearson_p']:.4f}), "
                      f"Spearman ρ={v['spearman_r']:+.4f}")

    print("\n-- Q2: Ablation --")
    abl = ablation(df)
    (OUT_DIR / "ablation.json").write_text(
        json.dumps(abl, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Folds: {abl.get('n_folds')}, cam-sufficient: {abl.get('n_cam_sufficient_folds')}")
    print(f"  Aggregate: {abl.get('aggregate')}")
    print(f"  Significance: {abl.get('significance')}")
    print(f"  Saved ablation.json")

    print("\n-- Report --")
    write_report(corr, abl)

    print("\nDone. Artifacts in:", OUT_DIR)


if __name__ == "__main__":
    main()
