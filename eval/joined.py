#!/usr/bin/env python3
"""Joined-feature challenger experiment: does weather or camera 2701 queue depth reduce 30-min MAE?

Data: ``features.build`` (Maps bins with v_training_set logic, weather, camera counts; all causal).
Rows scored: ``y_30`` present and ``after_gap = 0``.

Design: **rolling-origin daily folds**. For each SGT test day D (default 13-30 Sep, the same window
as the BQML audit), models are refit on rows whose label is fully observed before D starts
(``bin_ts + 40 min <= D 00:00 SGT``) and scored on D. Both directions, one model per fold with a
direction indicator.

Feature sets: ``maps`` (v_training_set columns), ``maps+weather`` (+ rain sums, forecast flags),
``maps+weather+camera`` (+ camera count, extent, age). **Layer A forecast arm:** ``maps+camfc`` adds
the camera-2701 queue forecast (``camera_forecast``: a profile learned from the Mar-Apr detections,
known in advance for every row) and ``maps+mpfc`` adds the same estimator fitted on each fold's Maps
rows (control: is any gain camera information, or just time-of-day shape?). These comparisons form a
second Holm family (``camfc``) so the weather / camera family is unchanged.

Candidates: persistence (``y_persistence``); **Maps typical** (Google's no-traffic duration for the
same bin, the "vs Maps" baseline); ridge regression (median imputation + missing indicators);
XGBoost (seeded); ensemble = mean(ridge, XGBoost).

Camera rows: the +camera models are compared with ``maps+weather`` **only on test rows that have a
camera value**, and camera coverage is reported. Below ``MIN_CAMERA_COVERAGE`` of test rows the
comparison is marked ``insufficient`` and must not be quoted as a result.

Significance: significance.py (DM + day-block bootstrap), Holm over the whole family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import bq_replica as bq
import camera_forecast as cf
import features as fx
from plots import write_series_csv

SGT = "Asia/Singapore"
DEFAULT_TEST_START = "2026-09-13"
DEFAULT_TEST_END = "2026-09-30"
MIN_CAMERA_COVERAGE = 0.6
SEED = 42
LABEL_LAG = pd.Timedelta(minutes=40)  # y_30 is the bin [t+30, t+40)
FEATURE_SETS: Dict[str, List[str]] = {
    "maps": fx.MAPS_FEATURES + ["is_my_to_sg"],
    "maps+weather": fx.MAPS_FEATURES + ["is_my_to_sg"] + fx.WEATHER_FEATURES,
    "maps+weather+camera": fx.MAPS_FEATURES + ["is_my_to_sg"] + fx.WEATHER_FEATURES + fx.CAMERA_FEATURES,
    "maps+camfc": fx.MAPS_FEATURES + ["is_my_to_sg"] + cf.CAMFC_FEATURES,
    "maps+mpfc": fx.MAPS_FEATURES + ["is_my_to_sg"] + cf.MP_FEATURES,
    "maps+mpfc+camfc": fx.MAPS_FEATURES + ["is_my_to_sg"] + cf.MP_FEATURES + cf.CAMFC_FEATURES,
}
BASELINES = ("persistence", "maps_typical")
MODELS = ("ridge", "xgb", "ensemble")
DIRECTIONS = ("both", "SG_TO_MY", "MY_TO_SG")


def scorable(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["y_30"].notna() & (frame["after_gap"] == 0)].copy()
    out["is_my_to_sg"] = (out["direction"] == "MY_TO_SG").astype(int)
    out["date_sgt"] = out["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    return out.sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def fold_days(frame: pd.DataFrame, start: str, end: str) -> List[str]:
    days = sorted(d for d in frame["date_sgt"].unique() if start <= d <= end)
    return days


def fold_split(frame: pd.DataFrame, day: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train rows have labels observed before ``day`` starts (SGT); test rows are that day."""
    day_start = pd.Timestamp(day).tz_localize(SGT).tz_convert("UTC")
    train = frame[frame["bin_ts"] + LABEL_LAG <= day_start]
    test = frame[frame["date_sgt"] == day]
    return train, test


def _ridge(cols: Sequence[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), Ridge(alpha=1.0))


def _xgb():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=10, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=1,
    )


def predict_fold(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> Dict[str, np.ndarray]:
    cols = list(cols)
    y = train["y_30"].to_numpy()
    ridge = _ridge(cols).fit(train[cols], y)
    xgb = _xgb().fit(train[cols], y)
    p_ridge = ridge.predict(test[cols])
    p_xgb = xgb.predict(test[cols])
    return {"ridge": p_ridge, "xgb": p_xgb, "ensemble": (p_ridge + p_xgb) / 2.0}


def run_folds(
    frame: pd.DataFrame,
    start: str = DEFAULT_TEST_START,
    end: str = DEFAULT_TEST_END,
    min_train_rows: int = 500,
    mp_harmonics: int = 4,
    replicas: bool = True,
) -> pd.DataFrame:
    """Out-of-fold predictions for every test row: columns ``<model>[<feature set>]`` plus baselines.

    Feature sets whose columns are absent are skipped. ``mp_*`` (Maps-profile control) is fitted on
    each fold's training rows only.

    ``replicas``: local copies of the BQML models (``bq_replica``). ``lin_bq[frozen]`` / ``xgb_bq[frozen]``
    are fitted once on BQML's 12 Sep training rows; ``lin_bq[daily]`` / ``xgb_bq[daily]`` refit the same
    estimators on every fold's earlier rows (the retraining effect, on one platform).
    """
    rows = []
    frozen = None
    if replicas:
        fr = bq.frozen_training_rows(frame)
        if len(fr) >= 100:
            frozen = (bq.fit_lin(fr), bq.fit_xgb(fr))
    for day in fold_days(frame, start, end):
        train, test = fold_split(frame, day)
        if len(train) < min_train_rows or test.empty:
            continue
        mp = cf.fold_maps_profile(train, mp_harmonics)
        train = cf.add_profile_features(train, mp, "mp")
        test = cf.add_profile_features(test, mp, "mp")
        out = test[["direction", "bin_ts", "date_sgt", "y_30", "y_persistence", "maps_typical_min", "cam_count"]].copy()
        out["train_rows"] = len(train)
        out["train_label_end"] = (train["bin_ts"] + LABEL_LAG).max()
        for fs, cols in FEATURE_SETS.items():
            if fs.endswith("camera") and train["cam_count"].notna().sum() == 0:
                continue  # no camera history yet: the model would ignore the columns
            if any(c not in train.columns for c in cols):
                continue
            for model, pred in predict_fold(train, test, cols).items():
                out[f"{model}[{fs}]"] = pred
        if replicas:
            if frozen is not None:
                out["lin_bq[frozen]"] = bq.predict_lin(frozen[0], test)
                out["xgb_bq[frozen]"] = bq.predict_xgb(frozen[1], test)
            tr_bq = bq.bqml_rows(train)
            if len(tr_bq) >= 100:
                out["lin_bq[daily]"] = bq.predict_lin(bq.fit_lin(tr_bq), test)
                out["xgb_bq[daily]"] = bq.predict_xgb(bq.fit_xgb(tr_bq), test)
        rows.append(out)
    if not rows:
        raise RuntimeError("no fold had enough training rows")
    oof = pd.concat(rows).sort_values(["direction", "bin_ts"]).reset_index(drop=True)
    oof["persistence"] = oof["y_persistence"]
    oof["maps_typical"] = oof["maps_typical_min"]
    return oof


def _metrics(oof: pd.DataFrame, candidates: Sequence[str]) -> List[dict]:
    from run_artifacts import metric_row

    out = []
    covered = oof["cam_count"].notna()
    for direction in DIRECTIONS:
        sub = oof if direction == "both" else oof[oof["direction"] == direction]
        for subset, mask in (("all", slice(None)), ("camera-covered", covered.loc[sub.index])):
            s = sub[mask]
            if s.empty:
                continue
            for c in candidates:
                if c not in s.columns:
                    continue
                ok = s[c].notna()
                if not ok.any():
                    continue
                err = s.loc[ok, "y_30"] - s.loc[ok, c]
                out.append(
                    metric_row(c, f"{direction}/{subset}", int(ok.sum()), float(err.abs().mean()), float(np.sqrt((err**2).mean())), direction=direction)
                )
    return out


def comparison_specs(oof: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """(challenger, reference, subset) tuples for the joined family."""
    specs = [
        ("xgb[maps]", "persistence", "all"),
        ("xgb[maps]", "maps_typical", "all"),
        ("xgb[maps+weather]", "xgb[maps]", "all"),
        ("ridge[maps+weather]", "ridge[maps]", "all"),
        ("ensemble[maps+weather]", "xgb[maps+weather]", "all"),
    ]
    if "xgb[maps+weather+camera]" in oof.columns:
        specs += [
            ("xgb[maps+weather+camera]", "xgb[maps+weather]", "camera-covered"),
            ("ridge[maps+weather+camera]", "ridge[maps+weather]", "camera-covered"),
        ]
    return specs


def replica_specs(oof: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Replica family: frozen vs persistence, the retraining effect, and harness vs BQML settings."""
    want = [
        ("lin_bq[frozen]", "persistence"),
        ("xgb_bq[frozen]", "persistence"),
        ("lin_bq[daily]", "lin_bq[frozen]"),
        ("xgb_bq[daily]", "xgb_bq[frozen]"),
        ("xgb[maps]", "xgb_bq[daily]"),
    ]
    return [(a, b, "all") for a, b in want if a in oof.columns and b in oof.columns]


def camfc_specs(oof: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Layer A forecast family: camera prior vs Maps-only, and vs the Maps-profile control."""
    specs = []
    for model in ("xgb", "ridge"):
        if f"{model}[maps+camfc]" in oof.columns:
            specs.append((f"{model}[maps+camfc]", f"{model}[maps]", "all"))
            if f"{model}[maps+mpfc]" in oof.columns:
                specs.append((f"{model}[maps+camfc]", f"{model}[maps+mpfc]", "all"))
            if f"{model}[maps+mpfc+camfc]" in oof.columns:
                specs.append((f"{model}[maps+mpfc+camfc]", f"{model}[maps+mpfc]", "all"))  # camera beyond the Maps prior
    return specs


def significance(oof: pd.DataFrame, *, alpha: float = 0.05, n_bootstrap: int = 4999, specs=None) -> Tuple[list, Dict[str, Any]]:
    from significance import apply_holm, compare_absolute_errors

    specs = comparison_specs(oof) if specs is None else specs
    comps, notes = [], {}
    coverage = {}
    for direction in DIRECTIONS:
        sub = oof if direction == "both" else oof[oof["direction"] == direction]
        coverage[direction] = float(sub["cam_count"].notna().mean()) if len(sub) else 0.0
        for ch, ref, subset in specs:
            s = sub[sub["cam_count"].notna()] if subset == "camera-covered" else sub
            s = s[s[ch].notna() & s[ref].notna()]
            if len(s) < 3 or s["date_sgt"].nunique() < 2:
                continue
            label = f"{ch} ({direction}/{subset})"
            comps.append(
                compare_absolute_errors(
                    s["y_30"], s[ch], s[ref],
                    label_challenger=label, label_reference=ref,
                    timestamps=list(s["bin_ts"]), groups=list(s["direction"]),
                    horizon_steps=3, alpha=alpha, n_bootstrap=n_bootstrap, day_utc_offset_hours=8.0,
                )
            )
            if subset == "camera-covered":
                notes[label] = "insufficient" if coverage[direction] < MIN_CAMERA_COVERAGE else "ok"
    return apply_holm(comps), {"camera_coverage": coverage, "camera_status": notes}


def joined_component(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    start: str = DEFAULT_TEST_START,
    end: str = DEFAULT_TEST_END,
    alpha: float = 0.05,
    n_bootstrap: int = 4999,
    inputs: Optional[Dict[str, Any]] = None,
    keep: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """``keep``: optional dict that receives the out-of-fold frame (``keep["oof"]``) for reuse."""
    from plots import plot_mae_diff_forest
    from run_artifacts import artifact_relpath, subdir
    from significance import format_comparison_table

    data = scorable(frame)
    cam_info = (inputs or {}).get("camera_forecast") or {}
    oof = run_folds(data, start, end, mp_harmonics=int(cam_info.get("harmonics", 4)))
    if keep is not None:
        keep["oof"] = oof
    candidates = list(BASELINES) + [c for c in oof.columns if "[" in c]
    metrics = _metrics(oof, candidates)
    comps, notes = significance(oof, alpha=alpha, n_bootstrap=n_bootstrap)
    print(format_comparison_table(comps))
    camfc_comps, _ = significance(oof, alpha=alpha, n_bootstrap=n_bootstrap, specs=camfc_specs(oof))
    replica_comps, _ = significance(oof, alpha=alpha, n_bootstrap=n_bootstrap, specs=replica_specs(oof))
    if replica_comps:
        print("BQML replica family (Holm within family):")
        print(format_comparison_table(replica_comps))
    if camfc_comps:
        print("Layer A forecast family (Holm within family):")
        print(format_comparison_table(camfc_comps))
    metrics += regime_metrics(oof, [c for c in ("persistence", "xgb[maps]", "xgb[maps+camfc]", "xgb[maps+mpfc]", "xgb[maps+mpfc+camfc]") if c in oof.columns])
    print("camera coverage of test rows:", notes["camera_coverage"])

    out = subdir(run_dir, "joined")
    written = [
        *plot_mae_diff_forest(comps, out / "joined-mae-diff.png", title="Joined features, 30 min: paired MAE difference (day-block CI; Holm)"),
        *_plot_mae_bars(metrics, out / "joined-mae-by-feature-set.png"),
    ]
    if camfc_comps:
        written.extend(plot_mae_diff_forest(camfc_comps, out / "camfc-mae-diff.png", title="Layer A queue forecast as input, 30 min: paired MAE difference (Holm within family)"))
    if replica_comps:
        written.extend(plot_mae_diff_forest(replica_comps, out / "replica-mae-diff.png", title="Local BQML replicas, 30 min: frozen vs daily refit (Holm within family)"))
    if cam_info.get("curves"):
        written.extend(cf.plot_profiles(cam_info["curves"], None, out / "camfc-profiles.png"))
    sig = []
    for c in comps:
        d = c.to_dict()
        d["family"] = "joined"
        status = notes["camera_status"].get(c.label_challenger)
        if status:
            d["camera_status"] = status
        sig.append(d)
    for c in camfc_comps:
        sig.append(dict(c.to_dict(), family="camfc"))
    for c in replica_comps:
        sig.append(dict(c.to_dict(), family="replica"))
    days = sorted(oof["date_sgt"].unique())
    meta = {
        "dataset": {
            **(inputs or {}),
            "rows_scorable": int(len(data)),
            "rows_scored": int(len(oof)),
            "rows_scored_per_direction": {d: int((oof["direction"] == d).sum()) for d in ("SG_TO_MY", "MY_TO_SG")},
            "camera_coverage_of_test_rows": notes["camera_coverage"],
            "min_camera_coverage": MIN_CAMERA_COVERAGE,
            "feature_coverage": fx.coverage_report(frame).to_dict(orient="records"),
        },
        "window": {
            "timezone": SGT,
            "basis": "rolling-origin daily folds; train labels end before each test day",
            "start": days[0],
            "end": days[-1],
            "folds": len(days),
        },
        "models": [
            "persistence (y_persistence)",
            "maps_typical (Maps duration without traffic, same bin)",
            "ridge (median impute + indicators, standardised, alpha=1)",
            f"xgb (XGBRegressor, seed {SEED})",
            "ensemble = mean(ridge, xgb)",
            "camfc: camera-2701 queue forecast from Mar-Apr detections (camera_forecast.py)",
            "mpfc: same estimator fitted on each fold's Maps rows (control)",
            "lin_bq / xgb_bq: local replicas of BQML lin_h30 / xgb_h30 (bq_replica.py); [frozen] = 12 Sep training rows, [daily] = refit per fold",
        ],
        "feature_sets": FEATURE_SETS,
        "replica_settings": bq.settings(),
        "horizon_minutes": 30,
        "metrics": metrics,
        "significance": sig,
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta


REGIME_THRESHOLD_MIN = 5.0


def regime_metrics(oof: pd.DataFrame, candidates: Sequence[str], threshold: float = REGIME_THRESHOLD_MIN) -> List[dict]:
    """MAE by observed 30-min change (rising / steady / falling; uses the label, diagnostic only)."""
    from run_artifacts import metric_row

    change = oof["y_30"] - oof["y_persistence"]
    regime = np.where(change > threshold, "rising", np.where(change < -threshold, "falling", "steady"))
    out = []
    for name in ("rising", "steady", "falling"):
        s = oof[regime == name]
        for c in candidates:
            err = (s["y_30"] - s[c]).dropna()
            if len(err):
                out.append(metric_row(c, f"both/regime={name}", int(len(err)), float(err.abs().mean()), float(np.sqrt((err**2).mean())), direction="both"))
    return out


def _plot_mae_bars(metrics: List[dict], path: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [m for m in metrics if m["slice"] == "both/all"]
    rows.sort(key=lambda m: m["mae_min"])
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(rows) + 1)))
    ax.barh([m["candidate"] for m in rows], [m["mae_min"] for m in rows], color="#4c72b0")
    for i, m in enumerate(rows):
        ax.text(m["mae_min"] + 0.02, i, f"{m['mae_min']:.2f}", va="center", fontsize=8)
    ax.set_xlabel("MAE (min), both directions, rolling-origin folds")
    ax.set_title("Joined features, 30 min: MAE by candidate")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = write_series_csv(
        path,
        [
            {"order_from_bottom": i, "candidate": m["candidate"], "slice": m["slice"], "mae_min": m["mae_min"]}
            for i, m in enumerate(rows)
        ],
        ("order_from_bottom", "candidate", "slice", "mae_min"),
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path, csv_path]


def cut_weather(rain: Optional[pd.DataFrame], fc: Optional[pd.DataFrame], cutoff) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Drop rainfall readings stamped after ``cutoff`` and forecasts acquired after it."""
    if cutoff is None:
        return rain, fc
    if rain is not None:
        rain = rain[rain["ts"] <= cutoff].reset_index(drop=True)
    if fc is not None:
        col = "available_at" if "available_at" in fc.columns else "issue_timestamp"
        fc = fc[fc[col] <= cutoff].reset_index(drop=True)
    return rain, fc


def load_inputs(refresh_bq: bool = False, data_cutoff=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from run_artifacts import file_fingerprint
    from timeseries_xgb import apply_data_cutoff, sync_canonical_travel_times

    tt = apply_data_cutoff(sync_canonical_travel_times(fx.TT_CACHE, refresh=refresh_bq), data_cutoff)
    rain = fx.load_rain() if fx.RAIN_CSV.exists() else None
    fc = fx.load_forecast() if fx.FORECAST_CSV.exists() else None
    rain, fc = cut_weather(rain, fc, data_cutoff)
    cam = fx.load_camera()
    info: Dict[str, Any] = {
        "travel_times": file_fingerprint(fx.TT_CACHE),
        "refreshed_from_bq": refresh_bq,
        "data_cutoff_utc": None if data_cutoff is None else data_cutoff.isoformat(),
    }
    for name, p in (("rainfall_S210", fx.RAIN_CSV), ("forecast_woodlands", fx.FORECAST_CSV), ("camera_2701", fx.CAMERA_CSV)):
        info[name] = file_fingerprint(p) if p.exists() else None
    frame = fx.build(tt, rain, fc, cam)
    history = cf.fetch_history(cf.HISTORY_CSV, refresh=False)  # read-only pull once, then cached
    profile, cam_info = cf.fit_camera_profile(history)
    cam_info["history"] = file_fingerprint(cf.HISTORY_CSV)
    cam_info["source"] = f"swiftborder.{cf.HISTORY_VIEW}"
    cam_info["curves"] = cf.profile_curves(profile)
    info["camera_forecast"] = cam_info
    return cf.add_profile_features(frame, profile, "camfc"), info


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=DEFAULT_TEST_START)
    parser.add_argument("--end", default=DEFAULT_TEST_END)
    parser.add_argument("--refresh-bq", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=4999)
    args = parser.parse_args(argv)

    from run_artifacts import create_run_dir, new_manifest, update_latest_pointer, write_manifest, write_run_readme

    frame, info = load_inputs(args.refresh_bq)
    run_dir = create_run_dir(components=["joined"], run_id=args.run_id, runs_root=args.runs_root)
    _, meta = joined_component(run_dir, frame, start=args.start, end=args.end, n_bootstrap=args.n_bootstrap, inputs=info)
    manifest = new_manifest(["joined"])
    manifest["joined"] = meta
    write_manifest(run_dir, manifest)
    write_run_readme(run_dir, manifest)
    update_latest_pointer(run_dir, manifest, runs_root=args.runs_root)
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
