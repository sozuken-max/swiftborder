#!/usr/bin/env python3
"""Camera-only estimate of *current* Maps duration against a calendar baseline.

This is the section 7b test. The target is the current bin's Maps
``duration_in_traffic`` (``y_persistence``), not the 30-minute-ahead label.
The calendar baseline is the Fourier profile of that duration, fit on September
2026 only. The camera model adds the current camera-2701 directional count and
does not read a Maps duration, lag, speed, or distance feature. October 1-4 is
the score window, which is the overlap of both series.

Missing frames stay missing. A scored frame with zero vehicles stays zero: the
backfill has no flag that separates fog, glare, or an empty road.

    python camera_current.py
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from harness.constants import SGT

from harness import significance as sig
from lb import camera_forecast as cf, features as fx, joined as jn
from studies import counts_substitute as cs, horizon_study as hs

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
BAR_MIN = 0.5
CALENDAR = ["tod_sin", "tod_cos", "dow", "is_weekend", "is_morning_peak", "is_evening_peak"]
RUN_ID = "20261017T084900Z_camera-current"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def attach_counts(maps: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Mean directional count in the same 10-minute bin. No frame means missing, not zero."""
    binned = cs.count_bins(counts)[["bin_ts", "direction", "dur_min"]].rename(columns={"dur_min": "cam_dir"})
    out = maps.merge(binned, on=["bin_ts", "direction"], how="left")
    return out


def october_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame["y_persistence"].notna()
        & frame["cam_dir"].notna()
        & (frame["after_gap"] == 0)
        & (frame["date_sgt"] >= "2026-10-01")
        & (frame["date_sgt"] <= "2026-10-04")
    )
    return frame.loc[mask].sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def september_rows(frame: pd.DataFrame) -> pd.DataFrame:
    day_start = pd.Timestamp("2026-10-01", tz="Asia/Singapore").tz_convert("UTC")
    mask = (
        frame["y_persistence"].notna()
        & frame["cam_dir"].notna()
        & (frame["after_gap"] == 0)
        & (frame["date_sgt"] >= "2026-09-06")
        & (frame["date_sgt"] < "2026-10-01")
        & (frame["bin_ts"] + pd.Timedelta(minutes=10) <= day_start)
    )
    return frame.loc[mask].sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def label_vs_calendar(low: Optional[float], high: Optional[float]) -> str:
    """Counts+calendar minus calendar-only. Negative means the counts model is better."""
    if low is None or high is None:
        return "inconclusive"
    if high < -BAR_MIN:
        return "improvement"
    if low > BAR_MIN:
        return "regression"
    if -BAR_MIN <= low and high <= BAR_MIN:
        return "no meaningful gain"
    return "inconclusive"


def paired(frame: pd.DataFrame, challenger: str, reference: str) -> sig.ComparisonResult:
    rows = frame.sort_values(["direction", "bin_ts"])
    return sig.compare_absolute_errors(
        rows["y_persistence"].to_numpy(),
        rows[challenger].to_numpy(),
        rows[reference].to_numpy(),
        label_challenger=challenger,
        label_reference=reference,
        timestamps=list(rows["bin_ts"]),
        groups=list(rows["direction"]),
        horizon_steps=1,
        alpha=0.05,
        n_bootstrap=cs.N_BOOTSTRAP,
        random_state=0,
        day_utc_offset_hours=8.0,
    )


def _interval(result: sig.ComparisonResult) -> Tuple[Optional[float], Optional[float]]:
    if result.joint_ci_low_min is not None and result.joint_ci_high_min is not None:
        return float(result.joint_ci_low_min), float(result.joint_ci_high_min)
    if result.n < 2 or not math.isfinite(result.bootstrap_ci_low_min):
        return None, None
    return float(result.bootstrap_ci_low_min), float(result.bootstrap_ci_high_min)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str], kind: str) -> np.ndarray:
    y = train["y_persistence"].to_numpy()
    model = jn._ridge(columns).fit(train[list(columns)], y) if kind == "ridge" else jn._xgb().fit(train[list(columns)], y)
    return model.predict(test[list(columns)])


def build_frame() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("reading Maps bins", flush=True)
    maps = fx.maps_features(cs.load_maps())
    print("reading camera counts", flush=True)
    joined = attach_counts(maps, cs.load_counts())
    train = september_rows(joined)
    test = october_rows(joined)
    print(f"train {len(train)}  october {len(test)}", flush=True)
    profile = cf.FourierProfile(harmonics=hs.PROFILE_HARMONICS).fit(
        train[["direction", "bin_ts", "y_persistence"]],
        value="y_persistence",
    )
    test = test.copy()
    test["calendar_profile"] = profile.predict(test["direction"].to_numpy(), test["bin_ts"] + cf.NOW_OFFSET)
    test["ridge_calendar"] = _fit_predict(train, test, CALENDAR, "ridge")
    test["ridge_counts"] = _fit_predict(train, test, [*CALENDAR, "cam_dir"], "ridge")
    test["xgb_counts"] = _fit_predict(train, test, [*CALENDAR, "cam_dir"], "xgb")
    return train, test, joined


def _mae(frame: pd.DataFrame, column: str) -> float:
    return float(np.mean(np.abs(frame["y_persistence"] - frame[column])))


def score(test: pd.DataFrame) -> List[dict]:
    rows = []
    for column, name in (
        ("ridge_counts", "ridge[counts+calendar]"),
        ("xgb_counts", "xgb[counts+calendar]"),
    ):
        result = paired(test, column, "calendar_profile")
        low, high = _interval(result)
        rows.append(
            {
                "id": name,
                "n": int(result.n),
                "mae_counts_calendar": _mae(test, column),
                "mae_calendar": _mae(test, "calendar_profile"),
                "mae_ridge_calendar": _mae(test, "ridge_calendar"),
                "diff": float(result.mean_ae_diff_min),
                "ci_low": low,
                "ci_high": high,
                "label": label_vs_calendar(low, high),
            }
        )
    return rows


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def write_plots(run_dir: Path, test: pd.DataFrame, rows: List[dict]) -> List[str]:
    names = []
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels = ["Fourier calendar", "ridge calendar", "ridge counts+calendar", "xgb counts+calendar"]
    values = [
        rows[0]["mae_calendar"],
        rows[0]["mae_ridge_calendar"],
        rows[0]["mae_counts_calendar"],
        rows[1]["mae_counts_calendar"],
    ]
    colors = ["#4c78a8", "#9ecae1", "#f58518", "#e45756"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("October MAE (minutes)")
    ax.set_title("Current Maps duration, 1-4 Oct 2026")
    _style(ax)
    fig.tight_layout()
    fig.savefig(run_dir / "mae_current.png", dpi=140)
    plt.close(fig)
    names.append("mae_current.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, row in enumerate(rows):
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color="#333333", linewidth=2)
        ax.scatter([row["diff"]], [i], color="#f58518", zorder=3)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.axvline(-BAR_MIN, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax.axvline(BAR_MIN, color="#d62728", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["id"] for row in rows])
    ax.set_xlabel("Counts+calendar MAE minus calendar MAE (minutes)")
    ax.set_title("Paired October difference, +/- 0.5 min bar")
    _style(ax)
    fig.tight_layout()
    fig.savefig(run_dir / "paired_current.png", dpi=140)
    plt.close(fig)
    names.append("paired_current.png")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True)
    for ax, direction in zip(axes, cs.DIRECTIONS):
        part = test[test["direction"] == direction].sort_values("bin_ts")
        clock = part["bin_ts"].dt.tz_convert(SGT)
        ax.plot(clock, part["y_persistence"], color="#1a1a1a", linewidth=0.9, label="Maps duration now")
        ax.plot(clock, part["calendar_profile"], color="#4c78a8", linewidth=0.8, label="Calendar profile")
        ax.plot(clock, part["xgb_counts"], color="#f58518", linewidth=0.8, label="xgb counts+calendar")
        ax.set_title(f"{direction}  n={len(part)}")
        ax.set_ylabel("Minutes")
        _style(ax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=SGT))
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Current Maps duration, 1-4 Oct 2026 (Asia/Singapore)", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "series_current.png", dpi=140)
    plt.close(fig)
    names.append("series_current.png")
    return names


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.3f}"


def write_report(run_dir: Path, payload: dict) -> None:
    lines = [
        "# Camera-only estimate of current travel time",
        "",
        payload["definition"],
        "",
        "## Window",
        "",
        payload["window"],
        "",
        "## Decision",
        "",
        "Paired difference is counts-plus-calendar MAE minus the Fourier calendar MAE, in minutes. Negative means the counts model is closer to the current Maps duration. The interval is the joint calendar-day bootstrap (`eval/significance.py`, 4,999 resamples, seed 0, Asia/Singapore). Four days make it coarse. The practical bar is 0.5 min. `improvement` means the interval lies entirely below -0.5; `regression` entirely above +0.5; `no meaningful gain` entirely inside +/-0.5.",
        "",
        "| Model | n | MAE counts+calendar | MAE Fourier calendar | MAE ridge calendar | Counts − calendar (95% CI) | Label |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["models"]:
        ci = f"{row['diff']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        lines.append(
            f"| `{row['id']}` | {row['n']} | {_fmt(row['mae_counts_calendar'])} | {_fmt(row['mae_calendar'])} | {_fmt(row['mae_ridge_calendar'])} | {ci} | {row['label']} |"
        )
    lines += ["", "## Figures", ""]
    captions = {
        "mae_current.png": "October MAE for current Maps duration.",
        "paired_current.png": "Paired difference against the calendar profile, with the +/- 0.5 minute bar.",
        "series_current.png": "Current Maps duration, the calendar profile, and the counts-plus-calendar XGBoost.",
    }
    for figure in payload["figures"]:
        lines.append(f"![{captions[figure]}]({figure})")
        lines.append("")
    lines += [
        "## Visibility",
        "",
        payload["visibility"],
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, test, _joined = build_frame()
    if test.groupby("direction").size().min() < 50:
        raise SystemExit(f"October overlap is under 50 rows in a direction: {test.groupby('direction').size().to_dict()}")
    rows = score(test)
    run_dir = EVAL_ROOT / "runs" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    figures = write_plots(run_dir, test, rows)
    n_by = {d: int((test["direction"] == d).sum()) for d in cs.DIRECTIONS}
    from studies.manifest import base_manifest, write_run_json

    study_body = {
        "question": "camera-only estimate of current Maps duration versus a calendar-only baseline",
        "target": "y_persistence (current bin duration_in_traffic, minutes)",
        "definition": (
            "The target is the current 10-minute Maps duration (`y_persistence`), not the 30-minute-ahead label. "
            "The calendar baseline is a Fourier profile of that duration, fit on September rows only. "
            "Ridge and XGBoost add the current camera-2701 directional count (`sg_my` or `my_sg` in the same bin) "
            "to calendar terms (`tod_sin`, `tod_cos`, `dow`, weekend, morning peak, evening peak). "
            "They do not read a Maps duration, lag, speed, congestion ratio, or distance at predict time."
        ),
        "window": (
            f"Train: {len(train)} September rows with a current Maps duration, a camera count, and `after_gap = 0` "
            f"(6-30 Sep 2026 Asia/Singapore). Eval: {len(test)} rows on 1-4 Oct "
            f"({n_by['SG_TO_MY']} SG_TO_MY, {n_by['MY_TO_SG']} MY_TO_SG). "
            "Counts end 4 Oct, so later October Maps bins are not in this score."
        ),
        "visibility": (
            "A bin with no scored frame is missing, not zero. A scored frame whose count is zero stays zero. "
            "`local_counts` has no column that separates fog, haze, glare, or an empty road, so that flag is not applied."
        ),
        "models": rows,
        "figures": figures,
        "packages": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    payload = base_manifest(RUN_ID, ["camera_current"])
    payload["camera_current"] = _jsonable(study_body)
    write_run_json(run_dir, _jsonable(payload))
    write_report(run_dir, {"run_id": RUN_ID, **study_body})
    print(json.dumps(rows, indent=2), flush=True)
    print(run_dir, flush=True)


if __name__ == "__main__":
    main()
