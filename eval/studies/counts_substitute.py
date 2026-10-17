#!/usr/bin/env python3
"""Counts-for-Maps substitution on every model forecast-api can call.

Train on September 2026 only (Asia/Singapore). Score October 2026 origins that have both a
directional camera count and a Maps ``y_30`` label. The target stays Google Maps
``duration_in_traffic`` at the 30-minute horizon. Nothing here writes to GCP or calls Roboflow.

Maps-version fits use ``joined._ridge`` / ``_xgb``, ``bq_replica.fit_lin`` / ``fit_xgb``,
``horizon_study``'s Fourier profile, and ``layer_b_fcm_xgb_mix.fit_day``. Counts-version fits use
those same estimators. The duration slots (current value, lags, rolling means, slope) hold the
directional count series produced by ``features.maps_features``. Congestion ratio and speed are
omitted on the standalone counts models. The mix keeps the module's 18-column layout and leaves
those two columns missing.

    python counts_substitute.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
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
from google.cloud import bigquery

from harness.metrics import mae
from harness import significance as sig
from lb import bq_replica, camera_forecast as cf, features as fx, joined as jn
from studies import horizon_study as hs
from warehouse import layer_b_fcm_mlp as fcm, layer_b_fcm_xgb_mix as mix

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
PROJECT = "swiftborder"
SGT = "Asia/Singapore"
BAR_MIN = 0.5
MIN_N_PER_DIRECTION = 50
N_BOOTSTRAP = 4999
SERVE_DAY = "2026-10-01"
MAPS_ONLY = ("congestion_ratio", "speed_kmh")
COUNT_FEATURES = [c for c in jn.FEATURE_SETS["maps"] if c not in MAPS_ONLY]
COUNT_LIN = [c for c in bq_replica.LIN_FEATURES if c not in MAPS_ONLY]
COUNT_XGB = [c for c in bq_replica.XGB_FEATURES if c not in MAPS_ONLY]
PROFILE_FEATURES = ["prof_now", "prof_h"]
DIRECTIONS = ("SG_TO_MY", "MY_TO_SG")


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


def _query(sql: str, location: str) -> pd.DataFrame:
    """Read-only BigQuery via ``google.cloud.bigquery``, same client as ``eval/layer_b.py``.

    Maps bins live in ``US`` and ``cam2701.local_counts`` lives in ``asia-southeast1``, so each
    query sets its own job location. No ``bq`` CLI call.
    """
    client = bigquery.Client(project=PROJECT, location=location)
    return client.query(sql).to_dataframe(create_bqstorage_client=False)


def load_maps() -> pd.DataFrame:
    sql = f"""
    SELECT route_id, direction, bin_ts, dur_min, congestion_ratio, speed_kmh, n_obs, gap_min
    FROM `{PROJECT}.traffic_prediction.v_bins_10min`
    WHERE bin_ts >= TIMESTAMP('2026-09-06', 'Asia/Singapore')
      AND bin_ts < TIMESTAMP('2026-10-06', 'Asia/Singapore')
    """
    frame = _query(sql, "US")
    frame["bin_ts"] = pd.to_datetime(frame["bin_ts"], utc=True)
    # The client returns gap_min as a nullable integer. The first bin's NA must be a float
    # NaN so features.maps_features can flag gaps; a pandas NA cannot cast to int.
    for column in ("dur_min", "congestion_ratio", "speed_kmh", "n_obs", "gap_min"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64").astype(float)
    frame["typical_min"] = np.nan
    return frame.sort_values(["route_id", "bin_ts"]).reset_index(drop=True)


def load_counts() -> pd.DataFrame:
    sql = f"""
    SELECT frame_ts, sg_my, my_sg
    FROM `{PROJECT}.cam2701.local_counts`
    WHERE frame_ts >= TIMESTAMP('2026-09-06', 'Asia/Singapore')
      AND frame_ts < TIMESTAMP('2026-10-06', 'Asia/Singapore')
    """
    frame = _query(sql, "asia-southeast1")
    frame["frame_ts"] = pd.to_datetime(frame["frame_ts"], utc=True)
    return frame


def count_bins(frames: pd.DataFrame) -> pd.DataFrame:
    """10-minute mean directional count. A bin with no frame is absent, not zero."""
    work = frames.copy()
    work["bin_ts"] = work["frame_ts"].dt.floor("10min")
    parts = []
    for direction, column in (("SG_TO_MY", "sg_my"), ("MY_TO_SG", "my_sg")):
        grouped = work.groupby("bin_ts", as_index=False)[column].mean()
        grouped["direction"] = direction
        grouped["route_id"] = direction
        grouped = grouped.rename(columns={column: "dur_min"})
        grouped["typical_min"] = np.nan
        grouped["congestion_ratio"] = np.nan
        grouped["speed_kmh"] = np.nan
        grouped["n_obs"] = 1
        parts.append(grouped)
    binned = pd.concat(parts, ignore_index=True).sort_values(["route_id", "bin_ts"])
    binned["gap_min"] = binned.groupby("route_id")["bin_ts"].diff().dt.total_seconds() / 60.0
    return binned.reset_index(drop=True)


def with_flag(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_my_to_sg"] = (out["direction"] == "MY_TO_SG").astype(int)
    out["date_sgt"] = out["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    return out


def september_train(frame: pd.DataFrame, *, require_lag60: bool) -> pd.DataFrame:
    """Origins in September whose 30-minute label closed before 1 Oct 00:00 SGT."""
    day_start = pd.Timestamp(SERVE_DAY, tz=SGT).tz_convert("UTC")
    mask = (
        frame["y_30"].notna()
        & (frame["after_gap"] == 0)
        & (frame["bin_ts"] + pd.Timedelta(minutes=40) <= day_start)
        & (frame["date_sgt"] >= "2026-09-01")
        & (frame["date_sgt"] < SERVE_DAY)
    )
    if require_lag60:
        mask = mask & frame["lag_60"].notna()
    return frame.loc[mask].sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def _fit_ridge(train: pd.DataFrame, columns: Sequence[str]):
    return jn._ridge(columns).fit(train[list(columns)], train["y_30"].to_numpy())


def _fit_xgb(train: pd.DataFrame, columns: Sequence[str]):
    return jn._xgb().fit(train[list(columns)], train["y_30"].to_numpy())


def _fit_bq_lin(train: pd.DataFrame, columns: Sequence[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="mean", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=bq_replica.LIN_L2),
    ).fit(train[list(columns)], train["y_30"].to_numpy())


def _fit_bq_xgb(train: pd.DataFrame, columns: Sequence[str]):
    from xgboost import XGBRegressor

    x, y = train[list(columns)], train["y_30"].to_numpy()
    return bq_replica.SeedMean(
        XGBRegressor(**bq_replica.XGB_PARAMS, random_state=int(seed), n_jobs=1).fit(x, y)
        for seed in bq_replica.XGB_SEEDS
    )


def _profile_value(profile: cf.FourierProfile, frame: pd.DataFrame) -> np.ndarray:
    """Profile of the target bin, same read as ``horizon_study.add_profile`` at 30 minutes."""
    return profile.predict(
        frame["direction"].to_numpy(),
        frame["bin_ts"] + pd.Timedelta(minutes=30) + cf.NOW_OFFSET,
    )


def attach_profile(frame: pd.DataFrame, profile: cf.FourierProfile) -> pd.DataFrame:
    out = frame.copy()
    direction = out["direction"].to_numpy()
    out["prof_now"] = profile.predict(direction, out["bin_ts"] + cf.NOW_OFFSET)
    out["prof_h"] = profile.predict(direction, out["bin_ts"] + pd.Timedelta(minutes=30) + cf.NOW_OFFSET)
    return out


def _place(frame: pd.DataFrame, grid: fcm.Grid, column: str, *, keep_nonpositive: bool) -> Dict[str, np.ndarray]:
    series = {d: np.full(grid.n, np.nan) for d in DIRECTIONS}
    for row in frame.itertuples(index=False):
        direction = row.direction
        if direction not in series:
            continue
        value = getattr(row, column)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        number = float(value)
        if not keep_nonpositive and number <= 0:
            continue
        offset = fcm.unix(row.bin_ts.to_pydatetime()) - grid.start_unix
        if offset % fcm.BIN_SECONDS or not 0 <= offset < grid.n * fcm.BIN_SECONDS:
            continue
        series[direction][offset // fcm.BIN_SECONDS] = number
    return series


def _stamp_maps_labels(data: Dict[str, mix.DirectionData], maps_dur: Dict[str, np.ndarray], grid: fcm.Grid) -> None:
    """Point the mix's labels at Maps duration while leaving count features in place."""
    horizon = next(iter(data.values())).frame.horizon
    steps = np.arange(1, horizon + 1, dtype=np.int64)
    for direction, dd in data.items():
        origin = ((dd.frame.origin_unix - grid.start_unix) // mix.BIN_SECONDS).astype(np.int64)
        dd.frame.actual = fcm._take(maps_dur[direction], origin[:, None] + steps[None, :])


def _mix_frame(models: mix.MixModels, data: Dict[str, mix.DirectionData]) -> pd.DataFrame:
    rows: List[dict] = []
    for direction, dd in data.items():
        idx = np.arange(dd.frame.n)
        predicted = mix.predict_direction(models, dd, idx)
        for i, origin in enumerate(dd.frame.origin_unix):
            rows.append(
                {
                    "direction": direction,
                    "bin_ts": pd.Timestamp(fcm.from_unix(int(origin))),
                    "mix": float(predicted["mix"][i]),
                    "fcm_mlp": float(predicted["fcm_mlp"][i]),
                    "xgb_member": float(predicted["xgb_maps"][i]),
                }
            )
    return pd.DataFrame(rows)


def fit_mix(maps_bins: pd.DataFrame, counts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One September fit of the served mix, then the same functions on count features."""
    cfg = mix.MixConfig()
    cfg.validate()
    serve_day = fcm.date_to_epoch_day(fcm.parse_sgt_date(SERVE_DAY))
    times = list(maps_bins["bin_ts"]) + list(counts["bin_ts"])
    grid = fcm.Grid.spanning(min(times).to_pydatetime(), max(times).to_pydatetime())
    maps_records = []
    for row in maps_bins.itertuples(index=False):
        dur = row.dur_min
        if dur is None or not math.isfinite(float(dur)) or float(dur) <= 0:
            continue
        maps_records.append(
            {
                "direction": row.direction,
                "bin_ts": row.bin_ts.to_pydatetime(),
                "dur_min": float(dur),
                "congestion_ratio": None if pd.isna(row.congestion_ratio) else float(row.congestion_ratio),
                "speed_kmh": None if pd.isna(row.speed_kmh) else float(row.speed_kmh),
            }
        )
    maps_on_grid = mix.bins_from_rows(maps_records, grid, cut=False)
    label_cutoff = fcm.as_utc(datetime(2026, 10, 6, tzinfo=timezone.utc))
    maps_data = mix.build_data(
        cfg, maps_on_grid, np.arange(grid.n), label_cutoff=label_cutoff, layer_a={}, protect_labels=False
    )
    print("fitting mix on Maps inputs (September only)", flush=True)
    maps_models, _, maps_diag = mix.fit_day(cfg, maps_data, serve_day)
    count_dur = _place(counts, grid, "dur_min", keep_nonpositive=True)
    empty = {d: np.full(grid.n, np.nan) for d in DIRECTIONS}
    count_on_grid = mix.Bins(grid, count_dur, empty, {d: empty[d].copy() for d in DIRECTIONS}, False)
    count_data = mix.build_data(
        cfg, count_on_grid, np.arange(grid.n), label_cutoff=label_cutoff, layer_a={}, protect_labels=False
    )
    _stamp_maps_labels(count_data, maps_on_grid.dur, grid)
    print("fitting mix on count inputs (September only)", flush=True)
    count_models, _, count_diag = mix.fit_day(cfg, count_data, serve_day)
    info = {
        "serve_day": SERVE_DAY,
        "history_start": cfg.history_start,
        "maps_diagnostics": _jsonable(maps_diag),
        "counts_diagnostics": _jsonable(count_diag),
        "counts_predict_time_maps_columns": "congestion_ratio and speed_kmh are NaN; duration slots are counts",
        "fcm_residual": "counts mix trains delta = Maps duration - directional count, then predicts count + delta",
    }
    return _mix_frame(maps_models, maps_data), _mix_frame(count_models, count_data), info


def paired(frame: pd.DataFrame, challenger: str, reference: str, direction: str) -> sig.ComparisonResult:
    rows = frame if direction == "both" else frame[frame["direction"] == direction]
    rows = rows.sort_values(["direction", "bin_ts"])
    return sig.compare_absolute_errors(
        rows["y_30"].to_numpy(),
        rows[challenger].to_numpy(),
        rows[reference].to_numpy(),
        label_challenger=challenger,
        label_reference=reference,
        timestamps=list(rows["bin_ts"]),
        groups=list(rows["direction"]),
        horizon_steps=3,
        alpha=0.05,
        n_bootstrap=N_BOOTSTRAP,
        random_state=0,
        day_utc_offset_hours=8.0,
    )


def _interval(result: sig.ComparisonResult) -> Tuple[Optional[float], Optional[float], str]:
    if result.joint_ci_low_min is not None and result.joint_ci_high_min is not None:
        return float(result.joint_ci_low_min), float(result.joint_ci_high_min), "joint_calendar_day"
    if result.n < 2 or not math.isfinite(result.bootstrap_ci_low_min):
        return None, None, "undefined"
    return float(result.bootstrap_ci_low_min), float(result.bootstrap_ci_high_min), "day_block"


def classify(
    n_by_direction: Dict[str, int],
    low: Optional[float],
    high: Optional[float],
    mae_counts: Optional[float],
    mae_persistence: Optional[float],
    *,
    emits_minutes: bool,
    reads_maps_at_predict: bool,
    calendar_days: Optional[int] = None,
) -> Tuple[str, str]:
    if calendar_days is not None and calendar_days < sig.MIN_BLOCKS:
        return (
            "inconclusive",
            f"only {calendar_days} shared calendar days (harness gate needs {sig.MIN_BLOCKS})",
        )
    if min(n_by_direction.values()) < MIN_N_PER_DIRECTION:
        return "inconclusive", "fewer than 50 October rows in a direction"
    if not emits_minutes:
        return "inconclusive", "the counts-series rule does not emit a duration in minutes"
    if low is None or high is None or mae_counts is None or mae_persistence is None:
        return "inconclusive", "paired interval undefined"
    if low > BAR_MIN:
        return "regression", "the paired interval lies entirely above +0.5 min"
    if high < -BAR_MIN:
        return "improvement", "the paired interval lies entirely below -0.5 min"
    if low >= -BAR_MIN and high <= BAR_MIN:
        if reads_maps_at_predict:
            return "inconclusive", "the counts version still reads Maps bins at predict time"
        if abs(mae_counts - mae_persistence) <= BAR_MIN:
            return (
                "substitute",
                "within +/-0.5 min of the Maps version and of persistence, and it does not read Maps bins at predict time",
            )
        return (
            "close to maps, outside persistence bar",
            "within +/-0.5 min of the Maps version, and more than 0.5 min from persistence",
        )
    return "inconclusive", "the paired interval crosses the +/-0.5 min bar"


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def score_block(
    frame: pd.DataFrame,
    model_id: str,
    maps_col: str,
    counts_col: Optional[str],
    *,
    emits_minutes: bool,
    reads_maps_at_predict: bool,
    train_n: dict,
    note: str,
) -> dict:
    n_by = {d: int((frame["direction"] == d).sum()) for d in DIRECTIONS}
    actual = frame["y_30"].to_numpy()
    mae_maps = mae(actual, frame[maps_col].to_numpy())
    mae_pers = mae(actual, frame["persistence_maps"].to_numpy())
    mae_counts = None
    if counts_col is not None and emits_minutes:
        mae_counts = mae(actual, frame[counts_col].to_numpy())
    raw_counts = None
    if counts_col is not None and not emits_minutes:
        raw_counts = mae(actual, frame[counts_col].to_numpy())

    slices = []
    for direction in ("both",) + DIRECTIONS:
        if emits_minutes and counts_col is not None:
            result = paired(frame, counts_col, maps_col, direction)
            low, high, kind = _interval(result)
            diff = float(result.mean_ae_diff_min)
        else:
            result = None
            low = high = diff = None
            kind = "not a minute comparison"
        part = frame if direction == "both" else frame[frame["direction"] == direction]
        y_part = part["y_30"].to_numpy()
        part_counts = mae(y_part, part[counts_col].to_numpy()) if counts_col and emits_minutes else None
        part_pers = mae(y_part, part["persistence_maps"].to_numpy())
        if direction == "both":
            label_n = n_by
            label_counts, label_pers = mae_counts, mae_pers
        else:
            # The other direction is not in this slice; only this direction's n can fail the floor.
            label_n = {d: MIN_N_PER_DIRECTION for d in DIRECTIONS}
            label_n[direction] = int(len(part))
            label_counts, label_pers = part_counts, part_pers
        cal_days = None if result is None else result.calendar_days
        label, reason = classify(
            label_n,
            low,
            high,
            label_counts,
            label_pers,
            emits_minutes=emits_minutes,
            reads_maps_at_predict=reads_maps_at_predict,
            calendar_days=cal_days,
        )
        y = part["y_30"].to_numpy()
        slices.append(
            {
                "slice": direction,
                "n": int(len(part)),
                "mae_maps_min": _round(mae(y, part[maps_col].to_numpy())),
                "mae_counts_min": _round(mae(y, part[counts_col].to_numpy())) if counts_col and emits_minutes else None,
                "mae_persistence_maps_min": _round(mae(y, part["persistence_maps"].to_numpy())),
                "paired_diff_counts_minus_maps_min": _round(diff),
                "interval_low_min": _round(low),
                "interval_high_min": _round(high),
                "interval": kind,
                "calendar_days": None if result is None else result.calendar_days,
                "label": label,
                "reason": reason,
                "raw_abs_diff_count_units_vs_minutes": _round(mae(y, part[counts_col].to_numpy()))
                if counts_col and not emits_minutes
                else None,
            }
        )
    return {
        "id": model_id,
        "train_n": train_n,
        "emits_minutes": emits_minutes,
        "reads_maps_at_predict": reads_maps_at_predict,
        "note": note,
        "slices": slices,
        "raw_counts_vs_maps_minutes": _round(raw_counts),
    }


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_ci(row: dict) -> str:
    if row["interval_low_min"] is None:
        return "—"
    return f"{row['paired_diff_counts_minus_maps_min']:.3f} [{row['interval_low_min']:.3f}, {row['interval_high_min']:.3f}]"


_MAPS_COLOR = "#1b4f72"
_COUNTS_COLOR = "#d35400"
_PERS_COLOR = "#7f8c8d"
_SGT_ZONE = SGT
_SLICE_TITLES = {
    "both": "Pooled",
    "SG_TO_MY": "SG_TO_MY",
    "MY_TO_SG": "MY_TO_SG",
}


def _slice_rows(payload: dict, slice_name: str) -> List[dict]:
    rows = []
    for model in payload["models"]:
        for sl in model["slices"]:
            if sl["slice"] == slice_name:
                rows.append({"id": model["id"], **sl})
    return rows


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e5e8eb", linewidth=0.6)
    ax.set_axisbelow(True)


def write_plots(run_dir: Path, payload: dict, frame: pd.DataFrame) -> List[str]:
    """October MAE bars, paired-difference intervals, and the served-model series."""
    names = [
        _plot_mae(run_dir, payload),
        _plot_paired_diff(run_dir, payload),
        _plot_series(run_dir, frame),
    ]
    return names


def _plot_mae(run_dir: Path, payload: dict) -> str:
    name = "mae_maps_vs_counts.png"
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.4), sharey=True)
    panels = ("both",) + DIRECTIONS
    for ax, panel in zip(axes, panels):
        rows = _slice_rows(payload, panel)
        y = np.arange(len(rows))
        height = 0.24
        maps_vals = np.array([row["mae_maps_min"] for row in rows], dtype=float)
        counts_vals = np.array(
            [np.nan if row["mae_counts_min"] is None else row["mae_counts_min"] for row in rows],
            dtype=float,
        )
        pers_vals = np.array([row["mae_persistence_maps_min"] for row in rows], dtype=float)
        finite = np.isfinite(counts_vals)
        ax.barh(y - height, maps_vals, height, color=_MAPS_COLOR, label="Maps inputs")
        ax.barh(y[finite], counts_vals[finite], height, color=_COUNTS_COLOR, label="Counts only")
        ax.barh(y + height, pers_vals, height, color=_PERS_COLOR, label="Maps persistence")
        ax.set_yticks(y)
        ax.set_yticklabels([row["id"] for row in rows])
        ax.set_xlabel("MAE (minutes)")
        n = rows[0]["n"] if rows else 0
        ax.set_title(f"{_SLICE_TITLES[panel]} (n={n})")
        ax.set_xlim(0, max(10.0, float(np.nanmax(maps_vals)), float(np.nanmax(counts_vals)), float(np.nanmax(pers_vals))) + 0.4)
        _style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("October 2026 MAE: Maps inputs, counts-only, and Maps persistence", fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.text(
        0.5,
        0.01,
        "Persistence and profile have no counts bar: those rules emit a vehicle count, not minutes. "
        "lin_bq and xgb_bq daily and sept-refit are the same September fit.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(run_dir / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def _plot_paired_diff(run_dir: Path, payload: dict) -> str:
    name = "paired_diff.png"
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.6), sharey=True)
    for ax, panel in zip(axes, ("both",) + DIRECTIONS):
        rows = [row for row in _slice_rows(payload, panel) if row["paired_diff_counts_minus_maps_min"] is not None]
        y = np.arange(len(rows))
        diffs = np.array([row["paired_diff_counts_minus_maps_min"] for row in rows], dtype=float)
        lows = np.array([row["interval_low_min"] for row in rows], dtype=float)
        highs = np.array([row["interval_high_min"] for row in rows], dtype=float)
        span_lo = min(-0.8, float(lows.min()) - 0.3)
        span_hi = max(0.8, float(highs.max()) + 0.3)
        ax.axvspan(-BAR_MIN, BAR_MIN, color="#e8f6ef", zorder=0)
        ax.axvline(-BAR_MIN, color="#1e8449", linestyle="--", linewidth=1, label="-0.5 min")
        ax.axvline(BAR_MIN, color="#922b21", linestyle="--", linewidth=1, label="+0.5 min")
        ax.axvline(0, color="#b0b7bf", linewidth=0.7)
        ax.errorbar(
            diffs,
            y,
            xerr=[diffs - lows, highs - diffs],
            fmt="o",
            color="#922b21",
            ecolor="#922b21",
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            label="Counts - Maps",
        )
        ax.set_yticks(y)
        ax.set_yticklabels([row["id"] for row in rows])
        ax.set_xlim(span_lo, span_hi)
        ax.set_xlabel("Counts MAE - Maps MAE (minutes)")
        ax.set_title(f"{_SLICE_TITLES[panel]} (n={rows[0]['n']})")
        _style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Paired October difference. The band is the +/- 0.5 minute bar.", fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.07))
    fig.text(
        0.5,
        0.01,
        "Positive means the counts version is worse. An interval entirely above +0.5 is a regression; "
        "entirely below -0.5 would be an improvement. Holm is not applied.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(run_dir / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def _plot_series(run_dir: Path, frame: pd.DataFrame) -> str:
    name = "series_xgb_mix.png"
    panels = (
        ("xgb[maps]", "xgb_maps", "xgb_counts"),
        ("mix[fcm_mlp+xgb]", "mix_maps", "mix_counts"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), sharex=True, sharey=True)
    for row_i, direction in enumerate(DIRECTIONS):
        part = frame[frame["direction"] == direction].sort_values("bin_ts")
        clock = part["bin_ts"].dt.tz_convert(_SGT_ZONE)
        for col_i, (title, maps_col, counts_col) in enumerate(panels):
            ax = axes[row_i, col_i]
            ax.plot(clock, part["y_30"], color="#1a1a1a", linewidth=0.9, label="Maps y_30")
            ax.plot(clock, part[maps_col], color=_MAPS_COLOR, linewidth=0.8, label="Maps-input forecast")
            ax.plot(clock, part[counts_col], color=_COUNTS_COLOR, linewidth=0.8, label="Counts-only forecast")
            ax.set_title(f"{title}  {direction}  n={len(part)}")
            _style_axes(ax)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=_SGT_ZONE))
            ax.xaxis.set_major_locator(mdates.DayLocator(tz=_SGT_ZONE))
            if row_i == 0 and col_i == 1:
                ax.legend(loc="upper right", frameon=False, fontsize=8)
            if col_i == 0:
                ax.set_ylabel("Minutes")
    fig.suptitle("Actual Maps duration and forecasts, 1-4 Oct 2026 (Asia/Singapore)", fontsize=12)
    fig.text(
        0.01,
        0.01,
        "10-minute origins that have both a camera count and a Maps y_30 label. "
        "The counts forecast does not read v_bins_10min at predict time.",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(run_dir / name, dpi=140)
    plt.close(fig)
    return name


def write_report(run_dir: Path, payload: dict) -> None:
    pooled = []
    directional = []
    for model in payload["models"]:
        for sl in model["slices"]:
            line = {"id": model["id"], **sl, "note": model["note"]}
            (pooled if sl["slice"] == "both" else directional).append(line)
    lines = [
        "# Counts substituted for Maps inputs",
        "",
        payload["definition"],
        "",
        "## Protocol",
        "",
        payload["protocol"],
        "",
        f"**Practical bar.** {payload['bar']}",
        "",
        "## Window",
        "",
        payload["window"],
        "",
        "## Decision table (pooled October rows)",
        "",
        "Paired difference is counts MAE minus Maps-version MAE, in minutes. Positive means the counts version is worse. The interval is the harness day-block bootstrap (`eval/significance.py`, 4,999 resamples, seed 0, Asia/Singapore day boundary). With both directions the interval is the joint calendar-day bootstrap. Holm is not applied across this family.",
        "",
        "| Model | n | MAE Maps | MAE counts | MAE persistence (Maps) | Counts − Maps (95% CI) | Label |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in pooled:
        lines.append(
            f"| `{row['id']}` | {row['n']} | {_fmt(row['mae_maps_min'])} | {_fmt(row['mae_counts_min'])} | {_fmt(row['mae_persistence_maps_min'])} | {_fmt_ci(row)} | {row['label']} |"
        )
    lines += [
        "",
        "## Per direction",
        "",
        "| Model | Direction | n | MAE Maps | MAE counts | MAE persistence (Maps) | Counts − Maps (95% CI) | Label |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in directional:
        lines.append(
            f"| `{row['id']}` | {row['slice']} | {row['n']} | {_fmt(row['mae_maps_min'])} | {_fmt(row['mae_counts_min'])} | {_fmt(row['mae_persistence_maps_min'])} | {_fmt_ci(row)} | {row['label']} |"
        )
    figures = payload.get("figures") or []
    if figures:
        lines += ["", "## Figures", ""]
        captions = {
            "mae_maps_vs_counts.png": "October MAE by model, pooled and by direction.",
            "paired_diff.png": "Paired difference (counts minus Maps) with the +/- 0.5 minute bar.",
            "series_xgb_mix.png": "Maps y_30 against the Maps-input and counts-only forecasts for xgb[maps] and the served mix.",
        }
        for figure in figures:
            lines.append(f"![{captions.get(figure, figure)}]({figure})")
            lines.append("")
    lines += ["## Notes", ""]
    for model in payload["models"]:
        lines.append(f"- `{model['id']}`: {model['note']}")
    if payload.get("univariate_raw"):
        lines += ["", payload["univariate_raw"]]
    lines += [
        "",
        "This run does not fill the live forecast-api 503, does not change the service, and does not show a crossing-time error. A counts model that still needed `v_bins_10min` at predict time would not be a substitute; the counts versions above do not read it then.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def coverage_text(maps_bins: pd.DataFrame, counts: pd.DataFrame) -> str:
    maps = maps_bins.copy()
    maps["date_sgt"] = maps["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    oct_maps = maps[maps["date_sgt"] >= SERVE_DAY]
    by_day = (
        oct_maps.groupby("date_sgt")
        .agg(n_bins=("bin_ts", "size"), avg_obs=("n_obs", "mean"))
        .reset_index()
    )
    counts = counts.copy()
    counts["date_sgt"] = counts["frame_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    count_days = counts.groupby("date_sgt").size()
    oct_count_days = {d: int(n) for d, n in count_days.items() if d >= SERVE_DAY}
    day_bits = [
        f"{row.date_sgt}: {int(row.n_bins)} bins, mean n_obs {row.avg_obs:.2f}"
        for row in by_day.itertuples(index=False)
    ]
    return (
        f"Maps `v_bins_10min` rows in the download: {len(maps_bins)} "
        f"({maps_bins['bin_ts'].min().isoformat()} to {maps_bins['bin_ts'].max().isoformat()}). "
        f"October SGT bin-days in that download: {'; '.join(day_bits)}. "
        f"`cam2701.local_counts` frames in the download: {len(counts)} "
        f"({counts['frame_ts'].min().isoformat()} to {counts['frame_ts'].max().isoformat()}). "
        f"October SGT frame counts by day: {oct_count_days}. "
        "Camera 2702 is not used: `features.py` builds counts from one camera (2701)."
    )


def main() -> int:
    print("querying BigQuery", flush=True)
    maps_raw = load_maps()
    count_raw = load_counts()
    print(f"maps bins {len(maps_raw)} count frames {len(count_raw)}", flush=True)
    maps_feats = with_flag(fx.maps_features(maps_raw))
    count_feats = with_flag(fx.maps_features(count_bins(count_raw)))
    count_feats = count_feats.drop(columns=["y_30", "y_60"]).merge(
        maps_feats[["direction", "bin_ts", "y_30"]], on=["direction", "bin_ts"], how="left"
    )
    day_start = pd.Timestamp(SERVE_DAY, tz=SGT).tz_convert("UTC")
    maps_profile = hs.fit_profile(maps_feats, day_start)
    count_profile = cf.FourierProfile(harmonics=hs.PROFILE_HARMONICS).fit(
        count_feats.loc[
            (count_feats["bin_ts"] + fx.BIN <= day_start) & (count_feats["date_sgt"] < SERVE_DAY),
            ["direction", "bin_ts", "y_persistence"],
        ].dropna(),
        value="y_persistence",
    )
    maps_scorable = jn.scorable(maps_feats)
    count_scorable = jn.scorable(count_feats)
    profile_rows_maps = int(((maps_feats["bin_ts"] + fx.BIN) <= day_start).sum())
    profile_rows_counts = int(
        ((count_feats["bin_ts"] + fx.BIN <= day_start) & (count_feats["date_sgt"] < SERVE_DAY) & count_feats["y_persistence"].notna()).sum()
    )
    october_maps = maps_scorable[maps_scorable["date_sgt"] >= SERVE_DAY]
    october_counts = count_scorable[count_scorable["date_sgt"] >= SERVE_DAY]
    joined = october_maps.merge(october_counts, on=["direction", "bin_ts"], suffixes=("_maps", "_counts"))
    joined = joined[joined["y_30_maps"].notna() & joined["y_30_counts"].notna()].copy()
    joined["y_30"] = joined["y_30_maps"]
    n_by = {d: int((joined["direction"] == d).sum()) for d in DIRECTIONS}
    print(f"October overlap rows { {d: n_by[d] for d in DIRECTIONS} }", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_counts-substitute"
    run_dir = EVAL_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    definition = (
        "The target on every row is Google Maps duration in traffic at the 30-minute horizon: "
        "the mean `duration_in_traffic` of the bin that starts 30 minutes after the origin (`y_30`), in minutes. "
        "The Maps version is that served estimator refit on September 2026 only, reading Maps duration, lags, "
        "rolling means, slope, congestion ratio and speed where that model uses them. "
        "The counts version puts the directional camera-2701 count in those duration slots "
        "(`sg_my` for SG_TO_MY, `my_sg` for MY_TO_SG) and does not read a Maps duration, lag, speed or distance "
        "feature at predict time. Calendar terms stay. "
        "Persistence-of-Maps repeats the current Maps duration. Persistence-of-counts repeats the current "
        "directional count and does not see Maps at predict time; that output is a vehicle count, so it is not "
        "scored as a minute forecast. The counts profile is the same Fourier rule fit on September counts and "
        "read at the target clock; that output is a typical count, not a duration. "
        "Learned counts models are fit to September rows whose label is Maps `y_30`, so they can emit minutes."
    )
    protocol = (
        "Callable forecast-api ids confirmed in `services/forecastapi/main.py`: `persistence`, `profile`, "
        "`ridge[maps]`, `xgb[maps]`, `xgb[maps+prof]`, `lin_bq[daily]`, `xgb_bq[daily]`, `lin_bq[frozen]`, "
        "`xgb_bq[frozen]`, and `mix[fcm_mlp+xgb]`. `served` at 30 minutes is the mix in both directions, so it is "
        "not a second fit. `fcm_mlp` alone and `timesfm` are not served. "
        f"`lin_bq[frozen]` and `xgb_bq[frozen]` are refit on all September rows and reported as "
        "`lin_bq[sept-refit]` and `xgb_bq[sept-refit]`, not as the 12 Sep production cutoff. "
        "Those September fits use the same rows and estimators as `lin_bq[daily]` and `xgb_bq[daily]` "
        "(BQ feature list, `lag_60` required), so each pair is one fit listed under both names. "
        "No October row is used to fit. The mix's L2 penalty still uses the module's last two September days "
        "as inner validation (`fit_day` for serving day 2026-10-01)."
    )
    bar = (
        "`docs/evaluation.md` states two different bars. \"What would count as success\" is the product target "
        "of mean absolute error at or below 15 minutes on crossing time; that target is not measured here, and "
        "this run does not claim it. The practical bar used for this decision is the one in the Significance "
        "section: a serving change needs a drop of at least 0.5 min (30 s), about the rounding of a forecast "
        "shown in whole minutes. The same +/-0.5 min band is the practical threshold elsewhere on that page. "
        "A difference inside that band is not a meaningful improvement. "
        "Substitution also requires the counts version to be able to run when Maps bins are absent "
        "(it does not read `v_bins_10min` at predict time) and its October MAE to sit within 0.5 min of both "
        "the Maps version and persistence-of-Maps. "
        "Label `inconclusive` when fewer than ten shared calendar days (harness significance gate) or fewer than "
        "50 rows in a direction. Otherwise label `regression` when the paired interval lies entirely above "
        "+0.5 min, `improvement` when it lies entirely below -0.5 min, `substitute` when the interval lies inside "
        "+/-0.5 min and the counts MAE is also within 0.5 min of persistence, and `inconclusive` when the "
        "interval crosses +/-0.5 min."
    )
    window = (
        "Clocks: bin timestamps are UTC. The month cut is the Asia/Singapore calendar date. "
        "Train: origin date 2026-09-06 through 2026-09-30 SGT (Maps logging and `local_counts` both start 6 Sep; "
        "1-5 Sep have neither series) and `bin_ts + 40 min` at or before 2026-10-01 00:00 SGT. "
        "Eval: October 2026 SGT origins that are scorable on both series (`y_30` present, `after_gap = 0`). "
        + coverage_text(maps_raw, count_raw)
    )
    payload: dict = {
        "schema_version": 2,
        "run_id": run_id,
        "components": ["counts_substitute"],
        "component": "counts_substitute",
        "provenance": {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "definition": definition,
        "protocol": protocol,
        "bar_min": BAR_MIN,
        "bar": bar,
        "window": window,
        "october_overlap_n": n_by,
        "models": [],
    }
    if min(n_by.values()) < MIN_N_PER_DIRECTION:
        payload["stopped"] = "October overlap is under 50 rows in a direction; no model was fit."
        payload["univariate_raw"] = ""
        (run_dir / "run.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        write_report(run_dir, payload)
        print(f"stopped, wrote {run_dir}", flush=True)
        return 0

    maps_train = september_train(maps_scorable, require_lag60=False)
    maps_train_bq = september_train(maps_scorable, require_lag60=True)
    count_train = september_train(count_scorable, require_lag60=False)
    count_train_bq = september_train(count_scorable, require_lag60=True)
    maps_train_p = attach_profile(maps_train, maps_profile)
    count_train_p = attach_profile(count_train, count_profile)
    october = joined.copy()
    # Rebuild predict frames from the suffixed join columns.
    def take(suffix: str, columns: Sequence[str]) -> pd.DataFrame:
        data = {"direction": october["direction"], "bin_ts": october["bin_ts"], "y_30": october["y_30"]}
        for column in columns:
            data[column] = october[f"{column}_{suffix}"]
        return pd.DataFrame(data)

    maps_x = take("maps", jn.FEATURE_SETS["maps"])
    count_x = take("counts", COUNT_FEATURES)
    maps_x = attach_profile(maps_x, maps_profile)
    count_x = attach_profile(count_x, count_profile)
    print(
        f"train rows maps {len(maps_train)} maps_bq {len(maps_train_bq)} counts {len(count_train)} counts_bq {len(count_train_bq)}",
        flush=True,
    )

    ridge_m = _fit_ridge(maps_train, jn.FEATURE_SETS["maps"])
    ridge_c = _fit_ridge(count_train, COUNT_FEATURES)
    xgb_m = _fit_xgb(maps_train, jn.FEATURE_SETS["maps"])
    xgb_c = _fit_xgb(count_train, COUNT_FEATURES)
    prof_cols_m = jn.FEATURE_SETS["maps"] + PROFILE_FEATURES
    prof_cols_c = COUNT_FEATURES + PROFILE_FEATURES
    xgb_prof_m = _fit_xgb(maps_train_p, prof_cols_m)
    xgb_prof_c = _fit_xgb(count_train_p, prof_cols_c)
    lin_m = bq_replica.fit_lin(maps_train_bq)
    lin_c = _fit_bq_lin(count_train_bq, COUNT_LIN)
    xgb_bq_m = bq_replica.fit_xgb(maps_train_bq)
    xgb_bq_c = _fit_bq_xgb(count_train_bq, COUNT_XGB)

    scored = pd.DataFrame(
        {
            "direction": october["direction"].to_numpy(),
            "bin_ts": pd.to_datetime(october["bin_ts"], utc=True),
            "y_30": october["y_30"].to_numpy(),
            "persistence_maps": october["y_persistence_maps"].to_numpy(),
            "persistence_counts": october["y_persistence_counts"].to_numpy(),
            "profile_maps": _profile_value(maps_profile, october),
            "profile_counts": _profile_value(count_profile, october),
            "ridge_maps": ridge_m.predict(maps_x[jn.FEATURE_SETS["maps"]]),
            "ridge_counts": ridge_c.predict(count_x[COUNT_FEATURES]),
            "xgb_maps": xgb_m.predict(maps_x[jn.FEATURE_SETS["maps"]]),
            "xgb_counts": xgb_c.predict(count_x[COUNT_FEATURES]),
            "xgb_prof_maps": xgb_prof_m.predict(maps_x[prof_cols_m]),
            "xgb_prof_counts": xgb_prof_c.predict(count_x[prof_cols_c]),
            "lin_maps": bq_replica.predict_lin(lin_m, maps_x),
            "lin_counts": lin_c.predict(count_x[COUNT_LIN]),
            "xgb_bq_maps": bq_replica.predict_xgb(xgb_bq_m, maps_x),
            "xgb_bq_counts": xgb_bq_c.predict(count_x[COUNT_XGB]),
        }
    )
    mix_note = "mix fit failed"
    try:
        mix_maps, mix_counts, mix_info = fit_mix(maps_raw, count_bins(count_raw))
        payload["mix"] = mix_info
        mix_maps["bin_ts"] = pd.to_datetime(mix_maps["bin_ts"], utc=True)
        mix_counts["bin_ts"] = pd.to_datetime(mix_counts["bin_ts"], utc=True)
        scored = scored.merge(
            mix_maps[["direction", "bin_ts", "mix"]].rename(columns={"mix": "mix_maps"}),
            on=["direction", "bin_ts"],
            how="left",
        )
        scored = scored.merge(
            mix_counts[["direction", "bin_ts", "mix"]].rename(columns={"mix": "mix_counts"}),
            on=["direction", "bin_ts"],
            how="left",
        )
        mix_note = (
            "Same `fit_day` / `predict_direction` as the service. Counts version: duration slots are directional "
            "counts, congestion and speed are missing, and the FCM residual is Maps minutes minus the count."
        )
    except Exception as exc:  # the rest of the table is still the experiment
        payload["mix_error"] = f"{type(exc).__name__}: {exc}"
        scored["mix_maps"] = np.nan
        scored["mix_counts"] = np.nan
        print(f"mix failed: {exc}", flush=True)

    minute_cols = [
        "ridge_maps",
        "ridge_counts",
        "xgb_maps",
        "xgb_counts",
        "xgb_prof_maps",
        "xgb_prof_counts",
        "lin_maps",
        "lin_counts",
        "xgb_bq_maps",
        "xgb_bq_counts",
        "profile_maps",
        "persistence_maps",
    ]
    if scored["mix_maps"].notna().any():
        minute_cols += ["mix_maps", "mix_counts"]
    mask = np.isfinite(scored[minute_cols].to_numpy(dtype=float)).all(axis=1)
    common = scored.loc[mask].sort_values(["direction", "bin_ts"]).reset_index(drop=True)
    payload["october_scored_n"] = {d: int((common["direction"] == d).sum()) for d in DIRECTIONS}
    payload["october_scored_days"] = sorted(common["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d").unique())
    print(f"scored rows {payload['october_scored_n']} days {payload['october_scored_days']}", flush=True)

    same_bq = (
        "Same September fit as the daily replica. Not the 12 Sep production frozen model."
    )
    specs = [
        ("persistence", "persistence_maps", "persistence_counts", False, False, {"maps": 0, "counts": 0},
         "Univariate. Counts version repeats the current vehicle count and is not a minute forecast."),
        ("profile", "profile_maps", "profile_counts", False, False,
         {"maps": profile_rows_maps, "counts": profile_rows_counts},
         "Univariate calendar rule. Counts version is fit on September counts and emits a typical count, not minutes. It does not read the live Maps bin."),
        ("ridge[maps]", "ridge_maps", "ridge_counts", True, False,
         {"maps": int(len(maps_train)), "counts": int(len(count_train))},
         "Harness ridge (`joined._ridge`). Counts features omit congestion ratio and speed."),
        ("xgb[maps]", "xgb_maps", "xgb_counts", True, False,
         {"maps": int(len(maps_train)), "counts": int(len(count_train))},
         "Harness XGBoost (`joined._xgb`). Counts features omit congestion ratio and speed."),
        ("xgb[maps+prof]", "xgb_prof_maps", "xgb_prof_counts", True, False,
         {"maps": int(len(maps_train_p)), "counts": int(len(count_train_p))},
         "Harness XGBoost plus the Fourier profile. The counts version uses the profile of the count series, not the Maps profile."),
        ("lin_bq[daily]", "lin_maps", "lin_counts", True, False,
         {"maps": int(len(maps_train_bq)), "counts": int(len(count_train_bq))},
         "BQ linear replica (`bq_replica.fit_lin` on the Maps side). September only, `lag_60` required."),
        ("lin_bq[sept-refit]", "lin_maps", "lin_counts", True, False,
         {"maps": int(len(maps_train_bq)), "counts": int(len(count_train_bq))},
         same_bq + " Callable id this stands in for: `lin_bq[frozen]`."),
        ("xgb_bq[daily]", "xgb_bq_maps", "xgb_bq_counts", True, False,
         {"maps": int(len(maps_train_bq)), "counts": int(len(count_train_bq))},
         "BQ XGBoost replica (`bq_replica.fit_xgb`, five seeds). September only, `lag_60` required."),
        ("xgb_bq[sept-refit]", "xgb_bq_maps", "xgb_bq_counts", True, False,
         {"maps": int(len(maps_train_bq)), "counts": int(len(count_train_bq))},
         same_bq + " Callable id this stands in for: `xgb_bq[frozen]`."),
        ("mix[fcm_mlp+xgb]", "mix_maps", "mix_counts", True, False,
         {"maps": "see mix.maps_diagnostics", "counts": "see mix.counts_diagnostics"},
         mix_note),
    ]
    if not np.isfinite(common["mix_maps"]).all():
        specs = [s for s in specs if s[0] != "mix[fcm_mlp+xgb]"]
        payload["models_omitted"] = ["mix[fcm_mlp+xgb]"]

    for spec in specs:
        payload["models"].append(
            score_block(
                common,
                spec[0],
                spec[1],
                spec[2],
                emits_minutes=spec[3],
                reads_maps_at_predict=spec[4],
                train_n=spec[5],
                note=spec[6],
            )
        )
    payload["figures"] = write_plots(run_dir, payload, common)
    raw_pers = mae(common["y_30"].to_numpy(), common["persistence_counts"].to_numpy())
    raw_prof = mae(common["y_30"].to_numpy(), common["profile_counts"].to_numpy())
    payload["univariate_raw"] = (
        f"Persistence-of-counts and the counts profile are not minute forecasts. "
        f"On these same October rows the mean absolute difference between Maps `y_30` (minutes) and the "
        f"vehicle-count output is {raw_pers:.3f} for persistence-of-counts and {raw_prof:.3f} for the counts profile. "
        "Those figures are not duration errors and are not labelled improvement or regression."
    )
    payload["train_n"] = {
        "maps_harness": int(len(maps_train)),
        "maps_bq": int(len(maps_train_bq)),
        "counts_harness": int(len(count_train)),
        "counts_bq": int(len(count_train_bq)),
    }
    (run_dir / "run.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    write_report(run_dir, payload)
    print(f"wrote {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
