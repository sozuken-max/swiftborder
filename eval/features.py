#!/usr/bin/env python3
"""Causal 10-minute feature table: Maps bins (v_training_set logic) + weather + camera 2701 counts.

One row per (direction, bin_ts) where ``bin_ts`` is the UTC start of a 10-minute bin.

**Availability time.** A bin's Maps mean (``dur_min`` = ``y_persistence``) includes observations
up to the bin end, so every row is treated as known at ``asof = bin_ts + 10 min``. External features
use only data stamped at or before ``asof``. Labels ``y_30`` / ``y_60`` are the bin means 3 and 6
bins later (the same definition as ``traffic_prediction.v_training_set``).

Maps columns reproduce ``v_bins_10min`` + ``v_training_set`` in pandas. Two deliberate differences
(both identical to the view when no bin is missing, which a 2026-09-30 query confirmed):
lags and labels are **time-based** (a missing bin gives NaN rather than the previous row), and
``after_gap`` is computed the same way as the view.

Weather (Causeway CSVs):
- ``rain_30`` / ``rain_60``: rainfall (mm) at station S210 (Woodlands Centre) in (asof-30, asof] and
  (asof-60, asof].
- ``fc_rain`` / ``fc_heavy``: the 2-hour forecast for Woodlands whose valid period covers the
  **target** time (bin_ts + 30 min), most recently **acquired by data.gov.sg** (``update_timestamp``)
  at or before ``asof``; rows without an acquisition time count as available 10 minutes after issue.
  This fixes the BigQuery view, which bins forecasts by issue time.

Camera 2701 (``backfill_camera_counts.py`` CSV), per direction:
- ``cam_count``, ``cam_extent``, ``cam_age_min`` from the latest scored frame with
  ``frame_ts <= asof`` and at most 30 minutes old. A scored frame with no vehicles is 0; a bin
  with no usable frame is NaN (the view's per-frame count drops empty frames).

    python features.py --build                 # writes eval/data/features_10min.parquet + coverage
    python features.py --build --refresh-bq    # re-download travel_times (read-only) first
    python features.py --parity                # compare Maps columns with the live view (read-only)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
TT_CACHE = EVAL_ROOT / "data" / "causeway_gdata.csv"
RAIN_CSV = REPO_ROOT / "Causeway" / "data" / "rainfall" / "S210_Woodlands_Centre.csv"
FORECAST_CSV = REPO_ROOT / "Causeway" / "data" / "forecast" / "Woodlands.csv"
CAMERA_CSV = EVAL_ROOT / "data" / "camera" / "cam2701_counts.csv"
OUT = EVAL_ROOT / "data" / "features_10min.parquet"

BIN = pd.Timedelta(minutes=10)
SGT = "Asia/Singapore"
CAMERA_DIRECTION = {"SG_TO_MY": "sg_my", "MY_TO_SG": "my_sg"}  # camdetect SG-MY / MY-SG columns
MAPS_FEATURES = [
    "y_persistence",
    "congestion_ratio",
    "speed_kmh",
    "lag_10",
    "lag_20",
    "lag_30",
    "lag_60",
    "roll_mean_30",
    "roll_mean_60",
    "slope_30",
    "tod_sin",
    "tod_cos",
    "tod_block",
    "dow",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
]
WEATHER_FEATURES = ["rain_30", "rain_60", "fc_rain", "fc_heavy"]
CAMERA_FEATURES = ["cam_count", "cam_extent", "cam_age_min"]
RAIN_RE = re.compile(r"rain|shower|thunder", re.I)
HEAVY_RE = re.compile(r"heavy|thunder", re.I)


def _utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="mixed")


# --- Maps -------------------------------------------------------------------------


def maps_bins(travel_times: pd.DataFrame) -> pd.DataFrame:
    """v_bins_10min: 10-minute means per route for status OK rows."""
    tt = travel_times[(travel_times["status"] == "OK") & travel_times["duration_in_traffic_sec"].notna()].copy()
    tt["bin_ts"] = _utc(tt["observed_at"]).dt.floor("10min")
    tt["dur_min"] = pd.to_numeric(tt["duration_in_traffic_sec"], errors="coerce") / 60.0
    if "duration_sec" in tt.columns:
        tt["typical_min"] = pd.to_numeric(tt["duration_sec"], errors="coerce") / 60.0
    else:
        tt["typical_min"] = np.nan
    binned = (
        tt.groupby(["route_id", "direction", "bin_ts"])
        .agg(
            dur_min=("dur_min", "mean"),
            typical_min=("typical_min", "mean"),
            congestion_ratio=("congestion_ratio", "mean"),
            speed_kmh=("speed_kmh", "mean"),
            n_obs=("dur_min", "size"),
        )
        .reset_index()
        .sort_values(["route_id", "bin_ts"])
    )
    binned["gap_min"] = binned.groupby("route_id")["bin_ts"].diff().dt.total_seconds() / 60.0
    return binned.reset_index(drop=True)


def _shift_time(g: pd.DataFrame, col: str, steps: int) -> pd.Series:
    """Value of ``col`` at bin_ts + steps*10 min within the same route (NaN if that bin is missing)."""
    lookup = g.set_index("bin_ts")[col]
    return pd.Series(lookup.reindex(g["bin_ts"] + steps * BIN).to_numpy(), index=g.index)


def maps_features(bins: pd.DataFrame) -> pd.DataFrame:
    """v_training_set columns (time-based lags; identical to the view when no bin is missing)."""
    parts = []
    for _, g in bins.groupby("route_id", sort=False):
        g = g.sort_values("bin_ts").copy()
        lag = {k: _shift_time(g, "dur_min", -k) for k in (1, 2, 3, 4, 5, 6)}
        f = pd.DataFrame(index=g.index)
        f["route_id"] = g["route_id"]
        f["direction"] = g["direction"]
        f["bin_ts"] = g["bin_ts"]
        f["y_persistence"] = g["dur_min"]
        f["maps_typical_min"] = g["typical_min"]  # Maps duration without traffic, same bin (baseline only)
        f["congestion_ratio"] = g["congestion_ratio"]
        f["speed_kmh"] = g["speed_kmh"]
        f["gap_min"] = g["gap_min"]
        f["lag_10"], f["lag_20"], f["lag_30"], f["lag_60"] = lag[1], lag[2], lag[3], lag[6]
        f["roll_mean_30"] = pd.concat([lag[1], lag[2], lag[3]], axis=1).mean(axis=1, skipna=True)
        f["roll_mean_60"] = pd.concat([lag[k] for k in range(1, 7)], axis=1).mean(axis=1, skipna=True)
        f["slope_30"] = lag[1] - lag[4]
        big_gap = (g["gap_min"] > 25).astype(int)
        f["after_gap"] = big_gap.rolling(7, min_periods=1).max().astype(int).to_numpy()
        f["y_30"] = _shift_time(g, "dur_min", 3)
        f["y_60"] = _shift_time(g, "dur_min", 6)
        parts.append(f)
    out = pd.concat(parts).sort_values(["direction", "bin_ts"]).reset_index(drop=True)
    for col in ("lag_10", "lag_20", "lag_30", "lag_60", "roll_mean_30", "roll_mean_60", "slope_30"):
        out.loc[out["after_gap"] == 1, col] = np.nan
    sgt = out["bin_ts"].dt.tz_convert(SGT)
    tod_min = sgt.dt.hour * 60 + sgt.dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * tod_min / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * tod_min / 1440)
    out["tod_block"] = tod_min / 5.0
    out["dow"] = (sgt.dt.dayofweek + 1) % 7 + 1  # BigQuery DAYOFWEEK: Sunday=1 .. Saturday=7
    out["is_weekend"] = out["dow"].isin([1, 7]).astype(int)
    out["is_morning_peak"] = sgt.dt.hour.between(6, 10).astype(int)
    out["is_evening_peak"] = sgt.dt.hour.between(16, 21).astype(int)
    out["date_sgt"] = sgt.dt.date.astype(str)
    out["asof"] = out["bin_ts"] + BIN
    return out


# --- weather ------------------------------------------------------------------------


def load_rain(path: Path = RAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = _utc(df["timestamp"])
    df["value_mm"] = pd.to_numeric(df["value_mm"], errors="coerce")
    return df[["ts", "value_mm"]].dropna().sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


FORECAST_FALLBACK_DELAY = pd.Timedelta(minutes=10)


def load_forecast(path: Path = FORECAST_CSV) -> pd.DataFrame:
    """Forecast rows with ``available_at``: data.gov.sg acquisition time (``update_timestamp``).

    Rows without an acquisition time are treated as available ``FORECAST_FALLBACK_DELAY`` after
    issue (conservative). Rows are sorted by ``available_at``.
    """
    df = pd.read_csv(path)
    for col in ("issue_timestamp", "valid_start", "valid_end"):
        df[col] = _utc(df[col])
    if "update_timestamp" in df.columns:
        upd = pd.to_datetime(df["update_timestamp"], utc=True, errors="coerce", format="mixed")
    else:
        upd = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    fallback = df["issue_timestamp"] + FORECAST_FALLBACK_DELAY
    df["available_at"] = upd.where(upd.notna(), fallback)
    df["available_at"] = df[["available_at", "issue_timestamp"]].max(axis=1)
    df["forecast"] = df["forecast"].fillna("")
    return df.sort_values(["available_at", "issue_timestamp"]).reset_index(drop=True)


def rain_sums(asof: pd.Series, rain: pd.DataFrame, window_min: int) -> np.ndarray:
    """Sum of readings with asof - window < ts <= asof (NaN when no reading falls in the window)."""
    ts = rain["ts"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(rain["value_mm"].to_numpy())])
    hi = np.searchsorted(ts, asof.to_numpy(), side="right")
    lo = np.searchsorted(ts, (asof - pd.Timedelta(minutes=window_min)).to_numpy(), side="right")
    sums = cs[hi] - cs[lo]
    return np.where(hi > lo, sums, np.nan)


def forecast_flags(asof: pd.Series, target: pd.Series, fc: pd.DataFrame) -> pd.DataFrame:
    """Most recently acquired forecast (available_at <= asof) whose valid period covers target."""
    avail_col = "available_at" if "available_at" in fc.columns else "issue_timestamp"
    fc = fc.sort_values(avail_col).reset_index(drop=True)
    avail = fc[avail_col].to_numpy()
    vs, ve = fc["valid_start"].to_numpy(), fc["valid_end"].to_numpy()
    text = fc["forecast"].to_numpy()
    rain, heavy = np.full(len(asof), np.nan), np.full(len(asof), np.nan)
    last = np.searchsorted(avail, asof.to_numpy(), side="right") - 1
    tgt = target.to_numpy()
    for i, j in enumerate(last):
        k = j
        while k >= 0 and k > j - 24:  # look back up to 24 rows (~6 h incl. revisions)
            if vs[k] <= tgt[i] < ve[k]:
                rain[i] = float(bool(RAIN_RE.search(text[k])))
                heavy[i] = float(bool(HEAVY_RE.search(text[k])))
                break
            k -= 1
    return pd.DataFrame({"fc_rain": rain, "fc_heavy": heavy})


def add_weather(frame: pd.DataFrame, rain: Optional[pd.DataFrame], fc: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = frame.copy()
    if rain is not None and len(rain):
        out["rain_30"] = rain_sums(out["asof"], rain, 30)
        out["rain_60"] = rain_sums(out["asof"], rain, 60)
    else:
        out["rain_30"] = out["rain_60"] = np.nan
    if fc is not None and len(fc):
        flags = forecast_flags(out["asof"], out["bin_ts"] + 3 * BIN, fc)
        out["fc_rain"], out["fc_heavy"] = flags["fc_rain"].to_numpy(), flags["fc_heavy"].to_numpy()
    else:
        out["fc_rain"] = out["fc_heavy"] = np.nan
    return out


# --- camera -------------------------------------------------------------------------


def load_camera(path: Path = CAMERA_CSV) -> pd.DataFrame:
    """Scored frames only (status ok); frame_ts is naive SGT in the backfill CSV."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["frame_ts", "sg_my", "my_sg", "sg_my_extent", "my_sg_extent"])
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["frame_ts"] = pd.to_datetime(df["frame_ts"]).dt.tz_localize(SGT).dt.tz_convert("UTC")
    return df.sort_values("frame_ts").drop_duplicates("frame_ts", keep="last").reset_index(drop=True)


def add_camera(frame: pd.DataFrame, cam: pd.DataFrame, max_age_min: int = 30) -> pd.DataFrame:
    out = frame.copy()
    for col in CAMERA_FEATURES:
        out[col] = np.nan
    if cam is None or cam.empty:
        return out
    ts = cam["frame_ts"].to_numpy()
    idx = np.searchsorted(ts, out["asof"].to_numpy(), side="right") - 1
    ok = idx >= 0
    age = np.full(len(out), np.nan)
    age[ok] = (out["asof"].to_numpy()[ok] - ts[idx[ok]]) / np.timedelta64(1, "m")
    fresh = ok & (age <= max_age_min)
    for direction, prefix in CAMERA_DIRECTION.items():
        rows = fresh & (out["direction"].to_numpy() == direction)
        src = idx[rows]
        out.loc[rows, "cam_count"] = pd.to_numeric(cam[prefix], errors="coerce").to_numpy()[src]
        out.loc[rows, "cam_extent"] = pd.to_numeric(cam[f"{prefix}_extent"], errors="coerce").to_numpy()[src]
        out.loc[rows, "cam_age_min"] = age[rows]
    return out


# --- build --------------------------------------------------------------------------


def build(
    travel_times: pd.DataFrame,
    rain: Optional[pd.DataFrame] = None,
    forecast: Optional[pd.DataFrame] = None,
    camera: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frame = maps_features(maps_bins(travel_times))
    frame = add_weather(frame, rain, forecast)
    return add_camera(frame, camera)


def coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of labelled rows (y_30 present) with each feature present, per direction."""
    labelled = frame[frame["y_30"].notna()]
    rows = []
    for direction, g in labelled.groupby("direction"):
        for col in MAPS_FEATURES + WEATHER_FEATURES + CAMERA_FEATURES:
            rows.append({"direction": direction, "feature": col, "coverage": float(g[col].notna().mean()), "n": len(g)})
    return pd.DataFrame(rows)


def parity(frame: pd.DataFrame, view: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, float]:
    """Max absolute difference per column against rows of v_training_set (joined on direction, bin_ts)."""
    cols = cols or ["y_persistence", "lag_10", "lag_20", "lag_30", "lag_60", "roll_mean_30", "roll_mean_60", "slope_30", "y_30", "tod_sin", "dow", "is_weekend", "is_morning_peak", "is_evening_peak", "after_gap"]
    v = view.copy()
    v["bin_ts"] = _utc(v["bin_ts"])
    merged = frame.merge(v, on=["direction", "bin_ts"], suffixes=("", "_view"))
    out = {"rows_compared": float(len(merged))}
    for c in cols:
        a = pd.to_numeric(merged[c], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_view"], errors="coerce")
        both_nan = a.isna() & b.isna()
        diff = (a - b).abs().where(~both_nan, 0.0)
        out[c] = float(diff.fillna(np.inf).max()) if len(diff) else 0.0
    return out


def parity_with_live_view(frame: pd.DataFrame, project: str = "swiftborder", client=None) -> Dict[str, float]:
    """Read-only pull of ``v_training_set`` up to the frame's last bin, then ``parity``."""
    from google.cloud import bigquery

    client = client or bigquery.Client(project=project)
    # The export's last bins can be incomplete (and their y_30 / y_60 not yet observed), while the
    # live view keeps growing. Compare only bins whose 60-min label was complete in the export.
    cutoff = frame["bin_ts"].max() - 7 * BIN
    subset = frame[frame["bin_ts"] <= cutoff]
    view = client.query(
        f"SELECT * FROM `{project}.traffic_prediction.v_training_set` WHERE bin_ts <= TIMESTAMP('{cutoff.isoformat()}')"
    ).to_dataframe()
    out = parity(subset, view)
    out["compared_through_bin_ts"] = cutoff.isoformat()
    return out


def _load_travel_times(refresh: bool) -> pd.DataFrame:
    from timeseries_xgb import sync_canonical_travel_times

    return sync_canonical_travel_times(TT_CACHE, refresh=refresh)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--parity", action="store_true", help="compare Maps columns with the live view (read-only)")
    parser.add_argument("--refresh-bq", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not (args.build or args.parity):
        parser.error("choose --build and/or --parity")

    tt = _load_travel_times(args.refresh_bq)
    rain = load_rain() if RAIN_CSV.exists() else None
    fc = load_forecast() if FORECAST_CSV.exists() else None
    cam = load_camera()
    frame = build(tt, rain, fc, cam)

    if args.build:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.out, index=False)
        cov = coverage_report(frame)
        print(f"Wrote {len(frame)} rows ({frame['bin_ts'].min()} .. {frame['bin_ts'].max()}) to {args.out}")
        print(cov.pivot(index="feature", columns="direction", values="coverage").round(3).to_string())

    if args.parity:
        for k, v in parity_with_live_view(frame).items():
            print(f"{k:>16}: {v:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
