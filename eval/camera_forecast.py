"""Layer A forecasts as Layer B inputs: a camera-2701 queue profile learned from historical detections.

Why: the only Layer A output in BigQuery (``cam2701.v_congestion_index_10min``, YOLO detections
counted per direction and 10-minute bin) covers 13 Mar - 22 Apr 2026, and the Maps label starts on
5 Sep. There is no row where both are observed. Instead of the observed count, Layer B gets a
**forecast** of the count: a model trained on the camera history that predicts the expected queue
for any direction, time of day and day type. Because it depends only on the calendar, it is known in
advance for every Maps row (causal by construction), and it was fitted on data months before the
test window (no leakage).

Model: per direction, ridge regression of the 10-minute vehicle count on Fourier terms of time of day
(``K`` harmonics), each also interacted with a weekend flag. ``K`` is chosen by leave-days-out
cross-validation against a global mean and an hour x day-type mean. Days with fewer than
``MIN_BINS_PER_DAY`` bins in a direction are dropped (partial capture days).

Features for a Layer B row with forecast origin bin ``t`` (label bin ``[t+30, t+40)``):

- ``camfc_now``: expected count at the origin bin (evaluated at ``t + 5 min``);
- ``camfc_30``: expected count at the label bin (``t + 35 min``);
- ``camfc_delta``: ``camfc_30 - camfc_now``, the expected queue build-up or clearing, the part the
  Maps lags cannot see.

Control (``mp_*``): the same estimator fitted **inside each fold on the Maps training rows**
(value = current travel time). If the camera prior helps only as much as a Maps-derived prior, the
gain is generic time-of-day shape, not camera information.

Limits: the view drops frames with zero vehicles in a direction (counts are biased up at night);
March and April 2026 differ from September (school terms, public holidays, festive traffic); this is a
prior, not a live sensor, so it cannot see today's incidents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from plots import write_series_csv

EVAL_ROOT = Path(__file__).resolve().parent
HISTORY_CSV = EVAL_ROOT / "data" / "cam2701_congestion_10min.csv"
HISTORY_VIEW = "cam2701.v_congestion_index_10min"
HISTORY_LOCATION = "asia-southeast1"
SGT = "Asia/Singapore"
DIRECTIONS = ("SG_TO_MY", "MY_TO_SG")
MIN_BINS_PER_DAY = 40
HARMONICS = (2, 4, 6, 8)
CV_FOLDS = 5
NOW_OFFSET = pd.Timedelta(minutes=5)
LABEL_OFFSET = pd.Timedelta(minutes=35)
CAMFC_FEATURES = ["camfc_now", "camfc_30", "camfc_delta"]
MP_FEATURES = ["mp_now", "mp_30", "mp_delta"]


# --- data -----------------------------------------------------------------------------


def fetch_history(path: Path = HISTORY_CSV, *, project: str = "swiftborder", refresh: bool = False, client=None) -> pd.DataFrame:
    """Cached read-only pull of the camera-2701 congestion view (direction, bin_ts, vis_count, ...)."""
    path = Path(path)
    if path.exists() and not refresh:
        df = pd.read_csv(path)
    else:
        from google.cloud import bigquery

        client = client or bigquery.Client(project=project)
        df = client.query(
            f"SELECT * FROM `{project}.{HISTORY_VIEW}` ORDER BY direction, bin_ts", location=HISTORY_LOCATION
        ).to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df["bin_ts"] = pd.to_datetime(df["bin_ts"], utc=True, format="mixed")
    df["vis_count"] = pd.to_numeric(df["vis_count"], errors="coerce")
    return df.dropna(subset=["vis_count"]).sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def clean_history(df: pd.DataFrame, min_bins_per_day: int = MIN_BINS_PER_DAY) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop (direction, day) groups with too few bins; return the kept rows and a summary."""
    out = df.copy()
    out["date_sgt"] = out["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    counts = out.groupby(["direction", "date_sgt"]).size()
    keep = counts[counts >= min_bins_per_day].index
    kept = out.set_index(["direction", "date_sgt"]).loc[lambda x: x.index.isin(keep)].reset_index()
    info = {
        "rows_in": int(len(df)),
        "rows_kept": int(len(kept)),
        "days_in": int(out["date_sgt"].nunique()),
        "days_kept_per_direction": {d: int(kept.loc[kept["direction"] == d, "date_sgt"].nunique()) for d in DIRECTIONS},
        "min_bins_per_day": min_bins_per_day,
        "range_sgt": [str(kept["bin_ts"].min().tz_convert(SGT)), str(kept["bin_ts"].max().tz_convert(SGT))] if len(kept) else None,
    }
    return kept.sort_values(["direction", "bin_ts"]).reset_index(drop=True), info


# --- profile models -----------------------------------------------------------------


def _calendar(ts: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sgt = pd.to_datetime(ts, utc=True).dt.tz_convert(SGT)
    tod = (sgt.dt.hour * 60 + sgt.dt.minute + sgt.dt.second / 60.0).to_numpy(dtype=float)
    weekend = (sgt.dt.dayofweek >= 5).to_numpy(dtype=float)
    return tod, weekend, sgt.dt.hour.to_numpy()


def fourier_design(tod_min: np.ndarray, weekend: np.ndarray, harmonics: int) -> np.ndarray:
    """[weekend, sin/cos(2 pi k tod / 1440) for k <= K, and each of those x weekend]."""
    cols = [weekend]
    for k in range(1, harmonics + 1):
        a = 2 * np.pi * k * tod_min / 1440.0
        s, c = np.sin(a), np.cos(a)
        cols += [s, c, s * weekend, c * weekend]
    return np.column_stack(cols)


@dataclass
class FourierProfile:
    """Per-direction ridge on Fourier x weekend terms. ``value`` column is the fitted quantity."""

    harmonics: int = 4
    alpha: float = 1.0
    models_: Dict[str, Any] = field(default_factory=dict)
    fallback_: Dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, value: str = "vis_count", time_offset: pd.Timedelta = NOW_OFFSET) -> "FourierProfile":
        from sklearn.linear_model import Ridge

        for d, g in df.groupby("direction"):
            tod, we, _ = _calendar(g["bin_ts"] + time_offset)
            y = g[value].to_numpy(dtype=float)
            ok = ~np.isnan(y)
            self.fallback_[d] = float(np.nanmean(y)) if ok.any() else float("nan")
            if ok.sum() >= 10:
                self.models_[d] = Ridge(alpha=self.alpha).fit(fourier_design(tod[ok], we[ok], self.harmonics), y[ok])
        return self

    def predict(self, direction: Sequence[str], ts: pd.Series) -> np.ndarray:
        direction = np.asarray(direction)
        tod, we, _ = _calendar(pd.Series(ts).reset_index(drop=True))
        out = np.full(len(direction), np.nan)
        for d in np.unique(direction):
            m = direction == d
            if d in self.models_:
                out[m] = self.models_[d].predict(fourier_design(tod[m], we[m], self.harmonics))
            else:
                out[m] = self.fallback_.get(d, np.nan)
        return out


@dataclass
class HourProfile:
    """Mean per direction x day type x hour (step function), with the direction mean as fallback."""

    table_: Dict[Tuple[str, float, int], float] = field(default_factory=dict)
    fallback_: Dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, value: str = "vis_count", time_offset: pd.Timedelta = NOW_OFFSET) -> "HourProfile":
        _, we, hour = _calendar(df["bin_ts"] + time_offset)
        g = pd.DataFrame({"d": df["direction"].to_numpy(), "we": we, "h": hour, "v": df[value].to_numpy(dtype=float)})
        self.table_ = g.groupby(["d", "we", "h"])["v"].mean().to_dict()
        self.fallback_ = g.groupby("d")["v"].mean().to_dict()
        return self

    def predict(self, direction: Sequence[str], ts: pd.Series) -> np.ndarray:
        _, we, hour = _calendar(pd.Series(ts).reset_index(drop=True))
        return np.array([self.table_.get((d, w, h), self.fallback_.get(d, np.nan)) for d, w, h in zip(direction, we, hour)])


@dataclass
class MeanProfile:
    fallback_: Dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, value: str = "vis_count", time_offset: pd.Timedelta = NOW_OFFSET) -> "MeanProfile":
        self.fallback_ = df.groupby("direction")[value].mean().to_dict()
        return self

    def predict(self, direction: Sequence[str], ts: pd.Series) -> np.ndarray:
        return np.array([self.fallback_.get(d, np.nan) for d in direction])


def candidate_models() -> Dict[str, Any]:
    out: Dict[str, Any] = {"direction mean": MeanProfile, "hour x day-type mean": HourProfile}
    for k in HARMONICS:
        out[f"fourier K={k}"] = (lambda k=k: FourierProfile(harmonics=k))
    return out


def cross_validate(df: pd.DataFrame, n_folds: int = CV_FOLDS, seed: int = 0) -> List[Dict[str, Any]]:
    """Leave-days-out CV (days shuffled into ``n_folds`` groups): MAE of the count per candidate and direction."""
    days = np.array(sorted(df["date_sgt"].unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(days), np.arange(len(days)) % n_folds))
    fold = df["date_sgt"].map(fold_of).to_numpy()
    rows = []
    for name, make in candidate_models().items():
        pred = np.full(len(df), np.nan)
        for f in range(n_folds):
            test = fold == f
            model = make().fit(df[~test])
            pred[test] = model.predict(
                df.loc[test, "direction"].to_numpy(),
                df.loc[test, "bin_ts"] + NOW_OFFSET,
            )
        err = np.abs(df["vis_count"].to_numpy() - pred)
        row = {"candidate": name, "mae_count": float(np.mean(err))}
        for d in DIRECTIONS:
            m = df["direction"].to_numpy() == d
            row[f"mae_count_{d}"] = float(np.mean(err[m])) if m.any() else float("nan")
        rows.append(row)
    return rows


def fit_camera_profile(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """Clean, cross-validate, pick the best Fourier ``K`` (or fall back), fit on all kept days."""
    kept, info = clean_history(df)
    cv = cross_validate(kept)
    fourier = [r for r in cv if r["candidate"].startswith("fourier")]
    best = min(fourier, key=lambda r: r["mae_count"])
    k = int(best["candidate"].split("=")[1])
    profile = FourierProfile(harmonics=k).fit(kept)
    info.update({"cv": cv, "cv_folds": CV_FOLDS, "chosen": best["candidate"], "harmonics": k})
    return profile, info


# --- features -----------------------------------------------------------------------


def add_profile_features(frame: pd.DataFrame, profile, prefix: str) -> pd.DataFrame:
    """``<prefix>_now`` / ``_30`` / ``_delta`` for rows with (direction, bin_ts = forecast origin)."""
    out = frame.copy()
    d = out["direction"].to_numpy()
    now = profile.predict(d, out["bin_ts"] + NOW_OFFSET)
    later = profile.predict(d, out["bin_ts"] + LABEL_OFFSET)
    out[f"{prefix}_now"] = now
    out[f"{prefix}_30"] = later
    out[f"{prefix}_delta"] = later - now
    return out


def fold_maps_profile(train: pd.DataFrame, harmonics: int) -> FourierProfile:
    """Control: the same estimator fitted on a fold's Maps rows (current travel time)."""
    return FourierProfile(harmonics=harmonics).fit(train[["direction", "bin_ts", "y_persistence"]].dropna(), value="y_persistence")


def profile_curves(profile, step_min: int = 10) -> Dict[str, Any]:
    """Profile values over one day (weekday Wednesday and weekend Saturday), for plots and run.json."""
    grid = np.arange(0, 1440, step_min)
    out: Dict[str, Any] = {"tod_min": grid.tolist()}
    for label, day in (("weekday", "2026-09-16"), ("weekend", "2026-09-19")):
        ts = pd.Series(pd.Timestamp(day, tz=SGT) + pd.to_timedelta(grid, unit="m")).dt.tz_convert("UTC") - NOW_OFFSET
        for d in DIRECTIONS:
            out[f"{d}/{label}"] = np.round(profile.predict([d] * len(ts), ts), 3).tolist()
    return out


def plot_profiles(curves: Dict[str, Any], maps_curves: Optional[Dict[str, Any]], path: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = np.array(curves["tod_min"]) / 60.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, d in zip(axes, DIRECTIONS):
        for label, style in (("weekday", "-"), ("weekend", "--")):
            ax.plot(hours, curves[f"{d}/{label}"], style, color="#4C78A8", label=f"camera forecast, {label}")
        ax.set_title(f"{d}: Layer A queue forecast (Mar-Apr detections)", fontsize=9)
        ax.set_xlabel("hour (SGT)")
        ax.set_ylabel("expected vehicles in frame")
        if maps_curves:
            ax2 = ax.twinx()
            for label, style in (("weekday", "-"), ("weekend", "--")):
                ax2.plot(hours, maps_curves[f"{d}/{label}"], style, color="#E45756", alpha=0.7, label=f"Maps profile, {label}")
            ax2.set_ylabel("Maps travel time (min)", color="#E45756")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DIRECTIONS:
        for label in ("weekday", "weekend"):
            camera = curves[f"{d}/{label}"]
            maps = None if not maps_curves else maps_curves[f"{d}/{label}"]
            for i, tod in enumerate(curves["tod_min"]):
                rows.append(
                    {
                        "tod_min": tod,
                        "hour_sgt": float(tod) / 60.0,
                        "direction": d,
                        "day_type": label,
                        "camera_forecast_vehicles": camera[i],
                        "maps_travel_time_min": "" if maps is None else maps[i],
                    }
                )
    csv_path = write_series_csv(
        path,
        rows,
        ("tod_min", "hour_sgt", "direction", "day_type", "camera_forecast_vehicles", "maps_travel_time_min"),
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path, csv_path]
