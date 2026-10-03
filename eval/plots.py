"""Comparison figures for Layer B evaluation (matplotlib, headless-safe)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from significance import ComparisonResult


def series_csv_path(figure: Path) -> Path:
    """CSV of the plotted series, same stem as the figure PNG."""
    return Path(figure).with_suffix(".csv")


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return ""
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_series_csv(figure: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    """Write the columns a figure plots beside ``figure`` (header row, no index dump)."""
    path = series_csv_path(figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_cell(row.get(col)) for col in columns})
    return path


def _finish_figure(fig, path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> List[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = write_series_csv(path, rows, columns)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path, csv_path]


def _slice_mae(slices: Sequence[dict], direction: str, time_of_day: str = "all") -> Optional[float]:
    for s in slices:
        if s["direction"] == direction and s["time_of_day"] == time_of_day:
            return float(s["mae"])
    return None


def plot_bqml_mae_comparison(
    slices_by_candidate: Mapping[str, List[dict]],
    path: Path,
    *,
    title: str = "30 min hold-out MAE (minutes)",
) -> List[Path]:
    """Grouped bars: candidates x direction (both / SG_TO_MY / MY_TO_SG)."""
    path = Path(path)
    directions = ["SG_TO_MY", "MY_TO_SG", "both"]
    candidates = list(slices_by_candidate.keys())
    x = np.arange(len(directions))
    width = 0.8 / max(len(candidates), 1)
    series_rows: List[Dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cand in enumerate(candidates):
        maes = [float(_slice_mae(slices_by_candidate[cand], d) or np.nan) for d in directions]
        for direction, mae in zip(directions, maes):
            series_rows.append({"candidate": cand, "direction": direction, "mae_min": mae})
        offset = (i - (len(candidates) - 1) / 2) * width
        bars = ax.bar(x + offset, maes, width, label=cand)
        for bar, val in zip(bars, maes):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.set_ylabel("MAE (min)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _finish_figure(fig, path, series_rows, ("candidate", "direction", "mae_min"))


def plot_mae_diff_forest(
    comparisons: Sequence[ComparisonResult],
    path: Path,
    *,
    title: str = "Mean paired AE difference vs persistence (min)",
) -> List[Path]:
    """Forest-style plot: challenger mean AE diff with block-bootstrap CI."""
    path = Path(path)
    comparisons = list(comparisons)[::-1]  # first comparison at the top
    labels = [f"{c.label_challenger}\nvs {c.label_reference} (n={c.n})" for c in comparisons]
    means = [c.mean_ae_diff_min for c in comparisons]
    # Clip at 0: a degenerate CI (one block) can sit a rounding error inside the mean.
    err_lo = [max(0.0, c.mean_ae_diff_min - c.bootstrap_ci_low_min) for c in comparisons]
    err_hi = [max(0.0, c.bootstrap_ci_high_min - c.mean_ae_diff_min) for c in comparisons]
    decisions = [getattr(c, "decision", "not significant") for c in comparisons]
    colours = {"challenger": "#2ca02c", "reference": "#d62728", "not significant": "#7f7f7f"}

    y = np.arange(len(comparisons))
    fig, ax = plt.subplots(figsize=(10, max(3, 0.55 * len(comparisons) + 1.5)))
    for yi, m, lo, hi, d in zip(y, means, err_lo, err_hi, decisions):
        ax.errorbar([m], [yi], xerr=[[lo], [hi]], fmt="o", capsize=4, color=colours.get(d, "#1f77b4"), ecolor="#555555")
    ax.axvline(0.0, color="black", lw=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |err_challenger| - |err_reference| (min); negative = challenger better")
    ax.set_title(title, fontsize=10, wrap=True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in colours.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Holm-adjusted decision", title_fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    # display_rank 0 is the top row (the first comparison passed in).
    displayed = list(reversed(comparisons))
    series_rows = [
        {
            "display_rank": rank,
            "challenger": c.label_challenger,
            "reference": c.label_reference,
            "n": c.n,
            "mean_ae_diff_min": c.mean_ae_diff_min,
            "bootstrap_ci_low_min": c.bootstrap_ci_low_min,
            "bootstrap_ci_high_min": c.bootstrap_ci_high_min,
            "decision": getattr(c, "decision", "not significant"),
        }
        for rank, c in enumerate(displayed)
    ]
    return _finish_figure(
        fig,
        path,
        series_rows,
        (
            "display_rank",
            "challenger",
            "reference",
            "n",
            "mean_ae_diff_min",
            "bootstrap_ci_low_min",
            "bootstrap_ci_high_min",
            "decision",
        ),
    )


def plot_backtest_method_means(
    scores_df,
    path: Path,
    *,
    title: str = "60 min backtest (22-24 Sep): mean MAE by method",
) -> List[Path]:
    """Bar chart of mean MAE_min grouped by method (pandas DataFrame from score_forecast_days)."""
    path = Path(path)
    grouped = scores_df.groupby("method")["MAE_min"].mean().sort_values()
    methods = grouped.index.tolist()
    values = grouped.values

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ca02c" if "XGB actual" in m else "#7f7f7f" for m in methods]
    bars = ax.barh(methods, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean MAE (min)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    series_rows = [
        {"order_from_bottom": i, "method": method, "mean_mae_min": float(value)}
        for i, (method, value) in enumerate(zip(methods, values))
    ]
    return _finish_figure(fig, path, series_rows, ("order_from_bottom", "method", "mean_mae_min"))


def plot_holdout_forecast_sample(
    observed_at: Sequence,
    actual_min: Sequence[float],
    series: Mapping[str, Sequence[float]],
    path: Path,
    *,
    title: str = "Hold-out sample: actual vs forecasts",
    max_points: int = 288,
) -> List[Path]:
    """Time-series overlay for a tail slice of the test window."""
    path = Path(path)
    n = min(len(actual_min), max_points)
    if n <= 0:
        raise ValueError("no points to plot")

    idx = slice(-n, None)
    t = list(observed_at)[idx]
    actual = np.asarray(actual_min, dtype=float)[idx]
    plotted = {name: np.asarray(vals, dtype=float)[idx] for name, vals in series.items()}
    columns = ["row", "target_ts", "actual_min", *plotted]
    series_rows = []
    for i in range(n):
        row: Dict[str, Any] = {"row": i, "target_ts": t[i], "actual_min": float(actual[i])}
        for name, vals in plotted.items():
            row[name] = float(vals[i])
        series_rows.append(row)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, actual, color="black", lw=1.6, label="Actual")
    palette = ["#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]
    for i, (name, vals) in enumerate(plotted.items()):
        ax.plot(t, vals, lw=1.2, alpha=0.85, color=palette[i % len(palette)], label=name)
    ax.set_ylabel("Duration (min)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _finish_figure(fig, path, series_rows, columns)
