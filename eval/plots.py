"""Comparison figures for Layer B evaluation (matplotlib, headless-safe)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from significance import ComparisonResult


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
) -> Path:
    """Grouped bars: candidates x direction (both / SG_TO_MY / MY_TO_SG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directions = ["SG_TO_MY", "MY_TO_SG", "both"]
    candidates = list(slices_by_candidate.keys())
    x = np.arange(len(directions))
    width = 0.8 / max(len(candidates), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cand in enumerate(candidates):
        maes = [float(_slice_mae(slices_by_candidate[cand], d) or np.nan) for d in directions]
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
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mae_diff_forest(
    comparisons: Sequence[ComparisonResult],
    path: Path,
    *,
    title: str = "Mean paired AE difference vs persistence (min)",
) -> Path:
    """Forest-style plot: challenger mean AE diff with block-bootstrap CI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{c.label_challenger}\n({c.n})" for c in comparisons]
    means = [c.mean_ae_diff_min for c in comparisons]
    err_lo = [c.mean_ae_diff_min - c.bootstrap_ci_low_min for c in comparisons]
    err_hi = [c.bootstrap_ci_high_min - c.mean_ae_diff_min for c in comparisons]

    y = np.arange(len(comparisons))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(comparisons) + 1)))
    ax.errorbar(
        means,
        y,
        xerr=[err_lo, err_hi],
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#555555",
    )
    ax.axvline(0.0, color="black", lw=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |err_ch| - |err_ref| (negative => challenger better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_backtest_method_means(
    scores_df,
    path: Path,
    *,
    title: str = "60 min backtest (22-24 Sep): mean MAE by method",
) -> Path:
    """Bar chart of mean MAE_min grouped by method (pandas DataFrame from score_forecast_days)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_holdout_forecast_sample(
    observed_at: Sequence,
    actual_min: Sequence[float],
    series: Mapping[str, Sequence[float]],
    path: Path,
    *,
    title: str = "Hold-out sample: actual vs forecasts",
    max_points: int = 288,
) -> Path:
    """Time-series overlay for a tail slice of the test window."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(actual_min), max_points)
    if n <= 0:
        raise ValueError("no points to plot")

    idx = slice(-n, None)
    t = observed_at[idx]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, actual_min[idx], color="black", lw=1.6, label="Actual")
    palette = ["#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]
    for i, (name, vals) in enumerate(series.items()):
        ax.plot(t, np.asarray(vals)[idx], lw=1.2, alpha=0.85, color=palette[i % len(palette)], label=name)
    ax.set_ylabel("Duration (min)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
