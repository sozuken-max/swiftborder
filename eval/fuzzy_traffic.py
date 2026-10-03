#!/usr/bin/env python3
"""Fuzzy traffic-level forecasting: light / moderate / heavy, 60 minutes ahead, both directions.

**Levels** are defined on the Google Maps ``duration_in_traffic`` travel time (minutes) at the
target time, fixed before looking at test data and shared by both directions:

    light < 20 min <= moderate < 35 min <= heavy

The route's free-flow time is about 10-13 min, so the cut points are roughly 1.5x and 2.5x free
flow. The crisp level of the raw observation at the target time is the ground truth.

**Fuzzy partition.** Trapezoidal memberships that sum to 1 (a Ruspini partition), crossing at
0.5 exactly at the crisp cut points, so defuzzifying a crisp number by maximum membership gives
the crisp level, while values near a boundary get graded membership (for example 0.6 moderate /
0.4 heavy).

**Classifiers** (all causal: inputs are known at the forecast origin):

- ``Persistence level``: the level of the travel time at the origin (baseline).
- ``Majority level``: the most frequent training level (baseline).
- ``Fuzzy rule base``: a learned fuzzy rule-based classifier (Ishibuchi-style). Inputs and terms:
  travel time now (light/moderate/heavy), 30-min trend (falling/steady/rising), target time of day
  (night/morning/midday/evening), day type (workday/non-workday, crisp) and the travel time at the
  target time one day earlier (light/moderate/heavy). Every antecedent combination is a candidate
  rule; its consequent is the class with the largest summed compatibility on the training rows and
  its weight is the certainty factor ``CF = (b_win - sum b_other) / sum b``. Rules with CF <= 0 or
  too little support are dropped. Inference is single-winner (largest compatibility x CF); if no
  rule fires, the persistence level is used. One rule base per direction.
- ``XGB forecast -> fuzzy level`` (hybrid): the offline XGBoost regressor's 60-min travel-time
  forecast, fuzzified with the same partition.

**Metrics:** accuracy, macro-F1, per-level recall, severe-error rate (light predicted as heavy or
the reverse), and the ranked probability score (RPS, ordinal, lower is better) of the normalised
class degrees. **Significance:** Diebold-Mariano on the 0/1 misclassification loss with day-block
bootstrap CIs and Holm (``significance.compare_absolute_errors``); ``mean_ae_diff_min`` is then a
difference in error rate (negative = challenger misclassifies less).

    python fuzzy_traffic.py                       # print tables (CSV cache, no network)
    python generate_comparison_plots.py --fuzzy   # write into a run folder / run.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from plots import write_series_csv

CACHE = Path(__file__).resolve().parent / "data" / "causeway_gdata.csv"
LEVELS: Tuple[str, ...] = ("light", "moderate", "heavy")
ROUTES: Tuple[str, ...] = ("jb_to_woodlands", "mandai_to_shell_jb")
ROUTE_DIRECTION = {"jb_to_woodlands": "MY_TO_SG", "mandai_to_shell_jb": "SG_TO_MY"}
XGB_SEED = 42
INF = float("inf")


# --- membership functions ---------------------------------------------------------


def trapezoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoid rising on [a, b], 1 on [b, c], falling on [c, d]. Infinite shoulders allowed."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    if np.isfinite(a):
        out = np.where(x <= a, 0.0, out)
        out = np.where((x > a) & (x < b), (x - a) / (b - a), out)
    if np.isfinite(d):
        out = np.where(x >= d, 0.0, out)
        out = np.where((x > c) & (x < d), (d - x) / (d - c), out)
    return np.where(np.isnan(x), np.nan, out)


@dataclass(frozen=True)
class LevelPartition:
    """Light / moderate / heavy on travel time (minutes). Crossings at ``light_max`` and ``heavy_min``."""

    light_max: float = 20.0
    heavy_min: float = 35.0
    half_width: float = 5.0

    def __post_init__(self) -> None:
        if not (0 < self.half_width and self.light_max + self.half_width <= self.heavy_min - self.half_width):
            raise ValueError("ramps overlap: need light_max + half_width <= heavy_min - half_width")

    def memberships(self, minutes: np.ndarray) -> np.ndarray:
        """(n, 3) degrees for light, moderate, heavy; rows sum to 1."""
        lo, hi, w = self.light_max, self.heavy_min, self.half_width
        m = np.asarray(minutes, dtype=float)
        return np.stack(
            [
                trapezoid(m, -INF, -INF, lo - w, lo + w),
                trapezoid(m, lo - w, lo + w, hi - w, hi + w),
                trapezoid(m, hi - w, hi + w, INF, INF),
            ],
            axis=1,
        )

    def crisp(self, minutes: np.ndarray) -> np.ndarray:
        """0 light, 1 moderate, 2 heavy (NaN input -> -1)."""
        m = np.asarray(minutes, dtype=float)
        out = np.digitize(m, [self.light_max, self.heavy_min]).astype(int)
        return np.where(np.isnan(m), -1, out)


def trend_memberships(delta_min: np.ndarray) -> np.ndarray:
    """(n, 3) falling / steady / rising for the 30-min change in minutes (Ruspini partition)."""
    d = np.asarray(delta_min, dtype=float)
    return np.stack(
        [trapezoid(d, -INF, -INF, -5.0, -1.0), trapezoid(d, -5.0, -1.0, 1.0, 5.0), trapezoid(d, 1.0, 5.0, INF, INF)],
        axis=1,
    )


def time_of_day_memberships(hour: np.ndarray) -> np.ndarray:
    """(n, 4) night / morning / midday / evening over decimal hour (cyclic, rows sum to 1)."""
    h = np.asarray(hour, dtype=float) % 24.0
    morning = trapezoid(h, 4.0, 6.0, 9.0, 11.0)
    midday = trapezoid(h, 9.0, 11.0, 14.0, 16.0)
    evening = trapezoid(h, 14.0, 16.0, 20.0, 23.0)
    night = np.clip(1.0 - morning - midday - evening, 0.0, 1.0)
    return np.stack([night, morning, midday, evening], axis=1)


def day_type_memberships(nonworkday: np.ndarray) -> np.ndarray:
    """(n, 2) workday / non-workday (crisp)."""
    nw = np.asarray(nonworkday, dtype=float)
    return np.stack([1.0 - nw, nw], axis=1)


INPUT_TERMS: Dict[str, Tuple[str, ...]] = {
    "now": LEVELS,
    "trend": ("falling", "steady", "rising"),
    "time": ("night", "morning", "midday", "evening"),
    "day": ("workday", "non-workday"),
    "yesterday": LEVELS,
}


# --- inputs -----------------------------------------------------------------------


@dataclass
class FuzzyInputs:
    now_min: np.ndarray
    trend_min: np.ndarray
    target_hour: np.ndarray
    nonworkday: np.ndarray
    yesterday_min: np.ndarray


def inputs_from_supervised(frame: pd.DataFrame) -> FuzzyInputs:
    """Causal inputs from a ``timeseries_xgb.build_supervised_frame`` table (needs lags 1 and 7)."""
    now = frame["persistence"].to_numpy(dtype=float) / 60.0
    lag1 = frame["target_lag_1"].to_numpy(dtype=float) / 60.0
    lag7 = frame["target_lag_7"].to_numpy(dtype=float) / 60.0
    d1 = frame["d1"].to_numpy(dtype=float) / 60.0
    return FuzzyInputs(
        now_min=now,
        trend_min=lag1 - lag7,
        target_hour=frame["hour"].to_numpy(dtype=float) + frame["minute"].to_numpy(dtype=float) / 60.0,
        nonworkday=frame["nonworkday"].to_numpy(dtype=float),
        yesterday_min=np.where(np.isnan(d1), now, d1),  # first day: no D-1 value, fall back to now
    )


def input_memberships(x: FuzzyInputs, partition: LevelPartition) -> List[np.ndarray]:
    return [
        partition.memberships(x.now_min),
        trend_memberships(x.trend_min),
        time_of_day_memberships(x.target_hour),
        day_type_memberships(x.nonworkday),
        partition.memberships(x.yesterday_min),
    ]


def firing_strengths(memberships: Sequence[np.ndarray]) -> np.ndarray:
    """(n, n_rules) product t-norm over every antecedent combination (C order over INPUT_TERMS)."""
    out = memberships[0]
    for m in memberships[1:]:
        out = (out[:, :, None] * m[:, None, :]).reshape(len(out), -1)
    return out


# --- rule base ---------------------------------------------------------------------


@dataclass
class FuzzyRuleClassifier:
    partition: LevelPartition = field(default_factory=LevelPartition)
    min_support: float = 2.0
    """Minimum summed compatibility (training rows) for a rule to be kept."""
    consequent_: Optional[np.ndarray] = None
    weight_: Optional[np.ndarray] = None
    support_: Optional[np.ndarray] = None

    def fit(self, x: FuzzyInputs, y_level: np.ndarray) -> "FuzzyRuleClassifier":
        y_level = np.asarray(y_level, dtype=int)
        mu = firing_strengths(input_memberships(x, self.partition))
        beta = np.stack([mu[y_level == c].sum(axis=0) for c in range(len(LEVELS))], axis=1)  # (rules, classes)
        total = beta.sum(axis=1)
        win = beta.argmax(axis=1)
        best = beta[np.arange(len(beta)), win]
        with np.errstate(invalid="ignore", divide="ignore"):
            cf = np.where(total > 0, (best - (total - best)) / total, 0.0)
        keep = (cf > 0) & (total >= self.min_support)
        self.consequent_ = np.where(keep, win, -1)
        self.weight_ = np.where(keep, cf, 0.0)
        self.support_ = total
        return self

    @property
    def n_rules(self) -> int:
        return int((self.consequent_ >= 0).sum()) if self.consequent_ is not None else 0

    def class_scores(self, x: FuzzyInputs) -> np.ndarray:
        """(n, 3) best compatibility x CF per class (0 when no rule of that class fires)."""
        if self.consequent_ is None:
            raise RuntimeError("fit first")
        act = firing_strengths(input_memberships(x, self.partition)) * self.weight_
        scores = np.zeros((act.shape[0], len(LEVELS)))
        for c in range(len(LEVELS)):
            cols = self.consequent_ == c
            if cols.any():
                scores[:, c] = act[:, cols].max(axis=1)
        return scores

    def predict(self, x: FuzzyInputs) -> np.ndarray:
        scores = self.class_scores(x)
        pred = scores.argmax(axis=1)
        fallback = self.partition.crisp(x.now_min)
        return np.where(scores.max(axis=1) > 0, pred, fallback)

    def predict_degrees(self, x: FuzzyInputs) -> np.ndarray:
        """Class degrees normalised to sum 1 (one-hot persistence level when no rule fires)."""
        scores = self.class_scores(x)
        s = scores.sum(axis=1, keepdims=True)
        fallback = np.eye(len(LEVELS))[np.clip(self.partition.crisp(x.now_min), 0, None)]
        return np.where(s > 0, scores / np.where(s > 0, s, 1.0), fallback)

    def describe_rules(self, top: int = 10) -> List[Dict[str, Any]]:
        """Readable rules sorted by support."""
        names = list(INPUT_TERMS)
        combos = list(product(*[range(len(t)) for t in INPUT_TERMS.values()]))
        order = np.argsort(-self.support_)
        out = []
        for q in order:
            if self.consequent_[q] < 0:
                continue
            terms = [INPUT_TERMS[n][i] for n, i in zip(names, combos[q])]
            text = " AND ".join(f"{n} is {t}" for n, t in zip(names, terms))
            out.append(
                {
                    "rule": f"IF {text} THEN {LEVELS[self.consequent_[q]]}",
                    "cf": round(float(self.weight_[q]), 3),
                    "support": round(float(self.support_[q]), 1),
                }
            )
            if len(out) >= top:
                break
        return out


# --- metrics -----------------------------------------------------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = len(LEVELS)) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return m


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, degrees: Optional[np.ndarray] = None) -> Dict[str, Any]:
    cm = confusion(y_true, y_pred)
    n = int(cm.sum())
    recall = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan") for i in range(len(LEVELS))]
    precision = [float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else 0.0 for i in range(len(LEVELS))]
    f1 = [
        (2 * p * r / (p + r)) if (p + r) > 0 and not np.isnan(r) else 0.0 for p, r in zip(precision, recall)
    ]
    out: Dict[str, Any] = {
        "n": n,
        "accuracy": float(np.trace(cm) / n) if n else float("nan"),
        "macro_f1": float(np.mean(f1)),
        "severe_error_rate": float((cm[0, 2] + cm[2, 0]) / n) if n else float("nan"),
        **{f"recall_{lvl}": r for lvl, r in zip(LEVELS, recall)},
        "confusion": cm.tolist(),
    }
    if degrees is not None:
        out["rps"] = ranked_probability_score(y_true, degrees)
    return out


def ranked_probability_score(y_true: np.ndarray, degrees: np.ndarray) -> float:
    """Mean RPS for ordered classes (0 = perfect); degrees rows are normalised to sum 1."""
    d = np.asarray(degrees, dtype=float)
    d = d / d.sum(axis=1, keepdims=True)
    obs = np.eye(d.shape[1])[np.asarray(y_true, dtype=int)]
    cum = np.cumsum(d, axis=1)[:, :-1] - np.cumsum(obs, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum**2, axis=1) / (d.shape[1] - 1)))


# --- component ---------------------------------------------------------------------


def _route_frames(export: pd.DataFrame, route: str):
    import timeseries_xgb as tsx

    config = tsx.TimeSeriesConfig(route_id=route)
    features = tsx.engineer_features(tsx.prepare_route_frame(export, config), config)
    return config, tsx.split_supervised(features, config)


def score_routes(
    export: pd.DataFrame,
    *,
    routes: Sequence[str] = ROUTES,
    partition: LevelPartition = LevelPartition(),
    min_support: float = 2.0,
) -> Dict[str, Any]:
    """Fit per route and return aligned test predictions (rows ordered by route, then target time)."""
    import timeseries_xgb as tsx

    rows: Dict[str, List[np.ndarray]] = {k: [] for k in ("y", "persist", "majority", "rules", "xgb", "route")}
    degrees: Dict[str, List[np.ndarray]] = {"persist": [], "rules": [], "xgb": []}
    timestamps: List[Any] = []
    info: Dict[str, Any] = {}
    config = None
    for route in routes:
        config, split = _route_frames(export, route)
        y_train = partition.crisp(split.train["y"].to_numpy() / 60.0)
        y_test = partition.crisp(split.test["y"].to_numpy() / 60.0)
        x_train, x_test = inputs_from_supervised(split.train), inputs_from_supervised(split.test)
        clf = FuzzyRuleClassifier(partition=partition, min_support=min_support).fit(x_train, y_train)
        xgb = tsx.train_xgb(split.train[split.feature_cols], split.train["y"].to_numpy(), config=config, random_state=XGB_SEED)
        xgb_min = xgb.predict(split.test[split.feature_cols]) / 60.0
        majority = int(np.bincount(y_train, minlength=len(LEVELS)).argmax())

        rows["y"].append(y_test)
        rows["persist"].append(partition.crisp(x_test.now_min))
        rows["majority"].append(np.full(len(y_test), majority))
        rows["rules"].append(clf.predict(x_test))
        rows["xgb"].append(partition.crisp(xgb_min))
        rows["route"].append(np.array([route] * len(y_test)))
        degrees["persist"].append(partition.memberships(x_test.now_min))
        degrees["rules"].append(clf.predict_degrees(x_test))
        degrees["xgb"].append(partition.memberships(xgb_min))
        timestamps.extend(list(split.test["target_ts"]))
        info[route] = {
            "direction": ROUTE_DIRECTION.get(route, route),
            "supervised_rows": {"train": int(len(split.train)), "test": int(len(split.test))},
            "boundary": str(split.boundary),
            "test_end": str(split.test["target_ts"].max()),
            "train_level_share": (np.bincount(y_train, minlength=3) / len(y_train)).round(3).tolist(),
            "test_level_share": (np.bincount(y_test, minlength=3) / len(y_test)).round(3).tolist(),
            "majority_level": LEVELS[majority],
            "rules_kept": clf.n_rules,
            "rules_possible": int(np.prod([len(t) for t in INPUT_TERMS.values()])),
            "top_rules": clf.describe_rules(10),
        }
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["degrees"] = {k: np.concatenate(v) for k, v in degrees.items()}
    out["timestamps"] = timestamps
    out["routes_info"] = info
    out["config"] = config
    return out


CANDIDATES = {
    "persist": "Persistence level",
    "majority": "Majority level (train)",
    "rules": "Fuzzy rule base",
    "xgb": "XGB forecast -> fuzzy level",
}


def _metric_row(candidate: str, slice_: str, m: Dict[str, Any]) -> Dict[str, Any]:
    row = {"candidate": candidate, "slice": slice_}
    row.update({k: v for k, v in m.items() if k != "confusion"})
    return row


def plot_memberships(partition: LevelPartition, path: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(5, 60, 551)
    mu = partition.memberships(x)
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (lvl, colour) in enumerate(zip(LEVELS, ("#54A24B", "#F58518", "#E45756"))):
        ax.plot(x, mu[:, i], label=lvl, color=colour)
    for cut in (partition.light_max, partition.heavy_min):
        ax.axvline(cut, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Maps travel time (min)")
    ax.set_ylabel("membership")
    ax.set_title("Traffic-level fuzzy partition (crossings at the crisp cut points)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "travel_time_min": float(minutes),
            "membership_light": float(mu[i, 0]),
            "membership_moderate": float(mu[i, 1]),
            "membership_heavy": float(mu[i, 2]),
            "light_max_min": partition.light_max,
            "heavy_min_min": partition.heavy_min,
        }
        for i, minutes in enumerate(x)
    ]
    csv_path = write_series_csv(
        path,
        rows,
        ("travel_time_min", "membership_light", "membership_moderate", "membership_heavy", "light_max_min", "heavy_min_min"),
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path, csv_path]


def plot_confusions(confusions: Dict[str, np.ndarray], path: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(confusions), figsize=(3.4 * len(confusions), 3.2))
    for ax, (label, cm) in zip(np.atleast_1d(axes), confusions.items()):
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color="black" if share[i, j] < 0.6 else "white")
        ax.set_xticks(range(3), LEVELS, fontsize=7)
        ax.set_yticks(range(3), LEVELS, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
        ax.set_title(label, fontsize=8)
    fig.suptitle("60-min traffic level, both directions, hold-out (counts; shade = row share)", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, cm in confusions.items():
        for i, actual in enumerate(LEVELS):
            for j, predicted in enumerate(LEVELS):
                rows.append({"model": label, "actual": actual, "predicted": predicted, "count": int(cm[i, j])})
    csv_path = write_series_csv(path, rows, ("model", "actual", "predicted", "count"))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path, csv_path]


def fuzzy_component(
    run_dir: Path,
    *,
    cache: Path = CACHE,
    project: str = "swiftborder",
    alpha: float = 0.05,
    partition: LevelPartition = LevelPartition(),
    min_support: float = 2.0,
) -> Tuple[List[Path], Dict[str, Any]]:
    import timeseries_xgb as tsx
    from run_artifacts import artifact_relpath, file_fingerprint, subdir
    from significance import apply_holm, compare_absolute_errors, format_comparison_table

    out_dir = subdir(run_dir, "fuzzy")
    export = tsx.sync_canonical_travel_times(cache, project=project, refresh=False)
    s = score_routes(export, partition=partition, min_support=min_support)
    config = s["config"]
    y = s["y"]
    metrics: List[Dict[str, Any]] = []
    confusions: Dict[str, np.ndarray] = {}
    for key, label in CANDIDATES.items():
        deg = s["degrees"].get(key)
        m = classification_metrics(y, s[key], deg)
        confusions[label] = np.array(m["confusion"])
        metrics.append(_metric_row(label, "holdout", m))
        for route in ROUTES:
            k = s["route"] == route
            mr = classification_metrics(y[k], s[key][k], None if deg is None else deg[k])
            metrics.append(_metric_row(label, f"holdout {ROUTE_DIRECTION[route]}", mr))

    err = {k: (s[k] != y).astype(float) for k in CANDIDATES}
    zeros = np.zeros(len(y))

    def cmp(ch: str, ref: str):
        return compare_absolute_errors(
            zeros, err[ch], err[ref], label_challenger=CANDIDATES[ch], label_reference=CANDIDATES[ref],
            horizon_steps=config.horizon_steps, timestamps=s["timestamps"], groups=list(s["route"]), alpha=alpha,
        )

    comparisons = apply_holm([cmp("rules", "persist"), cmp("xgb", "persist"), cmp("rules", "xgb")])
    print("Fuzzy significance on 0/1 loss (Holm over this family; diff = error-rate difference):")
    print(format_comparison_table(comparisons))

    written = [
        *plot_memberships(partition, out_dir / "fuzzy-memberships.png"),
        *plot_confusions({CANDIDATES[k]: confusions[CANDIDATES[k]] for k in ("persist", "rules", "xgb")}, out_dir / "fuzzy-confusion.png"),
    ]
    info = s["routes_info"]
    meta: Dict[str, Any] = {
        "dataset": {
            "source": f"{project}.causeway.travel_times",
            "cache": file_fingerprint(cache),
            "routes": info,
        },
        "window": {
            "timezone": "Asia/Singapore",
            "basis": "offline split per route (target time; train labels < start <= test labels)",
            "start": min(v["boundary"] for v in info.values()),
            "end": max(v["test_end"] for v in info.values()),
        },
        "models": list(CANDIDATES.values()),
        "horizon_minutes": config.horizon_minutes,
        "config": {
            "levels_min": {"light": f"< {partition.light_max}", "moderate": f"{partition.light_max}-{partition.heavy_min}", "heavy": f">= {partition.heavy_min}"},
            "partition": asdict(partition),
            "inputs": {k: list(v) for k, v in INPUT_TERMS.items()},
            "min_support": min_support,
            "t_norm": "product",
            "inference": "single winner (compatibility x CF); persistence level when no rule fires",
            "xgb_seed": XGB_SEED,
        },
        "loss_for_significance": "0/1 misclassification; mean_ae_diff_min is an error-rate difference",
        "confusion_labels": list(LEVELS),
        "confusion": {label: cm.tolist() for label, cm in confusions.items()},
        "metrics": metrics,
        "significance": [c.to_dict() for c in comparisons],
        "artifacts": [artifact_relpath(run_dir, p) for p in written],
    }
    return written, meta


def main(argv=None) -> int:
    import tempfile

    import timeseries_xgb as tsx

    export = tsx.sync_canonical_travel_times(CACHE, refresh=False)
    s = score_routes(export)
    for key, label in CANDIDATES.items():
        m = classification_metrics(s["y"], s[key], s["degrees"].get(key))
        rps = f", RPS {m['rps']:.3f}" if "rps" in m else ""
        print(f"{label}: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}, severe {m['severe_error_rate']:.3f}{rps}")
    for route, v in s["routes_info"].items():
        print(route, "rules kept", v["rules_kept"], "of", v["rules_possible"])
        for r in v["top_rules"][:5]:
            print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
