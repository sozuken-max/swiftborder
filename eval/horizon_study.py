"""Exploratory horizon study: how 30-minute-style forecasts degrade out to 24 hours.

Not a report run. It reads the same cache as the harness (``data/causeway_gdata.csv``), uses the
same rolling-origin daily folds and time-based Maps features as ``joined.py``, and the pooled
significance rule of ``significance.py`` (shared calendar days, joint day bootstrap). Results and
charts go to ``runs/horizon-study/`` (gitignored); findings are recorded in ``docs/horizon-study.md``.

For a horizon ``h`` (minutes, a multiple of 10) the label of origin bin ``t`` is the mean Maps
duration of the bin ``[t+h, t+h+10)``. A training row counts once that label bin has closed before
the test day starts (``bin_ts + h + 10 <= day_start``), the generalisation of ``joined.LABEL_LAG``.

Candidates (one direct model per horizon; nothing is chained):
- ``persistence``: the origin bin.
- ``same time yesterday`` / ``same time last week``: the bin at ``target - 1 d`` / ``target - 7 d``
  (known at the origin for every ``h <= 24 h``). At ``h = 24 h`` "yesterday" equals persistence.
- ``profile``: a per-direction Fourier (K = 8) x weekend ridge on every closed bin before the test day
  (``camera_forecast.FourierProfile`` on Maps rows), read at the target time. **Baseline only**: it
  ignores current traffic.
- ``xgb[maps]``: the harness XGBoost on the Maps features, refitted per fold and horizon.
- ``xgb[maps+prof]``: the same plus the profile at the origin and at the target.

Charts (always written): ``mae-by-horizon.png``, ``skill-vs-profile.png``, ``example-days.png``, each
with a CSV of the plotted series. ``--publish`` also copies the charts to ``docs/images/horizon-study/``
and writes the compact summary ``forecastapi/horizon_study.json`` that ``forecast-api`` serves.

    python horizon_study.py                         # 30 min .. 24 h, charts in runs/horizon-study/
    python horizon_study.py --publish               # + docs/images and forecastapi/horizon_study.json
    python horizon_study.py --horizons 60 120 360
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import camera_forecast as cf
import features as fx
import joined as jx
from significance import compare_absolute_errors

SGT = "Asia/Singapore"
DEFAULT_HORIZONS = tuple(range(30, 1441, 30))  # every 30 minutes, 30 min .. 24 h (the served curve's grid)
TICK_HORIZONS = (30, 60, 90, 120, 180, 240, 360, 480, 720, 1080, 1440)
TEST_START, TEST_END = "2026-09-13", "2026-09-30"
PROFILE_HARMONICS = 8
EVAL_ROOT = Path(__file__).resolve().parent
REPO = EVAL_ROOT.parent
CACHE = EVAL_ROOT / "data" / "causeway_gdata.csv"
OUT = EVAL_ROOT / "runs" / "horizon-study"
PUBLISH_IMAGES = REPO / "docs" / "images" / "horizon-study"
PUBLISH_JSON = REPO / "forecastapi" / "horizon_study.json"
EXAMPLE_DAYS = ("2026-09-24", "2026-09-25", "2026-09-26")  # Thu, Fri, Sat (SGT)
EXAMPLE_HORIZONS = (60, 240, 1440)
CANDIDATES = {
    "persistence": "y_persistence",
    "same time yesterday": "naive_d1",
    "same time last week": "naive_d7",
    "profile": "prof_h",
    "xgb[maps]": "xgb",
    "xgb[maps+prof]": "xgb_prof",
}
STYLE = {
    "persistence": dict(color="#7f7f7f", ls="-", marker="o"),
    "same time yesterday": dict(color="#bcbd22", ls=":", marker="v"),
    "same time last week": dict(color="#17becf", ls=":", marker="^"),
    "profile": dict(color="#ff7f0e", ls="--", marker="s"),
    "xgb[maps]": dict(color="#1f77b4", ls="-", marker="o"),
    "xgb[maps+prof]": dict(color="#2ca02c", ls="-", marker="D"),
}
COMPARISONS = (
    ("xgb", "y_persistence"),
    ("xgb", "prof_h"),
    ("prof_h", "y_persistence"),
    ("xgb", "naive_d7"),
    ("xgb_prof", "xgb"),
    ("xgb_prof", "prof_h"),
)
LABEL = {v: k for k, v in CANDIDATES.items()}


def _lookup(feats: pd.DataFrame, minutes: Sequence[int]) -> Dict[int, pd.Series]:
    """Value of y_persistence at bin_ts + m minutes, same route (NaN when that bin is missing)."""
    out = {}
    for m in minutes:
        parts = [fx._shift_time(g, "y_persistence", m // 10) for _, g in feats.groupby("route_id", sort=False)]
        out[m] = pd.concat(parts).reindex(feats.index)
    return out


def load() -> pd.DataFrame:
    tt = pd.read_csv(CACHE)
    feats = fx.maps_features(fx.maps_bins(tt))
    feats["is_my_to_sg"] = (feats["direction"] == "MY_TO_SG").astype(int)
    feats["date_sgt"] = feats["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    return feats


def fit_profile(feats: pd.DataFrame, day_start: pd.Timestamp) -> cf.FourierProfile:
    """The baseline profile of a fold: every closed bin before the test day."""
    closed = feats[feats["bin_ts"] + fx.BIN <= day_start][["direction", "bin_ts", "y_persistence"]].dropna()
    return cf.FourierProfile(harmonics=PROFILE_HARMONICS).fit(closed, value="y_persistence")


def add_profile(frame: pd.DataFrame, prof: cf.FourierProfile, h: int) -> pd.DataFrame:
    f = frame.copy()
    d = f["direction"].to_numpy()
    f["prof_now"] = prof.predict(d, f["bin_ts"] + cf.NOW_OFFSET)
    f["prof_h"] = prof.predict(d, f["bin_ts"] + pd.Timedelta(minutes=h) + cf.NOW_OFFSET)
    return f


def run_horizon(feats: pd.DataFrame, h: int, n_bootstrap: int = 999) -> Tuple[Dict, pd.DataFrame]:
    if h % 10:
        raise ValueError("horizon must be a multiple of 10 minutes")
    shifts = _lookup(feats, [h, h - 1440, h - 10080])
    frame = feats.assign(y_h=shifts[h], naive_d1=shifts[h - 1440], naive_d7=shifts[h - 10080])
    data = frame[frame["y_h"].notna() & (frame["after_gap"] == 0)]
    cols = jx.FEATURE_SETS["maps"]
    parts = []
    for day in pd.date_range(TEST_START, TEST_END).strftime("%Y-%m-%d"):
        day_start = pd.Timestamp(day).tz_localize(SGT).tz_convert("UTC")
        train = data[data["bin_ts"] + pd.Timedelta(minutes=h + 10) <= day_start]
        test = data[data["date_sgt"] == day]
        if len(train) < 200 or test.empty:
            continue
        assert (train["bin_ts"] + pd.Timedelta(minutes=h + 10)).max() <= day_start  # no label from the test day
        prof = fit_profile(feats, day_start)
        train, test = add_profile(train, prof, h), add_profile(test, prof, h)
        y = train["y_h"].to_numpy()
        out = test[["direction", "bin_ts", "date_sgt", "y_h", "y_persistence", "naive_d1", "naive_d7", "prof_h"]].copy()
        out["xgb"] = jx._xgb().fit(train[cols], y).predict(test[cols])
        pc = cols + ["prof_now", "prof_h"]
        out["xgb_prof"] = jx._xgb().fit(train[pc], y).predict(test[pc])
        out["train_rows"] = len(train)
        parts.append(out)
    oof = pd.concat(parts).sort_values(["direction", "bin_ts"]).reset_index(drop=True)

    row: Dict = {"horizon_min": h, "n": int(len(oof)), "test_days": int(oof["date_sgt"].nunique()),
                 "first_day_train_rows": int(oof["train_rows"].iloc[0]) if len(oof) else 0}
    for name, c in CANDIDATES.items():
        ok = oof[c].notna()
        row[f"mae/{name}"] = float((oof.loc[ok, "y_h"] - oof.loc[ok, c]).abs().mean())
        for d in ("SG_TO_MY", "MY_TO_SG"):
            s = oof[(oof["direction"] == d) & ok]
            row[f"mae/{name}/{d}"] = float((s["y_h"] - s[c]).abs().mean())
    for ch, ref in COMPARISONS:
        s = oof[oof[ch].notna() & oof[ref].notna()]
        r = compare_absolute_errors(
            s["y_h"], s[ch], s[ref], label_challenger=LABEL[ch], label_reference=LABEL[ref],
            timestamps=list(s["bin_ts"]), groups=list(s["direction"]), horizon_steps=h // 10,
            n_bootstrap=n_bootstrap, day_utc_offset_hours=8.0,
        )
        row[f"cmp/{LABEL[ch]} vs {LABEL[ref]}"] = {
            "n": r.n, "diff": r.mean_ae_diff_min, "joint_ci": [r.joint_ci_low_min, r.joint_ci_high_min],
            "calendar_days": r.calendar_days, "day_cluster_p": r.day_cluster_pvalue, "decision": r.decision,
        }
    return row, oof


# --- charts ---------------------------------------------------------------------------


def _hours_axis(ax, horizons: Sequence[int]) -> None:
    ax.set_xscale("log")
    hrs = [h / 60 for h in horizons if h in TICK_HORIZONS] or [h / 60 for h in horizons]
    ax.set_xticks(hrs)
    ax.set_xticklabels([f"{x:g}" for x in hrs], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Horizon (hours, log scale)")
    ax.grid(alpha=0.3)


def plot_mae_by_horizon(rows: List[Dict], path: Path) -> List[Path]:
    import plots

    horizons = [r["horizon_min"] for r in rows]
    fig, axes = plt_subplots(1, 3, figsize=(15, 4.6), sharey=True)
    series = []
    for ax, scope in zip(axes, ("both", "SG_TO_MY", "MY_TO_SG")):
        for name in CANDIDATES:
            key = f"mae/{name}" if scope == "both" else f"mae/{name}/{scope}"
            vals = [r[key] for r in rows]
            label = "profile (baseline, not a forecast)" if name == "profile" else name
            ax.plot([h / 60 for h in horizons], vals, label=label, lw=1.6, ms=4, **STYLE[name])
            series += [{"scope": scope, "candidate": name, "horizon_min": h, "mae_min": v} for h, v in zip(horizons, vals)]
        _hours_axis(ax, horizons)
        ax.set_title({"both": "Both directions", "SG_TO_MY": "SG -> JB (SG_TO_MY)", "MY_TO_SG": "JB -> SG (MY_TO_SG)"}[scope], fontsize=10)
    axes[0].set_ylabel("MAE of the Maps duration (min)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Exploratory horizon study, 13-30 Sep 2026 daily folds: MAE by horizon (lower is better)", fontsize=11)
    fig.tight_layout()
    return plots._finish_figure(fig, path, series, ("scope", "candidate", "horizon_min", "mae_min"))


def plot_skill_vs_profile(rows: List[Dict], path: Path) -> List[Path]:
    """MAE minus the profile baseline's MAE, with the joint calendar-day 95% CI."""
    import plots

    horizons = [r["horizon_min"] for r in rows]
    specs = (
        ("xgb[maps]", "cmp/xgb[maps] vs profile", 1.0),
        ("xgb[maps+prof]", "cmp/xgb[maps+prof] vs profile", 1.0),
        ("persistence", "cmp/profile vs persistence", -1.0),  # persistence - profile = -(profile - persistence)
    )
    fig, ax = plt_subplots(figsize=(10, 5))
    series = []
    offsets = {"xgb[maps]": 0.97, "xgb[maps+prof]": 1.03, "persistence": 1.0}
    for name, key, sign in specs:
        xs, ys, lo, hi, sig = [], [], [], [], []
        for h, r in zip(horizons, rows):
            c = r[key]
            d = sign * c["diff"]
            a, b = sorted(sign * v for v in c["joint_ci"])
            significant = c["decision"] in ("challenger", "reference")
            xs.append(h / 60 * offsets[name]); ys.append(d); lo.append(d - a); hi.append(b - d); sig.append(significant)
            series.append({"candidate": name, "horizon_min": h, "mae_minus_profile_min": d, "ci_low": a, "ci_high": b,
                           "significant": significant})
        st = STYLE[name]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=1.2, alpha=0.7)
        for x, y, l, u, s in zip(xs, ys, lo, hi, sig):
            ax.errorbar([x], [y], yerr=[[l], [u]], fmt=st["marker"], color=st["color"], ecolor=st["color"],
                        capsize=3, ms=6, mfc=st["color"] if s else "white")
        ax.plot([], [], color=st["color"], marker=st["marker"], ls=st["ls"], label=name)
    ax.axhline(0, color="#ff7f0e", lw=1.5, ls="--", label="profile baseline (0)")
    ax.axhspan(-100, 0, color="#2ca02c", alpha=0.05)
    ax.set_ylim(min(-3.5, min(s["ci_low"] for s in series if s["candidate"] != "persistence") - 0.3), 3.0)
    _hours_axis(ax, horizons)
    ax.set_ylabel("MAE minus profile MAE (min); below 0 = better than the baseline")
    ax.set_title("Skill over the calendar-profile baseline (95% joint day-bootstrap CI; filled = significant)\n"
                 "Persistence beyond the top edge is clipped", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return plots._finish_figure(fig, path, series, ("candidate", "horizon_min", "mae_minus_profile_min", "ci_low", "ci_high", "significant"))


def plot_example_days(oofs: Dict[int, pd.DataFrame], path: Path) -> List[Path]:
    """Actual vs forecasts by target time, for a few horizons and three test days."""
    import plots

    horizons = [h for h in EXAMPLE_HORIZONS if h in oofs]
    fig, axes = plt_subplots(len(horizons), 2, figsize=(15, 3.4 * len(horizons)), squeeze=False)
    series = []
    for i, h in enumerate(horizons):
        oof = oofs[h]
        model = "xgb_prof" if h >= 90 else "xgb"
        s_all = oof[oof["date_sgt"].isin(EXAMPLE_DAYS)].copy()
        s_all["target"] = (s_all["bin_ts"] + pd.Timedelta(minutes=h)).dt.tz_convert(SGT).dt.tz_localize(None)
        for j, d in enumerate(("SG_TO_MY", "MY_TO_SG")):
            ax = axes[i][j]
            s = s_all[s_all["direction"] == d]
            ax.plot(s["target"], s["y_h"], color="black", lw=1.4, label="actual (Maps)")
            ax.plot(s["target"], s[model], color=STYLE[LABEL[model]]["color"], lw=1.2, label=LABEL[model])
            ax.plot(s["target"], s["prof_h"], color=STYLE["profile"]["color"], lw=1.2, ls="--", label="profile (baseline)")
            ax.plot(s["target"], s["y_persistence"], color=STYLE["persistence"]["color"], lw=0.9, alpha=0.7, label="persistence")
            ax.set_title(f"{h / 60:g} h ahead, {d}", fontsize=9)
            ax.grid(alpha=0.25)
            for _, r in s.iterrows():
                series.append({"horizon_min": h, "direction": d, "target_sgt": r["target"], "actual_min": r["y_h"],
                               "model": LABEL[model], "model_min": r[model], "profile_min": r["prof_h"],
                               "persistence_min": r["y_persistence"]})
        axes[i][0].set_ylabel("min")
        axes[i][0].legend(fontsize=7, loc="upper left")
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelsize=7, labelrotation=20)
    fig.suptitle(f"Example test days {EXAMPLE_DAYS[0]} to {EXAMPLE_DAYS[-1]} (origins on these days; x = target time, SGT); exploratory", fontsize=11)
    fig.tight_layout()
    return plots._finish_figure(fig, path, series, ("horizon_min", "direction", "target_sgt", "actual_min", "model",
                                                    "model_min", "profile_min", "persistence_min"))


def plt_subplots(*args, **kwargs):
    import plots  # noqa: F401  (sets the Agg backend)
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


# --- outputs --------------------------------------------------------------------------


def markdown(rows: List[Dict]) -> str:
    names = list(CANDIDATES)
    lines = ["| h | days | rows | first-fold train rows | " + " | ".join(names) + " |", "| --- | --- | --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for r in rows:
        lines.append(f"| {r['horizon_min']} | {r['test_days']} | {r['n']} | {r['first_day_train_rows']} | " + " | ".join(f"{r[f'mae/{n}']:.2f}" for n in names) + " |")
    lines += ["", "| h | comparison | diff | joint CI | decision |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        for k, v in r.items():
            if k.startswith("cmp/"):
                lo, hi = v["joint_ci"]
                lines.append(f"| {r['horizon_min']} | {k[4:]} | {v['diff']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {v['decision']} |")
    return "\n".join(lines)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _study_code_dirty() -> bool:
    """True when this file differs from HEAD (the recorded commit is then not the exact code)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).resolve())], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return True


def summary(rows: List[Dict]) -> Dict:
    """Compact, served copy of the results (``forecast-api`` ``?list=horizon-study``)."""
    return {
        "status": "exploratory",
        "note": "Exploratory study on the 13-30 Sep 2026 daily folds; not a report run and not confirmed on Run B. "
                "The label is Google Maps duration_in_traffic, not a measured crossing time. "
                "The profile is a calendar baseline for comparison, not a forecast.",
        "window_sgt": [TEST_START, TEST_END],
        "cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        "code_commit": _git_sha(),
        "study_code_dirty": _study_code_dirty(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidates": list(CANDIDATES),
        "horizons": [
            {
                "horizon_min": r["horizon_min"],
                "rows": r["n"],
                "test_days": r["test_days"],
                "mae_min": {n: round(r[f"mae/{n}"], 3) for n in CANDIDATES},
                "mae_min_by_direction": {d: {n: round(r[f"mae/{n}/{d}"], 3) for n in CANDIDATES} for d in ("SG_TO_MY", "MY_TO_SG")},
                "vs_profile": {
                    k[4:].split(" vs ")[0]: {"diff_min": round(v["diff"], 3), "ci_min": [round(x, 3) for x in v["joint_ci"]], "decision": v["decision"]}
                    for k, v in r.items() if k.startswith("cmp/") and k.endswith(" vs profile")
                },
            }
            for r in rows
        ],
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--n-bootstrap", type=int, default=999)
    p.add_argument("--publish", action="store_true", help="copy charts to docs/images/horizon-study/ and write forecastapi/horizon_study.json")
    args = p.parse_args(argv)
    feats = load()
    rows, oofs = [], {}
    for h in args.horizons:
        row, oof = run_horizon(feats, h, args.n_bootstrap)
        rows.append(row)
        oofs[h] = oof
        print(f"h={h}: " + ", ".join(f"{n} {row[f'mae/{n}']:.2f}" for n in CANDIDATES), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (OUT / "results.md").write_text(markdown(rows) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary(rows), indent=1) + "\n", encoding="utf-8")
    charts = []
    charts += plot_mae_by_horizon(rows, OUT / "mae-by-horizon.png")
    charts += plot_skill_vs_profile(rows, OUT / "skill-vs-profile.png")
    charts += plot_example_days(oofs, OUT / "example-days.png")
    print(markdown(rows))
    print("charts:", ", ".join(str(c.relative_to(EVAL_ROOT)) for c in charts if c.suffix == ".png"))
    if args.publish:
        if tuple(args.horizons) != DEFAULT_HORIZONS:
            raise SystemExit("--publish needs the default horizons")
        PUBLISH_IMAGES.mkdir(parents=True, exist_ok=True)
        for c in charts:  # each chart and the CSV of its plotted series
            shutil.copy2(c, PUBLISH_IMAGES / c.name)
        shutil.copy2(OUT / "results.json", PUBLISH_IMAGES / "results.json")  # every comparison, per horizon
        shutil.copy2(OUT / "summary.json", PUBLISH_JSON)
        print("published:", PUBLISH_IMAGES.relative_to(REPO), PUBLISH_JSON.relative_to(REPO))


if __name__ == "__main__":
    main()
