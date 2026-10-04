"""Exploratory horizon study: how 30-minute-style forecasts degrade out to 24 hours.

Not a report run. It reads the same cache as the harness (``data/causeway_gdata.csv``), uses the
same rolling-origin daily folds and time-based Maps features as ``joined.py``, and the pooled
significance rule of ``significance.py`` (shared calendar days, joint day bootstrap). Results go to
``runs/horizon-study/`` (gitignored); findings are recorded in ``docs/horizon-study.md``.

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

    python horizon_study.py                         # 30 min .. 24 h
    python horizon_study.py --horizons 60 120 360
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import camera_forecast as cf
import features as fx
import joined as jx
from significance import compare_absolute_errors

SGT = "Asia/Singapore"
DEFAULT_HORIZONS = (30, 60, 90, 120, 180, 240, 360, 480, 720, 1080, 1440)
TEST_START, TEST_END = "2026-09-13", "2026-09-30"
OUT = Path(__file__).resolve().parent / "runs" / "horizon-study"
CANDIDATES = {
    "persistence": "y_persistence",
    "same time yesterday": "naive_d1",
    "same time last week": "naive_d7",
    "profile": "prof_h",
    "xgb[maps]": "xgb",
    "xgb[maps+prof]": "xgb_prof",
}
COMPARISONS = (
    ("xgb", "y_persistence"),
    ("xgb", "prof_h"),
    ("prof_h", "y_persistence"),
    ("xgb", "naive_d7"),
    ("xgb_prof", "xgb"),
    ("xgb_prof", "prof_h"),
)


def _lookup(feats: pd.DataFrame, minutes: Sequence[int]) -> Dict[int, pd.Series]:
    """Value of y_persistence at bin_ts + m minutes, same route (NaN when that bin is missing)."""
    out = {}
    for m in minutes:
        parts = [fx._shift_time(g, "y_persistence", m // 10) for _, g in feats.groupby("route_id", sort=False)]
        out[m] = pd.concat(parts).reindex(feats.index)
    return out


def load() -> pd.DataFrame:
    tt = pd.read_csv(fx.EVAL_ROOT / "data" / "causeway_gdata.csv")
    feats = fx.maps_features(fx.maps_bins(tt))
    feats["is_my_to_sg"] = (feats["direction"] == "MY_TO_SG").astype(int)
    feats["date_sgt"] = feats["bin_ts"].dt.tz_convert(SGT).dt.strftime("%Y-%m-%d")
    return feats


def run_horizon(feats: pd.DataFrame, h: int, n_bootstrap: int = 999) -> Dict:
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
        closed = feats[feats["bin_ts"] + fx.BIN <= day_start][["direction", "bin_ts", "y_persistence"]].dropna()
        prof = cf.FourierProfile(harmonics=8).fit(closed, value="y_persistence")

        def with_profile(f: pd.DataFrame) -> pd.DataFrame:
            f = f.copy()
            d = f["direction"].to_numpy()
            f["prof_now"] = prof.predict(d, f["bin_ts"] + cf.NOW_OFFSET)
            f["prof_h"] = prof.predict(d, f["bin_ts"] + pd.Timedelta(minutes=h) + cf.NOW_OFFSET)
            return f

        train, test = with_profile(train), with_profile(test)
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
    label = {v: k for k, v in CANDIDATES.items()}
    for ch, ref in COMPARISONS:
        s = oof[oof[ch].notna() & oof[ref].notna()]
        r = compare_absolute_errors(
            s["y_h"], s[ch], s[ref], label_challenger=label[ch], label_reference=label[ref],
            timestamps=list(s["bin_ts"]), groups=list(s["direction"]), horizon_steps=h // 10,
            n_bootstrap=n_bootstrap, day_utc_offset_hours=8.0,
        )
        row[f"cmp/{label[ch]} vs {label[ref]}"] = {
            "n": r.n, "diff": r.mean_ae_diff_min, "joint_ci": [r.joint_ci_low_min, r.joint_ci_high_min],
            "calendar_days": r.calendar_days, "day_cluster_p": r.day_cluster_pvalue, "decision": r.decision,
        }
    return row


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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--n-bootstrap", type=int, default=999)
    args = p.parse_args(argv)
    feats = load()
    rows = []
    for h in args.horizons:
        rows.append(run_horizon(feats, h, args.n_bootstrap))
        print(f"h={h}: " + ", ".join(f"{n} {rows[-1][f'mae/{n}']:.2f}" for n in CANDIDATES), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (OUT / "results.md").write_text(markdown(rows) + "\n", encoding="utf-8")
    print(markdown(rows))


if __name__ == "__main__":
    main()
