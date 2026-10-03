from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import ensemble as en


def test_lad_simplex_weights_recovers_a_convex_mix():
    rng = np.random.default_rng(0)
    P = rng.normal(20, 5, size=(300, 3))
    y = P @ np.array([0.7, 0.3, 0.0])
    w = en.lad_simplex_weights(P, y)
    np.testing.assert_allclose(w, [0.7, 0.3, 0.0], atol=1e-6)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)


def test_lad_simplex_weights_respects_sample_weights_and_empty():
    P = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    # Rows 0-1 favour member 0, rows 2-3 member 1; the weights decide.
    assert en.lad_simplex_weights(P, y, np.array([1, 1, 0, 0]))[0] == pytest.approx(1.0)
    assert en.lad_simplex_weights(P, y, np.array([0, 0, 1, 1]))[1] == pytest.approx(1.0)
    np.testing.assert_allclose(en.lad_simplex_weights(np.empty((0, 2)), np.empty(0)), [0.5, 0.5])


def _frame(days=5, per_day=6, directions=("SG_TO_MY", "MY_TO_SG")):
    rows = []
    start = pd.Timestamp("2026-09-12 16:00", tz="UTC")  # 13 Sep 00:00 SGT
    for d in directions:
        for i in range(days * per_day):
            ts = start + pd.Timedelta(hours=24 / per_day * i)
            y = 20 + (i % per_day)
            rows.append({"direction": d, "bin_ts": ts, "y": y, "a": y + 1.0, "b": y + (5.0 if d == "SG_TO_MY" else -5.0)})
    f = pd.DataFrame(rows)
    f["date_sgt"] = f["bin_ts"].dt.tz_convert("Asia/Singapore").dt.strftime("%Y-%m-%d")
    return f.sort_values(["direction", "bin_ts"]).reset_index(drop=True)


def test_rolling_stack_uses_only_earlier_days():
    f = _frame()
    res = en.rolling_stack(f, ["a", "b"])
    days = sorted(f["date_sgt"].unique())
    first = [w for w in res.weights if w["day"] in days[:2]]
    assert all(w["fit_days"] < en.MIN_FIT_DAYS for w in first)
    assert all(w["weights"] == [[0.5, 0.5]] for w in first)
    later = [w for w in res.weights if w["day"] == days[-1]]
    assert all(w["fit_days"] == len(days) - 1 for w in later)
    # changing the last day's labels must not change the last day's forecast (no look-ahead)
    g = f.copy()
    g.loc[g["date_sgt"] == days[-1], "y"] += 100
    np.testing.assert_allclose(en.rolling_stack(g, ["a", "b"]).prediction, res.prediction)


def test_rolling_stack_gate_and_select():
    f = _frame()
    gate = np.column_stack([np.ones(len(f)), np.zeros(len(f))])
    gated = en.rolling_stack(f, ["a", "b"], gate=gate)
    plain = en.rolling_stack(f, ["a", "b"])
    np.testing.assert_allclose(gated.prediction, plain.prediction, atol=1e-6)
    sel = en.rolling_select(f, ["b", "a"])
    days = sorted(f["date_sgt"].unique())
    assert [c["choice"] for c in sel.weights if c["day"] == days[0]] == ["b", "b"]  # no history: first member
    assert all(c["choice"] == "a" for c in sel.weights if c["day"] == days[-1])


def test_pool_30_joins_and_checks_labels():
    t0 = pd.Timestamp("2026-09-12 16:00", tz="UTC")
    rows = [
        {"direction": d, "bin_ts": (t0 + pd.Timedelta(minutes=10 * i)).to_pydatetime(), "y_30": 20.0 + i, "Persistence": 20.0,
         "lin_h30": 21.0, "xgb_h30": 22.0, "ensemble_mean": 21.5, "is_morning_peak": 0, "is_evening_peak": 0, "is_weekend": 0}
        for d in ("SG_TO_MY", "MY_TO_SG") for i in range(4)
    ]
    oof = pd.DataFrame([{"direction": r["direction"], "bin_ts": pd.Timestamp(r["bin_ts"]), "y_30": r["y_30"], "ridge[maps]": 23.0, "xgb[maps]": 24.0} for r in rows])
    m, info = en.pool_30(rows, oof)
    assert info["matched_rows"] == 8
    assert set(m.loc[m["direction"] == "SG_TO_MY", "served"]) == {21.0}  # lin_h30
    assert set(m.loc[m["direction"] == "MY_TO_SG", "served"]) == {20.0}  # persistence
    oof.loc[0, "y_30"] += 1
    with pytest.raises(RuntimeError):
        en.pool_30(rows, oof)


def test_candidates_30_produce_every_column():
    f = _frame(days=4, per_day=12)
    for c in en.POOL_30:
        f[c] = f["y"] + np.random.default_rng(1).normal(0, 1, len(f))
    f["served"] = f["lin_h30"]
    f["ensemble_mean"] = (f["lin_h30"] + f["xgb_h30"]) / 2
    out, logs = en.candidates_30(f)
    for c in ("mean[models]", "stack", "select", "fuzzy stack"):
        assert out[c].notna().all()
    assert logs["pool"] == list(en.POOL_30)


def test_error_regimes_split_by_observed_change():
    f = pd.DataFrame({"y": [30.0, 20.0, 10.0, 21.0], "Persistence": [20.0, 20.0, 20.0, 20.0], "m": [25.0, 20.0, 15.0, 20.0]})
    assert list(en.regime(f)) == ["rising", "steady", "falling", "steady"]
    out = en.error_regimes(f, ["Persistence", "m"])
    share = out["summary"]["error_share"]["Persistence"]
    assert share["rising"] == pytest.approx(10 / 21, abs=1e-4)
    assert {m["slice"] for m in out["metrics"]} == {"30min both/regime=rising", "30min both/regime=steady", "30min both/regime=falling"}
