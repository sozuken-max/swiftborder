"""Offline tests for layer_b_fcm_xgb_mix.py (no network, no GCP). Run from Causeway/: python -m pytest"""
import ast
import dataclasses
import datetime as dt
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

import layer_b_fcm_mlp as fcm  # noqa: E402 - after the optional-dependency checks
import layer_b_fcm_xgb_mix as m  # noqa: E402

UTC = dt.timezone.utc
SGT = fcm.SGT
REPO = Path(__file__).resolve().parents[2]
FAST = dict(ensemble_size=2, max_epochs=30, patience=6, mlp_hidden=(16,), mlp_alphas=(1e-3,), fcm_restarts=2,
            bootstrap_reps=300, protected_window=("", ""))
DATA_START = dt.date(2026, 8, 1)  # a Saturday
DATA_DAYS = 14                    # 1-14 Aug 2026 SGT
NOW = dt.datetime(2026, 8, 15, 1, 0, tzinfo=SGT)
HISTORY_START = DATA_START.isoformat()


# ---------------------------------------------------------------- helpers
def fast_cfg(lookback_days=7, validation_days=3, history_start=HISTORY_START, **base):
    return m.MixConfig(base=fcm.Config(**{**FAST, **base}), lookback_days=lookback_days, validation_days=validation_days,
                       history_start=history_start)


def _local_models():
    """forecastapi/local_models.py, the copy of eval/joined.py and eval/features.py that forecast-api serves."""
    pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO / "forecastapi"))
    try:
        import local_models
    finally:
        sys.path.remove(str(REPO / "forecastapi"))
    return local_models


def synthetic_rows(start=DATA_START, days=DATA_DAYS, seed=0, drop=()):
    """Daily peaks (evening SG->MY, morning MY->SG), weaker at weekends, AR(1) noise, plus ratio and speed."""
    rng = np.random.default_rng(seed)
    first = dt.datetime.combine(start, dt.time(0, 0), SGT).astimezone(UTC)
    n = days * 144
    t = int(first.timestamp()) + np.arange(n) * 600
    hour = (t + 28800) % 86400 / 3600.0
    dow = ((t + 28800) // 86400 + 3) % 7
    scale = np.where(dow >= 5, 0.4, 1.0)
    rows = []
    for d, centre, amp in (("SG_TO_MY", 18.5, 25.0), ("MY_TO_SG", 7.5, 30.0)):
        noise = np.zeros(n)
        shocks = rng.normal(0.0, 0.8, n)
        for j in range(1, n):
            noise[j] = 0.8 * noise[j - 1] + shocks[j]
        dur = np.maximum(12.0 + amp * scale * np.exp(-0.5 * ((hour - centre) / 1.3) ** 2) + noise, 6.0)
        for i in range(n):
            ts = dt.datetime.fromtimestamp(int(t[i]), UTC)
            if (d, ts) in drop:
                continue
            rows.append({"direction": d, "bin_ts": ts, "dur_min": float(dur[i]), "congestion_ratio": float(dur[i] / 12.0),
                         "speed_kmh": float(360.0 / dur[i])})
    return rows


def layer_a_rows_for(rows):
    """One accepted Layer A frame per bin and direction, its count tracking the travel time."""
    by_bin = {}
    for r in rows:
        by_bin.setdefault(r["bin_ts"], {})[r["direction"]] = r["dur_min"]
    out = []
    for ts, durs in by_bin.items():
        for d in durs:
            other = [v for k, v in durs.items() if k != d]
            count = round(durs[d] * 1.5)
            other_count = round(other[0] * 1.5) if other else 0
            out.append({"bin_ts": ts, "direction": d, "vehicle_count": count, "other_direction_count": other_count,
                        "total_count": count + other_count, "congestion_ordinal": min(3, count // 20), "rejected": False})
    return out


class FakeWarehouse:
    """Answers the module's queries from memory, checks every write against the table schema."""

    def __init__(self, bins=(), registry=None, layer_a_rows=None, monitor_rows=(), existing=()):
        self.bins, self.registry, self.layer_a_rows = list(bins), registry, layer_a_rows
        self.monitor_rows, self.existing = list(monitor_rows), list(existing)
        self.appended, self.ensured, self.queries, self.registry_params = {}, 0, [], None

    def query(self, sql, params=None, timeout_sec=600.0):
        self.queries.append(sql)
        if m.TABLE_REGISTRY in sql:
            self.registry_params = params
            if not self.registry or self.registry["created_at"] < params["since"]:
                return []
            return [self.registry]
        if m.TABLE_FORECASTS in sql:
            return [r for r in self.monitor_rows if params["start"] <= r["origin_ts"] <= params["end"]]
        if "SELECT DISTINCT bin_ts" in sql:
            return [{"bin_ts": b} for b in self.existing if params["start"] <= b <= params["end"]]
        if fcm.TABLE_LAYER_A in sql:
            if self.layer_a_rows is None:
                raise RuntimeError("Not found: Table layer_a_counts")
            return [r for r in self.layer_a_rows if params["start"] <= r["bin_ts"] <= params["end"]]
        if "AVG(congestion_ratio)" in sql:
            return [r for r in self.bins
                    if params["start"] <= r["bin_ts"] <= params["end"] and r["direction"] in params["directions"]]
        raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def ensure_tables(self):
        self.ensured += 1
        return list(m.ALL_SCHEMAS)

    def append(self, name, rows):
        for r in rows:
            for column, _, mode in m.ALL_SCHEMAS[name]:
                if mode == "REQUIRED":
                    assert r.get(column) is not None, f"{name}.{column} is REQUIRED"
        json.dumps([{c: m._jsonable(r.get(c)) for c, _, _ in m.ALL_SCHEMAS[name]} for r in rows])  # must serialise
        self.appended.setdefault(name, []).extend(rows)
        return len(rows)


def bins_for(rows, start, days):
    first = dt.datetime.combine(start, dt.time(0, 0), SGT).astimezone(UTC)
    grid = fcm.Grid(int(first.timestamp()), days * 144)
    return m.bins_from_rows(rows, grid)


def epoch_day(day):
    return fcm.date_to_epoch_day(day)


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


@pytest.fixture(scope="module")
def backtested(rows):
    return m.backtest(fast_cfg(), FakeWarehouse(rows), start=dt.date(2026, 8, 12), end=dt.date(2026, 8, 13), now=NOW)


@pytest.fixture(scope="module")
def trained(rows):
    return m.train(fast_cfg(), FakeWarehouse(rows), end=dt.date(2026, 8, 13), now=NOW)


# ---------------------------------------------------------------- configuration and constants
def test_config_defaults_and_environment():
    cfg = m.MixConfig()
    cfg.validate()
    assert (cfg.lookback_days, cfg.validation_days, cfg.history_start, cfg.base.horizon_steps) == (None, 7, "2026-09-06", 6)
    day = epoch_day(dt.date(2026, 10, 30))
    assert cfg.first_training_day(day) == epoch_day(dt.date(2026, 9, 6))  # all history: the evaluated daily refit
    env = {"MIX_LOOKBACK_DAYS": "21", "MIX_VALIDATION_DAYS": "5", "MIX_HISTORY_START": "2026-09-07",
           "SWIFTBORDER_PROJECT": "swiftborder"}
    cfg = m.MixConfig.from_env(env)
    assert (cfg.lookback_days, cfg.validation_days, cfg.history_start) == (21, 5, "2026-09-07")
    assert cfg.first_training_day(day) == day - 21
    assert m.MixConfig.from_env({"MIX_LOOKBACK_DAYS": " "}).lookback_days is None
    assert cfg.table(m.TABLE_FORECASTS) == "swiftborder.traffic_prediction.layer_b_mix_forecasts"


@pytest.mark.parametrize("env", [{"MIX_LOOKBACK_DAYS": "x"}, {"MIX_LOOKBACK_DAYS": "4"}, {"MIX_LOOKBACK_DAYS": "400"},
                                 {"MIX_VALIDATION_DAYS": "2"}, {"MIX_VALIDATION_DAYS": "29"}, {"FCM_CLUSTERS": "9"},
                                 {"MIX_HISTORY_START": "2026-13-01"}])
def test_config_rejects_bad_environment(env):
    with pytest.raises(m.ConfigError):
        m.MixConfig.from_env(env)


def test_constants_equal_the_harness_copy_in_forecast_api():
    """forecastapi/local_models.py is tested equal to eval/joined.py; the mix must use the same xgb[maps]."""
    tree = ast.parse((REPO / "forecastapi" / "local_models.py").read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in ("MAPS_FEATURES", "HARNESS_XGB_PARAMS"):
                found[target.id] = ast.literal_eval(node.value)
    assert tuple(found["MAPS_FEATURES"]) == m.MAPS_FEATURES
    assert dict(found["HARNESS_XGB_PARAMS"]) == dict(m.XGB_PARAMS)
    assert m.HORIZON_MIN == 30 and m.MIX_WEIGHTS == {"fcm_mlp": 0.5, "xgb_maps": 0.5}


def test_layer_a_table_is_shared_with_the_fcm_module():
    assert m.ALL_SCHEMAS[m.TABLE_LAYER_A] is fcm.SCHEMAS[fcm.TABLE_LAYER_A]
    assert set(m.SCHEMAS).isdisjoint(fcm.SCHEMAS)  # never writes the FCM module's own tables


# ---------------------------------------------------------------- Maps features
def test_after_gap_flags_follow_v_training_set():
    dur = np.arange(1.0, 31.0)
    dur[[10]] = np.nan          # one missing bin: a 20-minute step, not a gap
    dur[[20, 21]] = np.nan      # two missing bins: a 30-minute step, a gap
    flags = m.after_gap_flags(dur)
    assert not flags[:20].any()                       # the first bin read and the 20-minute step are not gaps
    assert flags[22:29].all() and not flags[29:].any()  # the gap bin and the next six observed bins
    assert not flags[[20, 21]].any()                  # missing bins carry no flag
    cut = m.after_gap_flags(np.arange(1.0, 11.0), first_is_gap=True)  # a read cut from a longer history
    assert cut[:7].all() and not cut[7:].any()


def test_maps_features_values_by_hand():
    first = dt.datetime(2026, 8, 2, 16, 0, tzinfo=UTC)  # Monday 3 Aug 00:00 SGT
    grid = fcm.Grid(int(first.timestamp()), 60)
    dur = np.arange(10.0, 70.0)
    dur[40] = np.nan
    bins = m.Bins(grid, {"SG_TO_MY": dur, "MY_TO_SG": dur + 1}, {d: np.full(60, 1.2) for d in m.DIRECTIONS},
                  {d: np.full(60, 30.0) for d in m.DIRECTIONS})
    x, after_gap = m.maps_features(bins, "MY_TO_SG", [45, 41])
    col = {name: x[:, k] for k, name in enumerate(m.MAPS_FEATURES)}
    # origin 45: value 56, lags 55, 54, 53 and (bin 39) 50; bin 40 missing so roll_mean_60 skips it
    assert col["y_persistence"][0] == 56 and col["lag_10"][0] == 55 and col["lag_30"][0] == 53 and col["lag_60"][0] == 50
    assert col["roll_mean_30"][0] == pytest.approx(54.0)
    assert col["roll_mean_60"][0] == pytest.approx(np.mean([55, 54, 53, 52, 50]))
    assert col["slope_30"][0] == pytest.approx(55 - 52)
    assert math.isnan(col["lag_10"][1]) and col["lag_20"][1] == 50  # time-based: the missing bin 40 stays missing
    hour = 45 * 10 // 60  # 07:30 SGT on a Monday
    assert col["tod_block"][0] == (hour * 60 + 30) / 5.0 and col["is_morning_peak"][0] == 1.0
    assert col["dow"][0] == 2.0 and col["is_weekend"][0] == 0.0 and col["is_my_to_sg"][0] == 1.0
    assert not after_gap.any()


def test_maps_features_blank_lags_after_a_gap():
    first = dt.datetime(2026, 8, 2, 16, 0, tzinfo=UTC)
    grid = fcm.Grid(int(first.timestamp()), 40)
    dur = np.linspace(20.0, 40.0, 40)
    dur[[10, 11, 12]] = np.nan
    ones = {d: np.ones(40) for d in m.DIRECTIONS}
    bins = m.Bins(grid, dict.fromkeys(m.DIRECTIONS, dur), ones, ones)
    x, after_gap = m.maps_features(bins, "SG_TO_MY", [14, 25])
    assert after_gap.tolist() == [True, False]
    lag_names = ("lag_10", "lag_20", "lag_30", "lag_60", "roll_mean_30", "roll_mean_60", "slope_30")
    lag_cols = [m.MAPS_FEATURES.index(c) for c in lag_names]
    assert np.isnan(x[0, lag_cols]).all() and np.isfinite(x[1, lag_cols]).all()


def test_maps_features_match_forecast_api_and_the_harness():
    """Bit-for-bit against forecastapi/local_models.features_from_bins (itself tested equal to eval/features.py)."""
    lm = _local_models()
    import pandas as pd
    drop = set()
    base = dt.datetime(2026, 8, 3, 0, 0, tzinfo=UTC)
    for k in (5, 6, 7, 30, 75, 76, 140):  # single and multi-bin gaps
        drop.add(("SG_TO_MY", base + dt.timedelta(minutes=10 * k)))
    rows = synthetic_rows(days=4, drop=drop)
    frame = lm.features_from_bins(pd.DataFrame([{**r, "route_id": r["direction"]} for r in rows]))
    bins = bins_for(rows, DATA_START, 4)
    for d in m.DIRECTIONS:
        ref = frame[frame["direction"] == d].reset_index(drop=True)
        idx = [bins.grid.index(ts.to_pydatetime()) for ts in ref["bin_ts"]]
        x, after_gap = m.maps_features(bins, d, idx)
        ref = ref.assign(is_my_to_sg=float(d == "MY_TO_SG"))
        expected = ref[list(m.MAPS_FEATURES)].to_numpy(dtype=float)
        np.testing.assert_array_equal(np.isnan(x), np.isnan(expected))
        np.testing.assert_allclose(np.nan_to_num(x), np.nan_to_num(expected), rtol=0, atol=1e-12)
        assert after_gap.tolist() == (ref["after_gap"] == 1).tolist()


def test_bins_from_rows_ignores_bad_rows():
    grid = fcm.Grid(int(dt.datetime(2026, 8, 3, tzinfo=UTC).timestamp()), 3)
    t0 = grid.time(0)
    rows = [{"direction": "SG_TO_MY", "bin_ts": t0, "dur_min": 20.0, "congestion_ratio": None, "speed_kmh": 30.0},
            {"direction": "SG_TO_MY", "bin_ts": t0 + dt.timedelta(minutes=5), "dur_min": 21.0},  # off the grid
            {"direction": "MY_TO_SG", "bin_ts": t0, "dur_min": -1.0},                          # not positive
            {"direction": "OTHER", "bin_ts": t0, "dur_min": 5.0}]
    bins = m.bins_from_rows(rows, grid)
    assert bins.dur["SG_TO_MY"][0] == 20.0 and math.isnan(bins.congestion["SG_TO_MY"][0]) and bins.speed["SG_TO_MY"][0] == 30.0
    assert np.isnan(bins.dur["MY_TO_SG"]).all() and np.isnan(bins.dur["SG_TO_MY"][1:]).all()


# ---------------------------------------------------------------- the mix and its decision rules
def test_mix_is_the_equal_average_and_never_worse_than_the_mean_member_error():
    rng = np.random.default_rng(3)
    a, b, y = rng.normal(30, 5, 500), rng.normal(30, 5, 500), rng.normal(30, 5, 500)
    mixed = m.mix_of(a, b)
    np.testing.assert_allclose(mixed, (a + b) / 2)
    assert (np.abs(mixed - y) <= (np.abs(a - y) + np.abs(b - y)) / 2 + 1e-12).all()
    assert np.isnan(m.mix_of(np.array([1.0, np.nan]), np.array([np.nan, 2.0]))).all()


def test_served_value_chain():
    row = {"persistence": 30.0, "mix": 28.0, "mix_layer_a": 27.0}
    assert m.served_value(row, "mix_layer_a", "mix") == ("mix_layer_a", 27.0)
    assert m.served_value({**row, "mix_layer_a": None}, "mix_layer_a", "mix") == ("mix", 28.0)
    assert m.served_value({**row, "mix_layer_a": None}, "mix_layer_a", "persistence") == ("persistence", 30.0)
    assert m.served_value({**row, "mix": None}, "mix") == ("persistence", 30.0)
    assert m.served_value(row, "mix_layer_a", "bogus") == ("mix_layer_a", 27.0)
    assert m.served_value({**row, "mix_layer_a": None}, "mix_layer_a", None) == ("persistence", 30.0)
    assert m.served_value(row, 42) == ("persistence", 30.0)


def _validation_rows(days, mix_err, vision_err=None, vision_share=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for day in range(days):
        for i in range(48):
            origin = dt.datetime(2026, 8, 3 + day, 1, 0, tzinfo=UTC) + dt.timedelta(minutes=10 * i)
            for d in m.DIRECTIONS:
                actual = 30.0 + rng.normal(0, 1)
                row = {"direction": d, "origin_ts": origin, "target_ts": origin + dt.timedelta(minutes=30), "step": 3,
                       "actual": actual, "persistence": actual + rng.choice([-3.0, 3.0]),
                       "mix": actual + rng.choice([-1.0, 1.0]) * mix_err, "xgb_maps": actual + 2.0, "fcm_mlp": actual - 2.0}
                if vision_err is not None and i < 48 * vision_share:
                    row["mix_layer_a"] = actual + rng.choice([-1.0, 1.0]) * vision_err
                    row["mix_same_rows"] = actual + rng.choice([-1.0, 1.0]) * mix_err
                out.append(row)
    return out


def test_decision_rules():
    cfg = fast_cfg()
    good = m.decide(cfg, _validation_rows(5, mix_err=1.0))
    assert all(n["served"] == "mix" and n["fallback"] == "mix" and n["candidate"] == "mix" for n in good.values())
    bad = m.decide(cfg, _validation_rows(5, mix_err=5.0))
    assert all(n["served"] == "persistence" and n["fallback"] == "persistence" for n in bad.values())
    vision = m.decide(cfg, _validation_rows(5, mix_err=1.0, vision_err=0.2))
    assert all(n["served"] == "mix_layer_a" and n["layer_a_coverage"] == 1.0 for n in vision.values())
    sparse = m.decide(cfg, _validation_rows(5, mix_err=1.0, vision_err=0.2, vision_share=0.3))
    assert all(n["candidate"] == "mix" for n in sparse.values())  # rule 2 needs half of the validation rows
    short = m.decide(cfg, _validation_rows(2, mix_err=0.1))
    assert all(n["served"] == "persistence" for n in short.values())  # fewer than three days never passes a gate
    empty = m.decide(cfg, [])
    assert all(n["served"] == "persistence" and n["rows"] == 0 for n in empty.values())


# ---------------------------------------------------------------- training rows and leakage
def _data(cfg, rows, first_day, last_day, layer_a=None):
    wh = FakeWarehouse(rows, layer_a_rows=layer_a)
    data, _ = m.load_window(cfg, wh, epoch_day(first_day), epoch_day(last_day), NOW, allow_protected=True)
    return data


def test_xgb_training_rows_are_in_the_harness_order_and_respect_the_cap(rows):
    day = epoch_day(dt.date(2026, 8, 12))
    capped = fast_cfg(lookback_days=5)
    x, _ = m.xgb_training_set(capped, _data(capped, rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12)), day, vision=False)
    flag = x[:, m.MAPS_FEATURES.index("is_my_to_sg")]
    split = int(np.argmin(flag))
    assert flag[:split].all() and not flag[split:].any()  # MY_TO_SG rows first, as eval/joined sorts them
    full = fast_cfg(lookback_days=None)
    _, y_full = m.xgb_training_set(full, _data(full, rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12)), day, vision=False)
    assert len(y_full) > len(flag)  # the default reads every day since the history start


def test_xgb_maps_equals_the_forecast_api_daily_fit():
    """Same training rows and identical predictions as forecastapi/local_models (equal to eval/joined.py)."""
    lm = _local_models()
    import pandas as pd
    base = dt.datetime(2026, 8, 6, 0, 0, tzinfo=UTC)
    drop = {(d, base + dt.timedelta(minutes=10 * k)) for d in m.DIRECTIONS for k in (3, 4, 5, 60, 200)}
    rows = synthetic_rows(drop=drop)
    cfg = fast_cfg(lookback_days=None)
    day_date = dt.date(2026, 8, 12)
    day = epoch_day(day_date)
    data = _data(cfg, rows, day_date, day_date)
    models, _, diag = m.fit_day(cfg, data, day)
    frame = lm.prepare(lm.features_from_bins(pd.DataFrame([{**r, "route_id": r["direction"]} for r in rows])))
    start = pd.Timestamp(day_date.isoformat()).tz_localize("Asia/Singapore").tz_convert("UTC")
    train = lm.training_rows(frame, "daily", start)
    assert diag["xgb_rows"]["maps"] == len(train)
    test = frame[(frame["date_sgt"] == day_date.isoformat()) & frame["y_30"].notna() & (frame["after_gap"] == 0)]
    keys = zip(test["direction"], (ts.to_pydatetime() for ts in test["bin_ts"]), strict=True)
    expected = dict(zip(keys, lm.fit("xgb[maps]", train).predict(test), strict=True))
    got = {(r["direction"], r["origin_ts"]): r["xgb_maps"] for r in m.evaluation_rows(data, models, None, day)}
    assert set(got) == set(expected)
    assert max(abs(got[k] - expected[k]) for k in got) == 0.0


def test_fit_networks_is_the_procedure_of_the_fcm_module(rows):
    """fcm_mlp here is byte-identical to layer_b_fcm_mlp.train_direction on the same days and labels."""
    cfg = fast_cfg(lookback_days=None, mlp_alphas=(1e-4, 1e-3))
    day = epoch_day(dt.date(2026, 8, 12))
    data = _data(cfg, rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12))
    for index, d in enumerate(m.DIRECTIONS):
        models, _, diag = m.fit_networks(cfg, data[d], day, index)
        sub = m.subset_frame(data[d].frame, np.flatnonzero(data[d].frame.origin_day <= day))
        steps = np.arange(1, sub.horizon + 1)
        sub.actual[(sub.origin_unix[:, None] + 600 * steps[None, :]) > m.day_start_unix(day) - 600] = np.nan
        inner = tuple(epoch_day(dt.date.fromisoformat(x)) for x in diag["inner_days"])
        fit_days = tuple(sorted({int(x) for x in sub.origin_day if x < inner[0]}))
        reference = fcm.train_direction(cfg.base, sub, fcm.DaySplit(fit_days, inner, (day,), ()), index)["models"]
        assert json.dumps(reference.to_json(), sort_keys=True) == json.dumps(models.to_json(), sort_keys=True)


def test_no_label_at_or_after_the_serving_day_reaches_a_fit(rows):
    """Corrupting every label whose bin closes after 00:00 SGT on the day leaves the fitted models unchanged."""
    cfg = fast_cfg()
    day = epoch_day(dt.date(2026, 8, 12))
    clean = _data(cfg, rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12))
    poisoned = _data(cfg, rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12))
    for dd in poisoned.values():
        steps = np.arange(1, dd.frame.horizon + 1)
        target = dd.frame.origin_unix[:, None] + 600 * steps[None, :]
        dd.frame.actual[target > m.day_start_unix(day) - 600] = 1e6
    a, _, diag = m.fit_day(cfg, clean, day)
    b, _, _ = m.fit_day(cfg, poisoned, day)
    assert diag["SG_TO_MY"]["inner_days"] == ["2026-08-10", "2026-08-11"]
    for d in m.DIRECTIONS:
        idx = np.flatnonzero(clean[d].frame.origin_day == day)
        pa, pb = m.predict_direction(a, clean[d], idx), m.predict_direction(b, clean[d], idx)
        for key in ("fcm_mlp", "xgb_maps", "mix"):
            np.testing.assert_array_equal(pa[key], pb[key])


def test_xgb_member_json_round_trip_is_exact(rows):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, len(m.MAPS_FEATURES)))
    x[rng.random(x.shape) < 0.05] = np.nan
    y = np.nansum(x[:, :3], axis=1) + rng.normal(0, 0.1, 600)
    member = m.XGBMember.fit(x, y, m.MAPS_FEATURES)
    again = m.XGBMember.from_json(json.loads(json.dumps(member.to_json())), m.MAPS_FEATURES)
    np.testing.assert_array_equal(member.predict(x), again.predict(x))
    with pytest.raises(ValueError):
        m.XGBMember.from_json(member.to_json(), m.MAPS_FEATURES + m.VISION_FEATURES)
    with pytest.raises(ValueError):
        member.predict(x[:, :5])


# ---------------------------------------------------------------- end to end
def test_backtest_scores_every_scorable_origin(backtested):
    rows = backtested["rows"]
    assert rows and {r["origin_ts"].astimezone(SGT).date() for r in rows} == {dt.date(2026, 8, 12), dt.date(2026, 8, 13)}
    assert all(r["mix"] is not None and r["xgb_maps"] is not None and r["fcm_mlp"] is not None for r in rows)
    assert all(r["target_ts"] - r["origin_ts"] == dt.timedelta(minutes=30) for r in rows)
    metrics = backtested["backtest"]["metrics"]
    mae = {k: metrics[k]["all"]["mae"] for k in ("mix", "fcm_mlp", "xgb_maps")}
    assert mae["mix"] <= (mae["fcm_mlp"] + mae["xgb_maps"]) / 2 + 1e-12
    assert backtested["backtest"]["pooled"]["mix_vs_persistence"]["days"] == 2
    assert set(m.headline(backtested["backtest"])["comparisons"]) == {"mix_vs_persistence", "mix_vs_xgb_maps", "mix_vs_fcm_mlp"}


def test_protected_window_is_refused_and_its_labels_are_never_scored(rows):
    cfg = fast_cfg(protected_window=("2026-08-14", "2026-08-20"))
    wh = FakeWarehouse(rows)
    with pytest.raises(m.ConfigError):
        m.backtest(cfg, wh, start=dt.date(2026, 8, 13), end=dt.date(2026, 8, 14), now=NOW)
    data = _data(dataclasses.replace(cfg), rows, dt.date(2026, 8, 13), dt.date(2026, 8, 13))  # allowed: full labels
    masked, _ = m.load_window(cfg, wh, epoch_day(dt.date(2026, 8, 13)), epoch_day(dt.date(2026, 8, 13)), NOW,
                              allow_protected=False)
    last_three = slice(-3, None)  # 23:30-23:50 SGT on 13 Aug: their 30-minute targets fall on 14 Aug
    for d in m.DIRECTIONS:
        assert np.isfinite(data[d].y30[last_three]).all()
        assert np.isnan(masked[d].y30[last_three]).all()


def test_train_registers_models_that_serve_the_same_forecasts(trained, rows):
    cfg = fast_cfg()
    assert trained["labels_before"] == dt.datetime(2026, 8, 14, 0, 0, tzinfo=SGT)
    assert trained["validation_start"] == dt.date(2026, 8, 11) and trained["validation_end"] == dt.date(2026, 8, 13)
    assert set(trained["decisions"]) == set(m.DIRECTIONS)
    assert all(r.get("served_model") in m.SERVABLE for r in trained["rows"])
    row = m.registry_row(cfg, trained, "train-test", NOW)
    models, problems = m.MixModels.from_json(json.loads(row["artefacts_json"]), cfg.base.horizon_steps)
    assert not problems and set(models.networks) == set(m.DIRECTIONS)
    data = _data(cfg, rows, dt.date(2026, 8, 14), dt.date(2026, 8, 14))
    for d in m.DIRECTIONS:
        idx = np.flatnonzero(data[d].frame.origin_day == epoch_day(dt.date(2026, 8, 14)))
        a, b = m.predict_direction(trained["models"], data[d], idx), m.predict_direction(models, data[d], idx)
        np.testing.assert_array_equal(a["mix"], b["mix"])


def _registry(cfg, trained, served=None):
    row = m.registry_row(cfg, trained, "train-test", NOW)
    if served:
        row["decisions_json"] = json.dumps({d: {"served": served, "candidate": "mix", "fallback": "mix"} for d in m.DIRECTIONS})
    return row


def test_run_cycle_serves_the_registered_mix(trained, rows):
    cfg = fast_cfg()
    wh = FakeWarehouse(rows, registry=_registry(cfg, trained, served="mix"))
    when = dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT)
    summary = m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    written = wh.appended[m.TABLE_FORECASTS]
    assert wh.ensured == 1 and len(written) == 2
    for r in written:
        assert r["served_model"] == "mix" and r["served_value"] == pytest.approx((r["fcm_mlp"] + r["xgb_maps"]) / 2)
        assert r["origin_ts"] == dt.datetime(2026, 8, 14, 17, 50, tzinfo=SGT)
        assert r["target_ts"] == r["origin_ts"] + dt.timedelta(minutes=30) and r["horizon_min"] == 30
        assert r["regime"] in ("light", "moderate", "heavy") and r["forecast_regime"] in ("light", "moderate", "heavy")
        assert r["labels_before"] == dt.datetime(2026, 8, 14, tzinfo=SGT) and not r["after_gap"]
    assert summary["model_age_days"] == 0


def test_run_cycle_falls_back_to_persistence(trained, rows):
    cfg = fast_cfg()
    when = dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT)
    wh = FakeWarehouse(rows)  # no registry entry
    m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    assert all(r["served_model"] == "persistence" and r["mix"] is None for r in wh.appended[m.TABLE_FORECASTS])

    broken = _registry(cfg, trained, served="mix")
    artefacts = json.loads(broken["artefacts_json"])
    artefacts["xgb_maps"]["booster_json"] = "{not json"
    broken["artefacts_json"] = json.dumps(artefacts)
    wh = FakeWarehouse(rows, registry=broken)
    m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    assert all(r["served_model"] == "persistence" and r["fcm_mlp"] is not None for r in wh.appended[m.TABLE_FORECASTS])

    gate_failed = _registry(cfg, trained, served="persistence")
    wh = FakeWarehouse(rows, registry=gate_failed)
    m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    assert all(r["served_model"] == "persistence" and r["mix"] is not None for r in wh.appended[m.TABLE_FORECASTS])


def test_run_cycle_after_a_gap_serves_persistence(trained):
    cfg = fast_cfg()
    when = dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT)
    origin = dt.datetime(2026, 8, 14, 17, 50, tzinfo=SGT).astimezone(UTC)
    gap = {(d, origin - dt.timedelta(minutes=10 * k)) for d in m.DIRECTIONS for k in (2, 3, 4)}
    wh = FakeWarehouse(synthetic_rows(drop=gap), registry=_registry(cfg, trained, served="mix"))
    m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    assert all(r["after_gap"] and r["served_model"] == "persistence" and r["xgb_maps"] is None
               for r in wh.appended[m.TABLE_FORECASTS])


def test_run_cycle_dry_run_and_stale_data(trained, rows):
    cfg = fast_cfg()
    wh = FakeWarehouse(rows, registry=_registry(cfg, trained))
    summary = m.run_cycle(cfg, wh, None, now=dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT), use_layer_a=False, write=False)
    assert not wh.appended and wh.ensured == 0 and len(summary["forecasts"]) == 2
    with pytest.raises(m.NoForecastError):
        m.run_cycle(cfg, wh, None, now=dt.datetime(2026, 8, 20, 12, 5, tzinfo=SGT), use_layer_a=False, write=False)


def test_layer_a_variants_are_fitted_on_frames_and_compared_on_the_same_rows(rows):
    cfg = fast_cfg()
    day = epoch_day(dt.date(2026, 8, 12))
    data = _data(cfg, rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12), layer_a=layer_a_rows_for(rows))
    models, same, diag = m.fit_day(cfg, data, day)
    assert models.xgb_maps_layer_a is not None and same.xgb_maps is not None
    assert all(models.networks[d].fcm_mlp_layer_a is not None and d in same.networks for d in m.DIRECTIONS)
    assert diag["xgb_rows"]["layer_a"] == diag["xgb_rows"]["maps"]
    fold = m.evaluation_rows(data, models, same, day)
    assert fold and all(r["mix_layer_a"] is not None and r["mix_same_rows"] is not None and r["layer_a_status"] == "ok"
                        for r in fold)
    decisions = m.decide(cfg, fold)
    assert all(n["layer_a_coverage"] == 1.0 and "mix_layer_a_vs_same_rows" in n for n in decisions.values())


def test_monitor_scores_logged_forecasts_and_respects_the_protected_window():
    now = dt.datetime(2026, 8, 20, 12, 0, tzinfo=SGT)
    logged = []
    for k in range(200):
        origin = dt.datetime(2026, 8, 17, 0, 0, tzinfo=SGT).astimezone(UTC) + dt.timedelta(minutes=10 * k)
        for d in m.DIRECTIONS:
            logged.append({"origin_ts": origin, "direction": d, "target_ts": origin + dt.timedelta(minutes=30),
                           "persistence": 31.0, "fcm_mlp": 29.0, "xgb_maps": 30.5, "mix": 29.75, "served_model": "mix",
                           "served_value": 29.75, "actual": 30.0})
    logged[0] = {**logged[0], "mix": None, "fcm_mlp": None, "xgb_maps": None, "served_model": "persistence",
                 "served_value": 31.0}  # e.g. an origin after a gap
    result = m.monitor(fast_cfg(), FakeWarehouse(monitor_rows=logged), days=7, now=now)
    assert result["rows"] == 400 and result["common_rows"] == 399
    assert result["metrics"]["persistence"]["all"]["mae"] == pytest.approx(1.0)  # on the 399 rows every model has
    assert result["metrics"]["mix"]["all"]["n"] == result["metrics"]["persistence"]["all"]["n"] == 399
    assert result["metrics_all_rows"]["served_value"]["all"]["mae"] == pytest.approx((399 * 0.25 + 1.0) / 400)
    with pytest.raises(m.ConfigError):
        m.monitor(fast_cfg(protected_window=("2026-08-18", "2026-08-25")), FakeWarehouse(monitor_rows=logged), days=7, now=now)
    with pytest.raises(m.ConfigError):
        m.monitor(fast_cfg(), FakeWarehouse(), days=0, now=now)


# ---------------------------------------------------------------- command line
def test_show_sql_needs_no_credentials(monkeypatch, capsys):
    monkeypatch.delenv("PROTECTED_WINDOW", raising=False)
    assert m.main(["show-sql"]) == m.EXIT_OK
    out = capsys.readouterr().out
    assert "layer_b_mix_registry" in out and "layer_a_counts" in out and "v_bins_10min" in out


def test_main_exit_codes(monkeypatch, capsys, rows):
    monkeypatch.setenv("PROTECTED_WINDOW", "")  # the default history start (6 Sep) is after these August days
    assert m.main(["train", "--end", "2026-08-03"], warehouse_factory=lambda cfg: FakeWarehouse(rows), now=NOW) == m.EXIT_CONFIG
    assert m.main(["train", "--end", "2026-08-15"], warehouse_factory=lambda cfg: FakeWarehouse(rows), now=NOW) == m.EXIT_CONFIG
    assert m.main(["run-cycle", "--no-layer-a"], warehouse_factory=lambda cfg: FakeWarehouse(rows),
                  now=dt.datetime(2026, 8, 30, tzinfo=SGT)) == m.EXIT_NO_FORECAST
    monkeypatch.setenv("MIX_LOOKBACK_DAYS", "nope")
    assert m.main(["show-sql"]) == m.EXIT_CONFIG
    assert '"severity": "ERROR"' in capsys.readouterr().out


def test_command_can_come_from_the_environment(monkeypatch, capsys):
    monkeypatch.setenv("LAYER_B_ARGS", "show-sql")
    monkeypatch.setattr(sys, "argv", ["layer_b_fcm_xgb_mix.py"])
    assert m.main() == m.EXIT_OK
    assert "-- travel-time bins" in capsys.readouterr().out


def test_main_backtest_writes_evaluation_rows(monkeypatch, capsys, rows):
    monkeypatch.setenv("PROTECTED_WINDOW", "")
    for key, value in (("MLP_HIDDEN", "16"), ("MLP_ENSEMBLE_SIZE", "1"), ("MIX_LOOKBACK_DAYS", "7"),
                       ("MIX_HISTORY_START", DATA_START.isoformat())):
        monkeypatch.setenv(key, value)
    wh = FakeWarehouse(rows)
    code = m.main(["backtest", "--start", "2026-08-13", "--end", "2026-08-13", "--write-eval"],
                  warehouse_factory=lambda cfg: wh, now=NOW)
    assert code == m.EXIT_OK
    written = wh.appended[m.TABLE_EVALUATION]
    assert written and all(r["split"] == "backtest" and r["model"] == m.MODEL_NAME for r in written)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{\"backtest\"")]
    assert printed and printed[-1]["headline"]["rows"] == len(written)


# ---------------------------------------------------------------- BigQuery wrapper, Layer A ingestion, edge cases
class FakeJob:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def result(self, timeout=None):
        return [type("Row", (), {"items": (lambda self, r=r: r.items())})() for r in self.rows]


class FakeClient:
    def __init__(self):
        self.created, self.loaded, self.queried = [], [], []

    def query(self, sql, job_config=None, location=None):
        self.queried.append((sql, job_config, location))
        return FakeJob([{"n": 1}])

    def create_table(self, table, exists_ok=False):
        self.created.append(table)

    def load_table_from_json(self, payload, table, job_config=None, location=None):
        self.loaded.append((payload, table, job_config, location))
        return FakeJob()


def test_warehouse_creates_partitioned_tables_and_loads_rows():
    client = FakeClient()
    wh = m.Warehouse(m.MixConfig(), client=client)
    assert wh.query("SELECT 1", {"start": dt.datetime(2026, 8, 1, tzinfo=UTC), "n": 3}) == [{"n": 1}]
    _, job_config, location = client.queried[0]
    assert location == "US" and job_config.labels == {"component": "layer-b-fcm-xgb-mix"}
    assert {p.name for p in job_config.query_parameters} == {"start", "n"}
    created = wh.ensure_tables()
    assert created == [f"swiftborder.traffic_prediction.{t}" for t in m.ALL_SCHEMAS]
    forecasts = next(t for t in client.created if t.table_id == m.TABLE_FORECASTS)
    assert forecasts.time_partitioning.field == "origin_ts" and forecasts.clustering_fields == ["direction", "origin_ts"]
    registry = next(t for t in client.created if t.table_id == m.TABLE_REGISTRY)
    assert registry.time_partitioning.field == "created_at" and registry.clustering_fields == ["model"]
    assert wh.append(m.TABLE_EVALUATION, []) == 0
    row = {"run_id": "r", "created_at": NOW, "split": "backtest", "origin_ts": NOW, "direction": "SG_TO_MY", "target_ts": NOW,
           "actual": 30.0, "mix": float("nan"), "layer_a_status": "missing", "labels_before": NOW, "model": m.MODEL_NAME,
           "not_a_column": 1}
    assert wh.append(m.TABLE_EVALUATION, [row]) == 1
    payload, table, job_config, _ = client.loaded[0]
    assert table.endswith(m.TABLE_EVALUATION) and payload[0]["mix"] is None and "not_a_column" not in payload[0]
    assert [f.name for f in job_config.schema] == [c for c, _, _ in m.SCHEMAS[m.TABLE_EVALUATION]]


README_PAYLOAD = {  # camdetect/README.md format=json example
    "camera_id": "2701", "date_time": "2026-08-14T18:01:00", "vehicle_count": 10, "predictions": [],
    "directions": {"available": True, "sg_my": {"count": 4, "congestion": "Quarter Way"},
                   "my_sg": {"count": 4, "congestion": "Half Way"}, "unknown": {"count": 2}},
}


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_run_cycle_logs_layer_a_and_forecasts_even_when_it_fails(trained, rows):
    cfg = fast_cfg(layer_a_url="https://swiftbackend.example.run.app")
    when = dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT)
    wh = FakeWarehouse(rows, registry=_registry(cfg, trained, served="mix"), layer_a_rows=layer_a_rows_for(rows))
    summary = m.run_cycle(cfg, wh, FakeSession([FakeResp(200, README_PAYLOAD)]), now=when)
    assert summary["layer_a_ingest"]["status"] == "logged" and len(wh.appended[m.TABLE_LAYER_A]) == 2
    assert all(r["layer_a_status"] == "ok" for r in wh.appended[m.TABLE_FORECASTS])
    wh = FakeWarehouse(rows, registry=_registry(cfg, trained, served="mix"))  # no Layer A table at all
    summary = m.run_cycle(cfg, wh, FakeSession([FakeResp(500)]), now=when)
    assert summary["layer_a_ingest"]["status"] == "failed" and len(wh.appended[m.TABLE_FORECASTS]) == 2
    summary = m.run_cycle(cfg, wh, FakeSession([]), now=when, write=False)
    assert summary["layer_a_ingest"] == {"status": "skipped_dry_run"}


def test_registry_problems_degrade_to_persistence(trained, rows):
    cfg = fast_cfg()
    when = dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT)
    partial = _registry(cfg, trained, served="mix")
    artefacts = json.loads(partial["artefacts_json"])
    del artefacts["networks"]["MY_TO_SG"]
    partial["artefacts_json"], partial["config_json"] = json.dumps(artefacts), "{not json"
    wh = FakeWarehouse(rows, registry=partial)
    m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    served = {r["direction"]: r["served_model"] for r in wh.appended[m.TABLE_FORECASTS]}
    assert served == {"SG_TO_MY": "mix", "MY_TO_SG": "persistence"}
    for bad in ({"decisions_json": "[]"}, {"artefacts_json": "{oops"}):
        wh = FakeWarehouse(rows, registry={**_registry(cfg, trained), **bad})
        m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
        assert all(r["served_model"] == "persistence" for r in wh.appended[m.TABLE_FORECASTS])
    odd = _registry(cfg, trained)
    odd["decisions_json"] = json.dumps({d: {"served": "bogus"} for d in m.DIRECTIONS})
    wh = FakeWarehouse(rows, registry=odd)
    m.run_cycle(cfg, wh, None, now=when, use_layer_a=False)
    assert all(r["decision"] == "persistence" for r in wh.appended[m.TABLE_FORECASTS])


def test_edge_cases_raise_clear_errors(rows):
    with pytest.raises(m.ConfigError):
        m.MixConfig(base=fcm.Config(horizon_steps=2)).validate()
    with pytest.raises(m.ConfigError):
        m.MixConfig(min_xgb_rows=10).validate()
    with pytest.raises(ValueError):
        m.XGBMember.fit(np.zeros((3, 2)), np.zeros(3), m.MAPS_FEATURES)
    with pytest.raises(m.ConfigError):
        m.backtest(fast_cfg(), FakeWarehouse(rows), start=dt.date(2026, 8, 13), end=dt.date(2026, 8, 12), now=NOW)
    with pytest.raises(m.ConfigError):
        m.plan_window(fast_cfg(), epoch_day(dt.date(2026, 9, 1)), epoch_day(dt.date(2026, 9, 2)), NOW)
    with pytest.raises(m.ConfigError):  # the window ends before the history start
        m.plan_window(m.MixConfig(), epoch_day(dt.date(2026, 8, 1)), epoch_day(dt.date(2026, 8, 2)), NOW)
    with pytest.raises(m.ConfigError):  # nothing to score: no bins on those days
        m.backtest(fast_cfg(), FakeWarehouse(rows), start=dt.date(2026, 8, 20), end=dt.date(2026, 8, 21),
                   now=dt.datetime(2026, 8, 25, tzinfo=SGT))
    data = _data(fast_cfg(), rows, dt.date(2026, 8, 12), dt.date(2026, 8, 12))
    with pytest.raises(m.ConfigError):  # too few XGBoost rows
        m.fit_day(dataclasses.replace(fast_cfg(), min_xgb_rows=10 ** 6), data, epoch_day(dt.date(2026, 8, 12)))
    assert m.served_value({}, "mix") == ("persistence", None)
    member = m.XGBMember.fit(np.random.default_rng(0).normal(size=(200, len(m.MAPS_FEATURES))), np.arange(200.0), m.MAPS_FEATURES)
    assert member.predict(np.empty((0, len(m.MAPS_FEATURES)))).shape == (0,)
    stored = member.to_json()
    stored["features"] = list(m.MAPS_FEATURES[:-1])
    with pytest.raises(ValueError):
        m.XGBMember.from_json(stored, m.MAPS_FEATURES[:-1])
    converted = m._jsonable({"a": np.array([1.5, np.nan]), "b": np.int64(3), "c": dt.date(2026, 8, 1)})
    assert converted == {"a": [1.5, None], "b": 3, "c": "2026-08-01"}


def test_main_train_registers_and_monitor_runs(monkeypatch, capsys, rows):
    for key, value in (("PROTECTED_WINDOW", ""), ("MLP_HIDDEN", "16"), ("MLP_ENSEMBLE_SIZE", "1"),
                       ("MIX_HISTORY_START", DATA_START.isoformat())):
        monkeypatch.setenv(key, value)
    wh = FakeWarehouse(rows)
    argv = ["train", "--end", "2026-08-13", "--lookback-days", "7", "--validation-days", "3", "--register", "--write-eval"]
    code = m.main(argv, warehouse_factory=lambda cfg: wh, now=NOW)
    assert code == m.EXIT_OK
    registered = wh.appended[m.TABLE_REGISTRY][0]
    assert registered["model"] == m.MODEL_NAME and registered["validation_start"] == dt.date(2026, 8, 11)
    assert json.loads(registered["config_json"])["lookback_days"] == 7
    assert {r["split"] for r in wh.appended[m.TABLE_EVALUATION]} == {"validation"}
    assert m.main(["ensure-tables"], warehouse_factory=lambda cfg: wh, now=NOW) == m.EXIT_OK
    assert m.main(["monitor", "--days", "3"], warehouse_factory=lambda cfg: wh, now=NOW) == m.EXIT_OK
    assert m.main(["train", "--lookback-days", "2"], warehouse_factory=lambda cfg: wh, now=NOW) == m.EXIT_CONFIG
    capsys.readouterr()


def test_run_cycle_still_serves_a_model_registered_before_a_long_pause(trained, rows):
    """The September model must outlive the protected window, when the daily job is refused by design."""
    cfg = fast_cfg()
    old = {**_registry(cfg, trained, served="mix"), "created_at": dt.datetime(2026, 8, 1, tzinfo=UTC)}
    wh = FakeWarehouse(rows, registry=old)
    summary = m.run_cycle(cfg, wh, None, now=dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT), use_layer_a=False)
    assert all(r["served_model"] == "mix" for r in wh.appended[m.TABLE_FORECASTS])
    registry_reads = [q for q in wh.queries if m.TABLE_REGISTRY in q]
    assert len(registry_reads) == 2 and wh.registry_params["since"] == dt.datetime(1970, 1, 1, tzinfo=UTC)
    assert summary["registry_run_id"] == "train-test"
    recent = FakeWarehouse(rows, registry=_registry(cfg, trained, served="mix"))
    m.run_cycle(cfg, recent, None, now=dt.datetime(2026, 8, 14, 18, 5, tzinfo=SGT), use_layer_a=False)
    assert len([q for q in recent.queries if m.TABLE_REGISTRY in q]) == 1  # recent partitions only
    assert recent.registry_params["since"] == dt.datetime(2026, 8, 7, 18, 5, tzinfo=SGT)
