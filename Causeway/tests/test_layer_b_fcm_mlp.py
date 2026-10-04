"""Offline tests for layer_b_fcm_mlp.py (no network, no GCP). Run from Causeway/: python -m pytest"""
import dataclasses
import datetime as dt
import json
import sys
import math

import pytest
import requests

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.neural_network import MLPRegressor  # noqa: E402

import layer_b_fcm_mlp as m  # noqa: E402 - after the optional-dependency checks

UTC = dt.timezone.utc
README_PAYLOAD = {  # camdetect/README.md format=json example, with directions available
    "camera_id": "2701",
    "date_time": "2025-12-01T07:36:21",
    "source_image": "https://images.data.gov.sg/x.jpg",
    "min_confidence": 0.1,
    "vehicle_count": 10,
    "predictions": [],
    "directions": {
        "available": True,
        "sg_my": {"count": 4, "congestion": "Quarter Way"},
        "my_sg": {"count": 4, "congestion": "Half Way"},
        "unknown": {"count": 2},
    },
}
FAST = dict(ensemble_size=2, max_epochs=30, patience=6, mlp_hidden=(16,), mlp_alphas=(1e-3,), fcm_restarts=2,
            bootstrap_reps=300, batch_size=64)


# ---------------------------------------------------------------- helpers
def synthetic_series(grid, seed=0):
    """Daily peaks (evening SG->MY, morning MY->SG), weaker at weekends, plus AR(1) noise."""
    rng = np.random.default_rng(seed)
    t = grid.start_unix + np.arange(grid.n) * 600
    hour = (t + 28800) % 86400 / 3600.0
    dow = ((t + 28800) // 86400 + 3) % 7
    scale = np.where(dow >= 5, 0.4, 1.0)
    out = {}
    for d, centre, amp in (("SG_TO_MY", 18.5, 25.0), ("MY_TO_SG", 7.5, 30.0)):
        noise = np.zeros(grid.n)
        shocks = rng.normal(0.0, 0.8, grid.n)
        for j in range(1, grid.n):
            noise[j] = 0.8 * noise[j - 1] + shocks[j]
        out[d] = 12.0 + amp * scale * np.exp(-0.5 * ((hour - centre) / 1.3) ** 2) + noise
    return out


def rows_from_series(grid, series):
    return [{"direction": d, "bin_ts": grid.time(i), "dur_min": float(v)}
            for d, values in series.items() for i, v in enumerate(values) if math.isfinite(v)]


class FakeWarehouse:
    """Answers the module's four queries from memory and records writes."""

    def __init__(self, bins=(), registry=None, layer_a_rows=(), existing=(), fail=()):
        self.bins, self.registry = list(bins), registry
        self.layer_a_rows, self.existing, self.fail = list(layer_a_rows), list(existing), set(fail)
        self.appended, self.ensured, self.queries = {}, 0, []

    def query(self, sql, params=None, timeout_sec=600.0):
        self.queries.append((sql, params))
        if "SELECT direction, bin_ts, dur_min" in sql:
            if "bins" in self.fail:
                raise RuntimeError("bins down")
            return [r for r in self.bins if params["start"] <= r["bin_ts"] <= params["end"] and r["direction"] in params["directions"]]
        if m.TABLE_REGISTRY in sql:
            if "registry" in self.fail:
                raise RuntimeError("registry table missing")
            return [self.registry] if self.registry else []
        if "ROW_NUMBER()" in sql:
            if "layer_a" in self.fail:
                raise RuntimeError("layer A table missing")
            return [r for r in self.layer_a_rows if params["start"] <= r["bin_ts"] <= params["end"]]
        if "SELECT DISTINCT bin_ts" in sql:
            return [{"bin_ts": b} for b in self.existing if params["start"] <= b <= params["end"]]
        raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def ensure_tables(self):
        self.ensured += 1
        return list(m.SCHEMAS)

    def append(self, name, rows):
        self.appended.setdefault(name, []).extend(rows)
        json.dumps([{c: m._jsonable(r.get(c)) for c, _, _ in m.SCHEMAS[name]} for r in rows])  # must serialise
        return len(rows)


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params), timeout))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def cfg_with_url(**kw):
    return m.Config(layer_a_url="https://swiftbackend.example.run.app", **kw)


@pytest.fixture(scope="module")
def trained():
    """One small end-to-end training run on synthetic data, shared by several tests."""
    cfg = m.Config(**FAST)
    first, last = m.sgt_day_bounds(dt.date(2026, 9, 6), dt.date(2026, 9, 27))
    grid = m.Grid.spanning(m.history_start(cfg, first), last + dt.timedelta(days=1))
    series = synthetic_series(grid)
    result = m.train(cfg, grid, series, {}, first=first, last=last, validation_days=4, test_days=4)
    return cfg, grid, series, first, last, result


# ---------------------------------------------------------------- config and time
def test_config_defaults_valid():
    cfg = m.Config.from_env({})
    assert cfg.view == "swiftborder.traffic_prediction.v_bins_10min"
    assert cfg.table(m.TABLE_FORECASTS) == "swiftborder.traffic_prediction.layer_b_fcm_forecasts"
    assert cfg.fcm_clusters == 3 and cfg.mlp_hidden == (32, 16)


def test_config_reads_model_settings():
    cfg = m.Config.from_env({"MLP_HIDDEN": "64, 32, 8", "MLP_ENSEMBLE_SIZE": "3", "FCM_CLUSTERS": "4",
                             "FCM_FUZZIFIER": "1.5", "NONWORK_DAYS": "2026-09-16, 2026-10-20"})
    assert cfg.mlp_hidden == (64, 32, 8) and cfg.ensemble_size == 3 and cfg.fcm_clusters == 4
    assert cfg.nonwork_days == ("2026-09-16", "2026-10-20")
    assert cfg.nonwork_epoch_days == [m.date_to_epoch_day(dt.date(2026, 9, 16)), m.date_to_epoch_day(dt.date(2026, 10, 20))]


@pytest.mark.parametrize("env", [
    {"FCM_CLUSTERS": "7"},
    {"FCM_FUZZIFIER": "1.0"},
    {"MLP_HIDDEN": "0"},
    {"MLP_HIDDEN": "a,b"},
    {"MLP_ENSEMBLE_SIZE": "0"},
    {"NONWORK_DAYS": "2026-13-01"},
    {"SWIFTBORDER_PROJECT": "Bad Project"},
    {"LAYER_B_DATASET": "x;DROP"},
    {"LAYER_A_URL": "http://insecure"},
    {"BINS_VIEW": "a.b"},
    {"PROTECTED_WINDOW": "2026-10-19..2026-10-01"},
])
def test_config_rejects_bad_env(env):
    with pytest.raises(m.ConfigError):
        m.Config.from_env(env)


def test_bins_days_and_grid():
    t = dt.datetime(2026, 10, 4, 1, 7, 33, tzinfo=UTC)
    assert m.floor_bin(t) == dt.datetime(2026, 10, 4, 1, 0, tzinfo=UTC)
    assert m.last_complete_bin(t) == dt.datetime(2026, 10, 4, 0, 50, tzinfo=UTC)
    first, last = m.sgt_day_bounds(dt.date(2026, 9, 13), dt.date(2026, 9, 30))
    assert first == dt.datetime(2026, 9, 12, 16, 0, tzinfo=UTC) and last == dt.datetime(2026, 9, 30, 15, 50, tzinfo=UTC)
    grid = m.Grid.spanning(first, last)
    assert grid.n == 18 * 144 and grid.time(grid.index(last)) == last
    with pytest.raises(ValueError):
        grid.index(first + dt.timedelta(minutes=5))
    with pytest.raises(ValueError):
        m.floor_bin(dt.datetime(2026, 10, 4))  # naive datetimes are refused
    assert m.epoch_day_to_date(m.sgt_epoch_day(m.unix(first))) == dt.date(2026, 9, 13)


def test_protected_window_guard():
    cfg = m.Config()
    m.check_protected_window(cfg, *m.sgt_day_bounds(dt.date(2026, 9, 6), dt.date(2026, 9, 30)), allow=False)
    bad = m.sgt_day_bounds(dt.date(2026, 9, 25), dt.date(2026, 10, 2))
    with pytest.raises(m.ConfigError):
        m.check_protected_window(cfg, *bad, allow=False)
    m.check_protected_window(cfg, *bad, allow=True)


# ---------------------------------------------------------------- Layer A (shared contract)
def test_layer_a_schema_matches_timesfm_module():
    timesfm = pytest.importorskip("layer_b_timesfm")
    assert m.SCHEMAS[m.TABLE_LAYER_A] == timesfm.SCHEMAS[timesfm.TABLE_LAYER_A]
    assert m.TABLE_LAYER_A == timesfm.TABLE_LAYER_A


def test_parse_readme_payload():
    obs = m.parse_layer_a_payload(README_PAYLOAD, "2701")
    assert obs.obs_ts == dt.datetime(2025, 11, 30, 23, 36, 21, tzinfo=UTC)  # SGT -> UTC
    rows = obs.to_rows(dt.datetime(2026, 1, 1, tzinfo=UTC))
    assert [r["direction"] for r in rows] == ["SG_TO_MY", "MY_TO_SG"]
    assert rows[0]["congestion_ordinal"] == 1 and rows[1]["other_direction_count"] == 4
    assert set(rows[0]) == {c for c, _, _ in m.SCHEMAS[m.TABLE_LAYER_A]}


@pytest.mark.parametrize("mutate", [
    lambda p: p["directions"].update(available=False),
    lambda p: p["directions"]["sg_my"].update(count=-1),
    lambda p: p.update(vehicle_count=11),
    lambda p: p.update(camera_id="2702"),
    lambda p: p.update(date_time="garbage"),
])
def test_parse_rejects_bad_payloads(mutate):
    payload = json.loads(json.dumps(README_PAYLOAD))
    mutate(payload)
    with pytest.raises(m.LayerAError):
        m.parse_layer_a_payload(payload, "2701")


def test_fetch_retry_policy():
    s = FakeSession([requests.ConnectionError(), FakeResp(503), FakeResp(200, README_PAYLOAD)])
    waits = []
    obs = m.fetch_layer_a(cfg_with_url(), s, when=dt.datetime(2026, 9, 13, 1, 5, tzinfo=UTC), sleep=waits.append)
    assert obs.total_count == 10 and waits == [2.0, 4.0]
    assert s.calls[0][1]["date_time"] == "2026-09-13T09:05:00"
    for failure in (requests.ReadTimeout(), FakeResp(502)):
        s = FakeSession([failure, FakeResp(200, README_PAYLOAD)])
        with pytest.raises(m.LayerAError):
            m.fetch_layer_a(cfg_with_url(), s, sleep=lambda _: None)
        assert len(s.calls) == 1  # possibly billed by Roboflow: never retried
    with pytest.raises(m.LayerANoFrame):
        m.fetch_layer_a(cfg_with_url(), FakeSession([FakeResp(404)]), sleep=lambda _: None)


def test_ingest_is_idempotent_and_backfill_defaults_to_dry_run():
    obs_bin = dt.datetime(2025, 11, 30, 23, 30, tzinfo=UTC)
    wh = FakeWarehouse(existing=[obs_bin])
    assert m.ingest_layer_a(cfg_with_url(), wh, FakeSession([FakeResp(200, README_PAYLOAD)]))["status"] == "already_logged"
    wh = FakeWarehouse()
    assert m.ingest_layer_a(cfg_with_url(), wh, FakeSession([FakeResp(200, README_PAYLOAD)]))["status"] == "logged"
    assert len(wh.appended[m.TABLE_LAYER_A]) == 2
    session = FakeSession([])
    plan = m.backfill_layer_a(cfg_with_url(), FakeWarehouse(), session, dt.date(2026, 9, 13), dt.date(2026, 9, 13),
                              every_minutes=60, max_calls=5, execute=False, now=dt.datetime(2026, 10, 4, tzinfo=UTC))
    assert plan == {"wanted": 24, "already_logged": 0, "planned_calls": 5, "execute": False} and not session.calls


# ---------------------------------------------------------------- features
def test_ffill_limited():
    x = np.array([np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, 2.0, np.nan])
    out = m.ffill_limited(x, 3)
    assert np.isnan(out[0]) and list(out[1:5]) == [1.0, 1.0, 1.0, 1.0]
    assert np.isnan(out[5]) and list(out[6:]) == [2.0, 2.0]


def test_build_frame_values():
    cfg = m.Config(nonwork_days=("2026-09-18",))
    origin = dt.datetime(2026, 9, 18, 0, 0, tzinfo=UTC)  # Friday 08:00 SGT, declared non-working
    grid = m.Grid.spanning(m.history_start(cfg, origin), origin + dt.timedelta(hours=2))
    raw = 10.0 + 0.01 * np.arange(grid.n)
    i = grid.index(origin)
    fr = m.build_frame(cfg, grid, raw, "SG_TO_MY", [i], label_cutoff=origin + dt.timedelta(minutes=40))
    x = dict(zip(m.BASE_FEATURES, fr.X[0]))
    assert fr.complete[0] and fr.now[0] == pytest.approx(raw[i])
    assert [x[f"delta_{k}"] for k in (10, 20, 30, 60)] == pytest.approx([0.01, 0.02, 0.03, 0.06])
    assert x["roll_mean_60_minus_now"] == pytest.approx(-0.025)
    assert x["roll_std_60"] == pytest.approx(0.01 * math.sqrt(35 / 12))
    assert (x["d1_gap"], x["d1_rise_mid"], x["d1_rise_end"], x["d1_missing"]) == pytest.approx((1.44, 0.03, 0.06, 0.0))
    assert (x["d7_gap"], x["d7_rise_end"], x["d7_missing"]) == pytest.approx((10.08, 0.06, 0.0))
    assert (x["tod_sin"], x["tod_cos"]) == pytest.approx((math.sin(2 * math.pi / 3), math.cos(2 * math.pi / 3)))
    assert x["dow_sin"] == pytest.approx(math.sin(2 * math.pi * 4 / 7)) and x["nonworkday"] == 1.0
    assert fr.actual[0, :4] == pytest.approx(raw[i + 1:i + 5]) and np.isnan(fr.actual[0, 4:]).all()  # cutoff at +40 min
    assert fr.seasonal_d1[0, 0] == pytest.approx(raw[i + 1 - 144])
    assert fr.seasonal_tod7[0, 0] == pytest.approx(raw[i + 1] - 5.76)
    assert m.epoch_day_to_date(fr.origin_day[0]) == dt.date(2026, 9, 18)


def test_build_frame_is_causal_and_handles_gaps():
    cfg = m.Config()
    origin = dt.datetime(2026, 9, 20, 4, 0, tzinfo=UTC)
    grid = m.Grid.spanning(m.history_start(cfg, origin), origin + dt.timedelta(hours=2))
    raw = 15.0 + np.sin(np.arange(grid.n) / 7.0)
    i = grid.index(origin)
    base = m.build_frame(cfg, grid, raw, "MY_TO_SG", [i], label_cutoff=grid.time(grid.n - 1))
    future = raw.copy()
    future[i + 1:] += 50.0
    changed = m.build_frame(cfg, grid, future, "MY_TO_SG", [i], label_cutoff=grid.time(grid.n - 1))
    assert np.array_equal(base.X, changed.X)            # features never read the future
    assert not np.allclose(base.actual, changed.actual)  # labels do
    gap = raw.copy()
    gap[i - 5:i - 1] = np.nan                            # four missing bins exceed the 3-bin fill limit
    assert not m.build_frame(cfg, grid, gap, "MY_TO_SG", [i], label_cutoff=origin).complete[0]
    stale = raw.copy()
    stale[i] = np.nan                                    # the origin bin itself must be observed
    assert not m.build_frame(cfg, grid, stale, "MY_TO_SG", [i], label_cutoff=origin).complete[0]
    no_week = raw.copy()
    no_week[: i - 144 * 6] = np.nan                      # D-7 missing: flagged, row still usable
    fr = m.build_frame(cfg, grid, no_week, "MY_TO_SG", [i], label_cutoff=origin)
    x = dict(zip(m.BASE_FEATURES, fr.X[0]))
    assert fr.complete[0] and x["d7_missing"] == 1.0 and x["d7_gap"] == 0.0


def test_series_from_rows_ignores_off_grid_and_bad_values():
    grid = m.Grid.spanning(dt.datetime(2026, 9, 1, tzinfo=UTC), dt.datetime(2026, 9, 1, 1, tzinfo=UTC))
    rows = [
        {"direction": "SG_TO_MY", "bin_ts": dt.datetime(2026, 9, 1, 0, 10, tzinfo=UTC), "dur_min": 12.5},
        {"direction": "SG_TO_MY", "bin_ts": dt.datetime(2026, 9, 1, 0, 15, tzinfo=UTC), "dur_min": 99.0},
        {"direction": "MY_TO_SG", "bin_ts": dt.datetime(2026, 9, 1, 0, 20, tzinfo=UTC), "dur_min": -1.0},
        {"direction": "OTHER", "bin_ts": dt.datetime(2026, 9, 1, 0, 20, tzinfo=UTC), "dur_min": 5.0},
    ]
    s = m.series_from_rows(rows, grid)
    assert s["SG_TO_MY"][1] == 12.5 and np.isnan(s["SG_TO_MY"]).sum() == grid.n - 1
    assert np.isnan(s["MY_TO_SG"]).all()


# ---------------------------------------------------------------- Fuzzy C-Means
def test_fcm_recovers_regimes_in_travel_time_order():
    rng = np.random.default_rng(1)
    truth = np.array([[40.0, 5.0, 2.0], [12.0, 0.0, 0.0], [25.0, -3.0, 1.0]])
    x = np.vstack([c + rng.normal(0, [1.5, 1.0, 0.5], (200, 3)) for c in truth])
    fcm = m.FuzzyCMeans.fit(x, features=m.FCM_FEATURES, c=3, m=2.0, seed=7, restarts=3)
    centres = fcm.centres_in_units()
    assert fcm.labels == ("light", "moderate", "heavy")
    assert centres[:, 0] == pytest.approx([12.0, 25.0, 40.0], abs=0.6)
    u = fcm.memberships(x)
    assert np.allclose(u.sum(axis=1), 1.0)
    assert (u[:200].argmax(axis=1) == 2).mean() > 0.97 and (u[200:400].argmax(axis=1) == 0).mean() > 0.97
    validity = fcm.validity(x)
    assert validity["partition_coefficient"] > 0.8 and validity["xie_beni"] < 0.2
    assert sum(validity["crisp_sizes"].values()) == 600


def test_fcm_membership_formula_and_edge_cases():
    fcm = m.FuzzyCMeans(("now", "delta_30", "d1_rise_end"), ("light", "heavy"), 2.0, np.zeros(3), np.ones(3),
                        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), 10, 0.0, 1)
    u = fcm.memberships(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]))
    assert u[0] == pytest.approx([1 / (1 + 1 / 4), 1 / (1 + 4)])  # d^2 = 1 and 4, m = 2
    assert u[1, 0] == pytest.approx(1.0)                          # on a centre
    assert np.isnan(u[2]).all()                                   # missing input
    assert fcm.level_memberships(np.array([0.0, 3.0]))[:, 0] == pytest.approx([1.0, 0.0])
    assert fcm.describe()["heavy"]["now"] == 3.0


def test_fcm_json_roundtrip_and_validation():
    rng = np.random.default_rng(2)
    fcm = m.FuzzyCMeans.fit(rng.normal(size=(90, 3)), features=m.FCM_FEATURES, c=3, m=2.0, seed=1, restarts=1)
    back = m.FuzzyCMeans.from_json(json.loads(json.dumps(fcm.to_json())))
    x = rng.normal(size=(5, 3))
    assert np.array_equal(back.memberships(x), fcm.memberships(x))
    data = fcm.to_json()
    data["centres"] = data["centres"][:2]
    with pytest.raises(ValueError):
        m.FuzzyCMeans.from_json(data)
    with pytest.raises(ValueError):
        m.FuzzyCMeans.fit(rng.normal(size=(5, 3)), features=m.FCM_FEATURES, c=3, m=2.0, seed=1)


# ---------------------------------------------------------------- MLP
@pytest.mark.filterwarnings("ignore::UserWarning")  # ConvergenceWarning from the deliberately short fit
def test_forward_pass_matches_scikit_learn():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(200, 5)), rng.normal(size=(200, 3))
    net = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=50, random_state=0).fit(x, y)
    assert np.allclose(m._forward(x, net.coefs_, net.intercepts_), net.predict(x))


@pytest.mark.parametrize("horizon", [1, 3])
def test_mlp_ensemble_learns_and_roundtrips_exactly(horizon):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(600, 4))
    y = np.column_stack([np.sin(x[:, 0]) + 0.5 * x[:, 1] * k for k in range(1, horizon + 1)])
    cfg = m.Config(**{**FAST, "max_epochs": 120, "patience": 15})
    ens, inner_mae = m.fit_mlp_ensemble(cfg, ("a", "b", "c", "d"), x_fit=x[:500], y_fit=y[:500], x_inner=x[500:],
                                        y_inner=y[500:], now_fit=np.full(500, 10.0), alpha=1e-3, seed=0)
    assert inner_mae < 0.5 * np.mean(np.abs(y[500:] - y[:500].mean(axis=0)))  # far better than the mean
    back = m.MLPEnsemble.from_json(json.loads(json.dumps(ens.to_json())))
    now = np.full(100, 20.0)
    assert np.array_equal(back.predict(x[500:], now)[0], ens.predict(x[500:], now)[0])  # stored == evaluated
    mean, spread = ens.predict(x[500:], now)
    assert mean.shape == (100, horizon) and (spread >= 0).all() and (mean >= ens.floor).all()
    assert len(ens.best_epochs) == 2 and all(1 <= e <= 120 for e in ens.best_epochs)


def test_mlp_ensemble_rejects_inconsistent_artefacts(trained):
    data = trained[5]["artefacts"]["SG_TO_MY"]["fcm_mlp"]
    broken = json.loads(json.dumps(data))
    broken["members"][0]["weights"][0] = broken["members"][0]["weights"][0][:-1]
    with pytest.raises(ValueError):
        m.MLPEnsemble.from_json(broken)
    broken = json.loads(json.dumps(data))
    broken["x_scale"][0] = 0.0
    with pytest.raises(ValueError):
        m.MLPEnsemble.from_json(broken)


# ---------------------------------------------------------------- splits and statistics
def test_split_days_and_ranks():
    days = list(range(100, 120))
    s = m.split_days(days, validation_days=5, test_days=5, inner_days=2)
    assert s.fit_days == tuple(range(100, 108)) and s.inner_days == (108, 109)
    assert s.validation_days == tuple(range(110, 115)) and s.test_days == tuple(range(115, 120))
    assert list(s.rank(np.array([100, 108, 110, 119]))) == [0, 1, 2, 3]
    assert list(m.split_days(days, validation_days=5, test_days=0, inner_days=2).rank(np.array([119]))) == [2]
    for kwargs in ({"validation_days": 2, "test_days": 0}, {"validation_days": 5, "test_days": 2},
                   {"validation_days": 8, "test_days": 8}):
        with pytest.raises(m.ConfigError):
            m.split_days(days, inner_days=2, **kwargs)


def test_labels_never_cross_a_split_boundary():
    cfg = m.Config()
    first, last = m.sgt_day_bounds(dt.date(2026, 9, 10), dt.date(2026, 9, 20))
    grid = m.Grid.spanning(m.history_start(cfg, first), last)
    raw = np.full(grid.n, 15.0)
    fr = m.build_frame(cfg, grid, raw, "SG_TO_MY", np.arange(grid.index(first), grid.index(last) + 1), label_cutoff=last)
    split = m.split_days(sorted(set(fr.origin_day.tolist())), validation_days=3, test_days=3, inner_days=2)
    labels = m.boundary_masked_labels(fr, split)
    o, t = split.rank(fr.origin_day), split.rank(fr.target_day())
    assert np.isnan(labels[t > o[:, None]]).all()
    assert np.isfinite(labels[(t == o[:, None]) & np.isfinite(fr.actual)]).all()
    assert np.isnan(labels[-1]).all()  # last origin: every target lies after the cutoff


def test_paired_day_bootstrap_and_gate():
    day = dt.date(2026, 9, 1)
    pairs = [(day + dt.timedelta(days=d), -1.0 + 0.1 * (k % 3)) for d in range(6) for k in range(20)]
    res = m.paired_day_bootstrap(pairs, reps=500, seed=1)
    assert res["days"] == 6 and res["ci_high"] < 0 and m.better(res)
    assert not m.better({**res, "days": 2})
    assert m.paired_day_bootstrap([], 100, 1)["n"] == 0


def _val_rows(days, fcm_err, vision_err=None, vision_share=1.0):
    rows = []
    for d in range(days):
        for k in range(40):
            origin = dt.datetime(2026, 9, 1, tzinfo=UTC) + dt.timedelta(days=d, minutes=10 * k)
            r = {"direction": "SG_TO_MY", "step": 1, "origin_ts": origin, "target_ts": origin + m.BIN, "actual": 20.0,
                 "persistence": 23.0, "mlp": 21.5, "fcm_mlp": 20.0 + fcm_err + 0.01 * k}
            if vision_err is not None and k < 40 * vision_share:
                r.update(fcm_mlp_layer_a=20.0 + vision_err, fcm_mlp_same_rows=20.0 + fcm_err + 0.01 * k)
            rows.append(r)
    return rows


def test_decision_rules():
    cfg = m.Config(horizon_steps=1, bootstrap_reps=300)
    d = m.decide(cfg, _val_rows(5, fcm_err=1.0))["SG_TO_MY"]["1"]
    assert d["served"] == "fcm_mlp" and better_than(d, "fcm_mlp_vs_mlp")
    assert m.decide(cfg, _val_rows(5, fcm_err=4.0))["SG_TO_MY"]["1"]["served"] == "persistence"
    assert m.decide(cfg, _val_rows(2, fcm_err=1.0))["SG_TO_MY"]["1"]["served"] == "persistence"  # too few days
    assert m.decide(cfg, _val_rows(5, fcm_err=1.0, vision_err=0.1))["SG_TO_MY"]["1"]["served"] == "fcm_mlp_layer_a"
    good = m.decide(cfg, _val_rows(5, fcm_err=1.0, vision_err=0.1))["SG_TO_MY"]["1"]
    assert good["fallback"] == "fcm_mlp"
    low = m.decide(cfg, _val_rows(5, fcm_err=1.0, vision_err=0.1, vision_share=0.3))["SG_TO_MY"]["1"]
    assert low["served"] == "fcm_mlp" and low["layer_a_coverage"] == pytest.approx(0.3)
    assert m.decide(cfg, [])["MY_TO_SG"]["1"]["served"] == "persistence"


def better_than(note, key):
    return m.better(note[key])


def test_served_value_chain():
    row = {"persistence": 20.0, "fcm_mlp": 18.0, "fcm_mlp_layer_a": None}
    assert m.served_value(row, "fcm_mlp_layer_a", "fcm_mlp") == ("fcm_mlp", 18.0)
    assert m.served_value(row, "fcm_mlp_layer_a", "persistence") == ("persistence", 20.0)
    assert m.served_value(row, "fcm_mlp_layer_a") == ("persistence", 20.0)  # no stored fallback: conservative
    assert m.served_value({**row, "fcm_mlp_layer_a": 17.0}, "fcm_mlp_layer_a", "persistence") == ("fcm_mlp_layer_a", 17.0)
    assert m.served_value({**row, "fcm_mlp": None}, "fcm_mlp") == ("persistence", 20.0)
    assert m.served_value(row, "something_else") == ("persistence", 20.0)
    assert m.served_value({}, "persistence") == ("persistence", None)


# ---------------------------------------------------------------- end-to-end training
def test_train_end_to_end(trained):
    cfg, _, _, _, last, result = trained
    assert result["train_start"] == dt.date(2026, 9, 6) and result["test_end"] == dt.date(2026, 9, 27)
    assert result["validation_start"] == dt.date(2026, 9, 20) and result["test_start"] == dt.date(2026, 9, 24)
    assert result["inner_validation_start"] == dt.date(2026, 9, 18) and result["train_end"] == dt.date(2026, 9, 19)
    rows = result["rows"]
    assert {r["split"] for r in rows} == {"validation", "test"}
    assert max(r["target_ts"] for r in rows) <= last                          # nothing after the cutoff
    test_start = dt.datetime(2026, 9, 23, 16, 0, tzinfo=UTC)
    assert all(r["target_ts"] < test_start for r in rows if r["split"] == "validation")
    for d in m.DIRECTIONS:
        diag = result["diagnostics"][d]
        assert list(diag["fcm"]["centres"]) == ["light", "moderate", "heavy"]
        levels = [diag["fcm"]["centres"][k]["now"] for k in ("light", "moderate", "heavy")]
        assert levels == sorted(levels)
        assert diag["rows"]["fit"] >= m.MIN_TRAIN_ROWS and diag["rows"]["with_layer_a"] == 0
        for step in range(1, cfg.horizon_steps + 1):
            note = result["decisions"][d][str(step)]
            assert note["served"] in ("fcm_mlp", "persistence") and note["layer_a_coverage"] == 0.0
    # the synthetic peaks are learnable an hour ahead: the network must beat persistence there
    v6 = result["metrics"]["validation"]["step_6"]
    assert v6["fcm_mlp"]["all"]["mae"] < v6["persistence"]["all"]["mae"]
    assert result["decisions"]["MY_TO_SG"]["6"]["served"] == "fcm_mlp"
    assert result["metrics"]["test"]["step_6"]["served_value"]["all"]["n"] > 0
    assert set(result["metrics"]["test_pooled"]) == {"fcm_mlp_vs_persistence", "fcm_mlp_vs_mlp", "served_value_vs_persistence"}


def test_registry_row_round_trips_into_serving_models(trained):
    cfg, _, _, _, _, result = trained
    row = m.registry_row(cfg, result, "train-x", dt.datetime(2026, 9, 28, tzinfo=UTC))
    assert set(row) == {c for c, _, _ in m.SCHEMAS[m.TABLE_REGISTRY]}
    for d in m.DIRECTIONS:
        models = m.DirectionModels.from_json(json.loads(row["artefacts_json"])[d], cfg.horizon_steps)
        assert models.fcm_mlp_layer_a is None and models.fcm.labels == ("light", "moderate", "heavy")
    assert json.loads(row["decisions_json"])["SG_TO_MY"]["1"]["served"] in ("fcm_mlp", "persistence")
    assert json.loads(row["config_json"])["base_features"] == list(m.BASE_FEATURES)
    with pytest.raises(ValueError):
        m.DirectionModels.from_json(json.loads(row["artefacts_json"])["SG_TO_MY"], cfg.horizon_steps + 1)
    eval_rows = m.evaluation_table_rows(result["rows"][:5], "train-x", dt.datetime(2026, 9, 28, tzinfo=UTC))
    FakeWarehouse().append(m.TABLE_EVALUATION, eval_rows)


def test_train_with_layer_a_fits_the_vision_networks():
    cfg = m.Config(**FAST)
    first, last = m.sgt_day_bounds(dt.date(2026, 9, 6), dt.date(2026, 9, 22))
    grid = m.Grid.spanning(m.history_start(cfg, first), last)
    series = synthetic_series(grid, seed=5)
    layer_a = {}
    for i in range(grid.index(first), grid.n):
        ts = grid.time(i)
        for d, other in (("SG_TO_MY", "MY_TO_SG"), ("MY_TO_SG", "SG_TO_MY")):
            layer_a[(d, ts)] = {"vehicle_count": int(series[d][i]), "other_direction_count": int(series[other][i]),
                                "congestion_ordinal": 1, "rejected": False}
    result = m.train(cfg, grid, series, layer_a, first=first, last=last, validation_days=3, test_days=3)
    for d in m.DIRECTIONS:
        assert result["artefacts"][d]["fcm_mlp_layer_a"] is not None
        note = result["decisions"][d]["6"]
        assert note["layer_a_coverage"] == pytest.approx(1.0) and "layer_a_vs_no_vision" in note
        assert note["served"] in m.SERVABLE


def test_train_reports_too_little_data():
    cfg = m.Config(**FAST)
    first, last = m.sgt_day_bounds(dt.date(2026, 9, 6), dt.date(2026, 9, 14))
    grid = m.Grid.spanning(m.history_start(cfg, first), last)
    with pytest.raises(m.ConfigError):
        m.train(cfg, grid, synthetic_series(grid), {}, first=first, last=last, validation_days=3, test_days=3)


# ---------------------------------------------------------------- live cycle
def _cycle_warehouse(trained, registry=True, **kw):
    cfg, grid, series, _, _, result = trained
    reg = m.registry_row(cfg, result, "train-x", dt.datetime(2026, 9, 28, tzinfo=UTC)) if registry else None
    return FakeWarehouse(bins=rows_from_series(grid, series), registry=reg, **kw)


NOW = dt.datetime(2026, 9, 27, 16, 25, tzinfo=UTC)  # 28 Sep 00:25 SGT, just after the training window


def test_run_cycle_serves_registered_model(trained):
    cfg, _, series, _, _, result = trained
    wh = _cycle_warehouse(trained)
    summary = m.run_cycle(cfg, wh, FakeSession([]), now=NOW)
    rows = wh.appended[m.TABLE_FORECASTS]
    assert wh.ensured == 1 and len(rows) == 2 * cfg.horizon_steps and summary["registry_run_id"] == "train-x"
    assert summary["layer_a_ingest"] == {"status": "disabled"}
    for r in rows:
        decision = result["decisions"][r["direction"]][str(r["step"])]["served"]
        assert r["served_model"] == decision and r["origin_ts"] == dt.datetime(2026, 9, 27, 16, 10, tzinfo=UTC)
        assert r["fcm_mlp"] is not None and r["regime"] in ("light", "moderate", "heavy")
        assert sum(json.loads(r["memberships_json"]).values()) == pytest.approx(1.0, abs=1e-3)
        assert r["forecast_regime"] in ("light", "moderate", "heavy") and r["target_ts"] == r["origin_ts"] + r["step"] * m.BIN
        assert set(r) == {c for c, _, _ in m.SCHEMAS[m.TABLE_FORECASTS]}


def test_run_cycle_falls_back_to_persistence(trained):
    cfg = trained[0]
    for wh in (_cycle_warehouse(trained, registry=False), _cycle_warehouse(trained, fail={"registry", "layer_a"})):
        session = FakeSession([FakeResp(500)])  # Layer A down: no retry, no sleep
        m.run_cycle(dataclass_replace(cfg, layer_a_url="https://x.example.run.app"), wh, session, now=NOW)
        rows = wh.appended[m.TABLE_FORECASTS]
        assert {r["served_model"] for r in rows} == {"persistence"} and all(r["regime"] is None for r in rows)
    corrupt = _cycle_warehouse(trained)
    corrupt.registry = {**corrupt.registry, "artefacts_json": json.dumps({"SG_TO_MY": {"fcm": {}}})}
    m.run_cycle(cfg, corrupt, FakeSession([]), now=NOW)
    assert {r["served_model"] for r in corrupt.appended[m.TABLE_FORECASTS]} == {"persistence"}


def dataclass_replace(cfg, **kw):
    return dataclasses.replace(cfg, **kw)


def test_run_cycle_dry_run_and_stale_data(trained):
    cfg = trained[0]
    wh = _cycle_warehouse(trained)
    summary = m.run_cycle(cfg, wh, FakeSession([]), now=NOW, write=False)
    assert not wh.appended and wh.ensured == 0 and len(summary["forecasts"]) == 12
    with pytest.raises(m.NoForecastError):
        m.run_cycle(cfg, FakeWarehouse(), FakeSession([]), now=NOW)


# ---------------------------------------------------------------- command line
def test_main_exit_codes(monkeypatch, capsys, trained):
    assert m.main(["show-sql"]) == m.EXIT_OK
    assert "v_bins_10min" in capsys.readouterr().out
    monkeypatch.setenv("FCM_CLUSTERS", "9")
    assert m.main(["ensure-tables"], warehouse_factory=lambda c: FakeWarehouse()) == m.EXIT_CONFIG
    monkeypatch.delenv("FCM_CLUSTERS")
    factory = lambda c: FakeWarehouse()  # noqa: E731
    assert m.main(["train", "--start", "2026-09-25", "--end", "2026-10-02"], warehouse_factory=factory) == m.EXIT_CONFIG
    assert m.main(["train", "--start", "2026-09-01", "--end", "2999-01-01"], warehouse_factory=factory) == m.EXIT_CONFIG
    assert m.main(["run-cycle"], warehouse_factory=factory) == m.EXIT_NO_FORECAST
    wh = _cycle_warehouse(trained)
    monkeypatch.setattr(m, "utc_now", lambda: NOW)
    assert m.main(["run-cycle", "--dry-run"], warehouse_factory=lambda c: wh) == m.EXIT_OK


def test_main_train_registers(monkeypatch, capsys, trained):
    _, grid, series, _, _, _ = trained
    wh = FakeWarehouse(bins=rows_from_series(grid, series), fail={"layer_a"})
    monkeypatch.setattr(m.Config, "from_env", classmethod(lambda cls, env=None: cls(**{**FAST, "ensemble_size": 1})))
    monkeypatch.setattr(m, "utc_now", lambda: NOW)
    code = m.main(["train", "--lookback-days", "22", "--validation-days", "4", "--test-days", "0", "--register", "--write-eval"],
                  warehouse_factory=lambda c: wh)
    assert code == m.EXIT_OK
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["test_start"] is None and "artefacts" not in printed and "rows" not in printed
    assert len(wh.appended[m.TABLE_REGISTRY]) == 1 and wh.appended[m.TABLE_EVALUATION]
    assert {r["split"] for r in wh.appended[m.TABLE_EVALUATION]} == {"validation"}


def test_jsonable_handles_numpy():
    assert m._jsonable({"a": np.float64(1.5), "b": np.int64(2), "c": np.array([1.0, np.nan]), "d": float("inf")}) == \
        {"a": 1.5, "b": 2, "c": [1.0, None], "d": None}


def test_command_can_come_from_the_environment(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["layer_b.py"])  # Procfile started the module without arguments
    monkeypatch.setenv("LAYER_B_ARGS", "show-sql")
    assert m.main() == m.EXIT_OK and "latest registry" in capsys.readouterr().out


def test_a_model_that_failed_the_persistence_gate_is_never_served():
    """Layer A adopted, fcm_mlp worse than persistence: without a frame, persistence is served."""
    cfg = m.Config(horizon_steps=1, bootstrap_reps=300)
    note = m.decide(cfg, _val_rows(5, fcm_err=4.0, vision_err=0.1, vision_share=0.6))["SG_TO_MY"]["1"]
    assert not m.better(note["fcm_mlp_vs_persistence"])
    assert note["served"] == "fcm_mlp_layer_a" and note["fallback"] == "persistence"
    no_frame = {"persistence": 23.0, "fcm_mlp": 24.0, "fcm_mlp_layer_a": None}
    assert m.served_value(no_frame, note["served"], note["fallback"]) == ("persistence", 23.0)


def test_memberships_saturate_beyond_the_outer_centres():
    fcm = m.FuzzyCMeans(m.FCM_FEATURES, ("light", "moderate", "heavy"), 2.0, np.zeros(3), np.ones(3),
                        np.array([[15.0, 0.0, 0.0], [30.0, -3.0, -3.0], [35.0, 4.0, 2.0]]), 10, 0.0, 1)
    jam = np.array([[120.0, 4.0, 2.0], [40.0, 4.0, 2.0]])
    assert fcm.memberships(jam)[:, 2] == pytest.approx([1.0, 1.0])               # saturated: fully heavy
    assert fcm.memberships(jam, saturate=False)[0, 2] < fcm.memberships(jam, saturate=False)[1, 2]  # textbook FCM
    assert fcm.level_memberships(np.array([120.0, 10.0])).argmax(axis=1).tolist() == [2, 0]
    assert fcm.level_memberships(np.array([120.0]))[0, 2] == pytest.approx(1.0)


def test_each_epoch_uses_a_new_minibatch_order(monkeypatch):
    import sklearn.neural_network._multilayer_perceptron as mlp_module
    orders = []
    real_shuffle = mlp_module.shuffle

    def spy(*arrays, **kw):
        out = real_shuffle(*arrays, **kw)
        orders.append(np.array(out).copy())
        return out

    monkeypatch.setattr(mlp_module, "shuffle", spy)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 3)), rng.normal(size=(200, 2))
    cfg = m.Config(**{**FAST, "ensemble_size": 1, "max_epochs": 3, "patience": 10})
    m.fit_mlp_ensemble(cfg, ("a", "b", "c"), x_fit=x[:150], y_fit=y[:150], x_inner=x[150:], y_inner=y[150:],
                       now_fit=np.full(150, 10.0), alpha=1e-3, seed=0)
    assert len(orders) == 3 and not np.array_equal(orders[1], orders[2]) and not np.array_equal(orders[0], orders[1])


def test_live_ingest_checks_the_table_before_paying_for_inference():
    now = dt.datetime(2026, 10, 4, 1, 3, tzinfo=UTC)
    wh = FakeWarehouse(existing=[m.floor_bin(now)])
    assert m.ingest_layer_a(cfg_with_url(), wh, FakeSession([]), now=now) == {"status": "already_logged", "bin_ts": m.floor_bin(now)}


def test_config_drift_is_reported_but_still_served(trained, capsys):
    cfg = dataclass_replace(trained[0], nonwork_days=("2026-09-16",))
    wh = _cycle_warehouse(trained)
    capsys.readouterr()
    m.run_cycle(cfg, wh, FakeSession([]), now=NOW)
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    drift = [entry for entry in logs if entry["message"].startswith("serving configuration differs")]
    assert drift and drift[0]["keys"] == ["nonwork_days"]
    assert m.config_drift(trained[0], json.loads(wh.registry["config_json"])) == []


def test_malformed_registry_fallback_serves_persistence_without_crashing(trained):
    cfg = trained[0]
    wh = _cycle_warehouse(trained)
    slim = json.loads(wh.registry["decisions_json"])
    for steps in slim.values():
        for note in steps.values():
            note.update(served="fcm_mlp_layer_a", fallback=["fcm_mlp"])
    wh.registry = {**wh.registry, "decisions_json": json.dumps(slim)}
    m.run_cycle(cfg, wh, FakeSession([]), now=NOW)  # no Layer A model or frame: falls through to persistence
    assert {r["served_model"] for r in wh.appended[m.TABLE_FORECASTS]} == {"persistence"}
    assert m.served_value({"persistence": 1.0}, {"bad": 1}, {"bad": 2}) == ("persistence", 1.0)


def test_config_drift_ignores_the_order_of_nonwork_days():
    a = m.Config(nonwork_days=("2026-09-16", "2026-10-20"))
    stored = json.loads(json.dumps(m._jsonable(m.model_config(m.Config(nonwork_days=("2026-10-20", "2026-09-16"))))))
    assert m.config_drift(a, stored) == [] and m.config_drift(m.Config(), stored) == ["nonwork_days"]
