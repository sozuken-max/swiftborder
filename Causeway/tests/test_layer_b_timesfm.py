"""Offline tests for layer_b_timesfm.py (no network, no GCP). Run from Causeway/: python -m pytest"""
import datetime as dt
import json
import sys
import random

import pytest
import requests

import layer_b_timesfm as m


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


# ---------------------------------------------------------------- config and time
def test_config_defaults_valid():
    cfg = m.Config.from_env({})
    assert cfg.view == "swiftborder.traffic_prediction.v_bins_10min"
    assert cfg.table("layer_a_counts") == "swiftborder.traffic_prediction.layer_a_counts"


@pytest.mark.parametrize("env", [
    {"TIMESFM_CONTEXT_WINDOW": "1000"},
    {"SWIFTBORDER_PROJECT": "Bad Project"},
    {"LAYER_B_DATASET": "x;DROP"},
    {"LAYER_A_URL": "http://insecure"},
    {"BINS_VIEW": "a.b"},
    {"LAYER_A_CONFIDENCE": "abc"},
    {"PROTECTED_WINDOW": "2026-10-19..2026-10-01"},
])
def test_config_rejects_bad_env(env):
    with pytest.raises(m.ConfigError):
        m.Config.from_env(env)


def test_bins_and_days():
    t = dt.datetime(2026, 10, 4, 1, 7, 33, tzinfo=UTC)
    assert m.floor_bin(t) == dt.datetime(2026, 10, 4, 1, 0, tzinfo=UTC)
    assert m.last_complete_bin(t) == dt.datetime(2026, 10, 4, 0, 50, tzinfo=UTC)
    first, last = m.sgt_day_bounds(dt.date(2026, 9, 13), dt.date(2026, 9, 30))
    assert first == dt.datetime(2026, 9, 12, 16, 0, tzinfo=UTC)
    assert last == dt.datetime(2026, 9, 30, 15, 50, tzinfo=UTC)
    with pytest.raises(ValueError):
        m.floor_bin(dt.datetime(2026, 10, 4))  # naive datetimes are refused


def test_period_and_day_type_use_singapore_time():
    assert m.period_of(dt.datetime(2026, 9, 14, 23, 0, tzinfo=UTC)) == "morning_peak"  # 07:00 SGT
    assert m.period_of(dt.datetime(2026, 9, 14, 9, 0, tzinfo=UTC)) == "evening_peak"   # 17:00 SGT
    assert m.day_type_of(dt.datetime(2026, 9, 18, 17, 0, tzinfo=UTC)) == "weekend"     # Sat 01:00 SGT


def test_protected_window_guard():
    cfg = m.Config()
    ok = m.sgt_day_bounds(dt.date(2026, 9, 13), dt.date(2026, 9, 30))
    m.check_protected_window(cfg, *ok, allow=False)  # 13-30 Sep is allowed
    bad = m.sgt_day_bounds(dt.date(2026, 9, 25), dt.date(2026, 10, 2))
    with pytest.raises(m.ConfigError):
        m.check_protected_window(cfg, *bad, allow=False)
    m.check_protected_window(cfg, *bad, allow=True)


# ---------------------------------------------------------------- Layer A
def test_parse_readme_payload():
    obs = m.parse_layer_a_payload(README_PAYLOAD, "2701")
    assert obs.obs_ts == dt.datetime(2025, 11, 30, 23, 36, 21, tzinfo=UTC)  # SGT -> UTC
    assert obs.counts == {"SG_TO_MY": 4, "MY_TO_SG": 4}
    rows = obs.to_rows(dt.datetime(2026, 1, 1, tzinfo=UTC))
    assert [r["direction"] for r in rows] == ["SG_TO_MY", "MY_TO_SG"]
    assert rows[0]["congestion_ordinal"] == 1 and rows[1]["congestion_ordinal"] == 2
    assert rows[0]["other_direction_count"] == 4 and rows[0]["bin_ts"] == dt.datetime(2025, 11, 30, 23, 30, tzinfo=UTC)
    assert set(rows[0]) == {c for c, _, _ in m.SCHEMAS[m.TABLE_LAYER_A]}


@pytest.mark.parametrize("mutate", [
    lambda p: p["directions"].update(available=False),
    lambda p: p["directions"]["sg_my"].update(count=-1),
    lambda p: p["directions"]["sg_my"].update(count=1.5),
    lambda p: p.update(vehicle_count=11),
    lambda p: p.update(camera_id="2702"),
    lambda p: p.update(date_time="garbage"),
    lambda p: p.pop("directions"),
])
def test_parse_rejects_bad_payloads(mutate):
    payload = json.loads(json.dumps(README_PAYLOAD))
    mutate(payload)
    with pytest.raises(m.LayerAError):
        m.parse_layer_a_payload(payload, "2701")


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


def test_fetch_retries_then_succeeds_and_sends_sgt_time():
    s = FakeSession([requests.ConnectionError(), FakeResp(503), FakeResp(200, README_PAYLOAD)])
    waits = []
    obs = m.fetch_layer_a(cfg_with_url(), s, when=dt.datetime(2026, 9, 13, 1, 5, tzinfo=UTC), sleep=waits.append)
    assert obs.total_count == 10 and waits == [2.0, 4.0]
    assert s.calls[0][1] == {"camera_id": "2701", "format": "json", "confidence": "0.25", "date_time": "2026-09-13T09:05:00"}


def test_fetch_does_not_retry_possibly_billed_failures():
    for failure in (requests.ReadTimeout(), FakeResp(502), FakeResp(500), FakeResp(504)):
        s = FakeSession([failure, FakeResp(200, README_PAYLOAD)])
        with pytest.raises(m.LayerAError):
            m.fetch_layer_a(cfg_with_url(), s, sleep=lambda x: None)
        assert len(s.calls) == 1


def test_fetch_404_and_4xx_are_not_retried():
    with pytest.raises(m.LayerANoFrame):
        m.fetch_layer_a(cfg_with_url(), FakeSession([FakeResp(404)]), sleep=lambda s: None)
    with pytest.raises(m.LayerAError):
        m.fetch_layer_a(cfg_with_url(), FakeSession([FakeResp(400)]), sleep=lambda s: None)
    with pytest.raises(m.LayerAError):
        m.fetch_layer_a(cfg_with_url(), FakeSession([FakeResp(500)] * 3), sleep=lambda s: None)
    with pytest.raises(m.ConfigError):
        m.fetch_layer_a(m.Config(), FakeSession([]))


def test_vision_features_quality_gate_and_age():
    cfg = m.Config()
    o = dt.datetime(2026, 9, 14, 0, 0, tzinfo=UTC)
    rec = {"vehicle_count": 30, "other_direction_count": 5, "congestion_ordinal": 2, "rejected": False}
    la = {("SG_TO_MY", o - m.BIN): rec, ("MY_TO_SG", o): {**rec, "rejected": True}}
    feats, status = m.vision_features(cfg, la, "SG_TO_MY", o)
    assert status == "ok" and feats == {"vehicle_count": 30.0, "other_direction_count": 5.0, "congestion_ordinal": 2.0}
    assert m.vision_features(cfg, la, "MY_TO_SG", o) == (None, "rejected_quality_gate")
    assert m.vision_features(cfg, {("SG_TO_MY", o - 3 * m.BIN): rec}, "SG_TO_MY", o) == (None, "missing")


# ---------------------------------------------------------------- statistics
def test_ridge_matches_numpy_closed_form():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3)) * [1, 5, 0.2] + [2, -1, 3]
    y = 1.5 + x @ [0.7, -0.2, 3.0] + rng.normal(scale=0.1, size=300)
    feats = ("a", "b", "c")
    rows = [dict(zip(feats, r)) for r in x]
    lam = 7.0
    model = m.RidgeResidual.fit(feats, rows, list(y), lam)
    mu, sd = x.mean(0), x.std(0)
    z = np.c_[np.ones(300), (x - mu) / sd]
    pen = np.diag([0, lam, lam, lam])
    beta = np.linalg.solve(z.T @ z + pen, z.T @ y)
    assert np.allclose(model.beta, beta, atol=1e-8)
    assert abs(model.predict(rows[0]) - z[0] @ beta) < 1e-8
    again = m.RidgeResidual.from_json(json.loads(json.dumps(model.to_json())))
    assert again.predict(rows[5]) == pytest.approx(model.predict(rows[5]))


def test_ridge_constant_feature_and_bad_json():
    rows = [{"a": 1.0, "b": float(i)} for i in range(50)]
    model = m.RidgeResidual.fit(("a", "b"), rows, [float(i) for i in range(50)], 1.0)
    assert model.std[0] == 1.0  # constant feature does not divide by zero
    with pytest.raises(ValueError):
        m.RidgeResidual(("a",), [0.0], [1.0], [0.0], 1)  # beta too short


def test_percentile_matches_numpy():
    np = pytest.importorskip("numpy")
    vals = [random.Random(1).random() for _ in range(101)]
    for q in (0, 10, 50, 90, 100):
        assert m.percentile(vals, q) == pytest.approx(np.percentile(vals, q))


def test_error_metrics():
    out = m.error_metrics([1.0, -2.0, 20.0, 0.0])
    assert out["mae"] == pytest.approx(23 / 4) and out["bias"] == pytest.approx(19 / 4)
    assert out["within_15min"] == 0.75 and out["median_ae"] == pytest.approx(1.5)


def test_bootstrap_is_deterministic_and_resamples_whole_days():
    pairs = [(dt.date(2026, 9, 13 + d), -0.5 + 0.1 * d) for d in range(10) for _ in range(20)]
    a = m.paired_day_bootstrap(pairs, 500, seed=3)
    b = m.paired_day_bootstrap(pairs, 500, seed=3)
    assert a == b and a["days"] == 10 and a["ci_low"] < a["mean"] < a["ci_high"]
    # duplicating rows within the same days must not narrow the interval (days are the unit)
    c = m.paired_day_bootstrap(pairs + pairs, 500, seed=3)
    assert c["ci_low"] == pytest.approx(a["ci_low"]) and c["ci_high"] == pytest.approx(a["ci_high"])
    assert m.better({"days": 10, "ci_high": -0.01}) and not m.better({"days": 2, "ci_high": -1.0})


# ---------------------------------------------------------------- SQL builders
def test_forecast_sql_is_parameterised_and_validated():
    cfg = m.Config()
    sql = m.forecast_sql(cfg)
    for name in ("@origin_start", "@origin_end", "@label_cutoff", "@directions", "@min_history_bins"):
        assert name in sql
    assert "model => 'TimesFM 2.5'" in sql and "context_window => 1024" in sql and "horizon => 6" in sql
    assert "act.bin_ts <= @label_cutoff" in sql  # labels never scored past the cutoff
    with pytest.raises(m.ConfigError):
        m.forecast_sql(cfg, stride_minutes=15)


def test_show_sql_needs_no_credentials(capsys):
    assert m.main(["show-sql"]) == 0
    assert "AI.FORECAST" in capsys.readouterr().out


# ---------------------------------------------------------------- end-to-end with fakes
ORIGIN = dt.datetime(2026, 10, 4, 0, 50, tzinfo=UTC)


def forecast_rows(origin=ORIGIN, actual=False):
    rows = []
    for d, base in (("SG_TO_MY", 24.0), ("MY_TO_SG", 18.0)):
        for k in range(1, 7):
            rows.append({"direction": d, "origin_ts": origin, "step": k, "target_ts": origin + k * m.BIN,
                         "actual": base + 1 if actual else None, "persistence": base, "seasonal_d1": base + 5,
                         "seasonal_d7": base + 2, "seasonal_tod7": None, "timesfm": base + 0.1 * k,
                         "timesfm_lower": base - 2, "timesfm_upper": base + 3, "confidence_level": 0.8,
                         "ai_forecast_status": "", "n_context_bins": 1024})
    return rows


class FakeWarehouse:
    def __init__(self, registry=None, layer_a=None, existing=None, forecasts=None):
        self.registry, self.layer_a, self.existing = registry or [], layer_a or [], existing or []
        self.forecasts = forecast_rows() if forecasts is None else forecasts
        self.appended, self.queries = {}, []

    def query(self, sql, params=None, timeout_sec=600.0):
        self.queries.append((sql, params))
        if "AI.FORECAST" in sql:
            return [dict(r) for r in self.forecasts]
        if m.TABLE_REGISTRY in sql:
            return self.registry
        if "SELECT DISTINCT bin_ts" in sql:
            return self.existing
        if "max_congestion_ratio" in sql:
            return self.layer_a
        raise AssertionError(sql)

    def append(self, name, rows):
        self.appended.setdefault(name, []).extend(rows)
        return len(rows)

    def ensure_tables(self):
        return list(m.SCHEMAS)


NOW = dt.datetime(2026, 10, 4, 1, 3, tzinfo=UTC)


def test_cycle_without_registry_serves_persistence_and_logs_all_candidates():
    wh = FakeWarehouse()
    payload = {**README_PAYLOAD, "date_time": "2026-10-04T09:03:00"}
    out = m.run_cycle(cfg_with_url(), wh, FakeSession([FakeResp(200, payload)]), now=NOW)
    assert out["origin_ts"] == ORIGIN and out["layer_a_ingest"]["status"] == "logged"
    rows = wh.appended[m.TABLE_FORECASTS]
    assert len(rows) == 12 and {r["served_model"] for r in rows} == {"persistence"}
    assert {c for c, _, _ in m.SCHEMAS[m.TABLE_FORECASTS]} == set(rows[0])
    assert len(wh.appended[m.TABLE_LAYER_A]) == 2
    params = [p for s, p in wh.queries if "AI.FORECAST" in s][0]
    assert params["origin_start"] == params["origin_end"] == params["label_cutoff"] == ORIGIN


def registry_entry(served="timesfm_layer_a", fallback="timesfm_calibrated"):
    spec = {"calibrated": m.RidgeResidual(("timesfm_minus_persistence",), [0.0], [1.0], [0.5, 0.0], 500).to_json(),
            "layer_a": m.RidgeResidual(m.BASE_FEATURES + m.VISION_FEATURES, [0.0] * 4, [1.0] * 4, [1.0, 0.0, 0.1, 0.0, 0.0], 500).to_json()}
    coefs = {d: {str(k): spec for k in range(1, 7)} for d in m.DIRECTIONS}
    decisions = {d: {str(k): {"served": served, "candidate": served, "fallback": fallback} for k in range(1, 7)} for d in m.DIRECTIONS}
    return [{"run_id": "train-x", "created_at": NOW, "decisions_json": json.dumps(decisions), "coefficients_json": json.dumps(coefs)}]


def test_cycle_applies_fusion_and_falls_back_when_layer_a_missing():
    la = [{"bin_ts": ORIGIN, "direction": "SG_TO_MY", "vehicle_count": 20, "other_direction_count": 3, "total_count": 25,
           "congestion_ordinal": 1, "rejected": False}]
    wh = FakeWarehouse(registry=registry_entry(), layer_a=la)
    out = m.run_cycle(m.Config(), wh, FakeSession([]), now=NOW)
    assert out["layer_a_ingest"]["status"] == "disabled"
    rows = {(r["direction"], r["step"]): r for r in wh.appended[m.TABLE_FORECASTS]}
    sg = rows[("SG_TO_MY", 3)]
    assert sg["served_model"] == "timesfm_layer_a" and sg["served_value"] == pytest.approx(24.3 + 1.0 + 0.1 * 20)
    my = rows[("MY_TO_SG", 3)]  # no Layer A frame -> the registered fallback (calibrated here)
    assert my["served_model"] == "timesfm_calibrated" and my["layer_a_status"] == "missing"
    assert my["served_value"] == pytest.approx(18.3 + 0.5)
    # with a plain-TimesFM fallback, a missing frame never serves the calibrated model
    wh2 = FakeWarehouse(registry=registry_entry(fallback="timesfm"), layer_a=la)
    m.run_cycle(m.Config(), wh2, FakeSession([]), now=NOW)
    my2 = {(r["direction"], r["step"]): r for r in wh2.appended[m.TABLE_FORECASTS]}[("MY_TO_SG", 3)]
    assert my2["served_model"] == "timesfm" and my2["served_value"] == pytest.approx(18.3)


def test_cycle_survives_layer_a_failure_and_dry_run_writes_nothing():
    wh = FakeWarehouse()
    out = m.run_cycle(cfg_with_url(), wh, FakeSession([FakeResp(500)] * 3), now=NOW, write=True)
    assert out["layer_a_ingest"]["status"] == "failed" and len(wh.appended[m.TABLE_FORECASTS]) == 12
    wh2 = FakeWarehouse()
    m.run_cycle(cfg_with_url(), wh2, FakeSession([]), now=NOW, write=False)
    assert wh2.appended == {}


def test_cycle_without_fresh_data_raises():
    with pytest.raises(m.NoForecastError):
        m.run_cycle(m.Config(), FakeWarehouse(forecasts=[]), FakeSession([]), now=NOW)


def test_ingest_is_idempotent_per_bin():
    payload = {**README_PAYLOAD, "date_time": "2026-10-04T09:03:00"}
    wh = FakeWarehouse(existing=[{"bin_ts": dt.datetime(2026, 10, 4, 1, 0, tzinfo=UTC)}])
    assert m.ingest_layer_a(cfg_with_url(), wh, FakeSession([FakeResp(200, payload)]))["status"] == "already_logged"
    assert wh.appended == {}


def test_backfill_dry_run_budget_and_resume():
    existing = [{"bin_ts": dt.datetime(2026, 9, 12, 16, 0, tzinfo=UTC)}]
    wh = FakeWarehouse(existing=existing)
    out = m.backfill_layer_a(cfg_with_url(), wh, FakeSession([]), dt.date(2026, 9, 13), dt.date(2026, 9, 13),
                             every_minutes=10, max_calls=5, execute=False, now=NOW)
    assert out == {"wanted": 144, "already_logged": 1, "planned_calls": 5, "execute": False}

    def payload_for(i):
        t = dt.datetime(2026, 9, 12, 16, 15, tzinfo=UTC) + i * m.BIN  # bin + 5 min, as requested
        return {**README_PAYLOAD, "date_time": t.astimezone(m.SGT).strftime("%Y-%m-%dT%H:%M:%S")}

    responses = [FakeResp(200, payload_for(0)), FakeResp(404), FakeResp(200, payload_for(2))]
    out = m.backfill_layer_a(cfg_with_url(), wh, FakeSession(responses), dt.date(2026, 9, 13), dt.date(2026, 9, 13),
                             every_minutes=10, max_calls=3, execute=True, now=NOW, sleep=lambda s: None)
    assert out["logged"] == 2 and out["no_frame"] == 1 and len(wh.appended[m.TABLE_LAYER_A]) == 4


def synthetic_backtest(days=12, vision_signal=True, seed=0):
    """TimesFM residual partly explained by the camera count when vision_signal is True."""
    rng = random.Random(seed)
    rows, la = [], {}
    start = dt.datetime(2026, 9, 12, 16, 0, tzinfo=UTC)
    for i in range(days * 144):
        origin = start + i * m.BIN
        for d in m.DIRECTIONS:
            count = rng.randint(0, 80)
            la[(d, origin)] = {"vehicle_count": count, "other_direction_count": rng.randint(0, 40), "total_count": count + 5,
                               "congestion_ordinal": rng.randint(0, 3), "rejected": False}
            pers = 20 + rng.random() * 10
            for k in (3,):
                tfm = pers + rng.gauss(0, 0.5)
                effect = 0.05 * (count - 40) if vision_signal else 0.0
                actual = tfm + effect + rng.gauss(0, 0.8)
                rows.append({"direction": d, "origin_ts": origin, "step": k, "target_ts": origin + k * m.BIN, "actual": actual,
                             "persistence": pers, "seasonal_d1": pers + 3, "seasonal_d7": pers + 2, "seasonal_tod7": pers + 1,
                             "timesfm": tfm, "timesfm_lower": tfm - 2, "timesfm_upper": tfm + 2})
    return rows, la


def test_train_adopts_layer_a_only_when_it_helps():
    cfg = m.Config(horizon_steps=3, bootstrap_reps=300)
    rows, la = synthetic_backtest(vision_signal=True)
    res = m.train(cfg, rows, la, validation_days=4)
    note = res["decisions"]["SG_TO_MY"]["3"]
    assert note["candidate"] == "timesfm_layer_a" and note["layer_a_coverage"] == 1.0
    assert note["layer_a_vs_no_vision"]["ci_high"] < 0
    rows, la = synthetic_backtest(vision_signal=False, seed=1)
    res = m.train(cfg, rows, la, validation_days=4)
    assert res["decisions"]["SG_TO_MY"]["3"]["candidate"] != "timesfm_layer_a"
    reg = m.registry_row(cfg, res, "train-1", NOW)
    assert set(reg) == {c for c, _, _ in m.SCHEMAS[m.TABLE_REGISTRY]}
    assert json.loads(reg["decisions_json"])["SG_TO_MY"]["3"]["served"] in m.SERVABLE


def test_train_validation_days_guard():
    rows, la = synthetic_backtest(days=4)
    with pytest.raises(m.ConfigError):
        m.train(m.Config(horizon_steps=3), rows, la, validation_days=4)
    with pytest.raises(m.ConfigError):
        m.train(m.Config(horizon_steps=3), rows, la, validation_days=2)


def test_main_exit_codes(monkeypatch):
    monkeypatch.setenv("TIMESFM_CONTEXT_WINDOW", "999")
    assert m.main(["show-sql"]) == m.EXIT_CONFIG
    monkeypatch.delenv("TIMESFM_CONTEXT_WINDOW")
    assert m.main(["run-cycle", "--no-layer-a"], warehouse_factory=lambda c: FakeWarehouse(forecasts=[]),
                  session=FakeSession([])) == m.EXIT_NO_FORECAST
    assert m.main(["backtest", "--start", "2026-09-25", "--end", "2026-10-02"],
                  warehouse_factory=lambda c: FakeWarehouse()) == m.EXIT_CONFIG  # protected window


def test_warehouse_against_real_client_objects():
    """Builds real google-cloud-bigquery objects (offline) through a stub client."""
    class Job:
        def result(self, timeout=None):
            return []

    class Client:
        def __init__(self):
            self.tables, self.loads, self.q = [], [], []

        def create_table(self, table, exists_ok=False):
            self.tables.append(table)

        def load_table_from_json(self, rows, dest, job_config=None, location=None):
            self.loads.append((rows, dest, job_config, location))
            return Job()

        def query(self, sql, job_config=None, location=None):
            self.q.append((sql, job_config, location))
            return Job()

    client = Client()
    wh = m.Warehouse(m.Config(), client=client)
    assert len(wh.ensure_tables()) == 4
    t = {x.table_id: x for x in client.tables}
    assert t["layer_a_counts"].time_partitioning.field == "bin_ts" and t["layer_a_counts"].clustering_fields == ["direction", "bin_ts"]
    wh.append(m.TABLE_FORECASTS, [{"run_id": "r", "created_at": NOW, "origin_ts": ORIGIN, "direction": "SG_TO_MY", "step": 3,
                                   "horizon_min": 30, "target_ts": ORIGIN, "served_model": "persistence",
                                   "layer_a_status": "missing", "model": m.TIMESFM_MODEL, "context_window": 1024,
                                   "timesfm": float("nan")}])
    rows, dest, cfgj, loc = client.loads[0]
    assert dest == "swiftborder.traffic_prediction.layer_b_forecasts" and loc == "US"
    assert rows[0]["created_at"] == "2026-10-04T01:03:00+00:00" and rows[0]["timesfm"] is None
    wh.query("SELECT 1", {"a": ORIGIN, "b": ["x"], "c": 3, "d": 1.5, "e": True, "f": "s", "g": [ORIGIN]})
    qp = {p.name: p for p in client.q[0][1].query_parameters}
    assert qp["a"].type_ == "TIMESTAMP" and qp["b"].array_type == "STRING" and qp["g"].array_type == "TIMESTAMP"
    assert qp["c"].type_ == "INT64" and qp["d"].type_ == "FLOAT64" and qp["e"].type_ == "BOOL"


def drift(days=12, val_days=4, seed=3):
    """Training days: TimesFM overshoots and the camera explains part of the residual.
    Validation days: no overshoot and a weaker camera effect (the reviewer's regime change)."""
    rng = random.Random(seed)
    rows, la = [], {}
    start = dt.datetime(2026, 9, 12, 16, 0, tzinfo=UTC)
    for i in range(days * 144):
        origin = start + i * m.BIN
        is_val = i >= (days - val_days) * 144
        for d in m.DIRECTIONS:
            c = rng.randint(0, 80)
            la[(d, origin)] = {"vehicle_count": c, "other_direction_count": rng.randint(0, 40), "total_count": c + 1,
                               "congestion_ordinal": rng.randint(0, 3), "rejected": False}
            pers = 20 + rng.random() * 10
            tfm = pers + rng.gauss(0, 2.0)
            resid = (0.03 * (c - 40) if is_val else -0.5 * (tfm - pers) + 0.05 * (c - 40)) + rng.gauss(0, 0.2)
            rows.append({"direction": d, "origin_ts": origin, "step": 1, "target_ts": origin + m.BIN, "actual": tfm + resid,
                         "persistence": pers, "seasonal_d1": None, "seasonal_d7": None, "seasonal_tod7": None,
                         "timesfm": tfm, "timesfm_lower": None, "timesfm_upper": None})
    return rows, la


def test_vision_is_never_judged_against_a_rejected_model():
    cfg = m.Config(horizon_steps=1, bootstrap_reps=500)
    rows, la = drift()
    res = m.train(cfg, rows, la, validation_days=4)
    for d in m.DIRECTIONS:
        note = res["decisions"][d]["1"]
        assert note["fallback"] == "timesfm"  # calibration rejected on validation
        assert not m.better(note["layer_a_vs_fallback"]) and note["candidate"] == "timesfm"
        mae = res["metrics"]["validation"]["step_1"]
        served = note["served"] if note["served"] != "persistence" else "persistence"
        assert mae[served][d]["mae"] <= mae["timesfm_layer_a"][d]["mae"]


def test_train_rows_never_take_labels_from_validation_days():
    rows, la = synthetic_backtest(days=8)
    last_train_origin = dt.datetime(2026, 9, 16, 15, 50, tzinfo=UTC)  # 23:50 SGT on the last training day
    rows.append({**rows[0], "origin_ts": last_train_origin, "target_ts": last_train_origin + 3 * m.BIN, "actual": 1e6})
    res = m.train(m.Config(horizon_steps=3, bootstrap_reps=200), rows, la, validation_days=4)
    for d in m.DIRECTIONS:  # the 1e6 label would wreck the intercept if it had leaked into training
        assert abs(m.RidgeResidual.from_json(res["coefficients"][d]["3"]["calibrated"]).beta[0]) < 5


class RaisingWarehouse(FakeWarehouse):
    def __init__(self, fail_on, **kw):
        super().__init__(**kw)
        self.fail_on = fail_on

    def query(self, sql, params=None, timeout_sec=600.0):
        if self.fail_on in sql:
            raise RuntimeError("404 Not found: Table")
        return super().query(sql, params, timeout_sec)


@pytest.mark.parametrize("fail_on", ["SELECT DISTINCT bin_ts", "max_congestion_ratio", m.TABLE_REGISTRY])
def test_cycle_keeps_forecasting_when_layer_a_or_registry_tables_fail(fail_on):
    payload = {**README_PAYLOAD, "date_time": "2026-10-04T09:03:00"}
    wh = RaisingWarehouse(fail_on, registry=registry_entry())
    m.run_cycle(cfg_with_url(), wh, FakeSession([FakeResp(200, payload)]), now=NOW)
    rows = wh.appended[m.TABLE_FORECASTS]
    assert len(rows) == 12 and all(r["served_value"] is not None for r in rows)


def test_corrupt_registry_falls_back_instead_of_crashing():
    bad_spec = {"features": ["timesfm_minus_persistence"], "mean": [0], "std": [0], "beta": [0, 0], "n": 1}
    bad = [{"run_id": "x", "created_at": NOW,
            "decisions_json": json.dumps({"SG_TO_MY": {"3": {"served": "timesfm_calibrated", "fallback": "timesfm"}}}),
            "coefficients_json": json.dumps({"SG_TO_MY": {"3": {"calibrated": bad_spec}}})}]
    wh = FakeWarehouse(registry=bad)
    m.run_cycle(m.Config(), wh, FakeSession([]), now=NOW)
    sg3 = {(r["direction"], r["step"]): r for r in wh.appended[m.TABLE_FORECASTS]}[("SG_TO_MY", 3)]
    assert sg3["served_model"] == "timesfm" and sg3["timesfm_calibrated"] is None


def test_small_context_windows_are_accepted():
    for cw in ("64", "128"):
        cfg = m.Config.from_env({"TIMESFM_CONTEXT_WINDOW": cw})
        assert m.forecast_params(cfg, ORIGIN, ORIGIN, ORIGIN)["min_history_bins"] == int(cw)


def test_backfill_stops_after_repeated_failures_and_cli_exits_nonzero():
    wh = FakeWarehouse()
    out = m.backfill_layer_a(cfg_with_url(), wh, FakeSession([FakeResp(500)] * 5), dt.date(2026, 9, 13), dt.date(2026, 9, 13),
                             every_minutes=10, max_calls=10, execute=True, now=NOW, sleep=lambda s: None)
    assert out["stopped_early"] and out["failed"] == 5
    import os
    os.environ["LAYER_A_URL"] = "https://swiftbackend.example.run.app"
    try:
        code = m.main(["backfill-layer-a", "--start", "2026-09-13", "--end", "2026-09-13", "--max-calls", "10", "--execute"],
                      warehouse_factory=lambda c: FakeWarehouse(), session=FakeSession([FakeResp(500)] * 5))
    finally:
        del os.environ["LAYER_A_URL"]
    assert code == m.EXIT_ERROR


def test_cycle_creates_tables_only_when_writing():
    class Counting(FakeWarehouse):
        calls = 0

        def ensure_tables(self):
            Counting.calls += 1
            return []

    m.run_cycle(m.Config(), Counting(), FakeSession([]), now=NOW, write=True)
    m.run_cycle(m.Config(), Counting(), FakeSession([]), now=NOW, write=False)
    assert Counting.calls == 1


def test_command_can_come_from_the_environment(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["layer_b.py"])  # Procfile started the module without arguments
    monkeypatch.setenv("LAYER_B_ARGS", "show-sql")
    assert m.main() == m.EXIT_OK and "latest registry" in capsys.readouterr().out


def test_train_cli_runs_before_the_layer_a_table_exists():
    rows, _ = synthetic_backtest(days=8)
    wh = RaisingWarehouse("max_congestion_ratio", forecasts=rows)  # 404: layer_a_counts not created yet
    code = m.main(["train", "--start", "2026-09-13", "--end", "2026-09-20", "--chunk-days", "30", "--validation-days", "3"],
                  warehouse_factory=lambda c: wh)
    assert code == m.EXIT_OK


def test_live_ingest_checks_the_table_before_paying_for_inference():
    wh = FakeWarehouse(existing=[{"bin_ts": dt.datetime(2026, 10, 4, 1, 0, tzinfo=UTC)}])
    out = m.ingest_layer_a(cfg_with_url(), wh, FakeSession([]), now=NOW)  # FakeSession([]) fails if called
    assert out["status"] == "already_logged" and wh.appended == {}
