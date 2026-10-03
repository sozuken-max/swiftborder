"""HTTP handler and local-model tests. BigQuery is never called (fetchers are replaced)."""

import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import pytest
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

import local_models as lm
import main

UTC = dt.timezone.utc


def _request(method="GET", query=None, body=None):
    builder = EnvironBuilder(method=method, query_string=query, json=body)
    return Request(builder.get_environ())


def _json(resp):
    body, status, headers = resp
    return json.loads(body), status, headers


def _rows(directions=("SG_TO_MY", "MY_TO_SG")):
    return {
        "source": "fixture",
        "rows": [
            {
                "direction": d,
                "forecast_min": 41.2 if d == "SG_TO_MY" else 25.0,
                "forecast_for": "2026-10-03T04:00:00Z",
                "origin_ts": "2026-10-03T03:30:00Z",
            }
            for d in directions
        ],
    }


def _install(monkeypatch):
    calls = []

    def fake(model_id, requested):
        calls.append((model_id, tuple(requested)))
        return _rows(requested)

    monkeypatch.setattr(main, "query_forecast", fake)
    return calls


def _view(start="2026-09-05 16:00", days=30, seed=0):
    """Synthetic v_training_set rows (both directions, 10-minute bins)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    rows = []
    for d in main.DIRECTIONS:
        for i in range(days * 144):
            ts = t0 + pd.Timedelta(minutes=10 * i)
            base = 25 + 8 * np.sin(2 * np.pi * i / 144) + rng.normal(0, 0.5)
            row = {"direction": d, "bin_ts": ts, "y_30": base + 1.0, "after_gap": 0}
            for c in lm.VIEW_COLUMNS:
                row[c] = base + rng.normal(0, 0.3)
            row["dow"] = float(ts.dayofweek)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _reset():
    main._cache.clear()
    main._fitted.clear()
    main._training.clear()
    main._clock = time.monotonic
    yield
    main._cache.clear()
    main._fitted.clear()
    main._training.clear()


def test_options_preflight():
    body, status, headers = main.forecast(_request("OPTIONS"))
    assert body == "" and status == 204
    assert headers["Access-Control-Allow-Methods"] == "GET, OPTIONS"


def test_success_with_fake_query(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, headers = _json(main.forecast(_request(query={"model": "xgb[maps]"})))
    assert status == 200
    assert payload["model"] == "xgb[maps]"
    assert payload["version"].startswith("labels_before=")
    assert payload["label"] == "maps_duration_in_traffic_min"
    assert calls == [("xgb[maps]", ("SG_TO_MY", "MY_TO_SG"))]


def test_catalog_states_and_served_selection(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"list": "models"})))
    assert status == 200
    by_id = {row["id"]: row for row in payload["models"]}
    assert by_id["served"]["deploy_state"] == "production"
    assert by_id["served"]["selection"] == main.SERVED_SELECTION
    for mid in list(lm.LOCAL_MODELS) + ["persistence"]:
        assert by_id[mid]["deploy_state"] == "local" and by_id[mid]["callable"] is True
    assert "lin_h30" not in by_id and "xgb_h30" not in by_id  # BigQuery ML is no longer called
    assert by_id["ensemble[maps]"]["callable"] is False
    assert by_id["lstm"]["deploy_state"] == "artifact"
    assert calls == []


def test_unknown_model_and_undeployed_are_400(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "not_a_model"})))
    assert status == 400 and payload["error"] == "unknown model"
    payload, status, _ = _json(main.forecast(_request(query={"model": "lstm"})))
    assert status == 400 and payload["error"] == "model is not deployed"
    assert calls == []


def test_version_must_be_current(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "lin_bq[frozen]", "version": "bqml-replica-2026-09-12"})))
    assert status == 200 and payload["version"] == "bqml-replica-2026-09-12"
    payload, status, _ = _json(main.forecast(_request(query={"model": "xgb[maps]", "version": "labels_before=1999-01-01T00:00:00+08:00"})))
    assert status == 400 and payload["error"] == "unknown version"
    assert len(calls) == 1


def test_cache_does_not_query_twice_inside_ttl(monkeypatch):
    calls = _install(monkeypatch)
    now = {"t": 1000.0}
    monkeypatch.setattr(main, "_clock", lambda: now["t"])
    first = _json(main.forecast(_request(query={"model": "served"})))
    now["t"] += 299
    second = _json(main.forecast(_request(query={"model": "served"})))
    assert first[1] == second[1] == 200 and first[0] == second[0]
    assert len(calls) == 1


def test_bad_direction_and_horizon_are_400(monkeypatch):
    calls = _install(monkeypatch)
    assert _json(main.forecast(_request(query={"model": "served", "direction": "north"})))[1] == 400
    payload, status, _ = _json(main.forecast(_request(query={"model": "served", "horizon_min": "1440"})))
    assert status == 400 and payload["error"] == "horizon_min must be 30"
    assert calls == []


def test_missing_direction_is_503_and_not_cached(monkeypatch):
    calls = []

    def empty(model_id, directions):
        calls.append(model_id)
        return {"source": "fixture", "rows": []}

    monkeypatch.setattr(main, "query_forecast", empty)
    for _ in range(2):
        payload, status, _ = _json(main.forecast(_request(query={"model": "persistence", "direction": "SG_TO_MY"})))
        assert status == 503 and payload == {"error": "No recent forecast for the requested direction"}
    assert len(calls) == 2


def test_query_failure_is_502_without_detail(monkeypatch):
    def boom(model_id, directions):
        raise RuntimeError("SELECT secret FROM `swiftborder.traffic_prediction.v_training_set`")

    monkeypatch.setattr(main, "query_forecast", boom)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 502 and payload == {"error": "Forecast query failed"}


# --- local models end to end (fetchers replaced) ---------------------------------------------


def _fake_bigquery(monkeypatch, view, now):
    reads = {"training": 0}

    def fetch_training(labels_before):
        reads["training"] += 1
        return view[view["bin_ts"] + pd.Timedelta(minutes=40) <= labels_before].copy()

    def fetch_latest(directions):
        past = view[(view["bin_ts"] <= pd.Timestamp(now["t"])) & view["direction"].isin(directions)]
        return past.sort_values("bin_ts").groupby("direction").tail(1).copy()

    monkeypatch.setattr(main, "fetch_training", fetch_training)
    monkeypatch.setattr(main, "fetch_latest", fetch_latest)
    monkeypatch.setattr(main, "_now", lambda: now["t"])
    return reads


def test_served_uses_the_selection_per_direction(monkeypatch):
    view = _view()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, view, now)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 200 and payload["source"] == "local model"
    for d, mid in main.SERVED_SELECTION.items():
        assert payload["directions"][d]["model"] == mid
    assert set(payload["model_meta"]) == set(main.SERVED_SELECTION.values())
    latest_my = view[(view["direction"] == "MY_TO_SG") & (view["bin_ts"] <= pd.Timestamp(now["t"]))].iloc[-1]
    assert payload["directions"]["MY_TO_SG"]["forecast_min"] == round(latest_my["y_persistence"], 1)


def test_daily_fit_is_reused_within_a_day_and_refit_next_day(monkeypatch):
    view = _view()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    reads = _fake_bigquery(monkeypatch, view, now)
    a = main.query_forecast("xgb[maps]", ["SG_TO_MY"])
    now["t"] = dt.datetime(2026, 10, 3, 15, 0, tzinfo=UTC)  # 23:00 SGT, same day
    main.query_forecast("xgb[maps]", ["SG_TO_MY"])
    assert reads["training"] == 1
    now["t"] = dt.datetime(2026, 10, 3, 16, 30, tzinfo=UTC)  # 00:30 SGT next day
    b = main.query_forecast("xgb[maps]", ["SG_TO_MY"])
    assert reads["training"] == 2
    assert a["meta"]["xgb[maps]"]["version"] != b["meta"]["xgb[maps]"]["version"]
    assert b["meta"]["xgb[maps]"]["training_rows"] > a["meta"]["xgb[maps]"]["training_rows"]


def test_training_rows_follow_the_harness_folds():
    view = lm.prepare(_view())
    day = pd.Timestamp("2026-10-02 16:00", tz="UTC")  # 00:00 SGT 3 Oct
    daily = lm.training_rows(view, "daily", day)
    assert (daily["bin_ts"] + lm.LABEL_LAG <= day).all()
    frozen = lm.training_rows(view, "frozen", day)
    assert (frozen["bin_ts"] + lm.LABEL_LAG <= lm.BQML_TRAINED_AT).all()
    assert not frozen["date_sgt"].isin(lm.BQML_VAL_DAYS).any()
    assert lm.version_for("xgb[maps]", dt.datetime(2026, 10, 3, 4, tzinfo=UTC)) == "labels_before=2026-10-03T00:00:00+08:00"
