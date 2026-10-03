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


ROUTES = {"SG_TO_MY": "woodlands_sg_to_my", "MY_TO_SG": "woodlands_my_to_sg"}
# 03:42 UTC: origin 03:30 closed at 03:40, the fixture target 04:00 is 18 minutes ahead
FIXTURE_NOW = dt.datetime(2026, 10, 3, 3, 42, tzinfo=UTC)


def _bins(start="2026-09-05 16:00", days=30, seed=0, drop=()):
    """Synthetic v_bins_10min rows (both directions, 10-minute bins); ``drop`` removes bin times."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    dropped = {pd.Timestamp(x, tz="UTC") for x in drop}
    rows = []
    for d in main.DIRECTIONS:
        for i in range(days * 144):
            ts = t0 + pd.Timedelta(minutes=10 * i)
            if ts in dropped:
                continue
            base = 25 + 8 * np.sin(2 * np.pi * i / 144) + rng.normal(0, 0.5)
            rows.append(
                {
                    "route_id": ROUTES[d], "direction": d, "bin_ts": ts, "dur_min": base,
                    "congestion_ratio": 1 + rng.normal(0, 0.05), "speed_kmh": 40 + rng.normal(0, 2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    main._cache.clear()
    main._fitted.clear()
    main._training.clear()
    main._clock = time.monotonic
    monkeypatch.setattr(main, "_now", lambda: FIXTURE_NOW)
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


def _fake_bigquery(monkeypatch, bins, now):
    """Replace both reads with the SQL filters applied to ``bins``."""
    reads = {"training": 0, "latest": []}

    def fetch_training(labels_before):
        reads["training"] += 1
        return bins[bins["bin_ts"] + main.BIN <= labels_before].copy()

    def fetch_latest(directions, at):
        reads["latest"].append(at)
        ts = bins["bin_ts"]
        keep = bins["direction"].isin(directions) & (ts >= at - main.LATEST_WINDOW) & (ts + main.BIN + main.INGEST_GRACE <= at)
        return bins[keep].copy()

    monkeypatch.setattr(main, "fetch_training", fetch_training)
    monkeypatch.setattr(main, "fetch_latest", fetch_latest)
    monkeypatch.setattr(main, "_now", lambda: now["t"])
    return reads


def _dur(bins, direction, ts):
    row = bins[(bins["direction"] == direction) & (bins["bin_ts"] == pd.Timestamp(ts, tz="UTC"))]
    return float(row["dur_min"].iloc[0])


def test_served_uses_the_selection_per_direction(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 200 and payload["source"] == "local model"
    for d, mid in main.SERVED_SELECTION.items():
        assert payload["directions"][d]["model"] == mid
    assert set(payload["model_meta"]) == set(main.SERVED_SELECTION.values())
    # 03:50 closes at 04:00 (+ grace): the newest servable origin is 03:40
    my = payload["directions"]["MY_TO_SG"]
    assert my["origin_ts"] == "2026-10-03T03:40:00Z" and my["forecast_for"] == "2026-10-03T04:10:00Z"
    assert my["forecast_min"] == round(_dur(bins, "MY_TO_SG", "2026-10-03 03:40"), 1)


def test_an_open_bin_is_never_an_origin(monkeypatch):
    # the readiness review's live case: generated 15:32:49 UTC used the 15:30 bin before it closed
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 15, 32, 49, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert status == 200
    for d in main.DIRECTIONS:
        row = payload["directions"][d]
        assert row["origin_ts"] == "2026-10-03T15:20:00Z" and row["origin_closed_at"] == "2026-10-03T15:30:00Z"
        assert row["forecast_for"] == "2026-10-03T15:50:00Z" and row["forecast_window_end"] == "2026-10-03T16:00:00Z"
        assert row["observation_age_min"] == 2.8 and row["lead_min"] == 17.2
    # and the grace minute: at 15:30:30 the 15:20 bin is not yet servable either
    main._cache.clear()
    now["t"] = dt.datetime(2026, 10, 3, 15, 30, 30, tzinfo=UTC)
    payload, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert payload["directions"]["SG_TO_MY"]["origin_ts"] == "2026-10-03T15:10:00Z"


def test_stale_observations_are_503_not_an_expired_forecast(monkeypatch):
    bins = _bins()
    bins = bins[bins["bin_ts"] < pd.Timestamp("2026-10-03 02:00", tz="UTC")]  # ingestion stops
    now = {"t": dt.datetime(2026, 10, 3, 2, 25, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    # 01:50 is the newest bin; its target 02:20 has started
    payload, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert status == 503 and payload == {"error": "No recent forecast for the requested direction"}
    # one missed fetch is still servable: at 02:15 the 01:50 origin targets 02:20
    now["t"] = dt.datetime(2026, 10, 3, 2, 15, tzinfo=UTC)
    payload, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert status == 200 and payload["directions"]["SG_TO_MY"]["lead_min"] == 5.0


def test_stale_direction_alone_is_503(monkeypatch):
    bins = _bins()
    cut = (bins["direction"] == "MY_TO_SG") & (bins["bin_ts"] >= pd.Timestamp("2026-10-03 02:00", tz="UTC"))
    now = {"t": dt.datetime(2026, 10, 3, 3, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins[~cut], now)
    assert _json(main.forecast(_request(query={"model": "persistence", "direction": "SG_TO_MY"})))[1] == 200
    assert _json(main.forecast(_request(query={"model": "persistence", "direction": "MY_TO_SG"})))[1] == 503
    assert _json(main.forecast(_request(query={"model": "persistence"})))[1] == 503


def test_origin_inside_an_hour_after_a_gap_is_503(monkeypatch):
    gap = [f"2026-10-03 02:{m:02d}" for m in (0, 10, 20)]  # 40-minute gap: > 25
    bins = _bins(drop=gap)
    now = {"t": dt.datetime(2026, 10, 3, 3, 0, tzinfo=UTC)}  # origin 02:40, right after the gap
    _fake_bigquery(monkeypatch, bins, now)
    assert _json(main.forecast(_request(query={"model": "persistence"})))[1] == 503
    main._cache.clear()
    now["t"] = dt.datetime(2026, 10, 3, 3, 52, tzinfo=UTC)  # origin 03:40: seven bins after the gap
    assert _json(main.forecast(_request(query={"model": "persistence"})))[1] == 200


def test_a_single_skipped_bin_gives_missing_lags_not_shifted_ones():
    bins = _bins(days=1, drop=["2026-09-06 02:00"])
    f = lm.features_from_bins(bins)
    sg = f[f["direction"] == "SG_TO_MY"].set_index("bin_ts")
    at = lambda s: sg.loc[pd.Timestamp(s, tz="UTC")]  # noqa: E731
    assert np.isnan(at("2026-09-06 02:10")["lag_10"])  # positional LAG would return 01:50
    assert at("2026-09-06 02:10")["lag_20"] == _dur(bins, "SG_TO_MY", "2026-09-06 01:50")
    assert np.isnan(at("2026-09-06 01:30")["y_30"])  # positional LEAD would return 02:10
    assert at("2026-09-06 02:10")["after_gap"] == 0  # a 20-minute gap is not > 25


def test_cached_forecast_expires_with_its_target(monkeypatch):
    calls = _install(monkeypatch)
    now = {"t": FIXTURE_NOW}
    monkeypatch.setattr(main, "_now", lambda: now["t"])
    clock = {"t": 1000.0}
    monkeypatch.setattr(main, "_clock", lambda: clock["t"])
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 200 and payload["directions"]["SG_TO_MY"]["lead_min"] == 18.0
    assert payload["directions"]["SG_TO_MY"]["observation_age_min"] == 2.0
    now["t"] += dt.timedelta(minutes=3)
    clock["t"] += 180
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 200 and payload["directions"]["SG_TO_MY"]["lead_min"] == 15.0 and len(calls) == 1
    now["t"] = dt.datetime(2026, 10, 3, 4, 0, 30, tzinfo=UTC)  # the fixture target has started
    clock["t"] += 60
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 503 and len(calls) == 2  # cache dropped, re-queried, still expired


def test_daily_fit_is_reused_within_a_day_and_refit_next_day(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    reads = _fake_bigquery(monkeypatch, bins, now)
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
    view = lm.prepare(lm.features_from_bins(_bins()))
    day = pd.Timestamp("2026-10-02 16:00", tz="UTC")  # 00:00 SGT 3 Oct
    daily = lm.training_rows(view, "daily", day)
    assert (daily["bin_ts"] + lm.LABEL_LAG <= day).all()
    frozen = lm.training_rows(view, "frozen", day)
    assert (frozen["bin_ts"] + lm.LABEL_LAG <= lm.BQML_TRAINED_AT).all()
    assert not frozen["date_sgt"].isin(lm.BQML_VAL_DAYS).any()
    assert lm.version_for("xgb[maps]", dt.datetime(2026, 10, 3, 4, tzinfo=UTC)) == "labels_before=2026-10-03T00:00:00+08:00"


def _install_selection(monkeypatch):
    calls = []

    def fake(chosen):
        calls.append(dict(chosen))
        return {
            "source": "fixture",
            "rows": [
                {"direction": d, "forecast_min": 30.0, "forecast_for": "2026-10-03T04:00:00Z", "origin_ts": "2026-10-03T03:30:00Z", "model": m}
                for d, m in chosen.items()
            ],
        }

    monkeypatch.setattr(main, "query_selection", fake)
    return calls


def test_model_defaults_to_served(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={})))
    assert status == 200 and payload["model"] == "served"
    assert calls == [("served", ("SG_TO_MY", "MY_TO_SG"))]


def test_per_direction_override(monkeypatch):
    calls = _install_selection(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "xgb[maps]", "model_my_to_sg": "persistence"})))
    assert status == 200 and payload["model"] == "custom"
    assert calls == [{"SG_TO_MY": "xgb[maps]", "MY_TO_SG": "persistence"}]
    assert payload["directions"]["SG_TO_MY"]["model"] == "xgb[maps]"
    assert payload["directions"]["MY_TO_SG"]["model"] == "persistence"
    assert payload["version"].startswith("SG_TO_MY=xgb[maps]:labels_before=")
    assert "MY_TO_SG=persistence:latest-bin" in payload["version"]


def test_override_with_served_base_resolves_the_selection(monkeypatch):
    calls = _install_selection(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model_sg_to_my": "xgb_bq[daily]"})))
    assert status == 200
    assert calls == [{"SG_TO_MY": "xgb_bq[daily]", "MY_TO_SG": main.SERVED_SELECTION["MY_TO_SG"]}]


def test_override_errors(monkeypatch):
    calls = _install_selection(monkeypatch)
    cases = [
        ({"model_sg_to_my": "nope"}, "unknown model in model_sg_to_my"),
        ({"model_sg_to_my": "lstm"}, "model in model_sg_to_my is not deployed"),
        ({"direction": "MY_TO_SG", "model_sg_to_my": "xgb[maps]"}, "a per-direction model is set for a direction that was not requested"),
        ({"model_sg_to_my": "xgb[maps]", "version": "x"}, "version cannot be combined with a per-direction model"),
    ]
    for query, message in cases:
        payload, status, _ = _json(main.forecast(_request(query=query)))
        assert status == 400 and payload["error"] == message, query
    assert calls == []


def test_catalog_lists_selection_parameters(monkeypatch):
    _install(monkeypatch)
    payload, _, _ = _json(main.forecast(_request(query={"list": "models"})))
    assert {"model", "model_sg_to_my", "model_my_to_sg"} <= set(payload["parameters"])


def test_live_path_with_override(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"model": "persistence", "model_sg_to_my": "ridge[maps]"})))
    assert status == 200 and payload["source"] == "local model"
    assert payload["directions"]["SG_TO_MY"]["model"] == "ridge[maps]"
    assert set(payload["model_meta"]) == {"ridge[maps]", "persistence"}
