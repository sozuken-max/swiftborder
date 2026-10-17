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

    def fake(model_id, requested, horizon=30):
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
    for mid in [m for m in lm.LOCAL_MODELS if m not in main.EXPLORATORY_MODELS] + ["persistence"]:
        assert by_id[mid]["deploy_state"] == "local" and by_id[mid]["callable"] is True
    assert by_id["xgb[maps+prof]"]["deploy_state"] == "exploratory" and by_id["xgb[maps+prof]"]["callable"] is True
    assert by_id["profile"]["deploy_state"] == "baseline" and by_id["profile"]["horizons_min"] == list(lm.HORIZONS)
    assert by_id["lin_bq[frozen]"]["horizons_min"] == [30]
    assert "lin_h30" not in by_id and "xgb_h30" not in by_id  # BigQuery ML is no longer called
    assert by_id["ensemble[maps]"]["callable"] is False
    assert by_id["lstm"]["deploy_state"] == "artifact"
    assert "timesfm" not in by_id and "fcm_mlp" not in by_id
    assert calls == []


def test_model_cards_match_the_allow_list(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"list": "cards"})))
    assert status == 200 and isinstance(payload, list)
    same, status_view, _ = _json(main.forecast(_request(query={"view": "cards"})))
    assert status_view == 200 and same == payload
    by_id = {card["id"]: card for card in payload}
    assert len(by_id) == len(payload)
    served = by_id["served"]
    assert served["kind"] == "mix" and served["callable"] is True
    assert {"lin_bq[frozen]", "persistence"} <= set(served["members"])
    assert "components" not in served
    assert any(card["kind"] == "mix" for card in payload)
    single = by_id["persistence"]
    assert single["kind"] == "single" and "components" not in single and "members" not in single
    profile = by_id["profile"]
    assert profile["kind"] == "single" and profile["name"] == "Typical day" and "members" not in profile
    for spec in main.MODELS.values():
        assert by_id[spec["id"]]["callable"] is spec["callable"]
    for mid in ("timesfm", "fcm_mlp"):
        assert by_id[mid]["callable"] is False and mid not in main.MODELS
        assert by_id[mid]["summary"].startswith("Not served.")
        assert "members" not in by_id[mid]
    assert "mae" not in json.dumps(payload).lower()
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
    payload, status, _ = _json(main.forecast(_request(query={"model": "served", "horizon_min": "45"})))
    assert status == 400 and payload["error"] == "horizon_min must be a multiple of 30 from 30 to 1440"
    payload, status, _ = _json(main.forecast(_request(query={"model": "lin_bq[frozen]", "horizon_min": "60"})))
    assert status == 400 and payload["error"] == "horizon_min must be 30"
    assert calls == []


def test_missing_direction_is_503_and_not_cached(monkeypatch):
    calls = []

    def empty(model_id, directions, horizon=30):
        calls.append(model_id)
        return {"source": "fixture", "rows": []}

    monkeypatch.setattr(main, "query_forecast", empty)
    for _ in range(2):
        payload, status, _ = _json(main.forecast(_request(query={"model": "persistence", "direction": "SG_TO_MY"})))
        assert status == 503 and payload == {"error": "No recent forecast for the requested direction"}
    assert len(calls) == 2


def test_query_failure_is_502_without_detail(monkeypatch):
    def boom(model_id, directions, horizon=30):
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
        assert "components" not in payload["directions"][d]
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

    def fake(chosen, horizon=30):
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


# --- exploratory horizons and the profile baseline (docs/horizon-study.md) -------------------


def test_exploratory_horizon_uses_the_study_selection_and_says_so(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"horizon_min": "120"})))
    assert status == 200 and payload["status"] == "exploratory" and payload["horizon_min"] == 120
    assert payload["version"] == main.EXPLORATORY_SELECTION_ID and payload["target_offset_min"] == [120, 130]
    for d in main.DIRECTIONS:
        row = payload["directions"][d]
        assert row["model"] == "xgb[maps+prof]"
        assert row["origin_ts"] == "2026-10-03T03:40:00Z" and row["forecast_for"] == "2026-10-03T05:40:00Z"
        assert row["lead_min"] == 100.0
    base = payload["baseline"]
    assert base["model"] == "profile" and base["status"] == "baseline" and "not a forecast" in base["label"]
    assert set(base["directions"]) == set(main.DIRECTIONS)
    if main.STUDY:
        assert {"xgb[maps+prof]", "profile", "persistence"} <= set(payload["study"]["mae_min"])
        assert payload["study"]["status"] == "exploratory"
    payload, status, _ = _json(main.forecast(_request(query={"horizon_min": "60"})))
    assert status == 200 and payload["directions"]["SG_TO_MY"]["model"] == "xgb[maps]"


def test_thirty_minutes_stays_evaluated_and_carries_the_baseline(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={})))
    assert status == 200 and payload["status"] == "evaluated" and payload["target_offset_min"] == [30, 40]
    assert payload["directions"]["SG_TO_MY"]["model"] == main.SERVED_SELECTION["SG_TO_MY"]
    assert payload["baseline"]["status"] == "baseline"


def test_profile_model_returns_the_baseline_value(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"model": "profile", "horizon_min": "240"})))
    assert status == 200 and payload["status"] == "baseline"
    for d in main.DIRECTIONS:
        assert payload["directions"][d]["forecast_min"] == payload["baseline"]["directions"][d]["forecast_min"]


def test_long_horizon_still_needs_a_fresh_origin(monkeypatch):
    bins = _bins()
    bins = bins[bins["bin_ts"] < pd.Timestamp("2026-10-03 02:00", tz="UTC")]
    now = {"t": dt.datetime(2026, 10, 3, 2, 25, tzinfo=UTC)}  # newest origin 01:50 is 35 min old
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"horizon_min": "1440"})))
    assert status == 503


def test_override_without_that_horizon_is_400(monkeypatch):
    _install_selection(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"horizon_min": "120", "model_sg_to_my": "lin_bq[frozen]"})))
    assert status == 400 and payload["error"] == "model in model_sg_to_my has no 120-minute horizon"


def test_baseline_curve(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 3, tzinfo=UTC)}
    reads = _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"baseline": "profile", "hours": "6"})))
    assert status == 200 and payload["status"] == "baseline" and "not a forecast" in payload["label"]
    for d in main.DIRECTIONS:
        pts = payload["directions"][d]
        assert len(pts) == 36 and pts[0]["bin_start"] == "2026-10-03T04:10:00Z" and pts[-1]["bin_start"] == "2026-10-03T10:00:00Z"
    assert reads["latest"] == []  # the curve needs no latest bins
    assert _json(main.forecast(_request(query={"baseline": "profile", "hours": "25"})))[1] == 400
    assert _json(main.forecast(_request(query={"baseline": "trend"})))[1] == 400


def test_horizon_study_listing():
    payload, status, _ = _json(main.forecast(_request(query={"list": "horizon-study"})))
    if main.STUDY is None:
        assert status == 503
        return
    assert status == 200 and payload["status"] == "exploratory"
    assert [h["horizon_min"] for h in payload["horizons"]] == list(lm.HORIZONS)


def test_training_rows_per_horizon():
    view = lm.prepare(lm.features_from_bins(_bins()))
    day = pd.Timestamp("2026-10-02 16:00", tz="UTC")
    for h in (60, 240, 1440):
        rows = lm.training_rows(view, "daily", day, h)
        assert (rows["bin_ts"] + pd.Timedelta(minutes=h + 10) <= day).all() and rows[lm.label_column(h)].notna().all()
    with pytest.raises(ValueError):
        lm.training_rows(view, "frozen", day, 60)


# --- forecast curve (?curve=forecast) -------------------------------------------------------


def test_forecast_curve_default_is_30_to_120_minutes(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast"})))
    assert status == 200 and payload["step_min"] == 30 and payload["hours"] == 2
    for d in main.DIRECTIONS:
        block = payload["directions"][d]
        assert block["origin_ts"] == "2026-10-03T03:40:00Z"
        pts = block["points"]
        assert [p["horizon_min"] for p in pts] == [30, 60, 90, 120]
        assert [p["forecast_for"] for p in pts] == ["2026-10-03T04:10:00Z", "2026-10-03T04:40:00Z", "2026-10-03T05:10:00Z", "2026-10-03T05:40:00Z"]
        assert pts[0]["model"] == main.SERVED_SELECTION[d] and pts[0]["status"] == "evaluated"
        assert pts[1]["model"] == "xgb[maps]" and pts[2]["model"] == "xgb[maps+prof]" and pts[1]["status"] == "exploratory"
        assert all("components" not in p for p in pts)
        assert [p["lead_min"] for p in pts] == [10.0, 40.0, 70.0, 100.0]
        assert all("baseline_min" in p for p in pts)


def test_forecast_curve_points_match_single_horizon_requests(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    curve, _, _ = _json(main.forecast(_request(query={"curve": "forecast", "hours": "2"})))
    for h in (60, 120):
        single, status, _ = _json(main.forecast(_request(query={"horizon_min": str(h)})))
        assert status == 200
        for d in main.DIRECTIONS:
            p = next(p for p in curve["directions"][d]["points"] if p["horizon_min"] == h)
            assert p["forecast_min"] == single["directions"][d]["forecast_min"]
            assert p["baseline_min"] == single["baseline"]["directions"][d]["forecast_min"]


def test_forecast_curve_switches_to_the_baseline_after_four_hours(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast", "hours": "24", "direction": "MY_TO_SG"})))
    assert status == 200 and list(payload["directions"]) == ["MY_TO_SG"]
    pts = payload["directions"]["MY_TO_SG"]["points"]
    assert len(pts) == 48 and pts[-1]["horizon_min"] == 1440
    for p in pts:
        if p["horizon_min"] > main.CURVE_MODEL_MAX_MIN:
            assert p["model"] == "profile" and p["status"] == "baseline" and p["forecast_min"] == p["baseline_min"]
        else:
            assert p["model"] != "profile"


def test_forecast_curve_errors_and_staleness(monkeypatch):
    bins = _bins()
    bins = bins[bins["bin_ts"] < pd.Timestamp("2026-10-03 02:00", tz="UTC")]
    now = {"t": dt.datetime(2026, 10, 3, 2, 25, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    assert _json(main.forecast(_request(query={"curve": "forecast"})))[1] == 503
    assert _json(main.forecast(_request(query={"curve": "forecast", "hours": "0"})))[1] == 400
    assert _json(main.forecast(_request(query={"curve": "forecast", "hours": "25"})))[1] == 400
    assert _json(main.forecast(_request(query={"curve": "baseline"})))[1] == 400


def test_cached_curve_expires_with_its_first_target(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    clock = {"t": 1000.0}
    monkeypatch.setattr(main, "_clock", lambda: clock["t"])
    calls = []
    real = main.forecast_curve
    monkeypatch.setattr(main, "forecast_curve", lambda d, h: calls.append(h) or real(d, h))
    assert _json(main.forecast(_request(query={"curve": "forecast"})))[1] == 200
    now["t"] = dt.datetime(2026, 10, 3, 4, 3, tzinfo=UTC)
    clock["t"] += 180
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast"})))
    assert status == 200 and len(calls) == 1 and payload["directions"]["SG_TO_MY"]["points"][0]["lead_min"] == 7.0
    now["t"] = dt.datetime(2026, 10, 3, 4, 10, 30, tzinfo=UTC)  # first target 04:10 has started
    clock["t"] += 60
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast"})))
    assert status == 200 and len(calls) == 2  # re-queried; the 03:50 origin is now servable
    assert payload["directions"]["SG_TO_MY"]["origin_ts"] == "2026-10-03T03:50:00Z"


# --- review fixes (PR #7) --------------------------------------------------------------------


def test_mixed_selection_labels_each_direction(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served", "model_my_to_sg": "profile"})))
    assert status == 200 and payload["status"] == "mixed"
    assert payload["directions"]["SG_TO_MY"]["status"] == "evaluated"
    assert payload["directions"]["MY_TO_SG"]["status"] == "baseline"
    payload, _, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert payload["status"] == "evaluated" and {r["status"] for r in payload["directions"].values()} == {"evaluated"}


def test_expired_cache_entries_are_evicted_on_put(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(main, "_clock", lambda: clock["t"])
    for i in range(144):  # one baseline curve per 10-minute bin for a day, each key read once
        main._cache_put(("baseline-curve", i), {"i": i})
        clock["t"] += 600
    assert len(main._cache) == 1


def test_browser_cache_never_outlives_the_first_target(monkeypatch):
    bins = _bins()
    # ingestion stops after 03:40: at 04:09:30 that origin's first target (04:10) is 30 s away
    stopped = bins[bins["bin_ts"] <= pd.Timestamp("2026-10-03 03:40", tz="UTC")]
    now = {"t": dt.datetime(2026, 10, 3, 4, 9, 30, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, stopped, now)
    payload, status, headers = _json(main.forecast(_request(query={"curve": "forecast"})))
    assert status == 200 and payload["directions"]["SG_TO_MY"]["points"][0]["forecast_for"] == "2026-10-03T04:10:00Z"
    assert headers["Cache-Control"] == "private, max-age=30"
    _, _, headers = _json(main.forecast(_request(query={"model": "persistence"})))
    assert headers["Cache-Control"] == "private, max-age=30"  # origin 03:40 + 30 min
    now["t"] = dt.datetime(2026, 10, 3, 3, 52, tzinfo=UTC)  # fresh origin 03:40: 18 min to target
    main._cache.clear()
    _, _, headers = _json(main.forecast(_request(query={"model": "persistence"})))
    assert headers["Cache-Control"] == "private, max-age=300"
    _, _, headers = _json(main.forecast(_request(query={"list": "models"})))
    assert headers["Cache-Control"] == "private, max-age=300"


# --- review fixes, round 2 (PR #7) -----------------------------------------------------------


def _new_route_bins():
    """MY_TO_SG starts at 00:00 SGT 3 Oct: no closed bin before the serving day, so no profile."""
    bins = _bins()
    day = pd.Timestamp("2026-10-02 16:00", tz="UTC")
    return bins[(bins["direction"] == "SG_TO_MY") | (bins["bin_ts"] >= day)]


def test_no_profile_history_never_sends_nan(monkeypatch):
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, _new_route_bins(), now)
    body, status, _ = main.forecast(_request(query={"model": "persistence"}))
    assert status == 200 and "NaN" not in body
    payload = json.loads(body)
    assert payload["baseline"]["directions"]["MY_TO_SG"]["forecast_min"] is None
    assert payload["baseline"]["directions"]["SG_TO_MY"]["forecast_min"] is not None
    for query in ({"model": "profile"}, {"baseline": "profile"}, {"curve": "forecast"}, {"horizon_min": "120"}):
        main._cache.clear()
        payload, status, _ = _json(main.forecast(_request(query=query)))
        assert status == 503 and payload == {"error": main.NO_PROFILE_ERROR}, query


def test_non_finite_numbers_are_a_502_not_invalid_json(monkeypatch):
    monkeypatch.setattr(main, "list_models", lambda: {"x": float("nan")})
    payload, status, _ = _json(main.forecast(_request(query={"list": "models"})))
    assert status == 502 and payload == {"error": "Forecast query failed"}


def test_single_forecast_cache_rolls_over_at_midnight_sgt(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 15, 59, tzinfo=UTC)}  # 23:59 SGT
    reads = _fake_bigquery(monkeypatch, bins, now)
    clock = {"t": 1000.0}
    monkeypatch.setattr(main, "_clock", lambda: clock["t"])
    first, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert status == 200 and len(reads["latest"]) == 1
    now["t"] = dt.datetime(2026, 10, 3, 16, 1, 30, tzinfo=UTC)  # 00:01:30 SGT, inside the 5-minute TTL
    clock["t"] += 150
    second, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert status == 200 and len(reads["latest"]) == 2  # not yesterday's cached answer
    assert first["baseline"]["version"] != second["baseline"]["version"]


def test_curve_metadata_is_kept_per_model_and_horizon(monkeypatch):
    bins = _bins()
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, bins, now)
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast", "hours": "4"})))
    assert status == 200
    meta = payload["model_meta"]
    for p in payload["directions"]["SG_TO_MY"]["points"]:
        m = meta[p["model_meta_key"]]
        assert m["model"] == p["model"] and m["horizon_min"] == p["horizon_min"]
    prof_fits = {k: v for k, v in meta.items() if v["model"] == "xgb[maps+prof]"}
    assert sorted(v["horizon_min"] for v in prof_fits.values()) == [90, 120, 150, 180, 210, 240]
    assert len({v["training_rows"] for v in prof_fits.values()}) > 1  # each fit has its own rows


# --- review fixes, round 3 (PR #7) -----------------------------------------------------------


def test_request_spanning_midnight_uses_one_serving_day(monkeypatch):
    """The clock advances 5 s on every read during the request, starting at 23:59:50 SGT."""
    bins = _bins()
    reads = _fake_bigquery(monkeypatch, bins, {"t": None})
    ticks = {"t": dt.datetime(2026, 10, 3, 15, 59, 50, tzinfo=UTC)}

    def advancing():
        ticks["t"] += dt.timedelta(seconds=5)
        return ticks["t"]

    monkeypatch.setattr(main, "_now", advancing)
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast", "hours": "4"})))
    assert status == 200 and ticks["t"] > dt.datetime(2026, 10, 3, 16, 0, tzinfo=UTC)  # the request crossed 00:00 SGT
    day = "labels_before=2026-10-03T00:00:00+08:00"
    daily = lambda vs: {v for v in vs if v.startswith("labels_before=")}  # noqa: E731 (frozen replicas have fixed versions)
    assert daily(m["version"] for m in payload["model_meta"].values()) == {day}
    assert daily(k[1] for k in main._fitted) == {day}  # every daily fit and the profile are from one serving day
    assert reads["training"] == 1 and len(reads["latest"]) == 1


@pytest.mark.parametrize("query", [
    {"curve": "forecast", "hours": "\u00b2"},
    {"baseline": "profile", "hours": "\u00b2"},
    {"horizon_min": "\u00b2"},
    {"curve": "forecast", "hours": "\u0663"},  # Arabic-Indic three: ASCII digits only
])
def test_non_ascii_digits_are_400(monkeypatch, query):
    _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query=query)))
    assert status == 400 and "error" in payload


def _fixture_rows(directions, model_for):
    """Rows a fake query can return. ``model_for(direction)`` is the producer, or (id, components)."""
    rows = []
    for d in directions:
        produced = model_for(d)
        components = None
        if isinstance(produced, tuple):
            produced, components = produced
        row = {
            "direction": d,
            "forecast_min": 41.2 if d == "SG_TO_MY" else 25.0,
            "forecast_for": "2026-10-03T04:00:00Z",
            "origin_ts": "2026-10-03T03:30:00Z",
            "model": produced,
        }
        if components is not None:
            row["components"] = components
        rows.append(row)
    return {"source": "fixture", "rows": rows}


def test_served_points_name_the_model_that_produced_them(monkeypatch):
    """Default ``served`` keeps that id at the top level and labels each direction with its producer."""
    def fake(model_id, directions, horizon=30):
        assert model_id == "served" and horizon == 30
        return _fixture_rows(directions, lambda d: main.SERVED_SELECTION[d])

    monkeypatch.setattr(main, "query_forecast", fake)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 200 and payload["model"] == "served"
    assert payload["directions"]["SG_TO_MY"]["model"] == "lin_bq[frozen]"
    assert payload["directions"]["MY_TO_SG"]["model"] == "persistence"
    assert payload["directions"]["SG_TO_MY"]["model"] != payload["directions"]["MY_TO_SG"]["model"]
    for row in payload["directions"].values():
        assert "components" not in row


def test_blend_point_lists_components_and_weights(monkeypatch):
    blend = [
        {"model": "lin_bq[frozen]", "weight": 0.5},
        {"model": "persistence", "weight": 0.5},
    ]

    def fake(model_id, directions, horizon=30):
        assert model_id == "served"
        return _fixture_rows(
            directions,
            lambda d: ("mean[models]", blend) if d == "SG_TO_MY" else "persistence",
        )

    monkeypatch.setattr(main, "query_forecast", fake)
    payload, status, _ = _json(main.forecast(_request()))
    assert status == 200 and payload["model"] == "served"
    sg = payload["directions"]["SG_TO_MY"]
    assert sg["model"] == "mean[models]" and sg["components"] == blend
    assert abs(sum(part["weight"] for part in sg["components"]) - 1) < 1e-9
    assert payload["directions"]["MY_TO_SG"]["model"] == "persistence"
    assert "components" not in payload["directions"]["MY_TO_SG"]


def test_blend_weights_that_do_not_sum_to_one_are_rejected(monkeypatch):
    def fake(model_id, directions, horizon=30):
        return _fixture_rows(
            directions,
            lambda d: ("mean[models]", [{"model": "lin_bq[frozen]", "weight": 0.2}, {"model": "persistence", "weight": 0.2}]),
        )

    monkeypatch.setattr(main, "query_forecast", fake)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 502 and payload == {"error": "Forecast query failed"}


def test_single_model_is_echoed_on_the_point_without_components(monkeypatch):
    def fake(model_id, directions, horizon=30):
        assert model_id == "lin_bq[frozen]"
        return _fixture_rows(directions, lambda d: "lin_bq[frozen]")

    monkeypatch.setattr(main, "query_forecast", fake)
    payload, status, _ = _json(main.forecast(_request(query={"model": "lin_bq[frozen]", "direction": "SG_TO_MY"})))
    assert status == 200 and payload["model"] == "lin_bq[frozen]"
    assert payload["directions"]["SG_TO_MY"]["model"] == "lin_bq[frozen]"
    assert "components" not in payload["directions"]["SG_TO_MY"]


def test_curve_points_name_the_model_at_each_lead(monkeypatch):
    """The public curve switches model by lead time; each point carries that id and no invented blend."""
    origin = pd.Timestamp("2026-10-03 03:30", tz="UTC")
    frame = pd.DataFrame({"direction": list(main.DIRECTIONS), "bin_ts": [origin, origin]})
    monkeypatch.setattr(main, "fetch_latest", lambda directions, now: frame)
    monkeypatch.setattr(main, "servable_origins", lambda bins, now: frame)
    monkeypatch.setattr(main, "fitted_profile", lambda: object())

    def predict(mid, sub, horizon, profile, now):
        return float(horizon), 12.0, {"eval_id": mid, "version": "fixture"}

    monkeypatch.setattr(main, "_predict_one", predict)
    payload = main.forecast_curve(list(main.DIRECTIONS), 2)
    sg = payload["directions"]["SG_TO_MY"]["points"]
    my = payload["directions"]["MY_TO_SG"]["points"]
    assert [p["model"] for p in sg] == ["lin_bq[frozen]", "xgb[maps]", "xgb[maps+prof]", "xgb[maps+prof]"]
    assert [p["model"] for p in my] == ["persistence", "xgb[maps]", "xgb[maps+prof]", "xgb[maps+prof]"]
    assert all("components" not in p for p in sg + my)


def test_curve_note_does_not_claim_the_model_never_won_later(monkeypatch):
    # the study has scattered, uncorrected wins after the cutoff: the note says "not consistent", not "never"
    now = {"t": dt.datetime(2026, 10, 3, 4, 0, tzinfo=UTC)}
    _fake_bigquery(monkeypatch, _bins(), now)
    payload, status, _ = _json(main.forecast(_request(query={"curve": "forecast"})))
    assert status == 200 and "not consistent" in payload["note"] and "no model beat it" not in payload["note"]
