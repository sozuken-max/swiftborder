"""HTTP handler tests. The BigQuery client is never called."""

import json
import time

from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

import main


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
                "direction": direction,
                "forecast_min": 41.2 if direction == "SG_TO_MY" else 25.0,
                "forecast_for": "2026-10-03T04:00:00Z",
                "origin_ts": "2026-10-03T03:30:00Z",
            }
            for direction in directions
        ],
    }


def _install(monkeypatch):
    calls = []

    def fake(model_id, requested):
        calls.append((model_id, tuple(requested)))
        return _rows(requested)

    monkeypatch.setattr(main, "query_forecast", fake)
    return calls


def setup_function():
    main._cache.clear()
    main._clock = time.monotonic


def teardown_function():
    main._cache.clear()
    main._clock = time.monotonic


def test_options_preflight():
    body, status, headers = main.forecast(_request("OPTIONS"))
    assert body == ""
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert headers["Access-Control-Allow-Methods"] == "GET, OPTIONS"


def test_success_with_fake_query(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, headers = _json(main.forecast(_request(query={"model": "lin_h30"})))
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert payload["model"] == "lin_h30"
    assert payload["source"] == "fixture"
    assert payload["label"] == "maps_duration_in_traffic_min"
    assert payload["horizon_min"] == 30
    assert "true wait" not in json.dumps(payload)
    assert calls == [("lin_h30", ("SG_TO_MY", "MY_TO_SG"))]


def test_unknown_model_is_400(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "not_a_model"})))
    assert status == 400
    assert payload["error"] == "unknown model"
    assert calls == []


def test_omitted_version_is_echoed(monkeypatch):
    _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "lin_h30"})))
    assert status == 200
    assert payload["version"] == main.LIN_H30_VERSION


def test_explicit_version_is_echoed(monkeypatch):
    _install(monkeypatch)
    payload, status, _ = _json(
        main.forecast(_request(query={"model": "xgb_h30", "version": main.XGB_H30_VERSION}))
    )
    assert status == 200
    assert payload["model"] == "xgb_h30"
    assert payload["version"] == main.XGB_H30_VERSION


def test_unknown_version_is_400(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(
        main.forecast(_request(query={"model": "lin_h30", "version": "1999-01-01T00:00:00Z"}))
    )
    assert status == 400
    assert payload["error"] == "unknown version"
    assert calls == []


def test_cache_does_not_query_twice_inside_ttl(monkeypatch):
    calls = _install(monkeypatch)
    now = {"t": 1000.0}
    monkeypatch.setattr(main, "_clock", lambda: now["t"])
    first, status1, _ = _json(main.forecast(_request(query={"model": "lin_h30"})))
    now["t"] += 299
    second, status2, _ = _json(main.forecast(_request(query={"model": "lin_h30"})))
    assert status1 == 200 and status2 == 200
    assert first == second
    assert len(calls) == 1


def test_both_directions_in_one_response(monkeypatch):
    _install(monkeypatch)
    payload, status, _ = _json(
        main.forecast(_request(query={"model": "persistence", "direction": "both"}))
    )
    assert status == 200
    assert set(payload["directions"]) == {"SG_TO_MY", "MY_TO_SG"}
    assert payload["directions"]["SG_TO_MY"]["forecast_min"] == 41.2
    assert payload["directions"]["MY_TO_SG"]["forecast_for"] == "2026-10-03T04:00:00Z"
    assert payload["version"] == main.PERSISTENCE_VERSION


def test_one_direction(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(
        main.forecast(_request(query={"model": "lin_h30", "direction": "MY_TO_SG"}))
    )
    assert status == 200
    assert list(payload["directions"]) == ["MY_TO_SG"]
    assert calls == [("lin_h30", ("MY_TO_SG",))]


def test_bad_direction_and_horizon_are_400(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(
        main.forecast(_request(query={"model": "lin_h30", "direction": "north"}))
    )
    assert status == 400
    assert "direction" in payload["error"]
    payload, status, _ = _json(
        main.forecast(_request(query={"model": "lin_h30", "horizon_min": "1440"}))
    )
    assert status == 400
    assert payload["error"] == "horizon_min must be 30"
    assert calls == []


def test_empty_string_body_falls_back_to_query(monkeypatch):
    _install(monkeypatch)
    payload, status, _ = _json(
        main.forecast(
            _request(
                query={"model": "xgb_h30", "direction": "SG_TO_MY"},
                body={"model": "", "direction": ""},
            )
        )
    )
    assert status == 200
    assert payload["model"] == "xgb_h30"
    assert list(payload["directions"]) == ["SG_TO_MY"]


def test_list_models_includes_a_non_bqml_model_that_is_not_callable(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"list": "models"})))
    assert status == 200
    by_id = {row["id"]: row for row in payload["models"]}
    assert by_id["xgb[maps]"]["family"] == "sklearn"
    assert by_id["xgb[maps]"]["deploy_state"] == "artifact"
    assert by_id["xgb[maps]"]["callable"] is False
    assert by_id["xgb[maps]"]["horizon_min"] == 30
    assert by_id["lin_h30"]["deploy_state"] == "api"
    assert by_id["served"]["deploy_state"] == "production"
    assert by_id["served"]["callable"] is True
    assert by_id["lin_h30"]["callable"] is True
    assert by_id["lstm"]["family"] == "deep"
    assert by_id["lstm"]["callable"] is False
    assert "camfc" not in by_id
    assert calls == []


def test_forecast_of_undeployed_model_is_400(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "xgb[maps]"})))
    assert status == 400
    assert payload["error"] == "model is not deployed"
    assert "forecast_min" not in payload
    assert calls == []


def test_missing_direction_is_503_and_not_cached(monkeypatch):
    calls = []

    def empty(model_id, directions):
        calls.append((model_id, tuple(directions)))
        return {"source": "fixture", "rows": []}

    monkeypatch.setattr(main, "query_forecast", empty)
    payload, status, _ = _json(main.forecast(_request(query={"model": "lin_h30", "direction": "SG_TO_MY"})))
    assert status == 503
    assert payload == {"error": "No recent forecast for the requested direction"}
    again, again_status, _ = _json(main.forecast(_request(query={"model": "lin_h30", "direction": "SG_TO_MY"})))
    assert again_status == 503
    assert again == payload
    assert len(calls) == 2


def test_partial_direction_set_is_503(monkeypatch):
    def partial(model_id, directions):
        return _rows(("SG_TO_MY",))

    monkeypatch.setattr(main, "query_forecast", partial)
    payload, status, _ = _json(main.forecast(_request(query={"model": "persistence"})))
    assert status == 503
    assert payload["error"] == "No recent forecast for the requested direction"


def test_query_failure_is_502_without_sql(monkeypatch):
    def boom(model_id, directions):
        raise RuntimeError("SELECT secret FROM `swiftborder.traffic_prediction.v_training_set`")

    monkeypatch.setattr(main, "query_forecast", boom)
    payload, status, _ = _json(main.forecast(_request(query={"model": "lin_h30"})))
    assert status == 502
    assert payload == {"error": "Forecast query failed"}
    assert "SELECT" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


def test_served_mirrors_the_live_view(monkeypatch):
    calls = _install(monkeypatch)
    payload, status, _ = _json(main.forecast(_request(query={"model": "served"})))
    assert status == 200
    assert payload["model"] == "served"
    assert payload["version"] == main.REGISTRY_VERSION
    assert calls == [("served", ("SG_TO_MY", "MY_TO_SG"))]
    sql = main.MODELS["served"]["sql"]
    assert "v_forecast_recent" in sql and "ML.PREDICT" not in sql


def test_only_production_and_api_states_are_callable():
    for spec in main.MODELS.values():
        assert spec["callable"] == (spec["deploy_state"] in main.CALLABLE_STATES)
    assert [m for m, s in main.MODELS.items() if s["deploy_state"] == "production"] == ["served"]
