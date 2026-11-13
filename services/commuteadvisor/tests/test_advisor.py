import datetime
import json

import pytest

import main

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 11, 13, 5, 0, tzinfo=UTC)  # 13:00 SGT, Friday


def _curve(values_sg, values_my, typical=30.0):
    def direction(values):
        origin = NOW - datetime.timedelta(minutes=20)
        return {
            "origin_ts": origin.isoformat(),
            "points": [
                {"horizon_min": 30 * (i + 1), "forecast_for": (origin + datetime.timedelta(minutes=30 * (i + 1))).isoformat(),
                 "forecast_min": v, "baseline_min": typical, "model": "xgb[maps+prof]"}
                for i, v in enumerate(values)
            ],
        }
    return {"directions": {"SG_TO_MY": direction(values_sg), "MY_TO_SG": direction(values_my)}}


TRAFFIC = {"series": {
    "mandai_to_shell_jb": {"points": [["2026-11-13T12:50:00", 2400, 30], ["2026-11-13T12:55:00", 2220, 31]]},
    "jb_to_woodlands": {"points": [["2026-11-13T12:55:00", 1800, 40]]},
}}

LLM_REPLY = {"headline": "Busier than usual towards Johor.", "tips": ["a", "b", "c"]}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    main._cache.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(main, "fetch_traffic", lambda: TRAFFIC)
    yield
    main._cache.clear()


class _Res:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


def _openai_ok(calls):
    def post(url, headers, json, timeout):
        calls.append(json)
        return _Res(200, {"choices": [{"message": {"content": __import__("json").dumps(LLM_REPLY)}}]})
    return post


def test_calendar_loads_and_knows_which_years_have_school_data():
    cal = main.load_calendar()
    assert datetime.date(2026, 11, 13) in cal["rows"]
    assert 2026 in cal["school_years"]["sg"] and 2026 in cal["school_years"]["my"]
    assert 2027 not in cal["school_years"]["sg"]  # the file has no 2027 school holidays


def test_calendar_facts_for_13_nov_2026():
    facts = main.calendar_facts(datetime.date(2026, 11, 13))
    assert facts["weekday"] == "Friday"
    assert facts["sg_school_holidays"] == "start 2026-11-21 (in 8 days)"
    assert facts["today"]["sg_public_holiday"] == []


def test_calendar_facts_on_a_holiday_and_inside_school_holidays():
    facts = main.calendar_facts(datetime.date(2026, 11, 9))
    assert facts["today"]["sg_public_holiday"] == ["Diwali (observed)"]
    assert facts["today"]["last_day_of_sg_long_weekend"] is True
    assert facts["my_school_holidays"] == "on now, 2026-11-08 to 2026-11-10"


def test_school_holidays_unknown_outside_covered_years():
    facts = main.calendar_facts(datetime.date(2027, 3, 1))
    assert facts["sg_school_holidays"].startswith("unknown")
    assert facts["my_school_holidays"].startswith("unknown")


def test_date_outside_the_calendar():
    assert main.calendar_facts(datetime.date(2030, 1, 1))["in_calendar"] is False


NOW_POINT = {"at": "12:55", "at_iso": "2026-11-13T12:55:00+08:00", "minutes": 37.0}


def test_best_window_worst_and_typical():
    values = [40, 38, 33, 32, 34, 45, 50, 41, 37, 36]
    facts = main.direction_facts(_curve(values, values)["directions"]["SG_TO_MY"], NOW_POINT)
    best = facts["best"]
    assert (best["leave_at"], best["crossing_min"], best["window"], best["leave_now"]) == ("14:40", 32.0, "14:10-15:10", False)
    assert best["window_start_iso"].startswith("2026-11-13T06:10")
    assert facts["worst"] == {"leave_at": "16:10", "crossing_min": 50.0}
    assert facts["vs_typical_avg_min"] == pytest.approx(8.6)
    assert facts["saving_vs_now_min"] == 5.0


def test_leaving_now_can_be_the_best_window():
    facts = main.direction_facts(_curve([26, 30, 40], [26, 30, 40])["directions"]["SG_TO_MY"],
                                 dict(NOW_POINT, minutes=25.0))
    assert facts["best"]["leave_now"] is True
    assert facts["best"]["window"] == "12:55-13:10"


def test_stale_now_reading_is_ignored():
    stale = {"series": {"mandai_to_shell_jb": {"points": [["2026-11-13T12:00:00", 1500, 30]]}}}
    assert main.latest_now(stale, "mandai_to_shell_jb", NOW) is None


def test_profile_points_are_not_forecasts():
    d = _curve([30, 31], [30, 31])["directions"]["SG_TO_MY"]
    d["points"][0]["model"] = "profile"
    assert len(main.direction_facts(d, None)["forecast"]) == 1


def test_latest_now_reads_the_newest_point():
    assert main.latest_now(TRAFFIC, "mandai_to_shell_jb", NOW) == NOW_POINT
    assert main.latest_now({}, "mandai_to_shell_jb") is None


def test_parse_advice_trims_tips_and_rejects_missing_fields():
    assert main.parse_advice(json.dumps(LLM_REPLY))["tips"] == ["a", "b"]
    with pytest.raises(ValueError):
        main.parse_advice(json.dumps({"tips": ["x"]}))


def test_advice_is_cached_per_origin(monkeypatch):
    calls = []
    curve = _curve([40, 35, 30], [30, 30, 30])
    monkeypatch.setattr(main, "fetch_curve", lambda: curve)
    monkeypatch.setattr(main.requests, "post", _openai_ok(calls))
    first = main.advise(NOW)
    main._cache.pop("recent")  # force the origin check
    second = main.advise(NOW)
    assert first is second and len(calls) == 1
    assert first["advice"]["headline"] == LLM_REPLY["headline"]
    sent = json.loads(calls[0]["messages"][1]["content"])
    assert sent["directions"]["SG_TO_MY"]["best"]["leave_at"] == "14:10"
    assert sent["calendar"]["sg_school_holidays"].startswith("start 2026-11-21")
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_llm_failure_still_returns_facts(monkeypatch):
    monkeypatch.setattr(main, "fetch_curve", lambda: _curve([30], [30]))
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: _Res(500, {"error": "boom"}))
    payload = main.advise(NOW)
    assert payload["advice"] is None and payload["advice_error"] == "Advice is unavailable right now."
    assert payload["facts"]["directions"]["MY_TO_SG"]["best"]["crossing_min"] == 30.0


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(main, "fetch_curve", lambda: _curve([30], [30]))
    assert "OPENAI_API_KEY" in main.advise(NOW)["advice_error"]


class _Req:
    def __init__(self, method="GET"):
        self.method = method


def test_http_ok_and_cors(monkeypatch):
    monkeypatch.setattr(main, "fetch_curve", lambda: _curve([30, 31], [30, 31]))
    monkeypatch.setattr(main.requests, "post", _openai_ok([]))
    body, status, headers = main.commute_advice(_Req())
    assert status == 200 and headers["Access-Control-Allow-Origin"] == "*"
    assert json.loads(body)["advice"]["headline"] == LLM_REPLY["headline"]


def test_http_upstream_failure_is_503(monkeypatch):
    def boom():
        raise main.UpstreamError("forecast-api returned 503")
    monkeypatch.setattr(main, "fetch_curve", boom)
    body, status, _ = main.commute_advice(_Req())
    assert status == 503 and "forecast-api" in json.loads(body)["error"]


def test_http_rejects_post_and_answers_preflight():
    assert main.commute_advice(_Req("POST"))[1] == 405
    assert main.commute_advice(_Req("OPTIONS"))[1] == 204
