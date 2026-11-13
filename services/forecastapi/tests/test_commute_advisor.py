import datetime
import json

import pytest

import commute_advisor as main  # the module under test (named main in the old service tests)
import main as api

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

LLM_REPLY = {"insight": "To Johor runs 8 min slower than usual,\nlikely the Friday evening rush.", "tips": ["a", "b", "c"]}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    main._cache.clear()
    main._key.update(value=None, at=-1e9, error=None)
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
    # leave-home times: each route slot minus APPROACH_MIN (30 min)
    assert (best["leave_at"], best["crossing_min"], best["window"], best["leave_now"]) == ("14:10", 32.0, "13:40-14:40", False)
    assert best["window_start_iso"].startswith("2026-11-13T05:40")
    assert facts["worst"] == {"leave_at": "15:40", "crossing_min": 50.0}
    assert facts["vs_typical_avg_min"] == pytest.approx(8.6)
    assert facts["biggest_gap_vs_typical"] == {"leave_at": "15:40", "crossing_min": 50.0, "typical_min": 30.0, "diff_min": 20.0}
    assert facts["saving_vs_now_min"] == 5.0


def test_first_slot_is_leaving_home_now():
    # route slot 13:10 SGT minus the 30-min drive: leave home at 12:40; the 12:55 route reading is context only
    facts = main.direction_facts(_curve([26, 30, 40], [26, 30, 40])["directions"]["SG_TO_MY"],
                                 dict(NOW_POINT, minutes=20.0))
    assert facts["best"]["leave_now"] is True and facts["best"]["window"] == "12:40-12:40"
    assert facts["forecast"][0]["leave_at"] == "12:40" and facts["now"]["minutes"] == 20.0
    assert facts["decision"]["scenario"] == "leave_now"


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
    advice = main.parse_advice(json.dumps(LLM_REPLY))
    assert advice["tips"] == ["a", "b"]
    assert advice["insight"] == "To Johor runs 8 min slower than usual, likely the Friday evening rush."  # one line
    with pytest.raises(ValueError):
        main.parse_advice(json.dumps({"tips": ["x"]}))


def test_advice_is_cached_per_origin(monkeypatch):
    calls = []
    curve = _curve([40, 35, 30], [30, 30, 30])
    monkeypatch.setattr(main.requests, "post", _openai_ok(calls))
    first = main.advise(curve, NOW)
    second = main.advise(curve, NOW)
    assert first is second and len(calls) == 1
    assert first["advice"]["insight"].startswith("To Johor runs 8 min slower")
    sent = json.loads(calls[0]["messages"][1]["content"])
    assert sent["directions"]["SG_TO_MY"]["best"]["leave_at"] == "13:40"  # route 14:10 minus the drive
    assert "leave home" in sent["directions"]["SG_TO_MY"]["times_are"]
    assert sent["calendar"]["sg_school_holidays"].startswith("start 2026-11-21")
    assert calls[0]["response_format"] == {"type": "json_object"}
    newer = _curve([40, 35, 30], [30, 30, 30])
    newer["directions"]["SG_TO_MY"]["origin_ts"] = "2026-11-13T04:50:00+00:00"
    main.advise(newer, NOW)
    assert len(calls) == 2  # a new forecast origin asks again


def test_llm_failure_still_returns_facts_and_is_not_retried_per_visit(monkeypatch):
    calls = []
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: calls.append(1) or _Res(500, {"error": "boom"}))
    curve = _curve([30], [30])
    payload = main.advise(curve, NOW)
    assert payload["advice"] is None and payload["advice_error"] == "Advice is unavailable right now."
    assert payload["facts"]["directions"]["MY_TO_SG"]["best"]["crossing_min"] == 30.0
    main.advise(curve, NOW)
    assert len(calls) == 1


def test_key_comes_from_secret_manager_when_not_in_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    reads = []
    monkeypatch.setattr(main, "_read_secret", lambda: reads.append(1) or "sm-key")
    assert main.openai_key() == "sm-key" and main.openai_key() == "sm-key" and len(reads) == 1


def test_missing_secret_is_reported_and_not_retried_per_visit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    reads = []

    def boom():
        reads.append(1)
        raise RuntimeError("Secret Manager returned 403 for openai-api-key")
    monkeypatch.setattr(main, "_read_secret", boom)
    payload = main.advise(_curve([30], [30]), NOW)
    assert payload["advice"] is None and payload["advice_error"] == "The OpenAI key is not available."
    with pytest.raises(RuntimeError):
        main.openai_key()
    assert len(reads) == 1  # the failed read is remembered for KEY_RETRY_S


# --- forecast-api endpoint (?advice=commute) -------------------------------------------------


def _get(query):
    from werkzeug.test import EnvironBuilder
    from werkzeug.wrappers import Request
    body, status, headers = api.forecast(Request(EnvironBuilder(method="GET", query_string=query).get_environ()))
    return json.loads(body), status, headers


def test_endpoint_uses_the_cached_5_hour_curve(monkeypatch):
    seen = []
    monkeypatch.setattr(api, "cached_curve", lambda d, h: seen.append((tuple(d), h)) or _curve([30, 31], [30, 31]))
    monkeypatch.setattr(main.requests, "post", _openai_ok([]))
    payload, status, headers = _get({"advice": "commute"})
    assert status == 200 and seen == [(("SG_TO_MY", "MY_TO_SG"), 5)]
    assert "likely" in payload["advice"]["insight"] and headers["Access-Control-Allow-Origin"]


def test_endpoint_errors(monkeypatch):
    assert _get({"advice": "other"})[1] == 400

    def none(d, h):
        raise api.NoRecentForecast(d)
    monkeypatch.setattr(api, "cached_curve", none)
    assert _get({"advice": "commute"})[1] == 503


# --- decision tree (decide) ------------------------------------------------------------------


def _slots(now_min, values):
    t0 = datetime.datetime(2026, 11, 13, 6, 40, tzinfo=UTC)  # 14:40 SGT
    return [{"at": t0, "min": now_min}] + [
        {"at": t0 + datetime.timedelta(minutes=30 * (i + 1)), "min": v} for i, v in enumerate(values)]


def _sgt(iso):
    return datetime.datetime.fromisoformat(iso).astimezone(main.SGT).strftime("%H:%M")


def test_decide_leave_now_before_it_gets_worse():
    # live 13 Nov 14:40, To Johor: 4:10 PM is only 1.9 min faster than now
    d = main.decide(_slots(37.2, [36.5, 38.7, 35.3, 39.3, 45.7, 47.1, 48.6, 47.6, 46.8, 46.2]))
    assert d["scenario"] == "leave_now" and d["crossing_min"] == 37.2
    assert (_sgt(d["window_end_iso"]), _sgt(d["worse_from_iso"]), _sgt(d["peak"]["at_iso"])) == ("16:40", "17:10", "18:10")


def test_decide_leave_now_or_after_the_peak():
    # live 13 Nov 14:40, To Singapore: 7:40 PM (28.5) is no better than now (28.6)
    d = main.decide(_slots(28.6, [31.8, 35.6, 44.8, 50.5, 49.6, 47.3, 43.6, 36.9, 31.9, 28.5]))
    assert d["scenario"] == "leave_now_or_later"
    assert (_sgt(d["window_end_iso"]), _sgt(d["worse_from_iso"]), _sgt(d["reopen_iso"])) == ("15:10", "15:40", "19:40")
    assert d["peak"]["crossing_min"] == 50.5


def test_decide_wait_for_a_real_saving():
    d = main.decide(_slots(48.0, [47.0, 44.0, 38.0, 37.5, 39.0, 45.0]))
    assert d["scenario"] == "wait" and d["saving_min"] == 10.5 and d["crossing_min"] == 37.5
    assert (_sgt(d["window_start_iso"]), _sgt(d["window_end_iso"])) == ("16:10", "17:10")


def test_decide_steady():
    d = main.decide(_slots(25.0, [26.0, 24.0, 27.5, 25.5]))
    assert d["scenario"] == "steady" and d["range_min"] == [24.0, 27.5]


def test_decide_worse_from_the_first_slot():
    d = main.decide(_slots(30.0, [36.0, 40.0, 42.0]))
    assert d["scenario"] == "leave_now" and d["window_end_iso"] == d["window_start_iso"]


def test_decision_is_in_the_facts():
    facts = main.direction_facts(_curve([40, 35, 30], [30, 30, 30])["directions"]["SG_TO_MY"], NOW_POINT)
    assert facts["decision"]["scenario"] == "wait"


def test_live_13_nov_with_the_drive_to_the_checkpoint():
    # live 13 Nov 14:40 SGT origin; route slots from 15:10, so leave-home times from 14:40
    t0 = datetime.datetime(2026, 11, 13, 6, 40, tzinfo=UTC)

    def curve(vals):
        return {"origin_ts": t0.isoformat(), "points": [
            {"forecast_for": (t0 + datetime.timedelta(minutes=30 * (i + 1))).isoformat(), "forecast_min": v,
             "baseline_min": v, "model": "xgb"} for i, v in enumerate(vals)]}
    jb = main.direction_facts(curve([36.5, 38.7, 35.3, 39.3, 45.7, 47.1, 48.6, 47.6, 46.8, 46.2]), None)["decision"]
    sg = main.direction_facts(curve([31.8, 35.6, 44.8, 50.5, 49.6, 47.3, 43.6, 36.9, 31.9, 28.5]), None)["decision"]
    assert jb["scenario"] == "leave_now" and _sgt(jb["window_end_iso"]) == "16:10" and _sgt(jb["worse_from_iso"]) == "16:40"
    assert sg["scenario"] == "leave_now_or_later"
    assert (_sgt(sg["window_end_iso"]), _sgt(sg["reopen_iso"]), sg["reopen_crossing_min"]) == ("15:10", "18:40", 31.9)
