"""HTTP handler tests with data.gov.sg and Roboflow mocked (no network, no real key)."""

import io
import json

import pytest
import requests
from PIL import Image
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

import main

FRAME_URL = "https://images.example.test/2701.jpg"


def _jpeg(size=(1920, 1080)):
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 40, 40)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeResp:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} upstream detail https://secret.example/path")

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


PREDICTIONS = [
    # foot y = 300 + 20 = 320, well above the 2701 line -> SG-MY
    {"x": 1500, "y": 300, "width": 60, "height": 40, "class": "car", "confidence": 0.9},
    # foot y = 1050 + 20 = 1070 at x=300, below the line (~1051) -> MY-SG
    {"x": 300, "y": 1050, "width": 60, "height": 40, "class": "car", "confidence": 0.8},
    # low confidence
    {"x": 900, "y": 500, "width": 60, "height": 40, "class": "car", "confidence": 0.05},
]


@pytest.fixture
def upstream(monkeypatch):
    """Mock data.gov.sg + frame download + Roboflow. Returns a dict to tweak per test."""
    state = {
        "images_payload": {"items": [{"cameras": [{"camera_id": "2701", "image": FRAME_URL}]}]},
        "frame": FakeResp(content=_jpeg()),
        "workflow": FakeResp(payload={"outputs": [{"predictions": {"predictions": PREDICTIONS}}]}),
        "calls": [],
    }

    def fake_get(url, params=None, timeout=None):
        state["calls"].append(("GET", url, params))
        if url == main.TRAFFIC_IMAGES_API:
            p = state["images_payload"]
            if isinstance(p, Exception):
                raise p
            return FakeResp(payload=p)
        return state["frame"]

    def fake_post(url, json=None, headers=None, timeout=None):
        state["calls"].append(("POST", url, None))
        wf = state["workflow"]
        if isinstance(wf, Exception):
            raise wf
        return wf

    monkeypatch.setattr(main.requests, "get", fake_get)
    monkeypatch.setattr(main.requests, "post", fake_post)
    monkeypatch.setattr(main, "ROBOFLOW_API_KEY", "test-key")
    return state


def _request(method="GET", query=None, body=None):
    builder = EnvironBuilder(method=method, query_string=query, json=body)
    return Request(builder.get_environ())


def _json(resp):
    body, status, headers = resp
    return json.loads(body), status, headers


def test_options_preflight():
    body, status, headers = main.detect(_request("OPTIONS"))
    assert status == 204
    assert "X-Vehicle-Count" in headers["Access-Control-Expose-Headers"]


def test_json_mode_counts_and_directions(upstream):
    payload, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 200
    assert payload["vehicle_count"] == 2
    assert payload["directions"]["sg_my"]["count"] == 1
    assert payload["directions"]["my_sg"]["count"] == 1
    assert "extent" in payload["directions"]["sg_my"]
    assert payload["dividing_line"][0] == [176.0, 1074.0]


def test_body_wins_over_query_and_zero_confidence_is_honoured(upstream):
    payload, _, _ = _json(main.detect(_request("POST", query={"confidence": "0.85"}, body={"format": "json", "confidence": 0})))
    assert payload["min_confidence"] == 0.0
    assert payload["vehicle_count"] == 3  # the 0.05 detection is kept at confidence 0


def test_empty_string_body_values_fall_back_to_query(upstream):
    payload, status, _ = _json(
        main.detect(_request("POST", query={"date_time": "2026-09-20T08:00:00", "format": "json"}, body={"date_time": "", "format": ""}))
    )
    assert status == 200
    assert payload["date_time"] == "2026-09-20T08:00:00"


def test_query_confidence_used_when_body_omits_it(upstream):
    payload, _, _ = _json(main.detect(_request("POST", query={"confidence": "0.85"}, body={"format": "json"})))
    assert payload["min_confidence"] == 0.85
    assert payload["vehicle_count"] == 1


def test_bad_confidence_falls_back_to_default(upstream):
    payload, _, _ = _json(main.detect(_request(query={"format": "json", "confidence": "abc"})))
    assert payload["min_confidence"] == main.DEFAULT_CONFIDENCE


@pytest.mark.parametrize("fmt", ["image", "directional", "unknown-format"])
def test_image_modes_return_jpeg_with_headers(upstream, fmt):
    body, status, headers = main.detect(_request(query={"format": fmt, "date_time": "2026-09-20T08:00:00"}))
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert Image.open(io.BytesIO(body)).format == "JPEG"
    assert headers["X-Vehicle-Count"] == "2"
    assert headers["X-Vehicle-Count-SG-MY"] == "1"
    assert headers["X-Frame-Datetime"] == "2026-09-20T08:00:00"


def test_camera_not_found_is_404(upstream):
    payload, status, _ = _json(main.detect(_request(query={"camera_id": "9999", "format": "json"})))
    assert status == 404


def test_camera_id_matches_integer_upstream_ids(upstream):
    upstream["images_payload"] = {"items": [{"cameras": [{"camera_id": 2701, "image": FRAME_URL}]}]}
    _, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 200


def test_upstream_failures_are_502_without_leaking_detail(upstream):
    upstream["images_payload"] = requests.ConnectionError("dns failure for internal-host")
    payload, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 502
    assert "internal-host" not in payload["error"]


def test_frame_http_error_is_502_and_skips_inference(upstream):
    upstream["frame"] = FakeResp(status=403, content=b"<html>denied</html>")
    payload, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 502
    assert "secret.example" not in payload["error"]
    assert not any(c[0] == "POST" for c in upstream["calls"])


@pytest.mark.parametrize("fmt", ["json", "image", "directional"])
def test_non_image_frame_is_502_and_skips_inference(upstream, fmt):
    upstream["frame"] = FakeResp(content=b"<html>not an image</html>")
    body, status, _ = main.detect(_request(query={"format": fmt}))
    assert status == 502
    assert "valid image" in json.loads(body)["error"]
    assert not any(c[0] == "POST" for c in upstream["calls"])


def test_inference_failures_are_502(upstream):
    upstream["workflow"] = requests.Timeout("roboflow timeout")
    _, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 502
    upstream["workflow"] = FakeResp(payload=ValueError("not json"))
    _, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 502


@pytest.mark.parametrize("fmt", ["json", "image", "directional"])
def test_malformed_predictions_do_not_crash(upstream, fmt):
    upstream["workflow"] = FakeResp(
        payload={
            "outputs": [
                {
                    "predictions": {
                        "predictions": [
                            "not a dict",
                            {"x": 100, "confidence": 0.9},  # no box geometry
                            {"x": "a", "y": 1, "width": 1, "height": 1, "confidence": "high"},
                            {"x": 1500, "y": 300, "width": 60, "height": 40, "confidence": 0.9},
                        ]
                    }
                }
            ]
        }
    )
    body, status, headers = main.detect(_request(query={"format": fmt}))
    assert status == 200
    if fmt == "json":
        payload = json.loads(body)
        assert payload["vehicle_count"] == 2  # non-dict dropped; "high" confidence -> 0.0 dropped
        d = payload["directions"]
        assert d["sg_my"]["count"] + d["my_sg"]["count"] + d["unknown"]["count"] == 2
    else:
        assert headers["X-Vehicle-Count"] == "2"


@pytest.mark.parametrize("bad", ["2026-09-20 08:00:00", "yesterday", "2026-13-40T99:00:00", "2026-09-20T08:00:00\r\nX: y"])
def test_bad_date_time_is_400_before_any_upstream_call(upstream, bad):
    payload, status, _ = _json(main.detect(_request(query={"date_time": bad, "format": "json"})))
    assert status == 400
    assert upstream["calls"] == []


def test_non_string_date_time_in_body_is_400(upstream):
    _, status, _ = _json(main.detect(_request("POST", body={"date_time": 20260920, "format": "json"})))
    assert status == 400


def test_missing_key_is_500_and_makes_no_calls(upstream, monkeypatch):
    monkeypatch.setattr(main, "ROBOFLOW_API_KEY", "")
    payload, status, _ = _json(main.detect(_request(query={"format": "json"})))
    assert status == 500
    assert "not configured" in payload["error"]
    assert upstream["calls"] == []


def test_non_object_json_body_is_ignored(upstream):
    builder = EnvironBuilder(method="POST", data="[1, 2]", content_type="application/json", query_string={"format": "json"})
    payload, status, _ = _json(main.detect(Request(builder.get_environ())))
    assert status == 200


def test_detect_frame_matches_handler(upstream):
    result = main.detect_frame(_jpeg(), "2701", 0.1)
    payload, _, _ = _json(main.detect(_request(query={"format": "json", "confidence": "0.1"})))
    assert len(result["kept"]) == payload["vehicle_count"]
    assert result["summary"] == payload["directions"]
    assert result["image_size"] == (1920, 1080)


def test_detect_frame_rejects_non_image_before_inference(upstream):
    with pytest.raises(main.InvalidImage):
        main.detect_frame(b"nope", "2701")
    assert upstream["calls"] == []


def test_env_float_is_safe(monkeypatch):
    monkeypatch.setenv("X_CONF", "abc")
    assert main._env_float("X_CONF", 0.3) == 0.3
    monkeypatch.setenv("X_CONF", "nan")
    assert main._env_float("X_CONF", 0.3) == 0.3
    monkeypatch.setenv("X_CONF", "0.25")
    assert main._env_float("X_CONF", 0.3) == 0.25


def test_congestion_extent_is_reported():
    assert main._congestion_extent([100, 600], 1000) == pytest.approx(0.5)
    assert main._congestion_extent([100], 1000) == 0.0


def _post_urls(state):
    return [url for method, url, _ in state["calls"] if method == "POST"]


def test_default_model_uses_configured_workflow(upstream):
    body, status, headers = main.detect(_request(query={"format": "directional"}))
    assert status == 200
    assert _post_urls(upstream) == [main._workflow_url(main.ROBOFLOW_WORKFLOW_ID)]
    assert headers["X-Workflow-Id"] == main.ROBOFLOW_WORKFLOW_ID


@pytest.mark.parametrize("model", ["v4", "V6"])
def test_model_param_picks_an_allowlisted_workflow(upstream, model):
    payload, status, _ = _json(main.detect(_request(query={"format": "json", "model": model})))
    expected = main.WORKFLOW_VERSIONS[model.lower()]
    assert status == 200
    assert payload["workflow_id"] == expected
    assert _post_urls(upstream) == [main._workflow_url(expected)]


@pytest.mark.parametrize("model", ["v5", "some-other-workflow", 6])
def test_unknown_model_is_400_before_any_upstream_call(upstream, model):
    payload, status, _ = _json(main.detect(_request(method="POST", body={"model": model})))
    assert status == 400
    assert "model must be one of" in payload["error"]
    assert upstream["calls"] == []


def test_workflow_id_header_is_exposed():
    _, _, headers = main.detect(_request("OPTIONS"))
    assert "X-Workflow-Id" in headers["Access-Control-Expose-Headers"]


def _summary_stub():
    return {"sg_my": {"count": 1, "congestion": "Free Flow"}, "my_sg": {"count": 0, "congestion": "Free Flow"}}


def test_directional_boxes_are_one_pixel_and_unlabelled():
    image = Image.new("RGB", (1920, 1080), (40, 40, 40))
    pred = {"x": 1000, "y": 500, "width": 200, "height": 100, "confidence": 0.9, "direction": main.DIR_SG_MY}
    main._draw_directional(image, [pred], None, _summary_stub())
    x1, y1, x2, y2 = main._box(pred)
    red = main.DIR_COLORS[main.DIR_SG_MY]
    assert image.getpixel((x1 + 50, y1)) == red          # top edge drawn
    assert image.getpixel((x1 + 50, y1 + 1)) == (40, 40, 40)  # 1 px only
    assert image.getpixel((x1 + 50, y1 - 5)) == (40, 40, 40)  # no filled label above the box


def test_directional_banner_is_the_colour_key():
    def banner_colours(preds):
        image = Image.new("RGB", (1920, 1080), (40, 40, 40))
        main._draw_directional(image, preds, None, _summary_stub())
        strip = image.crop((0, 0, 1920, 20))  # top rows: inside the banner, above any box
        return {c for _, c in strip.getcolors(maxcolors=100000)}

    known = {"x": 100, "y": 500, "width": 50, "height": 50, "confidence": 0.9, "direction": main.DIR_MY_SG}
    colours = banner_colours([known])
    assert main.DIR_COLORS[main.DIR_SG_MY] in colours and main.DIR_COLORS[main.DIR_MY_SG] in colours
    assert main.DIR_COLORS[main.DIR_UNKNOWN] not in colours

    unknown = {**known, "x": 300, "direction": main.DIR_UNKNOWN}
    assert main.DIR_COLORS[main.DIR_UNKNOWN] in banner_colours([known, unknown])
