"""model=local: the in-container ONNX model. Roboflow and data.gov.sg stay mocked; no key, no network."""

import os

import numpy as np
import pytest

import main
import test_handler
from test_handler import _jpeg, _json, _post_urls, _request, upstream  # noqa: F401 (fixture)


def _raw(*detections, anchors=8):
    """Head output (7, anchors): cx, cy, w, h, then bus/car/truck scores, in letterboxed pixels."""
    raw = np.zeros((7, anchors), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(detections):
        raw[:4, i] = (cx, cy, w, h)
        raw[4 + cls, i] = score
    return raw


class FakeInput:
    name = "images"
    shape = [1, 3, 736, 1280]


class FakeSession:
    """Stands in for onnxruntime: one car at the letterboxed centre of a 1920x1080 frame."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, feeds):
        self.calls += 1
        assert feeds["images"].shape == (1, 3, 736, 1280)
        # Frame (900..1020, 500..580) -> letterboxed x * 2/3, y * 2/3 + 8
        return [_raw((640, 368, 80, 160 / 3, 1, 0.9))[None]]


@pytest.fixture
def fake_session(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(main, "_local_session", session)
    return session


def test_letterbox_matches_ultralytics_geometry():
    tensor, scale, pad = main._letterbox(_jpeg((1920, 1080)), (736, 1280))
    assert tensor.shape == (1, 3, 736, 1280) and tensor.dtype == np.float32
    assert scale == pytest.approx(2 / 3)
    assert pad == (0, 8)
    assert tensor[0, :, 0, 0] == pytest.approx([114 / 255] * 3)  # grey padding band
    assert tensor[0, :, 735, 0] == pytest.approx([114 / 255] * 3)


def test_resize_halves_by_averaging_pixel_pairs():
    pixels = np.array([[[0], [100], [200], [40]]] * 2, dtype=np.uint8)  # 2 x 4, one channel
    out = main._resize_linear(pixels, 2, 1)
    assert out[:, :, 0].tolist() == [[50, 120]]


def test_resize_keeps_flat_image_flat():
    out = main._resize_linear(np.full((1080, 1920, 3), 77, dtype=np.uint8), 1280, 720)
    assert out.shape == (720, 1280, 3)
    assert (out == 77).all()


def test_predictions_are_unletterboxed_into_frame_pixels():
    preds = main._local_predictions(_raw((640, 368, 80, 160 / 3, 1, 0.9)), 2 / 3, (0, 8), (1920, 1080))
    assert len(preds) == 1
    p = preds[0]
    assert (p["x"], p["y"], p["width"], p["height"]) == pytest.approx((960, 540, 120, 80), abs=0.1)
    assert p["class"] == "car" and p["class_id"] == 1 and p["confidence"] == pytest.approx(0.9)


def test_nms_suppresses_within_a_class_only():
    raw = _raw(
        (100, 100, 50, 50, 1, 0.9),
        (102, 101, 50, 50, 1, 0.6),  # same car, lower score: suppressed
        (101, 100, 50, 50, 2, 0.5),  # same place, other class: kept (class-aware, like Ultralytics)
        (400, 300, 40, 40, 1, 0.0005),  # under the candidate floor
    )
    preds = main._local_predictions(raw, 1.0, (0, 0), (1280, 736))
    assert [(p["class"], p["confidence"]) for p in preds] == [("car", 0.9), ("truck", 0.5)]


def test_boxes_are_clipped_and_padding_only_boxes_dropped():
    raw = _raw((5, 100, 30, 20, 0, 0.8), (640, 2, 40, 4, 1, 0.7))  # second box lies in the top padding
    preds = main._local_predictions(raw, 2 / 3, (0, 8), (1920, 1080))
    assert len(preds) == 1
    assert preds[0]["x"] - preds[0]["width"] / 2 == pytest.approx(0)


def test_handler_local_needs_no_roboflow_key_and_makes_no_billed_call(upstream, fake_session, monkeypatch):
    monkeypatch.setattr(main, "ROBOFLOW_API_KEY", "")
    payload, status, _ = _json(main.detect(_request(query={"format": "json", "model": "LOCAL"})))
    assert status == 200
    assert payload["workflow_id"] == main.LOCAL_MODEL_ID
    assert payload["vehicle_count"] == 1
    assert payload["predictions"][0]["direction"] in (main.DIR_SG_MY, main.DIR_MY_SG)
    assert _post_urls(upstream) == []
    assert fake_session.calls == 1


@pytest.mark.parametrize("fmt", ["image", "directional"])
def test_handler_local_image_headers(upstream, fake_session, fmt):
    _, status, headers = main.detect(_request(query={"format": fmt, "model": "local"}))
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["X-Workflow-Id"] == main.LOCAL_MODEL_ID
    assert headers["X-Vehicle-Count"] == "1"


def test_unknown_model_error_lists_local(upstream):
    payload, status, _ = _json(main.detect(_request(query={"model": "v5"})))
    assert status == 400
    assert "local" in payload["error"]


def test_missing_model_file_is_503(upstream, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_local_session", None)
    monkeypatch.setattr(main, "LOCAL_MODEL_PATH", str(tmp_path / "missing.onnx"))
    payload, status, _ = _json(main.detect(_request(query={"format": "json", "model": "local"})))
    assert status == 503
    assert payload["error"] == "Local model is not available"
    assert _post_urls(upstream) == []


def test_detect_frame_local_matches_handler(upstream, fake_session):
    result = main.detect_frame(_jpeg(), "2701", 0.1, main.LOCAL_MODEL_ID)
    payload, _, _ = _json(main.detect(_request(query={"format": "json", "model": "local"})))
    assert result["summary"] == payload["directions"]


def test_checked_in_model_loads_and_runs(monkeypatch):
    """The ONNX file the Dockerfile copies: loads, has the 736x1280 input, and scores a frame."""
    assert os.path.exists(main.LOCAL_MODEL_PATH)
    monkeypatch.setattr(main, "_local_session", None)
    session = main._get_local_session()
    assert session.get_inputs()[0].shape == [1, 3, 736, 1280]
    assert session.get_outputs()[0].shape == [1, 7, 19320]
    result = main.detect_frame(_jpeg(), "2701", 0.5, main.LOCAL_MODEL_ID)
    assert result["image_size"] == (1920, 1080)
    assert isinstance(result["kept"], list)


def test_truncated_frame_is_502_on_local(upstream, fake_session):
    whole = _jpeg()
    upstream["frame"].content = whole[: len(whole) // 3]  # header intact, body cut off
    payload, status, _ = _json(main.detect(_request(query={"format": "json", "model": "local"})))
    assert status == 502
    assert payload["error"] == "Source frame is not a valid image"
    assert fake_session.calls == 0


def _pred(x, y, w, h, conf, cls="car"):
    return {"x": x, "y": y, "width": w, "height": h, "confidence": conf, "class": cls}


def test_overlap_suppression_drops_cross_class_duplicates_only():
    preds = [
        _pred(100, 100, 50, 40, 0.9, "truck"),
        _pred(101, 100, 50, 40, 0.4, "bus"),  # same vehicle, other class: each 98% inside the other
        _pred(140, 100, 50, 40, 0.6),  # beside it: 10/50 of its width overlaps -> kept
        _pred(400, 400, 50, 40, 0.2),
    ]
    kept, dropped = main._suppress_overlaps(preds, 0.8)
    assert dropped == 1
    assert [p["confidence"] for p in kept] == [0.9, 0.6, 0.2]  # input order kept


def test_small_box_inside_big_box_is_never_dropped():
    # Car fully inside a truck's box: 100% of the car is covered, but only 12% of the truck.
    truck = _pred(100, 50, 200, 100, 0.8, "truck")
    car = _pred(120, 70, 60, 40, 0.6)
    kept, dropped = main._suppress_overlaps([truck, car], 0.8)
    assert dropped == 0 and len(kept) == 2


def test_both_boxes_must_be_covered_past_the_cut():
    # 60x40 and 66x44 on one car: the smaller is fully inside, the larger 83% covered -> one vehicle.
    big, small = _pred(100, 100, 66, 44, 0.7), _pred(100, 100, 60, 40, 0.3)
    assert main._suppress_overlaps([big, small], 0.8) == ([big], 1)
    # 60x40 inside 70x50: the larger is only 69% covered -> both kept.
    big = _pred(100, 100, 70, 50, 0.7)
    assert main._suppress_overlaps([big, small], 0.8) == ([big, small], 0)


def test_neighbours_in_a_queue_survive():
    # Two 60x40 cars, the second 30 px along and 10 px up: 37.5% overlap.
    kept, dropped = main._suppress_overlaps([_pred(100, 100, 60, 40, 0.7), _pred(130, 90, 60, 40, 0.6)], 0.8)
    assert dropped == 0 and len(kept) == 2


class TwinSession(FakeSession):
    """One car scored twice: as car and as truck at the same place (class-aware NMS keeps both)."""

    def run(self, _outputs, feeds):
        self.calls += 1
        return [_raw((640, 368, 80, 160 / 3, 1, 0.9), (641, 368, 80, 160 / 3, 2, 0.4))[None]]


@pytest.fixture
def twin_session(monkeypatch):
    session = TwinSession()
    monkeypatch.setattr(main, "_local_session", session)
    return session


def test_local_suppresses_overlaps_by_default(upstream, twin_session):
    payload, status, headers = _json(main.detect(_request(query={"format": "json", "model": "local"})))
    assert status == 200
    assert payload["vehicle_count"] == 1
    assert payload["max_overlap"] == main.LOCAL_MAX_OVERLAP
    assert payload["overlap_suppressed"] == 1
    assert payload["predictions"][0]["class"] == "car"


def test_overlap_param_one_turns_suppression_off(upstream, twin_session):
    payload, _, _ = _json(main.detect(_request(query={"format": "json", "model": "local", "overlap": "1"})))
    assert payload["vehicle_count"] == 2
    assert payload["max_overlap"] is None
    assert payload["overlap_suppressed"] == 0


def test_roboflow_models_do_not_suppress_unless_asked(upstream):
    dup = {"x": 1500, "y": 300, "width": 60, "height": 40, "class": "truck", "confidence": 0.5}
    upstream["workflow"]._payload["outputs"][0]["predictions"]["predictions"] = [*test_handler.PREDICTIONS, dup]
    payload, _, _ = _json(main.detect(_request(query={"format": "json", "model": "v6"})))
    assert payload["vehicle_count"] == 3 and payload["max_overlap"] is None
    payload, _, _ = _json(main.detect(_request(query={"format": "json", "model": "v6", "overlap": "0.8"})))
    assert payload["vehicle_count"] == 2 and payload["overlap_suppressed"] == 1


@pytest.mark.parametrize("raw,expected", [(None, None), ("", None), ("abc", None), ("-1", None), ("nan", None), ("0.5", 0.5), ("3", 1.0)])
def test_parse_overlap(raw, expected):
    assert main._parse_overlap(raw) == expected


def test_default_cut_drops_a_shifted_echo_on_one_car():
    # Confident box plus a weak echo shifted 12 px and 6 px: each covers 68% of the other (0.8 kept both).
    box, echo = _pred(100, 100, 60, 40, 0.7), _pred(112, 106, 60, 40, 0.15)
    assert main._mutual_overlap(main._box(box), main._box(echo)) == pytest.approx(0.8 * 0.85)
    assert main._suppress_overlaps([box, echo], main.LOCAL_MAX_OVERLAP) == ([box], 1)
    # Two cars in a queue, 30 px along and 10 px up (37.5% overlap), both stay at the default.
    kept, dropped = main._suppress_overlaps([box, _pred(130, 90, 60, 40, 0.6)], main.LOCAL_MAX_OVERLAP)
    assert dropped == 0 and len(kept) == 2
