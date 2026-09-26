"""Geometry and congestion tests. No network and no Roboflow key."""

import main


LINE = [(0.0, 100.0), (100.0, 0.0)]
FRAME = (1920, 1080)


def _box(x, y, height=20.0, width=10.0):
    return {"x": x, "y": y, "width": width, "height": height, "confidence": 0.9, "class": "car"}


def test_y_on_line_interpolates_midpoint():
    assert main._y_on_line(50.0, LINE) == 50.0


def test_y_on_line_is_none_outside_the_span():
    assert main._y_on_line(-1.0, LINE) is None
    assert main._y_on_line(101.0, LINE) is None


def test_classify_direction_uses_the_foot_point():
    # Centre y=10, height 20 -> foot y=20, which is above the line at x=0 (y=100).
    assert main._classify_direction(_box(0, 10), LINE) == main.DIR_SG_MY
    # Centre y=100, height 20 -> foot y=110, on or below the line.
    assert main._classify_direction(_box(0, 100), LINE) == main.DIR_MY_SG
    # Foot exactly on the line is MY-SG.
    assert main._classify_direction(_box(0, 90), LINE) == main.DIR_MY_SG


def test_classify_direction_unknown_when_off_the_line_or_unreadable():
    assert main._classify_direction(_box(-5, 10), LINE) == main.DIR_UNKNOWN
    assert main._classify_direction({"x": 0}, LINE) == main.DIR_UNKNOWN
    assert main._classify_direction(_box(0, 10), None) == main.DIR_UNKNOWN
    assert main._classify_direction(_box(0, 10), []) == main.DIR_UNKNOWN


def test_congestion_bands_and_empty_queue():
    assert main._congestion_level([], 1080) == "Free Flow"
    assert main._congestion_level([0], 0) == "Free Flow"
    height = 1000
    assert main._congestion_level([0, 200], height) == "Free Flow"
    assert main._congestion_level([0, 250], height) == "Quarter Way"
    assert main._congestion_level([0, 500], height) == "Half Way"
    assert main._congestion_level([0, 750], height) == "Back to Back"


def test_summarize_directions_counts_sum_to_detections():
    predictions = [
        _box(0, 10),
        _box(0, 100),
        _box(-5, 10),
    ]
    summary = main._summarize_directions(predictions, LINE, (200, 1000))
    total = summary["sg_my"]["count"] + summary["my_sg"]["count"] + summary["unknown"]["count"]
    assert total == len(predictions)
    assert summary["available"] is True
    assert {p["direction"] for p in predictions} == {
        main.DIR_SG_MY,
        main.DIR_MY_SG,
        main.DIR_UNKNOWN,
    }


def test_dividing_line_scales_the_2701_reference_frame():
    points = main._dividing_line("2701", FRAME)
    assert points is not None
    assert points[0] == (176.0, 1074.0)
    half = main._dividing_line("2701", (960, 540))
    assert half[0] == (88.0, 537.0)


def test_dividing_line_rejects_unknown_camera_and_unsorted_x():
    assert main._dividing_line("2702", FRAME) is None
    assert main._dividing_line("2701", None) is None
    unsorted = {
        "2701": {
            "reference_size": [100, 100],
            "points": [[10, 10], [5, 20]],
        }
    }
    original = main.DIVIDING_LINES
    main.DIVIDING_LINES = unsorted
    try:
        assert main._dividing_line("2701", (100, 100)) is None
    finally:
        main.DIVIDING_LINES = original


def test_extract_predictions_degrades_on_a_changed_payload():
    assert main._extract_predictions([{"predictions": {"predictions": [{"class": "car"}]}}]) == [
        {"class": "car"}
    ]
    assert main._extract_predictions([]) == []
    assert main._extract_predictions([{"predictions": {}}]) == []


def test_load_dividing_lines_falls_back_on_bad_json(monkeypatch):
    monkeypatch.setenv("DIVIDING_LINES", "{")
    assert main._load_dividing_lines() == main.DEFAULT_DIVIDING_LINES
    monkeypatch.setenv(
        "DIVIDING_LINES",
        '{"2702": {"reference_size": [10, 10], "points": [[0, 1], [2, 3]]}}',
    )
    loaded = main._load_dividing_lines()
    assert "2702" in loaded
