"""Layer A scorer on a hand-built fixture with known IoUs (see fixtures/layer_a)."""

from pathlib import Path

import pytest
from PIL import Image

import layer_a as la
from run_artifacts import validate_manifest

FIX = Path(__file__).parent / "fixtures" / "layer_a"
# car AP@0.5: sorted .9 TP, .8 FP, .7 TP, .6 FP, .05 FP with 2 GT -> 101-point AP = (51 + 50 * 2/3) / 101
CAR_AP50 = (51 + 50 * 2 / 3) / 101
# offset box IoU = 81/119 = 0.68: TP at 0.50-0.65 (4 thresholds), FP at 0.70-0.95 (6) -> AP 51/101
CAR_AP_STRICT = 51 / 101


@pytest.fixture
def data():
    return la.load_coco(FIX / "_annotations.coco.json"), la.load_predictions(FIX / "predictions.json"), la.load_frame_times(FIX / "frame_times.csv")


def test_iou():
    a = la.Box(0, 0, 10, 10, "car")
    assert la.iou(a, la.Box(1, 1, 11, 11, "car")) == pytest.approx(81 / 119)
    assert la.iou(a, la.Box(20, 20, 30, 30, "car")) == 0.0
    assert la.iou(a, a) == 1.0


def test_average_precision_hand_computed():
    scored = [(0.9, True), (0.8, False), (0.7, True), (0.6, False), (0.05, False)]
    assert la.average_precision(scored, 2) == pytest.approx(CAR_AP50)
    assert la.average_precision([], 2) == 0.0
    assert la.average_precision([(0.5, False)], 0) != la.average_precision([(0.5, False)], 0)  # NaN


def test_all_slice_matches_hand_calculation(data):
    gt, preds, times = data
    rows = la.score(gt, preds, confidence=0.1, camera=None, times=times)
    all_ = next(r for r in rows if r["slice"] == "all")
    assert all_["n"] == 3
    assert all_["map50"] == pytest.approx((CAR_AP50 + 1.0) / 2)
    car_mean = (4 * CAR_AP50 + 6 * CAR_AP_STRICT) / 10
    assert all_["map50_95"] == pytest.approx((car_mean + 1.0) / 2)
    assert all_["precision"] == pytest.approx(3 / 5)  # conf >= 0.1: 3 TP, 2 FP
    assert all_["recall"] == pytest.approx(1.0)
    assert all_["count_mae"] == pytest.approx(2 / 3)
    assert all_["count_bias"] == pytest.approx(2 / 3)


def test_day_night_split_uses_frame_times(data):
    gt, preds, times = data
    rows = {r["slice"]: r for r in la.score(gt, preds, times=times, camera=None)}
    assert rows["day"]["n"] == 2 and rows["night"]["n"] == 1
    assert rows["day"]["map50"] == pytest.approx(CAR_AP50)
    assert rows["night"]["map50"] == pytest.approx(1.0)  # car has no GT at night -> not averaged
    assert rows["night"]["count_mae"] == 0.0  # the 0.05 car is below the confidence cut


def test_brightness_fallback(tmp_path, data):
    gt, preds, _ = data
    Image.new("L", (32, 32), 10).save(tmp_path / "img_night.jpg")
    Image.new("L", (32, 32), 200).save(tmp_path / "img_day.jpg")
    assert la.light_of("img_night.jpg", gt["img_night.jpg"], {}, tmp_path) == "night"
    assert la.light_of("img_day.jpg", gt["img_day.jpg"], {}, tmp_path) == "day"
    assert la.light_of("img_empty.jpg", gt["img_empty.jpg"], {}, tmp_path) == "unknown"


def test_yolo_and_coco_give_the_same_boxes(tmp_path, data):
    gt, _, _ = data
    labels = tmp_path / "labels"
    labels.mkdir()
    classes = ["car", "truck"]
    for name, rec in gt.items():
        lines = []
        for b in rec.boxes:
            cx, cy = (b.x1 + b.x2) / 2 / rec.width, (b.y1 + b.y2) / 2 / rec.height
            w, h = (b.x2 - b.x1) / rec.width, (b.y2 - b.y1) / rec.height
            lines.append(f"{classes.index(b.cls)} {cx} {cy} {w} {h}")
        (labels / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    yolo = la.load_yolo(labels, classes, image_size=(1920, 1080))
    assert set(yolo) == set(gt)
    for name in gt:
        a = sorted((b.cls, round(b.x1, 6), round(b.y1, 6), round(b.x2, 6), round(b.y2, 6)) for b in gt[name].boxes)
        b = sorted((x.cls, round(x.x1, 6), round(x.y1, 6), round(x.x2, 6), round(x.y2, 6)) for x in yolo[name].boxes)
        assert a == b


def test_per_direction_count_error_uses_the_2701_line():
    rec = la.ImageRecord("f.jpg", 1920, 1080, [la.Box.from_center(1500, 300, 60, 40, "car"), la.Box.from_center(300, 1050, 60, 40, "car")])
    preds = {"f.jpg": [la.Box.from_center(1500, 300, 60, 40, "car", 0.9)]}  # misses the MY-SG car
    out = la.count_errors({"f.jpg": rec}, preds, ["f.jpg"], camera="2701")
    assert out["count_mae_SG-MY"] == 0.0
    assert out["count_mae_MY-SG"] == 1.0


def test_malformed_predictions_are_skipped(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"a.jpg": [{"x": 1}, {"x": 1, "y": 1, "width": 2, "height": 2, "confidence": 0.5}]}', encoding="utf-8")
    assert len(la.load_predictions(p)["a.jpg"]) == 1


def test_cli_writes_a_valid_layer_a_component(tmp_path, capsys):
    out = tmp_path / "layer_a.json"
    code = la.main(
        ["--coco", str(FIX / "_annotations.coco.json"), "--predictions", str(FIX / "predictions.json"), "--frame-times", str(FIX / "frame_times.csv"), "--json-out", str(out)]
    )
    assert code == 0
    assert "map50" in capsys.readouterr().out
    import json

    comp = json.loads(out.read_text())
    manifest = {
        "schema_version": 2, "run_id": "r", "created_at_utc": "x", "components": ["layer_a"],
        "provenance": {"git_sha": "x", "git_dirty": False, "code_sha256": "y"}, "layer_a": comp,
    }
    assert validate_manifest(manifest) == []
    assert comp["window"]["start"].startswith("2026-09-20T08:00")
