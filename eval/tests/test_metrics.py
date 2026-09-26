import math

from metrics import DIRECTIONS, mae, peak_time_of_day, rmse, score_slices


def test_mae_rmse():
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]
    assert mae(actual, predicted) == (2 + 2 + 3) / 3
    assert rmse(actual, predicted) == math.sqrt((4 + 4 + 9) / 3)


def test_peak_time_of_day_labels():
    assert peak_time_of_day(1, 0) == "morning peak"
    assert peak_time_of_day("1", "0") == "morning peak"
    assert peak_time_of_day(0, 1) == "evening peak"
    assert peak_time_of_day(0, 0) == "other"


def test_direction_names():
    assert "SG_TO_MY" in DIRECTIONS
    assert "MY_TO_SG" in DIRECTIONS


def test_score_slices_by_direction_and_tod():
    rows = [
        {
            "direction": "SG_TO_MY",
            "is_morning_peak": 1,
            "is_evening_peak": 0,
            "y_30": 10.0,
            "predicted": 11.0,
        },
        {
            "direction": "MY_TO_SG",
            "is_morning_peak": 0,
            "is_evening_peak": 1,
            "y_30": 20.0,
            "predicted": 18.0,
        },
    ]
    slices = score_slices(rows, "y_30", "predicted")
    keys = {(s["direction"], s["time_of_day"]) for s in slices}
    assert ("SG_TO_MY", "morning peak") in keys
    assert ("MY_TO_SG", "evening peak") in keys
    assert ("both", "all") in keys
