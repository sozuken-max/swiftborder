import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from camera_forecast import plot_profiles
from deep_forecast import plot_seed_mae
from fuzzy_traffic import LevelPartition, plot_confusions, plot_memberships
from joined import _plot_mae_bars
from plots import plot_backtest_method_means, plot_bqml_mae_comparison, plot_holdout_forecast_sample, plot_mae_diff_forest
from run_artifacts import artifact_relpath
from significance import ComparisonResult


def _header(path: Path) -> List[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _comparison() -> ComparisonResult:
    return ComparisonResult(
        label_challenger="xgb",
        label_reference="persist",
        n=100,
        mean_ae_diff_min=-0.1,
        paired_t_pvalue=0.05,
        bootstrap_ci_low_min=-0.2,
        bootstrap_ci_high_min=0.05,
        block_size=6,
        alpha=0.05,
    )


def test_plot_bqml_mae_comparison_writes_file(tmp_path: Path):
    slices = {
        "Persistence": [{"direction": "both", "time_of_day": "all", "mae": 2.5, "rmse": 3.0, "n": 10}],
        "xgb_h30": [{"direction": "both", "time_of_day": "all", "mae": 2.3, "rmse": 2.9, "n": 10}],
    }
    out = tmp_path / "mae.png"
    png, series = plot_bqml_mae_comparison(slices, out)
    assert png == out and out.is_file() and out.stat().st_size > 100
    assert series == out.with_suffix(".csv") and series.is_file()
    assert _header(series) == ["candidate", "direction", "mae_min"]
    rows = _rows(series)
    assert rows[2]["candidate"] == "Persistence" and rows[2]["direction"] == "both" and rows[2]["mae_min"] == "2.5"
    assert rows[0]["mae_min"] == ""  # SG_TO_MY is not in the tiny frame


def test_plot_mae_diff_forest_writes_file(tmp_path: Path):
    out = tmp_path / "forest.png"
    png, series = plot_mae_diff_forest([_comparison()], out)
    assert png == out and out.is_file() and out.stat().st_size > 100
    assert series == out.with_suffix(".csv")
    assert _header(series) == [
        "display_rank",
        "challenger",
        "reference",
        "n",
        "mean_ae_diff_min",
        "bootstrap_ci_low_min",
        "bootstrap_ci_high_min",
        "decision",
    ]
    row = _rows(series)[0]
    assert row["challenger"] == "xgb" and row["reference"] == "persist" and row["n"] == "100"
    assert row["mean_ae_diff_min"] == "-0.1"
    assert row["decision"] == "not significant"


def test_backtest_and_holdout_series_match_the_plotted_slice(tmp_path: Path):
    run = tmp_path / "run"
    scores = pd.DataFrame({"method": ["XGB actual", "Persistence", "XGB actual"], "MAE_min": [1.0, 3.0, 2.0]})
    png, series = plot_backtest_method_means(scores, run / "offline" / "backtest-mae.png")
    assert png.is_file() and series.is_file()
    assert _header(series) == ["order_from_bottom", "method", "mean_mae_min"]
    assert [row["method"] for row in _rows(series)] == ["XGB actual", "Persistence"]
    assert _rows(series)[0]["mean_mae_min"] == "1.5"
    assert artifact_relpath(run, series) == "offline/backtest-mae.csv"

    stamps = [datetime(2026, 9, 26, 1, 35), datetime(2026, 9, 26, 1, 40), datetime(2026, 9, 26, 1, 45)]
    png, series = plot_holdout_forecast_sample(
        stamps,
        [10.0, 11.0, 12.0],
        {"XGB": [9.0, 10.5, 11.5]},
        run / "offline" / "holdout-sample.png",
        max_points=2,
    )
    assert png.is_file() and _header(series) == ["row", "target_ts", "actual_min", "XGB"]
    plotted = _rows(series)
    assert len(plotted) == 2
    assert plotted[0]["target_ts"] == "2026-09-26T01:40:00"
    assert plotted[0]["actual_min"] == "11.0" and plotted[1]["XGB"] == "11.5"
    assert ":" not in artifact_relpath(run, png).split("/", 1)[0]


def test_seed_membership_confusion_profile_and_joined_bars_write_series(tmp_path: Path):
    png, series = plot_seed_mae(
        {"lstm": {"label": "LSTM(64)", "seed_mean_prediction_mae_min": 4.0, "mae_by_seed": [3.5, 4.5], "runs": [{"seed": 0}, {"seed": 1}]}},
        {"Persistence T-60": 4.7},
        tmp_path / "deep-mae-by-seed.png",
        title="seeds",
    )
    assert png.is_file() and series.is_file()
    assert _header(series) == ["kind", "model_index", "model", "seed", "mae_min"]
    kinds = [row["kind"] for row in _rows(series)]
    assert kinds == ["bar", "seed", "seed", "reference"]
    assert _rows(series)[1]["seed"] == "0" and _rows(series)[3]["model"] == "Persistence T-60"

    png, series = plot_memberships(LevelPartition(), tmp_path / "fuzzy-memberships.png")
    assert png.is_file() and series.is_file()
    assert _header(series) == [
        "travel_time_min",
        "membership_light",
        "membership_moderate",
        "membership_heavy",
        "light_max_min",
        "heavy_min_min",
    ]
    assert _rows(series)[0]["light_max_min"] == "20.0"

    cm = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    png, series = plot_confusions({"Fuzzy rule base": cm}, tmp_path / "fuzzy-confusion.png")
    assert png.is_file()
    assert _header(series) == ["model", "actual", "predicted", "count"]
    assert _rows(series)[0] == {"model": "Fuzzy rule base", "actual": "light", "predicted": "light", "count": "1"}

    curves = {
        "tod_min": [0, 600],
        "SG_TO_MY/weekday": [1.0, 2.0],
        "SG_TO_MY/weekend": [1.5, 2.5],
        "MY_TO_SG/weekday": [3.0, 4.0],
        "MY_TO_SG/weekend": [3.5, 4.5],
    }
    png, series = plot_profiles(curves, None, tmp_path / "camfc-profiles.png")
    assert png.is_file() and series.is_file()
    assert _header(series) == ["tod_min", "hour_sgt", "direction", "day_type", "camera_forecast_vehicles", "maps_travel_time_min"]
    first = _rows(series)[0]
    assert first["direction"] == "SG_TO_MY" and first["day_type"] == "weekday"
    assert first["camera_forecast_vehicles"] == "1.0" and first["maps_travel_time_min"] == ""

    png, series = _plot_mae_bars(
        [
            {"candidate": "xgb[maps]", "slice": "both/all", "mae_min": 2.2},
            {"candidate": "persistence", "slice": "both/all", "mae_min": 2.6},
            {"candidate": "xgb[maps]", "slice": "SG_TO_MY/all", "mae_min": 9.0},
        ],
        tmp_path / "joined-mae-by-feature-set.png",
    )
    assert png.is_file()
    assert _header(series) == ["order_from_bottom", "candidate", "slice", "mae_min"]
    assert [row["candidate"] for row in _rows(series)] == ["xgb[maps]", "persistence"]
