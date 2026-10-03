import csv
import json
from pathlib import Path

import numpy as np
import pytest

from camera_forecast import plot_profiles
from deep_forecast import plot_seed_mae
from fuzzy_traffic import LevelPartition, plot_confusions, plot_memberships
from plots import plot_bqml_mae_comparison, plot_mae_diff_forest
from replay_series import REPORT_SNAPSHOT, replay_series_csvs
from significance import ComparisonResult


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _comparison() -> ComparisonResult:
    return ComparisonResult(
        label_challenger="xgb",
        label_reference="persist",
        n=4,
        mean_ae_diff_min=-0.25,
        paired_t_pvalue=0.04,
        bootstrap_ci_low_min=-0.4,
        bootstrap_ci_high_min=-0.05,
        block_size=6,
        alpha=0.05,
        dm_stat=-2.0,
        dm_pvalue=0.01,
        dm_pvalue_reference=0.99,
        blocks_per_resample=12,
        holm_adjusted_p=0.02,
    )


def _manifest() -> dict:
    compared = _comparison().to_dict()
    curves = {
        "tod_min": [0, 600],
        "SG_TO_MY/weekday": [1.0, 2.0],
        "SG_TO_MY/weekend": [1.5, 2.5],
        "MY_TO_SG/weekday": [3.0, 4.0],
        "MY_TO_SG/weekend": [3.5, 4.5],
    }
    return {
        "schema_version": 2,
        "components": ["offline", "bqml", "joined", "deep", "fuzzy", "ensemble"],
        "offline": {
            "metrics": [
                {"candidate": "XGB actual", "slice": "backtest mean of days", "n": 2, "mae_min": 1.5, "rmse_min": 2.0},
                {"candidate": "Persistence", "slice": "backtest mean of days", "n": 2, "mae_min": 3.0, "rmse_min": 4.0},
                {"candidate": "XGB actual", "slice": "backtest 09-27 Sun", "n": 1, "mae_min": 1.0, "rmse_min": 1.0},
            ],
            "significance": [compared],
            "artifacts": [
                "offline/backtest-mae.png",
                "offline/holdout-sample.png",
                "offline/holdout-mae-diff.png",
            ],
        },
        "bqml": {
            "metrics": [
                {"candidate": "Persistence", "slice": "both/all", "direction": "both", "n": 2, "mae_min": 2.5, "rmse_min": 3.0},
                {"candidate": "xgb_h30", "slice": "SG_TO_MY/all", "direction": "SG_TO_MY", "n": 1, "mae_min": 2.0, "rmse_min": 2.1},
                {"candidate": "xgb_h30", "slice": "both/all", "direction": "both", "n": 2, "mae_min": 2.2, "rmse_min": 2.4},
                {"candidate": "Persistence", "slice": "SG_TO_MY/all", "direction": "SG_TO_MY", "n": 1, "mae_min": 2.4, "rmse_min": 2.8},
            ],
            "significance": [{**compared, "family": "headline"}, {**compared, "family": "slices", "label_challenger": "slice-only"}],
            "artifacts": ["bqml/mae-by-direction.png", "bqml/mae-diff-ci.png"],
        },
        "joined": {
            "dataset": {"camera_forecast": {"curves": curves}},
            "metrics": [
                {"candidate": "persistence", "slice": "both/all", "n": 2, "mae_min": 2.6, "rmse_min": 3.0},
                {"candidate": "xgb[maps]", "slice": "both/all", "n": 2, "mae_min": 2.2, "rmse_min": 2.5},
                {"candidate": "xgb[maps]", "slice": "SG_TO_MY/all", "n": 1, "mae_min": 9.0, "rmse_min": 9.0},
            ],
            "significance": [
                {**compared, "family": "joined"},
                {**compared, "family": "camfc", "label_challenger": "camfc"},
            ],
            "artifacts": [
                "joined/joined-mae-diff.png",
                "joined/joined-mae-by-feature-set.png",
                "joined/camfc-mae-diff.png",
                "joined/camfc-profiles.png",
            ],
        },
        "deep": {
            "seed_summary": {
                "lstm": {
                    "label": "LSTM(64)",
                    "seed_mean_prediction_mae_min": 4.0,
                    "mae_by_seed": [3.5, 4.5],
                    "runs": [{"seed": 0}, {"seed": 1}],
                }
            },
            "metrics": [
                {"candidate": "Persistence T-60", "slice": "holdout", "n": 4, "mae_min": 4.7, "rmse_min": 6.0},
                {"candidate": "XGB (sklearn)", "slice": "holdout", "n": 4, "mae_min": 3.4, "rmse_min": 4.9},
                {"candidate": "LSTM(64), seed mean", "slice": "holdout", "n": 4, "mae_min": 4.0, "rmse_min": 5.0},
            ],
            "significance": [compared],
            "artifacts": ["deep/deep-mae-by-seed.png", "deep/deep-mae-diff.png"],
        },
        "fuzzy": {
            "config": {"partition": {"light_max": 20.0, "heavy_min": 35.0, "half_width": 5.0}},
            "confusion_labels": ["light", "moderate", "heavy"],
            "confusion": {
                "Persistence level": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                "Majority level (train)": [[9, 0, 0], [0, 0, 0], [0, 0, 0]],
                "Fuzzy rule base": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "XGB forecast -> fuzzy level": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            },
            "artifacts": ["fuzzy/fuzzy-memberships.png", "fuzzy/fuzzy-confusion.png"],
        },
        "ensemble": {
            "significance": [
                {**compared, "family": "30min", "label_challenger": "stack"},
                {**compared, "family": "60min", "label_challenger": "mean[deep]"},
            ],
            "artifacts": ["ensemble/ensemble-30min-mae-diff.png", "ensemble/ensemble-60min-mae-diff.png"],
        },
    }


def test_replay_writes_csv_from_manifest_and_skips_holdout_tail(tmp_path: Path):
    manifest = _manifest()
    (tmp_path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = (tmp_path / "run.json").read_bytes()
    result = replay_series_csvs(tmp_path)
    assert (tmp_path / "run.json").read_bytes() == before
    written = {path.name for path in result.written}
    assert "holdout-sample.csv" not in written
    assert not (tmp_path / "offline" / "holdout-sample.csv").exists()
    assert any("holdout-sample" in note for note in result.skipped)
    assert "mae-diff-ci.csv" in written

    expected = tmp_path / "expected"
    plot_mae_diff_forest([_comparison()], expected / "holdout-mae-diff.png")
    assert _read(tmp_path / "offline" / "holdout-mae-diff.csv") == _read(expected / "holdout-mae-diff.csv")

    plot_bqml_mae_comparison(
        {
            "Persistence": [
                {"direction": "SG_TO_MY", "time_of_day": "all", "mae": 2.4},
                {"direction": "both", "time_of_day": "all", "mae": 2.5},
            ],
            "xgb_h30": [
                {"direction": "SG_TO_MY", "time_of_day": "all", "mae": 2.0},
                {"direction": "both", "time_of_day": "all", "mae": 2.2},
            ],
        },
        expected / "mae-by-direction.png",
    )
    assert _read(tmp_path / "bqml" / "mae-by-direction.csv") == _read(expected / "mae-by-direction.csv")
    assert all(row["challenger"] != "slice-only" for row in _rows(tmp_path / "bqml" / "mae-diff-ci.csv"))

    assert [row["candidate"] for row in _rows(tmp_path / "joined" / "joined-mae-by-feature-set.csv")] == ["xgb[maps]", "persistence"]
    assert _rows(tmp_path / "joined" / "camfc-mae-diff.csv")[0]["challenger"] == "camfc"
    plot_profiles(manifest["joined"]["dataset"]["camera_forecast"]["curves"], None, expected / "camfc-profiles.png")
    assert _read(tmp_path / "joined" / "camfc-profiles.csv") == _read(expected / "camfc-profiles.csv")

    plot_seed_mae(
        manifest["deep"]["seed_summary"],
        {"Persistence T-60": 4.7, "XGB (sklearn)": 3.4},
        expected / "deep-mae-by-seed.png",
        title="seeds",
    )
    assert _read(tmp_path / "deep" / "deep-mae-by-seed.csv") == _read(expected / "deep-mae-by-seed.csv")

    plot_memberships(LevelPartition(), expected / "fuzzy-memberships.png")
    assert _read(tmp_path / "fuzzy" / "fuzzy-memberships.csv") == _read(expected / "fuzzy-memberships.csv")
    plot_confusions(
        {
            "Persistence level": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "Fuzzy rule base": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "XGB forecast -> fuzzy level": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        },
        expected / "fuzzy-confusion.png",
    )
    assert _read(tmp_path / "fuzzy" / "fuzzy-confusion.csv") == _read(expected / "fuzzy-confusion.csv")
    assert "Majority" not in _read(tmp_path / "fuzzy" / "fuzzy-confusion.csv")

    assert _rows(tmp_path / "ensemble" / "ensemble-30min-mae-diff.csv")[0]["challenger"] == "stack"
    assert _rows(tmp_path / "ensemble" / "ensemble-60min-mae-diff.csv")[0]["challenger"] == "mean[deep]"
    assert _rows(tmp_path / "offline" / "backtest-mae.csv")[0]["method"] == "XGB actual"


def test_replay_refuses_promoted_report_snapshot():
    with pytest.raises(ValueError, match="promoted report"):
        replay_series_csvs(REPORT_SNAPSHOT)


def test_replay_output_dir_leaves_source_and_still_refuses_report(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    dest = tmp_path / "out"
    result = replay_series_csvs(source, output_dir=dest)
    assert (dest / "offline" / "backtest-mae.csv").is_file()
    assert any(path.parent.parent == dest for path in result.written)
    assert not (source / "offline").exists()
    with pytest.raises(ValueError, match="promoted report"):
        replay_series_csvs(source, output_dir=REPORT_SNAPSHOT)
    with pytest.raises(ValueError, match="promoted report"):
        replay_series_csvs(tmp_path, output_dir=REPORT_SNAPSHOT / "nested")
