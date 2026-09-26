from pathlib import Path

from plots import plot_bqml_mae_comparison, plot_mae_diff_forest
from significance import ComparisonResult


def test_plot_bqml_mae_comparison_writes_file(tmp_path: Path):
    slices = {
        "Persistence": [{"direction": "both", "time_of_day": "all", "mae": 2.5, "rmse": 3.0, "n": 10}],
        "xgb_h30": [{"direction": "both", "time_of_day": "all", "mae": 2.3, "rmse": 2.9, "n": 10}],
    }
    out = tmp_path / "mae.png"
    plot_bqml_mae_comparison(slices, out)
    assert out.is_file() and out.stat().st_size > 100


def test_plot_mae_diff_forest_writes_file(tmp_path: Path):
    comp = ComparisonResult(
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
    out = tmp_path / "forest.png"
    plot_mae_diff_forest([comp], out)
    assert out.is_file() and out.stat().st_size > 100
