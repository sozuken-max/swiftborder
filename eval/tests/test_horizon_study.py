"""eval/horizon_study.py guards (no cache needed)."""

import pandas as pd
import pytest

import horizon_study as hs


@pytest.mark.parametrize("h", [0, -30, 5, 1450, 2880])
def test_horizons_outside_10_to_1440_are_rejected(h):
    # beyond 24 h "same time yesterday" (target - 1 d) would be after the origin: future information
    with pytest.raises(ValueError):
        hs.check_horizon(h)


def test_valid_horizons_pass():
    assert [hs.check_horizon(h) for h in (10, 30, 1440)] == [10, 30, 1440]
    assert hs.DEFAULT_HORIZONS == tuple(range(30, 1441, 30))


def test_cli_rejects_a_bad_horizon_before_loading(monkeypatch):
    monkeypatch.setattr(hs, "load", lambda: pytest.fail("loaded the cache"))
    with pytest.raises(SystemExit):
        hs.main(["--horizons", "2880"])


def test_example_chart_is_skipped_without_example_horizons(tmp_path):
    oof = pd.DataFrame({"direction": [], "bin_ts": [], "date_sgt": []})
    assert hs.plot_example_days({120: oof}, tmp_path / "example-days.png") == []
    assert not (tmp_path / "example-days.png").exists()
