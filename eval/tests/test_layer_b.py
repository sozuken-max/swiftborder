"""layer_b harness with a fake BigQuery source (no network)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import layer_b as lb
from run_artifacts import validate_manifest

UTC = timezone.utc
START_UTC = datetime(2026, 9, 12, 16, 0, tzinfo=UTC)  # 2026-09-13 00:00 SGT


class FakeSource:
    def __init__(self, days=3, trained=datetime(2026, 9, 12, 6, 15, tzinfo=UTC), gaps=(), after_gap=()):
        self.trained = trained
        self.rows = []
        rng = np.random.default_rng(0)
        n = days * 144
        for direction in lb.DIRECTIONS:
            for i in range(n):
                ts = START_UTC + timedelta(minutes=10 * i)
                y = 25 + 5 * np.sin(i / 20) + rng.normal(0, 1)
                self.rows.append(
                    {
                        "direction": direction,
                        "bin_ts": ts,
                        "is_morning_peak": int(6 <= (ts.hour + 8) % 24 <= 10),
                        "is_evening_peak": int(16 <= (ts.hour + 8) % 24 <= 21),
                        "is_weekend": int(ts.astimezone(lb.SGT).weekday() >= 5),
                        "after_gap": int(i in after_gap),
                        "lead_gap_min": 40 if i in gaps else 30,
                        "y_30": y,
                        "y_persistence": y + rng.normal(0, 3),
                    }
                )
        self.latest = self.rows[-1]["bin_ts"]
        self.queried = []

    def latest_labelled_bin(self):
        return self.latest

    def fetch_rows(self, start, end):
        self.queried.append((start, end))
        return [r for r in self.rows if start <= r["bin_ts"] <= end]

    def fetch_predictions(self, model, start, end):
        noise = 1.0 if model == "lin_h30" else 1.5
        rng = np.random.default_rng(1 if model == "lin_h30" else 2)
        return [
            {"direction": r["direction"], "bin_ts": r["bin_ts"], "predicted": r["y_30"] + rng.normal(0, noise)}
            for r in self.fetch_rows(start, end)
        ]

    def model_metadata(self, model):
        return {
            "model": model,
            "model_type": "LINEAR_REGRESSION" if model == "lin_h30" else "BOOSTED_TREE_REGRESSOR",
            "created": self.trained,
            "modified": self.trained,
            "training_run_starts": [self.trained - timedelta(minutes=1)],
            "data_split": {"dataSplitMethod": "CUSTOM", "dataSplitColumn": "is_val"},
        }


def test_parse_sgt_defaults_to_singapore_time():
    assert lb.parse_sgt("2026-09-13 00:00:00") == START_UTC
    assert lb.parse_sgt("2026-09-13T00:00:00+00:00") == datetime(2026, 9, 13, tzinfo=UTC)


def test_window_defaults_to_13_sep_sgt_and_latest_bin():
    src = FakeSource()
    result = lb.run_harness(src, n_bootstrap=200)
    assert result["holdout"].window_start == START_UTC
    assert result["holdout"].window_end == src.latest
    assert src.queried[0] == (START_UTC, src.latest)


def test_explicit_window_end_is_used():
    src = FakeSource()
    result = lb.run_harness(src, window_end="2026-09-14 00:00", n_bootstrap=200)
    assert result["holdout"].window_end == datetime(2026, 9, 13, 16, 0, tzinfo=UTC)
    assert all(r["bin_ts"] <= result["holdout"].window_end for r in result["holdout"].rows)


def test_rejects_models_trained_inside_the_window():
    src = FakeSource(trained=START_UTC + timedelta(hours=1))
    with pytest.raises(lb.LeakageError):
        lb.run_harness(src, n_bootstrap=200)


def test_gap_filters_drop_rows_for_every_candidate():
    src = FakeSource(gaps={5, 6}, after_gap={10})
    result = lb.run_harness(src, n_bootstrap=200)
    ex = result["holdout"].excluded
    assert ex["lead_gap_not_30"] == 4  # 2 rows x 2 directions
    assert ex["after_gap"] == 2
    assert len(result["holdout"].rows) == len(src.rows) - 6


def test_missing_prediction_drops_the_row():
    src = FakeSource(days=1)
    base = src.fetch_rows(START_UTC, src.latest)
    preds = {m: src.fetch_predictions(m, START_UTC, src.latest) for m in lb.MODELS}
    preds["xgb_h30"] = preds["xgb_h30"][1:]
    h = lb.prepare_rows(base, preds, START_UTC, src.latest)
    assert h.excluded["missing_prediction"] == 1


def test_ensemble_is_the_mean_of_the_two_models():
    result = lb.run_harness(FakeSource(days=1), n_bootstrap=200)
    r = result["holdout"].rows[0]
    assert r[lb.ENSEMBLE] == pytest.approx((r["lin_h30"] + r["xgb_h30"]) / 2)


def test_slices_cover_directions_and_dimensions():
    names = {s[0] for s in lb.slice_definitions()}
    assert "both/all" in names and "MY_TO_SG/light=night" in names and "SG_TO_MY/day_type=weekend" in names
    assert len(names) == 3 * (1 + 3 + 2 + 2)


def test_slice_math_matches_manual_mae():
    result = lb.run_harness(FakeSource(days=2), n_bootstrap=200)
    rows = result["holdout"].rows
    sub = [r for r in rows if r["direction"] == "SG_TO_MY" and lb.light(r) == "night"]
    manual = sum(abs(r["y_30"] - r["lin_h30"]) for r in sub) / len(sub)
    got = next(m for m in result["metrics"] if m["candidate"] == "lin_h30" and m["slice"] == "SG_TO_MY/light=night")
    assert got["mae_min"] == pytest.approx(manual)
    assert got["n"] == len(sub)


def test_light_uses_sgt_hours():
    row = {"bin_ts": datetime(2026, 9, 13, 23, 0, tzinfo=UTC)}  # 07:00 SGT next day
    assert lb.light(row) == "day"
    row = {"bin_ts": datetime(2026, 9, 13, 11, 0, tzinfo=UTC)}  # 19:00 SGT
    assert lb.light(row) == "night"


def test_families_are_holm_adjusted_separately():
    # "both" needs >= 10 shared calendar days; 6 days x 2 directions is not enough any more
    result = lb.run_harness(FakeSource(days=11), n_bootstrap=200)
    head = result["families"]["headline"]
    assert len(head) == 3 * 3 + 2
    assert all(c.family_size == len(head) for c in head)
    assert all(c.block == "day" and c.block_size == 144 for c in head if "/all" in c.label_challenger and "both" not in c.label_challenger)
    # Models are ~3x more accurate than the fake persistence, so they should win the headline.
    both = next(c for c in head if c.label_challenger == "lin_h30 (both/all)")
    assert both.decision == "challenger" and both.calendar_days >= 10
    short = lb.run_harness(FakeSource(days=6), n_bootstrap=200)["families"]["headline"]
    assert next(c for c in short if c.label_challenger == "lin_h30 (both/all)").decision == "insufficient data"


def test_component_validates_as_schema_v2(tmp_path):
    src = FakeSource(days=2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths, meta = lb.bqml_component(run_dir, source=src, n_bootstrap=200)
    manifest = {
        "schema_version": 2,
        "run_id": "run",
        "created_at_utc": "2026-09-30T00:00:00Z",
        "components": ["bqml"],
        "provenance": {"git_sha": "x", "git_dirty": False, "code_sha256": "y"},
        "bqml": meta,
    }
    assert validate_manifest(manifest, run_dir) == []
    assert meta["window"]["start_sgt"].startswith("2026-09-13T00:00")
    assert meta["model_metadata"][0]["data_split"]["dataSplitColumn"] == "is_val"
    assert {s["family"] for s in meta["significance"]} == {"headline", "slices"}
    assert all(p.is_file() for p in paths)


def test_removed_holdout_days_flag_fails_loudly():
    assert lb.main(["--holdout-days", "3"]) == 2
