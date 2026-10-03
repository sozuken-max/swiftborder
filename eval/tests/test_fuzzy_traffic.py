import numpy as np
import pandas as pd
import pytest

import fuzzy_traffic as ft
from fuzzy_traffic import (
    FuzzyInputs,
    FuzzyRuleClassifier,
    LevelPartition,
    classification_metrics,
    firing_strengths,
    input_memberships,
    ranked_probability_score,
    time_of_day_memberships,
    trapezoid,
    trend_memberships,
)


def test_trapezoid_shape_and_infinite_shoulders():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(trapezoid(x, 1, 2, 3, 4), [0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(trapezoid(np.array([1.5, 3.5]), 1, 2, 3, 4), [0.5, 0.5])
    np.testing.assert_allclose(trapezoid(x, -np.inf, -np.inf, 2, 4), [1, 1, 1, 0.5, 0, 0])
    assert np.isnan(trapezoid(np.array([np.nan]), 0, 1, 2, 3)[0])


def test_level_partition_is_ruspini_and_crosses_at_cut_points():
    p = LevelPartition()
    x = np.linspace(0, 80, 801)
    mu = p.memberships(x)
    np.testing.assert_allclose(mu.sum(axis=1), 1.0)
    np.testing.assert_allclose(p.memberships(np.array([20.0, 35.0])), [[0.5, 0.5, 0], [0, 0.5, 0.5]])
    # Max membership away from the exact crossings equals the crisp level.
    off = x[(np.abs(x - 20) > 1e-9) & (np.abs(x - 35) > 1e-9)]
    np.testing.assert_array_equal(p.memberships(off).argmax(axis=1), p.crisp(off))


def test_crisp_levels_and_nan():
    p = LevelPartition()
    np.testing.assert_array_equal(p.crisp(np.array([10, 19.99, 20, 34.99, 35, 60, np.nan])), [0, 0, 1, 1, 2, 2, -1])


def test_partition_rejects_overlapping_ramps():
    with pytest.raises(ValueError):
        LevelPartition(light_max=20, heavy_min=25, half_width=5)


def test_trend_and_time_partitions_sum_to_one():
    np.testing.assert_allclose(trend_memberships(np.linspace(-20, 20, 81)).sum(axis=1), 1.0)
    tod = time_of_day_memberships(np.linspace(0, 24, 97))
    np.testing.assert_allclose(tod.sum(axis=1), 1.0)
    assert tod[np.searchsorted(np.linspace(0, 24, 97), 7.5), 1] == 1.0  # 07:30 is fully "morning"
    assert time_of_day_memberships(np.array([2.0]))[0, 0] == 1.0  # 02:00 is "night"


def _inputs(now, trend=0.0, hour=12.0, nonwork=0, yesterday=None):
    now = np.asarray(now, dtype=float)
    k = len(now)
    return FuzzyInputs(
        now_min=now,
        trend_min=np.full(k, trend),
        target_hour=np.full(k, hour),
        nonworkday=np.full(k, nonwork, dtype=float),
        yesterday_min=now if yesterday is None else np.asarray(yesterday, dtype=float),
    )


def test_firing_strengths_cover_every_rule_and_sum_to_one():
    x = _inputs([12.0, 27.0, 50.0], trend=3.0, hour=10.0)
    mu = firing_strengths(input_memberships(x, LevelPartition()))
    assert mu.shape == (3, 3 * 3 * 4 * 2 * 3)
    np.testing.assert_allclose(mu.sum(axis=1), 1.0)  # product of Ruspini partitions


def test_rule_classifier_learns_a_rising_trend_rule():
    # Moderate now and rising -> heavy next hour; moderate and steady -> stays moderate.
    rng = np.random.default_rng(0)
    now = rng.uniform(26, 29, 400)
    trend = np.where(np.arange(400) % 2 == 0, 10.0, 0.0)
    x = FuzzyInputs(now, trend, np.full(400, 12.0), np.zeros(400), now)
    y = np.where(trend > 5, 2, 1)
    clf = FuzzyRuleClassifier().fit(x, y)
    assert clf.n_rules >= 2
    test = FuzzyInputs(np.array([27.5, 27.5]), np.array([12.0, 0.0]), np.full(2, 12.0), np.zeros(2), np.array([27.5, 27.5]))
    np.testing.assert_array_equal(clf.predict(test), [2, 1])
    deg = clf.predict_degrees(test)
    np.testing.assert_allclose(deg.sum(axis=1), 1.0)
    assert any("THEN heavy" in r["rule"] and "trend is rising" in r["rule"] for r in clf.describe_rules(5))


def test_rule_classifier_falls_back_to_persistence_when_no_rule_fires():
    x = _inputs(np.full(50, 12.0), hour=2.0)
    clf = FuzzyRuleClassifier().fit(x, np.zeros(50, dtype=int))
    unseen = _inputs(np.array([50.0]), hour=18.0, nonwork=1)  # heavy, evening, non-workday: no rule
    assert clf.class_scores(unseen).max() == 0
    assert clf.predict(unseen)[0] == 2
    np.testing.assert_allclose(clf.predict_degrees(unseen), [[0, 0, 1]])


def test_min_support_drops_rare_rules():
    x = _inputs(np.array([12.0]), hour=2.0)
    clf = FuzzyRuleClassifier(min_support=2.0).fit(x, np.array([0]))
    assert clf.n_rules == 0


def test_classification_metrics_and_severe_errors():
    y = np.array([0, 0, 1, 2, 2])
    p = np.array([0, 2, 1, 2, 0])
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["severe_error_rate"] == pytest.approx(0.4)
    assert m["confusion"] == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    assert m["recall_moderate"] == 1.0


def test_rps_is_zero_for_perfect_and_ordinal():
    y = np.array([0, 2])
    assert ranked_probability_score(y, np.eye(3)[y]) == 0.0
    near = ranked_probability_score(np.array([2]), np.array([[0, 1, 0]]))
    far = ranked_probability_score(np.array([2]), np.array([[1, 0, 0]]))
    assert 0 < near < far == pytest.approx(1.0)


def _export(days: int = 6) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-09-01 00:00:00")
    for route, base in (("jb_to_woodlands", 15.0), ("mandai_to_shell_jb", 13.0)):
        for i in range(days * 288):
            t = start + pd.Timedelta(minutes=5 * i)
            peak = 25.0 if 7 <= t.hour < 10 else (12.0 if 17 <= t.hour < 20 else 0.0)
            rows.append(
                {
                    "observed_at_sgt": t.strftime("%Y-%m-%dT%H:%M:%S"),
                    "route_id": route,
                    "status": "OK",
                    "duration_sec": 600.0,
                    "duration_in_traffic_sec": (base + peak) * 60.0,
                    "error_message": None,
                }
            )
    return pd.DataFrame(rows)


def test_score_routes_aligns_rows_by_route_then_time():
    s = ft.score_routes(_export())
    n = len(s["y"])
    assert all(len(s[k]) == n for k in ("persist", "majority", "rules", "xgb", "route"))
    assert len(s["timestamps"]) == n
    assert list(dict.fromkeys(s["route"])) == list(ft.ROUTES)
    for route in ft.ROUTES:
        ts = [t for t, r in zip(s["timestamps"], s["route"]) if r == route]
        assert ts == sorted(ts)
    assert set(np.unique(s["y"])) <= {0, 1, 2}


def test_fuzzy_component_writes_valid_manifest_block(tmp_path, monkeypatch):
    import run_artifacts as ra

    cache = tmp_path / "cache.csv"
    _export().to_csv(cache, index=False)
    monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths, meta = ft.fuzzy_component(run_dir, cache=cache)
    assert all(p.is_file() for p in paths)
    manifest = {
        "schema_version": 2, "run_id": "run", "created_at_utc": "x", "components": ["fuzzy"],
        "provenance": {"git_sha": "x", "git_dirty": False, "code_sha256": "y"}, "fuzzy": meta,
    }
    assert ra.validate_manifest(manifest, run_dir) == []
    assert {m["candidate"] for m in meta["metrics"]} == set(ft.CANDIDATES.values())
    assert len(meta["significance"]) == 3
    readme = ra.write_run_readme(run_dir, manifest).read_text(encoding="utf-8")
    assert "Macro-F1" in readme and "error-rate difference" in readme
