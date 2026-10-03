import numpy as np
import pandas as pd
import pytest

import deep_forecast as df_
import run_artifacts as ra
from deep_forecast import MODEL_SPECS, SequenceFitConfig, residual_target
from timeseries_transformer import TransformerConfig, n_patches


def test_residual_target_anchor():
    np.testing.assert_allclose(residual_target(np.array([10.0, 12.0]), np.array([9.0, 15.0])), [1.0, -3.0])
    np.testing.assert_allclose(residual_target(np.array([10.0]), None), [10.0])


def test_transformer_config_validation_and_patches():
    assert n_patches(36, 6) == 6
    with pytest.raises(ValueError):
        n_patches(36, 5)
    with pytest.raises(ValueError):
        TransformerConfig(d_model=30, n_heads=4)


def test_model_specs_cover_ablation():
    assert MODEL_SPECS["transformer"].anchor and not MODEL_SPECS["transformer_raw"].anchor
    assert set(df_.DEFAULT_MODELS) <= set(MODEL_SPECS)


def test_manifest_accepts_deep_and_fuzzy_components():
    assert "deep" in ra.KNOWN_COMPONENTS and "fuzzy" in ra.KNOWN_COMPONENTS
    manifest = {
        "schema_version": 2, "run_id": "r", "created_at_utc": "x", "components": ["fuzzy"],
        "provenance": {"git_sha": "x", "git_dirty": False, "code_sha256": "y"},
        "fuzzy": {
            "dataset": {}, "window": {"start": "a", "end": "b"}, "models": [], "horizon_minutes": 60,
            "metrics": [{"candidate": "c", "slice": "holdout", "n": 1, "accuracy": 1.0}], "significance": [], "artifacts": [],
        },
    }
    problems = ra.validate_manifest(manifest)
    assert any("macro_f1" in p for p in problems)


def _export(days: int = 5) -> pd.DataFrame:
    start = pd.Timestamp("2026-09-01 00:00:00")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(days * 288):
        t = start + pd.Timedelta(minutes=5 * i)
        level = 900 + 600 * np.sin(2 * np.pi * (t.hour * 60 + t.minute) / 1440) + rng.normal(0, 30)
        rows.append(
            {
                "observed_at_sgt": t.strftime("%Y-%m-%dT%H:%M:%S"),
                "route_id": "jb_to_woodlands",
                "status": "OK",
                "duration_sec": 600.0,
                "duration_in_traffic_sec": float(level),
                "error_message": None,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_transformer_builds_and_seeds_differ():
    pytest.importorskip("tensorflow")
    from timeseries_transformer import build_transformer_model

    a = build_transformer_model((36, 9), TransformerConfig(seed=1))
    b = build_transformer_model((36, 9), TransformerConfig(seed=2))
    c = build_transformer_model((36, 9), TransformerConfig(seed=1))
    assert a.output_shape == (None, 1)
    assert 15_000 < a.count_params() < 25_000
    assert not np.allclose(a.get_weights()[0], b.get_weights()[0])
    np.testing.assert_allclose(a.get_weights()[0], c.get_weights()[0])


@pytest.mark.slow
def test_recurrent_seed_is_respected():
    pytest.importorskip("tensorflow")
    from timeseries_lstm import LSTMTrainConfig, build_recurrent_model

    w = [build_recurrent_model((12, 3), LSTMTrainConfig(lstm_units=8, seed=s)).get_weights()[0] for s in (0, 1)]
    assert not np.allclose(w[0], w[1])


@pytest.mark.slow
def test_deep_component_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    cache = tmp_path / "cache.csv"
    _export().to_csv(cache, index=False)
    monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fit = SequenceFitConfig(epochs=2, patience=1, reduce_lr_patience=1)
    paths, meta = df_.deep_component(
        run_dir, cache=cache, models=("gru", "transformer"), seeds=(0, 1), fit_config=fit, recurrent_units=8,
        transformer_config=TransformerConfig(d_model=8, n_heads=2, ff_dim=16, n_layers=1),
    )
    assert all(p.is_file() for p in paths)
    manifest = {
        "schema_version": 2, "run_id": "run", "created_at_utc": "x", "components": ["deep"],
        "provenance": {"git_sha": "x", "git_dirty": False, "code_sha256": "y"}, "deep": meta,
    }
    assert ra.validate_manifest(manifest, run_dir) == []
    assert len(meta["seed_summary"]["gru"]["mae_by_seed"]) == 2
    slices = {m["slice"] for m in meta["metrics"]}
    assert {"holdout", "holdout seed 0", "holdout seed 1"} <= slices
    # 2 models x (vs persistence, vs XGB) + transformer vs ... (no lstm / raw here)
    assert len(meta["significance"]) == 4
