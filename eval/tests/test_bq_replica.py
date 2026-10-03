import numpy as np
import pandas as pd
import pytest

import bq_replica as bq
import ensemble as en
import joined as jx
from tests.test_joined import _frame


def _maps_frame(days=10, seed=0):
    f = _frame(days=days, seed=seed)
    rng = np.random.default_rng(seed)
    f["lag_60"] = f["y_persistence"] + rng.normal(0, 0.2, len(f))
    f["tod_block"] = np.arange(len(f)) % 288
    f["date_sgt"] = f["bin_ts"].dt.tz_convert("Asia/Singapore").dt.strftime("%Y-%m-%d")
    return f


def test_frozen_rows_match_bqml_training_cut():
    f = _maps_frame()
    rows = bq.frozen_training_rows(f)
    assert (rows["bin_ts"] + bq.LABEL_LAG <= bq.BQML_TRAINED_AT).all()
    assert not rows["date_sgt"].isin(bq.BQML_VAL_DAYS).any()
    assert rows["lag_60"].notna().all() and (rows["after_gap"] == 0).all()


def test_replicas_fit_and_predict():
    f = _maps_frame()
    tr = bq.frozen_training_rows(f)
    lin, xgb = bq.fit_lin(tr), bq.fit_xgb(tr, seeds=(0, 1))
    p1, p2 = bq.predict_lin(lin, f), bq.predict_xgb(xgb, f)
    assert p1.shape == p2.shape == (len(f),)
    assert np.isfinite(p1).all() and np.isfinite(p2).all()
    assert bq.XGB_PARAMS["base_score"] == 0.5 and bq.XGB_PARAMS["n_estimators"] == 28
    assert bq.XGB_FEATURES == bq.LIN_FEATURES[:12] + ["tod_block"] + bq.LIN_FEATURES[12:]


def test_joined_adds_replica_columns_and_family():
    data = jx.scorable(_maps_frame(days=12))
    oof = jx.run_folds(data, "2026-09-13", "2026-09-15", min_train_rows=200, mp_harmonics=2)
    for c in ("lin_bq[frozen]", "xgb_bq[frozen]", "lin_bq[daily]", "xgb_bq[daily]"):
        assert c in oof.columns and oof[c].notna().all()
    specs = jx.replica_specs(oof)
    assert ("xgb_bq[daily]", "xgb_bq[frozen]", "all") in specs
    assert not any("bq[" in s[0] for s in jx.comparison_specs(oof))


def test_ensemble_local_pool_uses_replicas():
    data = jx.scorable(_maps_frame(days=12))
    oof = jx.run_folds(data, "2026-09-13", "2026-09-15", min_train_rows=200, mp_harmonics=2)
    m, info = en.pool_30_local(oof)
    assert info["source"].startswith("local replicas")
    np.testing.assert_allclose(m["ensemble_mean"], (m["lin_h30"] + m["xgb_h30"]) / 2)
    sg = m["direction"] == "SG_TO_MY"
    np.testing.assert_allclose(m.loc[sg, "served"], m.loc[sg, "lin_h30"])
    np.testing.assert_allclose(m.loc[~sg, "served"], m.loc[~sg, "Persistence"])
    assert en.local_labels()["lin_h30"].endswith("(local replica)")
    with pytest.raises(RuntimeError):
        en.pool_30_local(oof.drop(columns=["lin_bq[frozen]"]))


def test_parity_frame_checks_labels_and_agreement():
    import bqml_parity as bp

    f = _maps_frame(days=12)
    test = f[f["date_sgt"] >= "2026-09-13"].head(50)
    rows = [{"direction": r.direction, "bin_ts": r.bin_ts.to_pydatetime(), "y_30": r.y_30, "Persistence": r.y_persistence,
             "lin_h30": r.y_persistence, "xgb_h30": r.y_persistence} for r in test.itertuples()]
    m = bp.parity_frame(rows, f)
    assert len(m) == 50
    rows[0]["y_30"] += 1
    with pytest.raises(RuntimeError):
        bp.parity_frame(rows, f)
    a = bp.agreement(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]))
    assert a["mean_abs_diff_min"] == pytest.approx(1 / 6) and a["signed_mean_diff_min"] == pytest.approx(-1 / 6)


def test_flags_parity_needs_bqml_and_skip_offline():
    import generate_comparison_plots as gcp

    with pytest.raises(SystemExit):
        gcp.main(["--bqml-parity"])
    with pytest.raises(SystemExit):
        gcp.main(["--skip-offline"])  # nothing left to run
