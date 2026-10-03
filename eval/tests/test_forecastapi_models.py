"""forecastapi/local_models.py must stay identical to the harness models it serves (ADR 0004)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bq_replica as bq
import features as fx
import joined as jx

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("forecastapi_local_models", REPO / "forecastapi" / "local_models.py")
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)


def test_feature_lists_and_settings_match_eval():
    assert lm.MAPS_FEATURES == jx.FEATURE_SETS["maps"]
    assert lm.BQ_LIN_FEATURES == bq.LIN_FEATURES
    assert lm.BQ_XGB_FEATURES == bq.XGB_FEATURES
    assert lm.BQ_XGB_PARAMS == bq.XGB_PARAMS
    assert tuple(lm.BQ_XGB_SEEDS) == tuple(bq.XGB_SEEDS)
    assert lm.BQ_LIN_L2 == bq.LIN_L2
    assert lm.BQML_TRAINED_AT == bq.BQML_TRAINED_AT and tuple(lm.BQML_VAL_DAYS) == tuple(bq.BQML_VAL_DAYS)
    assert lm.LABEL_LAG == jx.LABEL_LAG
    harness = jx._xgb().get_params()
    for k, v in lm.HARNESS_XGB_PARAMS.items():
        assert harness[k] == v, k
    ridge = jx._ridge(jx.FEATURE_SETS["maps"])
    assert ridge.steps[-1][1].alpha == lm.HARNESS_RIDGE_ALPHA
    assert ridge.steps[0][1].strategy == "median" and ridge.steps[0][1].add_indicator is True
    assert set(lm.LOCAL_MODELS) == {"ridge[maps]", "xgb[maps]", "lin_bq[daily]", "xgb_bq[daily]", "lin_bq[frozen]", "xgb_bq[frozen]"}


def _view(days=14, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-09-05 16:00", tz="UTC")
    rows = []
    for d in ("SG_TO_MY", "MY_TO_SG"):
        for i in range(days * 144):
            ts = t0 + pd.Timedelta(minutes=10 * i)
            base = 25 + 8 * np.sin(2 * np.pi * i / 144) + rng.normal(0, 0.5)
            row = {"direction": d, "bin_ts": ts, "y_30": base + rng.normal(1, 0.5), "after_gap": int(i % 97 == 0)}
            for c in lm.VIEW_COLUMNS:
                row[c] = base + rng.normal(0, 0.3)
            if i % 50 == 0:
                row["lag_60"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_id", ["ridge[maps]", "xgb[maps]", "lin_bq[daily]", "xgb_bq[daily]", "lin_bq[frozen]", "xgb_bq[frozen]"])
def test_service_fit_predicts_exactly_what_the_harness_predicts(model_id):
    raw = _view()
    day = "2026-09-17"
    day_start = pd.Timestamp(day).tz_localize("Asia/Singapore").tz_convert("UTC")

    # harness side (joined.run_folds): scorable frame, fold split, then the estimator
    data = jx.scorable(raw.assign(date_sgt=raw["bin_ts"].dt.tz_convert("Asia/Singapore").dt.strftime("%Y-%m-%d")))
    train_h, test_h = jx.fold_split(data, day)
    if model_id in ("ridge[maps]", "xgb[maps]"):
        harness = jx.predict_fold(train_h, test_h, jx.FEATURE_SETS["maps"])[model_id.split("[")[0]]
    elif model_id.endswith("[daily]"):
        tr = bq.bqml_rows(train_h)
        harness = bq.predict_lin(bq.fit_lin(tr), test_h) if model_id.startswith("lin") else bq.predict_xgb(bq.fit_xgb(tr), test_h)
    else:
        fr = bq.frozen_training_rows(data)
        harness = bq.predict_lin(bq.fit_lin(fr), test_h) if model_id.startswith("lin") else bq.predict_xgb(bq.fit_xgb(fr), test_h)

    # service side (forecastapi): prepared view rows, training_rows, fit
    view = lm.prepare(raw)
    train_s = lm.training_rows(view, lm.LOCAL_MODELS[model_id]["rows"], day_start)
    served = lm.fit(model_id, train_s).predict(lm.prepare(test_h.drop(columns=["is_my_to_sg", "date_sgt"])))
    np.testing.assert_allclose(served, harness, rtol=0, atol=1e-6)


def test_service_features_equal_the_harness_features_with_missing_bins():
    # v_training_set uses positional LAG/LEAD; the service builds features like eval/features.py instead
    rng = np.random.default_rng(3)
    t0 = pd.Timestamp("2026-09-05 16:00", tz="UTC")
    skipped = {t0 + pd.Timedelta(minutes=10 * i) for i in (40, 41, 42, 43, 200, 333)}  # a 50-minute gap and two single bins
    rows = []
    for d, route in (("SG_TO_MY", "r_sg"), ("MY_TO_SG", "r_my")):
        for i in range(3 * 144):
            ts = t0 + pd.Timedelta(minutes=10 * i)
            if ts in skipped:
                continue
            rows.append({"route_id": route, "direction": d, "bin_ts": ts, "dur_min": 25 + rng.normal(0, 3),
                         "typical_min": 20.0, "congestion_ratio": 1 + rng.normal(0, 0.1), "speed_kmh": 40 + rng.normal(0, 3)})
    bins = pd.DataFrame(rows)
    bins["gap_min"] = bins.groupby("route_id")["bin_ts"].diff().dt.total_seconds() / 60.0

    harness = fx.maps_features(bins)
    service = lm.features_from_bins(bins.drop(columns=["gap_min", "typical_min"]))
    cols = ["direction", "bin_ts", *lm.VIEW_COLUMNS, "after_gap", "y_30"]
    pd.testing.assert_frame_equal(
        service[cols].reset_index(drop=True),
        harness[cols].reset_index(drop=True),
        check_dtype=False,
    )
    single = harness[harness["bin_ts"] == t0 + pd.Timedelta(minutes=10 * 201)]
    assert single["lag_10"].isna().all() and (single["after_gap"] == 0).all()
