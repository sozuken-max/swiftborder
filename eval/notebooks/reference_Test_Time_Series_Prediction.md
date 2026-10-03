# Reference: teammate Colab notebook

Source file (local): `Test_Time_Series_Prediction (1).ipynb` (2026-09-26).

**Integrated into repo:** [`../timeseries_xgb.py`](../timeseries_xgb.py) and [`causeway_xgb_timeseries.ipynb`](causeway_xgb_timeseries.ipynb). Do not maintain a second copy of the full `.ipynb` in git unless the teammate exports a new version.

## Route scope

**JB → SG only:** filters `route_id == jb_to_woodlands`. **SG → JB** (reverse row in `travel_times`) is **not** in this notebook or in [`timeseries_xgb.py`](../timeseries_xgb.py). Bidirectional **30 min** BQML: [`layer_b.py`](../layer_b.py).

## Teammate notes (2026-09-26)

- Best offline scores so far with **updated XGB hyperparameters** (see `XGBTrainConfig`).
- **DWT (`USE_DWT`) off** on the scored path: the 2026-09-26 notebook saw only a marginal gain, and the Maps history is short. The repo experiment that replaced that summary is `compare_window_feature_sets` in [`timeseries_xgb.py`](../timeseries_xgb.py): lags (A) vs lags plus a causal difference and a 60-minute rolling mean (B) vs lags plus a z-scored **db2 level 2** wavelet (C). Level 3 is optional. db4 is not the default because it smooths the bends. See [eval/README.md](../README.md).
- **Slew-rate limit** 300 s per 5-minute step on `duration_in_traffic_sec` before training.

## Superseded numbers

The teammate's 22–24 Sep backtest table is **not** reportable. The 2026-09-30 review found that the notebook pipeline:

- labelled the value 115 minutes before the target as "persistence T-60";
- used `duration_sec` from the target row (same Maps call as the label);
- scored interpolated, bfilled, slew-limited labels instead of raw observations;
- backtested days that overlapped the 80% training split.

`timeseries_xgb.py` fixes all four (tests in `../tests/test_timeseries_xgb.py`). Report numbers come only from [`../runs/report/run.json`](../runs/report/run.json); see [`../../docs/evaluation.md`](../../docs/evaluation.md).
