# Reference: teammate Colab notebook

Source file (local): `Test_Time_Series_Prediction (1).ipynb` (2026-09-26).

**Integrated into repo:** [`../timeseries_xgb.py`](../timeseries_xgb.py) and [`causeway_xgb_timeseries.ipynb`](causeway_xgb_timeseries.ipynb). Do not maintain a second copy of the full `.ipynb` in git unless the teammate exports a new version.

## Route scope

**JB → SG only:** filters `route_id == jb_to_woodlands`. **SG → JB** (reverse row in `travel_times`) is **not** in this notebook or in [`timeseries_xgb.py`](../timeseries_xgb.py). Bidirectional **30 min** BQML: [`layer_b.py`](../layer_b.py).

## Teammate notes (2026-09-26)

- Best offline scores so far with **updated XGB hyperparameters** (see `XGBTrainConfig`).
- **DWT (`USE_DWT`) off** for now — only marginal gain; short Maps history is likely insufficient for wavelet features to help reliably.
- **Slew-rate limit** 300 s per 5-minute step on `duration_in_traffic_sec` before training.

## Reported 60 min backtest (22–24 Sep 2026, mean columns)

| Method | RMSE (min) mean | MAE (min) mean |
| --- | --- | --- |
| Persistence T-60 | 5.30 | 3.58 |
| Naive blend | 4.24 | 2.71 |
| **XGB actual window** | **3.45** | **2.28** |

Reconcile with [`../../docs/evaluation.md`](../../docs/evaluation.md) after `python generate_comparison_plots.py` and `python promote_report_run.py`.
