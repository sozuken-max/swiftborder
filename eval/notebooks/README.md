# eval notebooks

Exploratory analysis that does **not** deploy `swiftbackend` or change BigQuery models.

| Notebook | Purpose |
| --- | --- |
| [causeway_xgb_timeseries.ipynb](causeway_xgb_timeseries.ipynb) | sklearn **XGBoost** on 5-minute Maps durations, **`jb_to_woodlands` only (JB → SG)**. **SG → JB is not implemented.** **60-minute** horizon (`H=12`). Logic in [`../timeseries_xgb.py`](../timeseries_xgb.py). Both directions at 30 min: [`../layer_b.py`](../layer_b.py). |

**Not the live serve path:** production uses `v_forecast_recent` (30 minutes, BQML `lin_h30` or persistence). Score production candidates with [`../layer_b.py`](../layer_b.py).

## Run

```bash
cd eval
pip install -r requirements-notebook.txt
jupyter notebook notebooks/causeway_xgb_timeseries.ipynb
```

**Data:** canonical `causeway.travel_times` via `sync_canonical_travel_times`, cached at `eval/data/causeway_gdata.csv` (gitignored; see [`../data/README.md`](../data/README.md)). Set `REFRESH_FROM_BQ = True` in the notebook to refresh from BigQuery (read-only).

**Evaluation rules the notebook follows** (enforced in `timeseries_xgb.py` and its tests): inputs are causal (forward-fill up to 30 min, no bfill, slew cap on inputs only); labels are raw observations; persistence is the last observed value 60 minutes before the label; `duration_sec` is taken at the forecast origin; backtest days are full days after the train/test boundary.

**LSTM:** section 3 uses [`timeseries_lstm.py`](../timeseries_lstm.py) on the same train/test rows as XGB, with a pluggable custom `keras.layers.Layer` (example `ResidualGatedLSTM`).
