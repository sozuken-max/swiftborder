# eval notebooks

Exploratory analysis that does **not** deploy `swiftbackend` or change BigQuery models.

| Notebook | Purpose |
| --- | --- |
| [causeway_xgb_timeseries.ipynb](causeway_xgb_timeseries.ipynb) | sklearn **XGBoost** on 5-minute Maps durations — **`jb_to_woodlands` only (JB → SG)**. **SG → JB is not implemented.** **60-minute** horizon (`H=12`). Logic in [`../timeseries_xgb.py`](../timeseries_xgb.py). Both directions at 30 min: [`../layer_b.py`](../layer_b.py). |

**Not the live serve path:** production uses `v_forecast_recent` (30 minutes, BQML `lin_h30` or persistence). Score production candidates with [`../layer_b.py`](../layer_b.py).

## Run

```bash
cd eval
pip install -r requirements-dev.txt
jupyter notebook notebooks/causeway_xgb_timeseries.ipynb
```

**Data:** live download of canonical `causeway.travel_times` via `sync_canonical_travel_times`, cached at [`../data/causeway_gdata.csv`](../data/causeway_gdata.csv). Set `REFRESH_FROM_BQ = True` in the notebook to refresh from BigQuery.

**Setup:** `pip install -r ../requirements-notebook.txt` (adds `db-dtypes`, `pyarrow`, optional `tensorflow` for LSTM).

**LSTM:** section 3 uses [`timeseries_lstm.py`](../timeseries_lstm.py) and a pluggable custom `keras.layers.Layer` (example `ResidualGatedLSTM` in the notebook).
