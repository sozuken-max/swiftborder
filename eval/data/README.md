# Local data (not in git)

The canonical dataset is **`swiftborder.causeway.travel_times`** in BigQuery. The notebook downloads it with [`sync_canonical_travel_times`](../timeseries_xgb.py) and caches here.

## Cache file

| Path | Role |
| --- | --- |
| `causeway_gdata.csv` | Full-table snapshot (all routes). Refreshed when the notebook runs with `REFRESH_FROM_BQ = True`. |

`*.csv` is gitignored.

## Without the notebook

```bash
cd eval
python -c "
from pathlib import Path
import timeseries_xgb as t
t.sync_canonical_travel_times('data/causeway_gdata.csv', refresh=True)
"
```

Requires Application Default Credentials and `pip install -r requirements-dev.txt` (`db-dtypes`, `pyarrow`).

Route-level frames use `prepare_route_frame(export, config)` (default **`jb_to_woodlands` — JB → SG only**; reverse route not in the offline notebook).

## Other eval paths

- **30 min BQML:** [`layer_b.py`](../layer_b.py) on `traffic_prediction.v_training_set` (live BigQuery, not this CSV).
