# BigQuery SQL (project `swiftborder`)

Checked-in definitions for Layer B views, side-feature views, BQML models, and serving registry reference data. Exported or reconstructed **2026-09-26** from live BigQuery (`INFORMATION_SCHEMA.TABLES`, `bq show --model`, `model_registry` query).

**Authority:** GCP project `swiftborder` wins if these files drift. Re-export with the commands in [bigquery/EXPORT.md](bigquery/EXPORT.md).

## Apply order

Run in BigQuery (location **US** for `traffic_prediction`; **asia-southeast1** for `cam2701` / `weatherforecast` views).

| Step | File | Notes |
| --- | --- | --- |
| 1 | [bigquery/traffic_prediction/v_bins_10min.sql](bigquery/traffic_prediction/v_bins_10min.sql) | Needs `causeway.travel_times` |
| 2 | [bigquery/traffic_prediction/v_training_set.sql](bigquery/traffic_prediction/v_training_set.sql) | Maps-only features |
| 3 | [bigquery/traffic_prediction/bqml_lin_h30.sql](bigquery/traffic_prediction/bqml_lin_h30.sql) | Trains on `v_training_set`; CUSTOM split |
| 4 | [bigquery/traffic_prediction/bqml_xgb_h30.sql](bigquery/traffic_prediction/bqml_xgb_h30.sql) | Optional; not served by `v_forecast_recent` |
| 5 | [bigquery/traffic_prediction/model_registry_reference.sql](bigquery/traffic_prediction/model_registry_reference.sql) | Serving metadata only |
| 6 | [bigquery/traffic_prediction/v_forecast_recent.sql](bigquery/traffic_prediction/v_forecast_recent.sql) | Needs models + registry |
| — | [bigquery/weatherforecast/v_weather_features_10min.sql](bigquery/weatherforecast/v_weather_features_10min.sql) | Not joined to training today |
| — | [bigquery/cam2701/v_congestion_index_10min.sql](bigquery/cam2701/v_congestion_index_10min.sql) | Not joined to training today |

**Not in this folder:** `causeway.travel_times` ingest (Maps fetcher), `traffic_images` backfill, rainfall/weather table loaders. See [docs/roadmap.md](../docs/roadmap.md).

## Models vs evaluation

- `bqml_*.sql` OPTIONS match `bq show --model` on 2026-09-26. The `is_val` dates mirror `model_registry.test_window` (`2026-09-11..12`). That is the **training-time** split, not the trailing hold-out in [`eval/layer_b.py`](../eval/layer_b.py).
- Report numbers should come from `eval/layer_b.py` (skill on the Maps series), not from BigQuery training metrics in the SQL headers.
