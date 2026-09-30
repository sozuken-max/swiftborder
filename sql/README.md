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

**Not in this folder:** `causeway.travel_times` ingest (Maps fetcher), `traffic_images` backfill. The weather tables are appended by [`Causeway/load_bigquery.py`](../Causeway/load_bigquery.py) (manual; adds nullable `weatherforecast.update_timestamp`). See [docs/roadmap.md](../docs/roadmap.md).

## Models vs evaluation

- `bqml_*.sql` OPTIONS match `bq show --model` (2026-09-26; models unchanged on 2026-09-30: both created 2026-09-12, never retrained). The `is_val` dates mirror `model_registry.test_window` (`2026-09-11..12`). That is the **training-time** split.
- Report numbers come from [`eval/layer_b.py`](../eval/layer_b.py) on the fixed out-of-sample window from 13 Sep, not from BigQuery training metrics in the SQL headers.
- Live `v_training_set` and `v_forecast_recent` SQL matched these files on 2026-09-30. `eval/features.py` reproduces `v_training_set` in pandas with exact parity.

## Known view issues (documented, not changed)

- `v_training_set` computes lags and `y_30` / `y_60` by **row** offset. With no missing bins this equals a time offset (0 exceptions on 2026-09-30); `eval/layer_b.py` additionally checks that the label is exactly 30 minutes ahead.
- `v_weather_features_10min` bins forecasts by **issue** time, not valid period, and uses `ANY_VALUE(forecast)`.
- `v_congestion_index_10min` builds per-frame counts from detections only, so frames with zero vehicles are dropped and `vis_count` is biased upward.
- The weather and camera views are in `asia-southeast1`; `traffic_prediction` is in `US`, so they cannot be joined in one query. The joined experiment is offline in `eval/`.
