# Data & Warehouse (`data/`)

This directory groups offline data ingestion scripts and checked-in BigQuery warehouse DDL:

| Directory | Purpose | Key Files |
| --- | --- | --- |
| [`sql/`](sql/README.md) | BigQuery dataset DDL, views, and historical BQML reference statements for project `swiftborder` | [`sql/bigquery/EXPORT.md`](sql/bigquery/EXPORT.md), [`sql/bigquery/traffic_prediction/`](sql/bigquery/traffic_prediction/v_training_set.sql), [`sql/bigquery/cam2701/`](sql/bigquery/cam2701/local_counts.sql), [`sql/bigquery/weatherforecast/`](sql/bigquery/weatherforecast/v_weather_features_10min.sql) |
| [`weather/`](weather/README.md) | NEA data.gov.sg historical rainfall (`S210` Woodlands Centre) and 2-hour area forecast (`Woodlands`) fetchers, filters, and BigQuery loader | [`weather/fetch_rainfall_history.py`](weather/fetch_rainfall_history.py), [`weather/fetch_forecast_history.py`](weather/fetch_forecast_history.py), [`weather/load_bigquery.py`](weather/load_bigquery.py) |

Live Google Maps travel-time ingestion runs as a Cloud Run service in [`../services/gmapfetcher/`](../services/gmapfetcher/README.md), and offline evaluation caches live under [`../eval/data/`](../eval/data/README.md).
