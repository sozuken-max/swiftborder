# Re-export SQL from BigQuery

Project: `swiftborder`. Run from a machine with `bq` and read access.

## Views (canonical DDL)

```bash
bq query --project_id=swiftborder --use_legacy_sql=false --format=json \
  "SELECT table_name, ddl FROM swiftborder.traffic_prediction.INFORMATION_SCHEMA.TABLES WHERE table_type = 'VIEW'"
```

Repeat for `swiftborder.weatherforecast` and `swiftborder.cam2701`.

Replace `CREATE VIEW` with `CREATE OR REPLACE VIEW` when saving under `sql/bigquery/<dataset>/`.

## BQML models

```bash
bq show --format=prettyjson --model swiftborder:traffic_prediction.lin_h30
bq show --format=prettyjson --model swiftborder:traffic_prediction.xgb_h30
```

Update `sql/bigquery/traffic_prediction/bqml_*.sql` OPTIONS and feature lists to match `trainingOptions` and `featureColumns`. The original `CREATE MODEL ... AS SELECT` text is not always available via API; keep the training query documented in those files.

## Serving registry

```bash
bq query --project_id=swiftborder --use_legacy_sql=false \
  "SELECT * FROM swiftborder.traffic_prediction.model_registry"
```
