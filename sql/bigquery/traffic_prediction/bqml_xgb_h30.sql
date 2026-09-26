-- BQML model `traffic_prediction.xgb_h30` (BOOSTED_TREE_REGRESSOR).
-- Training OPTIONS and feature list from `bq show --model swiftborder:traffic_prediction.xgb_h30` (2026-09-26).
-- Same CUSTOM `is_val` split as `bqml_lin_h30.sql` (dates match `model_registry.test_window` 2026-09-11..12).
-- Not called by `v_forecast_recent`. BigQuery-reported training metrics on the CUSTOM eval split: MAE ~2.87 min, R^2 ~0.84.

CREATE OR REPLACE MODEL `swiftborder.traffic_prediction.xgb_h30`
OPTIONS (
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['y_30'],
  data_split_method = 'CUSTOM',
  data_split_col = 'is_val',
  num_parallel_tree = 1,
  max_iterations = 80,
  max_tree_depth = 4,
  learn_rate = 0.1,
  subsample = 0.8,
  l2_reg = 1,
  early_stop = TRUE,
  min_relative_progress = 0.01,
  category_encoding_method = 'LABEL_ENCODING'
) AS
SELECT
  direction,
  y_persistence,
  congestion_ratio,
  speed_kmh,
  lag_10,
  lag_20,
  lag_30,
  lag_60,
  roll_mean_30,
  roll_mean_60,
  slope_30,
  tod_sin,
  tod_cos,
  tod_block,
  dow,
  is_weekend,
  is_morning_peak,
  is_evening_peak,
  y_30,
  DATE(bin_sgt) IN (DATE '2026-09-11', DATE '2026-09-12') AS is_val
FROM `swiftborder.traffic_prediction.v_training_set`
WHERE y_30 IS NOT NULL
  AND after_gap = 0
  AND lag_60 IS NOT NULL;
