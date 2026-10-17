-- BQML model `traffic_prediction.lin_h30` (LINEAR_REGRESSION).
-- Training OPTIONS and feature list from `bq show --model swiftborder:traffic_prediction.lin_h30` (2026-09-26).
-- The original Console training query is not stored in git; this script reproduces the same OPTIONS,
-- features, label (`y_30`), and CUSTOM split on `is_val` aligned with `model_registry.test_window` (2026-09-11..12).
-- BigQuery-reported training metrics (not the same as eval/layer_b.py hold-out): MAE ~3.04 min, R^2 ~0.84 on the CUSTOM eval split.

CREATE OR REPLACE MODEL `swiftborder.traffic_prediction.lin_h30`
OPTIONS (
  model_type = 'LINEAR_REG',
  input_label_cols = ['y_30'],
  data_split_method = 'CUSTOM',
  data_split_col = 'is_val',
  l2_reg = 0.1,
  optimize_strategy = 'NORMAL_EQUATION',
  calculate_p_values = FALSE,
  category_encoding_method = 'ONE_HOT_ENCODING'
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
