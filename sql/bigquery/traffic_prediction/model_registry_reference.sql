-- Snapshot of `traffic_prediction.model_registry` from live BigQuery on 2026-09-26.
-- This table is operational metadata (which model the serve view may use per direction).
-- Re-score with eval/layer_b.py before changing serving_model or reason.

-- Example schema (create only if missing in a fresh project):
-- CREATE TABLE IF NOT EXISTS `swiftborder.traffic_prediction.model_registry` (
--   direction STRING,
--   serving_model STRING,
--   reason STRING,
--   decided_on DATE,
--   test_window STRING
-- );

MERGE `swiftborder.traffic_prediction.model_registry` AS t
USING (
  SELECT 'MY_TO_SG' AS direction, 'persistence' AS serving_model,
    'lin_h30 loses in 2 of 3 periods (morning -27.5%, off-peak -22.5%, bias +2.7 min); not promoted' AS reason,
    DATE '2026-09-12' AS decided_on, '2026-09-11..12' AS test_window
  UNION ALL
  SELECT 'SG_TO_MY', 'lin_h30',
    'beats persistence in all 3 periods on held-out days (skill +1.5% evening, +6.1% morning, +17.1% off-peak)',
    DATE '2026-09-12', '2026-09-11..12'
) AS s
ON t.direction = s.direction
WHEN MATCHED THEN UPDATE SET
  serving_model = s.serving_model,
  reason = s.reason,
  decided_on = s.decided_on,
  test_window = s.test_window
WHEN NOT MATCHED THEN INSERT ROW;
