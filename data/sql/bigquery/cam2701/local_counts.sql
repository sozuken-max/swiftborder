-- Per-frame model=local vehicle counts by direction, written by camdetect/backfill_local_counts.py
-- (--bq-table swiftborder.cam2701.local_counts). The script creates this table if it is missing;
-- keep this DDL in step with BQ_SCHEMA there. Dataset cam2701 is in asia-southeast1, so this table
-- cannot be joined in one query with causeway.travel_times (US).
CREATE TABLE IF NOT EXISTS `swiftborder.cam2701.local_counts` (
  camera_id STRING NOT NULL,
  frame_datetime_sgt DATETIME NOT NULL,  -- capture time from the object name, SGT wall clock
  frame_ts TIMESTAMP NOT NULL,           -- same instant, UTC
  sg_my INT64 NOT NULL,                  -- Singapore -> Malaysia (above the dividing line)
  my_sg INT64 NOT NULL,                  -- Malaysia -> Singapore
  unknown INT64 NOT NULL,                -- no direction (sg_my + my_sg + unknown = total)
  total INT64 NOT NULL,
  model_id STRING NOT NULL,              -- e.g. local:yolo26s-v6-boxfix
  min_confidence FLOAT64 NOT NULL,
  max_overlap FLOAT64,                   -- dedupe cut; NULL when off
  source STRING NOT NULL,                -- gs:// URI of the scored frame (resume key)
  scored_at TIMESTAMP NOT NULL
)
CLUSTER BY camera_id
OPTIONS (description = 'Per-frame model=local vehicle counts by direction (camdetect/backfill_local_counts.py).');
