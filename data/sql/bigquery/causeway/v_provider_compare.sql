-- Google Maps vs TomTom, one row per route per 10-minute bin, from the first TomTom row on.
-- Written 2026-10-30 for the TomTom trial (services/gmapfetcher 1.4.x). Run in location US.
-- Real readings only: carried-forward copies are dropped. A bin where only one provider has an OK
-- reading is kept with the other side NULL, so failed calls show up as gaps.
-- same_road: the two routes are within 300 m of each other. TomTom jb_to_woodlands rows from before
-- 1.4.1 (vehicleHeading fix, about 8,450 m) come out FALSE; filter on it before comparing durations.
-- diff_min / diff_pct are TomTom minus Google (positive = TomTom slower).

CREATE OR REPLACE VIEW `swiftborder.causeway.v_provider_compare`
AS
WITH obs AS (
  SELECT 'gmaps' AS provider, route_id, direction, observed_at, distance_m, duration_in_traffic_sec, speed_kmh
  FROM `swiftborder.causeway.travel_times`
  WHERE status = 'OK'
    AND duration_in_traffic_sec IS NOT NULL
    AND NOT STARTS_WITH(IFNULL(error_message, ''), 'carried_forward')
    AND observed_at >= (SELECT MIN(observed_at) FROM `swiftborder.causeway.travel_times_tomtom`)
  UNION ALL
  SELECT 'tomtom' AS provider, route_id, direction, observed_at, distance_m, duration_in_traffic_sec, speed_kmh
  FROM `swiftborder.causeway.travel_times_tomtom`
  WHERE status = 'OK'
    AND duration_in_traffic_sec IS NOT NULL
    AND NOT STARTS_WITH(IFNULL(error_message, ''), 'carried_forward')
),
binned AS (
  SELECT
    provider,
    route_id,
    direction,
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(observed_at), 600) * 600) AS bin_ts,
    AVG(duration_in_traffic_sec) / 60.0 AS dur_min,
    AVG(distance_m) AS distance_m,
    AVG(speed_kmh) AS speed_kmh
  FROM obs
  GROUP BY 1, 2, 3, 4
),
g AS (SELECT * EXCEPT (provider) FROM binned WHERE provider = 'gmaps'),
t AS (SELECT * EXCEPT (provider) FROM binned WHERE provider = 'tomtom')
SELECT
  DATETIME(bin_ts, 'Asia/Singapore') AS bin_sgt,
  route_id,
  direction,
  ROUND(g.dur_min, 1) AS gmaps_min,
  ROUND(t.dur_min, 1) AS tomtom_min,
  ROUND(t.dur_min - g.dur_min, 1) AS diff_min,
  ROUND(SAFE_DIVIDE(t.dur_min - g.dur_min, g.dur_min) * 100, 1) AS diff_pct,
  CAST(g.distance_m AS INT64) AS gmaps_m,
  CAST(t.distance_m AS INT64) AS tomtom_m,
  ABS(t.distance_m - g.distance_m) <= 300 AS same_road,
  ROUND(g.speed_kmh, 1) AS gmaps_kmh,
  ROUND(t.speed_kmh, 1) AS tomtom_kmh,
  bin_ts
FROM g
FULL OUTER JOIN t USING (route_id, direction, bin_ts);
