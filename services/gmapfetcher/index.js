import { http } from '@google-cloud/functions-framework';
import { Client } from '@googlemaps/google-maps-services-js';
import { BigQuery } from '@google-cloud/bigquery';
import { Storage } from '@google-cloud/storage';
import { fillMidpoints } from './snapshot.js';
import { CARRY_TAG, carryForward, parseInterval, toSgtParts } from './carry.js';

const MAPS_KEY = process.env.GOOGLE_MAPS_API_KEY;
const BQ_DATASET = process.env.BQ_DATASET || 'causeway';
const BQ_TABLE = process.env.BQ_TABLE || 'travel_times';
const BQ_LOCATION = process.env.BQ_LOCATION || 'US';
const SNAPSHOT_BUCKET = process.env.SNAPSHOT_BUCKET || 'swiftborder-public';
const SNAPSHOT_OBJECT = process.env.SNAPSHOT_OBJECT || 'traffic-24h.json';
// Must match the Cloud Scheduler `Gmap-Woodlands` cadence.
const FETCH_INTERVAL_MIN = parseInterval(process.env.FETCH_INTERVAL_MIN);

const maps = new Client({});
const bigquery = new BigQuery();
const storage = new Storage();

const ROUTES = [
  {
    route_id: 'mandai_to_shell_jb',
    route_name: 'Mandai Rd to Shell after Woodlands checkpoint',
    direction: 'SG_TO_MY',
    origin: { lat: 1.4122921412065066, lng: 103.77127314705015 },
    destination: { lat: 1.4706219584484974, lng: 103.76692293761168 },
  },
  {
    route_id: 'jb_to_woodlands',
    route_name: 'JB approach to Woodlands (post-checkpoint)',
    direction: 'MY_TO_SG',
    origin: { lat: 1.4884465059461982, lng: 103.78625911225137 },
    destination: { lat: 1.4401375461664851, lng: 103.76842889302142 },
  },
];

function baseRow(route, observedAt) {
  const sgt = toSgtParts(observedAt);
  return {
    observed_at: observedAt.toISOString(),
    observed_at_sgt: sgt.datetime,
    observed_date_sgt: sgt.date,
    route_id: route.route_id,
    route_name: route.route_name,
    direction: route.direction,
    origin_lat: route.origin.lat,
    origin_lng: route.origin.lng,
    dest_lat: route.destination.lat,
    dest_lng: route.destination.lng,
    status: 'ERROR',
    distance_m: null,
    duration_sec: null,
    duration_in_traffic_sec: null,
    congestion_ratio: null,
    speed_kmh: null,
    error_message: null,
  };
}

async function fetchRoute(route, observedAt) {
  const row = baseRow(route, observedAt);

  try {
    const res = await maps.distancematrix({
      params: {
        origins: [route.origin],
        destinations: [route.destination],
        mode: 'driving',
        departure_time: observedAt,
        traffic_model: 'best_guess',
        key: MAPS_KEY,
      },
      timeout: 10000,
    });

    if (res.data.status !== 'OK') {
      row.status = res.data.status;
      row.error_message = res.data.error_message || null;
      return row;
    }

    const el = res.data.rows?.[0]?.elements?.[0];
    if (!el) {
      row.status = 'NO_ELEMENT';
      return row;
    }

    row.status = el.status;
    if (el.status !== 'OK') return row;

    const distanceM = el.distance?.value ?? null;
    const durationSec = el.duration?.value ?? null;
    const trafficSec = el.duration_in_traffic?.value ?? null;
    const effectiveSec = trafficSec ?? durationSec;

    row.distance_m = distanceM;
    row.duration_sec = durationSec;
    row.duration_in_traffic_sec = trafficSec;
    row.congestion_ratio =
      trafficSec && durationSec ? +(trafficSec / durationSec).toFixed(4) : null;
    row.speed_kmh =
      distanceM && effectiveSec
        ? +((distanceM / 1000) / (effectiveSec / 3600)).toFixed(2)
        : null;

    return row;
  } catch (err) {
    row.error_message = String(err?.message || err).slice(0, 1000);
    return row;
  }
}

// Query the last 24h and overwrite the public JSON snapshot the website reads.
async function writeSnapshot(observedAt) {
  const query = `
    SELECT
      FORMAT_DATETIME('%Y-%m-%dT%H:%M', observed_at_sgt) AS t,
      route_id,
      duration_in_traffic_sec AS sec,
      speed_kmh
    FROM \`${BQ_DATASET}.${BQ_TABLE}\`
    WHERE observed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
      AND status = 'OK'
      AND duration_in_traffic_sec IS NOT NULL
      AND NOT STARTS_WITH(IFNULL(error_message, ''), '${CARRY_TAG}')
    ORDER BY observed_at
  `;

  const [rows] = await bigquery.query({ query, location: BQ_LOCATION });

  const series = {};
  for (const r of ROUTES) {
    series[r.route_id] = { name: r.route_name, direction: r.direction, points: [] };
  }
  for (const row of rows) {
    if (!series[row.route_id]) continue;
    series[row.route_id].points.push([row.t, row.sec, row.speed_kmh]);
  }
  for (const s of Object.values(series)) s.points = fillMidpoints(s.points, { gapMin: FETCH_INTERVAL_MIN });

  const payload = {
    updated_at_sgt: toSgtParts(observedAt).datetime,
    window_hours: 24,
    point_format: ['datetime_sgt', 'duration_in_traffic_sec', 'speed_kmh', 'interpolated'],
    series,
  };

  await storage
    .bucket(SNAPSHOT_BUCKET)
    .file(SNAPSHOT_OBJECT)
    .save(JSON.stringify(payload), {
      contentType: 'application/json',
      metadata: { cacheControl: 'public, max-age=300' },
    });

  return rows.length;
}

http('fetchTravelTimes', async (req, res) => {
  if (!MAPS_KEY) {
    res.status(500).json({ error: 'GOOGLE_MAPS_API_KEY not set' });
    return;
  }

  const observedAt = new Date();
  const rows = await Promise.all(ROUTES.map((r) => fetchRoute(r, observedAt)));
  const carried = carryForward(rows, FETCH_INTERVAL_MIN);

  try {
    await bigquery.dataset(BQ_DATASET).table(BQ_TABLE).insert([...rows, ...carried]);
  } catch (err) {
    const detail = err?.errors
      ? JSON.stringify(err.errors).slice(0, 2000)
      : String(err?.message || err);
    console.error('BigQuery insert failed:', detail);
    res.status(500).json({ error: 'bigquery_insert_failed', detail });
    return;
  }

  // A snapshot failure must not fail the run: the data is already safely in
  // BigQuery and the next scheduled run regenerates the file from scratch.
  let snapshotPoints = null;
  let snapshotError = null;
  try {
    snapshotPoints = await writeSnapshot(observedAt);
  } catch (err) {
    snapshotError = String(err?.message || err).slice(0, 500);
    console.error('Snapshot write failed:', snapshotError);
  }

  const okCount = rows.filter((r) => r.status === 'OK').length;
  console.log(
    JSON.stringify({
      observed_at: observedAt.toISOString(),
      inserted: rows.length,
      ok: okCount,
      carried_forward: carried.length,
      snapshot_points: snapshotPoints,
      snapshot_error: snapshotError,
    })
  );

  res.status(200).json({
    observed_at_sgt: toSgtParts(observedAt).datetime,
    inserted: rows.length,
    ok: okCount,
    carried_forward: carried.length,
    snapshot_points: snapshotPoints,
    snapshot_error: snapshotError,
    rows: rows.map((r) => ({
      route_id: r.route_id,
      status: r.status,
      distance_m: r.distance_m,
      duration_in_traffic_sec: r.duration_in_traffic_sec,
      speed_kmh: r.speed_kmh,
    })),
  });
});
