import { http } from '@google-cloud/functions-framework';
import { Client } from '@googlemaps/google-maps-services-js';
import { BigQuery } from '@google-cloud/bigquery';
import { Storage } from '@google-cloud/storage';
import { fillMidpoints } from './snapshot.js';
import { CARRY_TAG, carryForward, parseFlag, parseInterval, toSgtParts } from './carry.js';
import { applyTomTomResponse, tomtomUrl } from './tomtom.js';

const MAPS_KEY = process.env.GOOGLE_MAPS_API_KEY;
const TOMTOM_KEY = process.env.TOMTOM_API_KEY;
// Each provider is switched on or off on its own. Google stays on and TomTom off unless set.
const GMAPS_ENABLED = parseFlag('GMAPS_ENABLED', process.env.GMAPS_ENABLED, true);
const TOMTOM_ENABLED = parseFlag('TOMTOM_ENABLED', process.env.TOMTOM_ENABLED, false);
const BQ_DATASET = process.env.BQ_DATASET || 'causeway';
const BQ_TABLE = process.env.BQ_TABLE || 'travel_times';
// TomTom rows go to their own table: v_bins_10min averages every OK row in a bin, so sharing
// travel_times would blend the two providers into the training set and forecast-api.
const TOMTOM_BQ_TABLE = process.env.TOMTOM_BQ_TABLE || 'travel_times_tomtom';
const BQ_LOCATION = process.env.BQ_LOCATION || 'US';
const SNAPSHOT_BUCKET = process.env.SNAPSHOT_BUCKET || 'swiftborder-public';
const SNAPSHOT_OBJECT = process.env.SNAPSHOT_OBJECT || 'traffic-24h.json';
const TOMTOM_SNAPSHOT_OBJECT = process.env.TOMTOM_SNAPSHOT_OBJECT || 'traffic-24h-tomtom.json';
// Must match the Cloud Scheduler `Gmap-Woodlands` cadence.
const FETCH_INTERVAL_MIN = parseInterval(process.env.FETCH_INTERVAL_MIN);

if (TOMTOM_BQ_TABLE === BQ_TABLE) {
  throw new Error(`TOMTOM_BQ_TABLE must differ from BQ_TABLE (${BQ_TABLE})`);
}
if (TOMTOM_SNAPSHOT_OBJECT === SNAPSHOT_OBJECT) {
  throw new Error(`TOMTOM_SNAPSHOT_OBJECT must differ from SNAPSHOT_OBJECT (${SNAPSHOT_OBJECT})`);
}

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

async function fetchTomTomRoute(route, observedAt) {
  const row = { ...baseRow(route, observedAt), provider: 'tomtom' };

  try {
    const res = await fetch(tomtomUrl(route, TOMTOM_KEY), { signal: AbortSignal.timeout(10000) });
    const body = await res.json().catch(() => null);
    return applyTomTomResponse(row, res.status, body);
  } catch (err) {
    // The key is in the URL; keep it out of BigQuery and the logs.
    row.error_message = String(err?.message || err).split(TOMTOM_KEY).join('***').slice(0, 1000);
    return row;
  }
}

// Google rows are written exactly as before (no `provider` column in travel_times; null = Google).
const PROVIDERS = [
  {
    name: 'gmaps',
    enabled: GMAPS_ENABLED,
    key: MAPS_KEY,
    keyVar: 'GOOGLE_MAPS_API_KEY',
    fetchRoute,
    table: BQ_TABLE,
    snapshotObject: SNAPSHOT_OBJECT,
  },
  {
    name: 'tomtom',
    enabled: TOMTOM_ENABLED,
    key: TOMTOM_KEY,
    keyVar: 'TOMTOM_API_KEY',
    fetchRoute: fetchTomTomRoute,
    table: TOMTOM_BQ_TABLE,
    snapshotObject: TOMTOM_SNAPSHOT_OBJECT,
  },
];

// Query the last 24h of `table` and overwrite its public JSON snapshot (the website reads Google's).
async function writeSnapshot(observedAt, table, snapshotObject) {
  const query = `
    SELECT
      FORMAT_DATETIME('%Y-%m-%dT%H:%M', observed_at_sgt) AS t,
      route_id,
      duration_in_traffic_sec AS sec,
      speed_kmh
    FROM \`${BQ_DATASET}.${table}\`
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
    .file(snapshotObject)
    .save(JSON.stringify(payload), {
      contentType: 'application/json',
      metadata: { cacheControl: 'public, max-age=300' },
    });

  return rows.length;
}

// One provider's run: fetch both routes, insert into its table, rebuild its snapshot.
// Never throws, so one provider failing cannot stop the other.
async function runProvider(p, observedAt) {
  if (!p.key) return { error: `${p.keyVar} not set` };

  const rows = await Promise.all(ROUTES.map((r) => p.fetchRoute(r, observedAt)));
  const carried = carryForward(rows, FETCH_INTERVAL_MIN);

  try {
    await bigquery.dataset(BQ_DATASET).table(p.table).insert([...rows, ...carried]);
  } catch (err) {
    const detail = err?.errors
      ? JSON.stringify(err.errors).slice(0, 2000)
      : String(err?.message || err);
    console.error(`[${p.name}] BigQuery insert failed:`, detail);
    return { error: 'bigquery_insert_failed', detail };
  }

  // A snapshot failure must not fail the run: the data is already safely in
  // BigQuery and the next scheduled run regenerates the file from scratch.
  let snapshotPoints = null;
  let snapshotError = null;
  try {
    snapshotPoints = await writeSnapshot(observedAt, p.table, p.snapshotObject);
  } catch (err) {
    snapshotError = String(err?.message || err).slice(0, 500);
    console.error(`[${p.name}] Snapshot write failed:`, snapshotError);
  }

  return {
    inserted: rows.length,
    ok: rows.filter((r) => r.status === 'OK').length,
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
  };
}

http('fetchTravelTimes', async (req, res) => {
  const observedAt = new Date();
  const enabled = PROVIDERS.filter((p) => p.enabled);
  const results = await Promise.all(enabled.map((p) => runProvider(p, observedAt)));

  const providers = {};
  enabled.forEach((p, i) => {
    providers[p.name] = results[i];
  });
  if (enabled.length === 0) console.warn('No provider enabled (GMAPS_ENABLED and TOMTOM_ENABLED are both off)');

  const summary = {};
  for (const [name, r] of Object.entries(providers)) {
    const { rows, ...rest } = r;
    summary[name] = rest;
  }
  console.log(JSON.stringify({ observed_at: observedAt.toISOString(), providers: summary }));

  // 500 if any enabled provider failed to write; the others have still written their rows.
  const failed = results.some((r) => r.error);
  res.status(failed ? 500 : 200).json({
    observed_at_sgt: toSgtParts(observedAt).datetime,
    providers,
  });
});
