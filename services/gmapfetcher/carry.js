// Pure helpers for the BigQuery rows; no Google client imports, so tests run without npm install.

const MINUTE_MS = 60 * 1000;
const BIN_MIN = 10; // traffic_prediction.v_bins_10min

// Marks a carried-forward row. The table has no flag column, so the marker lives in error_message.
export const CARRY_TAG = 'carried_forward';

// SGT is UTC+8, no DST, so a fixed offset is safe.
export function toSgtParts(date) {
  const iso = new Date(date.getTime() + 8 * 3600 * 1000).toISOString();
  return { datetime: iso.slice(0, 19), date: iso.slice(0, 10) };
}

// Minutes between scheduled runs: a multiple of 10, at least 10. Anything else throws.
export function parseInterval(value, fallback = 20) {
  const n = value == null || value === '' ? fallback : Number(value);
  if (!Number.isInteger(n) || n < BIN_MIN || n % BIN_MIN !== 0) {
    throw new Error(`FETCH_INTERVAL_MIN must be a multiple of ${BIN_MIN}, got ${value}`);
  }
  return n;
}

// The fetcher runs every `intervalMin`, but v_bins_10min, v_training_set and forecast-api expect one
// row per route in every 10-min bin (y_30 is 3 bins ahead; with 20-min rows it would always be empty).
// For each OK row observed at T, add a copy of the same reading at T+10, T+20, ... up to the next run.
// A copy uses only the reading at T (last observation carried forward), so no bin ever sees a later
// observation. Failed rows are not carried: a failed run leaves its bins empty, as before.
export function carryForward(rows, intervalMin) {
  const out = [];
  for (const row of rows) {
    if (row.status !== 'OK') continue;
    const t = Date.parse(row.observed_at);
    for (let m = BIN_MIN; m < intervalMin; m += BIN_MIN) {
      const at = new Date(t + m * MINUTE_MS);
      const sgt = toSgtParts(at);
      out.push({
        ...row,
        observed_at: at.toISOString(),
        observed_at_sgt: sgt.datetime,
        observed_date_sgt: sgt.date,
        error_message: `${CARRY_TAG} from ${row.observed_at}`,
      });
    }
  }
  return out;
}
