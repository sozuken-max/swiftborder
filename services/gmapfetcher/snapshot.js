// Pure helpers for the public snapshot; no Google client imports, so tests run without npm install.

const MINUTE_MS = 60 * 1000;

// "YYYY-MM-DDTHH:MM" (SGT wall clock) <-> epoch ms. The offset cancels out, so UTC parsing is fine.
function parseMinute(t) {
  return Date.parse(`${t}:00Z`);
}

function formatMinute(ms) {
  return new Date(ms).toISOString().slice(0, 16);
}

// Real points arrive every `gapMin` minutes (FETCH_INTERVAL_MIN). Between two real points whose gap is
// within 1 min of a multiple of 5 min, from 10 min up to `gapMin` (a run can slip across a minute
// boundary), add straight-line points every 5 min after the earlier one so the website keeps its 5-min
// grid; older 10-min history is filled too when `gapMin` is 20. Points exist only in the JSON, never in
// BigQuery. Points are [t, sec, speed_kmh, interpolated]. The newest point is always real.
export function fillMidpoints(points, { gapMin = 10, marginMin = 1, stepMin = 5 } = {}) {
  const out = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (i > 0) {
      const prev = points[i - 1];
      const gap = (parseMinute(p[0]) - parseMinute(prev[0])) / MINUTE_MS;
      const n = Math.round(gap / stepMin);
      if (n >= 2 && n * stepMin <= gapMin && Math.abs(gap - n * stepMin) <= marginMin) {
        for (let k = 1; k < n; k++) {
          const frac = (k * stepMin) / gap;
          const lerp = (a, b, digits) =>
            a == null || b == null ? null : +(a + (b - a) * frac).toFixed(digits);
          out.push([
            formatMinute(parseMinute(prev[0]) + k * stepMin * MINUTE_MS),
            lerp(prev[1], p[1], 0),
            lerp(prev[2], p[2], 2),
            true,
          ]);
        }
      }
    }
    out.push([p[0], p[1], p[2], false]);
  }
  return out;
}
