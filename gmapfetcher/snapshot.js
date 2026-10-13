// Pure helpers for the public snapshot; no Google client imports, so tests run without npm install.

const MINUTE_MS = 60 * 1000;

// "YYYY-MM-DDTHH:MM" (SGT wall clock) <-> epoch ms. The offset cancels out, so UTC parsing is fine.
function parseMinute(t) {
  return Date.parse(`${t}:00Z`);
}

function formatMinute(ms) {
  return new Date(ms).toISOString().slice(0, 16);
}

// Real points arrive every 10 min. Between two real points 9-11 min apart (10 min, with a 1-min margin
// for a run that slips across a minute boundary), add the straight-line point 5 min after the earlier
// one so the website keeps its 5-min grid. Midpoints exist only in the JSON, never in BigQuery.
// Points are [t, sec, speed_kmh, interpolated]. The newest point is always real.
export function fillMidpoints(points, { gapMin = 10, marginMin = 1, stepMin = 5 } = {}) {
  const out = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (i > 0) {
      const prev = points[i - 1];
      const gap = (parseMinute(p[0]) - parseMinute(prev[0])) / MINUTE_MS;
      if (Math.abs(gap - gapMin) <= marginMin) {
        const frac = stepMin / gap;
        const lerp = (a, b, digits) =>
          a == null || b == null ? null : +(a + (b - a) * frac).toFixed(digits);
        out.push([
          formatMinute(parseMinute(prev[0]) + stepMin * MINUTE_MS),
          lerp(prev[1], p[1], 0),
          lerp(prev[2], p[2], 2),
          true,
        ]);
      }
    }
    out.push([p[0], p[1], p[2], false]);
  }
  return out;
}
