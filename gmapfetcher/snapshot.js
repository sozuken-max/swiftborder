// Pure helpers for the public snapshot; no Google client imports, so tests run without npm install.

const MINUTE_MS = 60 * 1000;

// "YYYY-MM-DDTHH:MM" (SGT wall clock) <-> epoch ms. The offset cancels out, so UTC parsing is fine.
function parseMinute(t) {
  return Date.parse(`${t}:00Z`);
}

function formatMinute(ms) {
  return new Date(ms).toISOString().slice(0, 16);
}

// Real points arrive every 10 min. Between two real points 10 min apart, add the straight-line
// midpoint so the website keeps its 5-min grid. Midpoints exist only in the JSON, never in BigQuery.
// Points are [t, sec, speed_kmh, interpolated]. The newest point is always real.
export function fillMidpoints(points, { gapMin = 10, stepMin = 5 } = {}) {
  const out = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (i > 0) {
      const prev = points[i - 1];
      const gap = (parseMinute(p[0]) - parseMinute(prev[0])) / MINUTE_MS;
      if (gap === gapMin) {
        const mid = (a, b, digits) =>
          a == null || b == null ? null : +((a + b) / 2).toFixed(digits);
        out.push([
          formatMinute(parseMinute(prev[0]) + stepMin * MINUTE_MS),
          mid(prev[1], p[1], 0),
          mid(prev[2], p[2], 2),
          true,
        ]);
      }
    }
    out.push([p[0], p[1], p[2], false]);
  }
  return out;
}
