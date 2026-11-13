// SwiftBorder front end: Woodlands Causeway travel times, forecast and live cameras.
//
// Two views on one page, picked by the URL hash:
//   "#"               summary: one card per direction + AI vehicle detection
//   "#forecast"       Causeway: trend chart, forecast vs typical day, LTA cameras
//   "#forecast/tuas"  Tuas: LTA cameras only (no travel-time feed for Tuas)
//
// Data sources (all public, read from the browser):
//   traffic-24h.json  Google Maps Distance Matrix travel times, written every 5 min
//   forecast-api      forecast curve and typical-day profile (docs/runbooks/forecast-api.md)
//                     and ?advice=commute: the Commute Buddy card (services/forecastapi/README.md)
//   data.gov.sg       LTA traffic camera images
//   swiftbackend      annotated camera frame + per-direction vehicle counts

// ── Endpoints ───────────────────────────────────────────────────────────────
const TRAFFIC_API = 'https://storage.googleapis.com/swiftborder-public/traffic-24h.json';
const FORECAST_API = 'https://forecast-api-1095552466513.asia-southeast1.run.app/';
const ADVISOR_API = `${FORECAST_API}?advice=commute`; // same forecast-api service, separate request
const LTA_API_URL = 'https://api.data.gov.sg/v1/transport/traffic-images';
const BACKEND_URL = 'https://swiftbackend-1095552466513.europe-west1.run.app/';

const SGT = 'Asia/Singapore';

// ── Directions ──────────────────────────────────────────────────────────────
// `series` is the key in traffic-24h.json, `apiKey` the key in forecast-api,
// `aiKey` the direction suffix on swiftbackend's count headers. `color` must
// match --dir-sg / --dir-jb in style.css: blue and violet, kept clear of the
// green / amber / red congestion colours so a hue means one thing.
const DIRECTIONS = [
    { key: 'sg', label: 'To Johor', series: 'mandai_to_shell_jb', apiKey: 'SG_TO_MY', aiKey: 'SG-MY', color: '#5b9dff' },
    { key: 'jb', label: 'To Singapore', series: 'jb_to_woodlands', apiKey: 'MY_TO_SG', aiKey: 'MY-SG', color: '#c58aff' }
];

// ── Checkpoints and cameras (data.gov.sg camera IDs) ────────────────────────
const CHECKPOINTS = {
    woodlands: { name: 'Woodlands Causeway', cameras: ['2701', '2702', '2703', '2704', '2705', '2706'] },
    tuas: { name: 'Tuas Second Link', cameras: ['4703', '4712', '4713', '4714', '4716'] }
};

// ── Tuning ──────────────────────────────────────────────────────────────────
// Congestion colour: fixed cut-offs on the whole minutes shown, both directions.
const LEVEL_WARN_MIN = 30;       // 30-50 min amber
const LEVEL_DANGER_MIN = 50;     // over 50 min red

// Summary trend arrow: mean of the last 30 min against the 60 min before it.
const TREND_RECENT_MIN = 30;
const TREND_PRIOR_MIN = 60;
const TREND_MIN_POINTS = 3;      // per half; fewer => steady
const TREND_STABLE_PCT = 10;     // within +/-10% => steady

const NO_DATA_AFTER_MIN = 60;    // newest sample older than this => card shows no data
const STALE_AFTER_MIN = 20;      // newest sample older than this => "(n min old)"

// Forecast: only the 30-min point is evaluated; later points are exploratory and the
// model held up to about 5.5 h in the study, so the page asks for 5 h. The comparison
// chart spans the full day: that 5 h forecast over the 24 h typical-day profile.
const FORECAST_HOURS = 5;
const COMPARE_HOURS = 24;
const FORECAST_MAX_SLOTS = 66;           // 5.5 h of 5-min slots on the trend chart
// A cold forecast-api instance retrains every horizon's model before it answers: 91.5 s on
// 2026-11-13. The page tells the user, counts down two minutes, and keeps waiting after that.
const CURVE_TIMEOUT_MS = 240000;
const RETRAIN_COUNTDOWN_MS = 120000;
const PENDING_NOTICE_DELAY_MS = 2000;    // a warm curve answers in ~3 s: no notice for those
const BASELINE_REFRESH_MS = 3600000;     // the typical-day profile is refitted once per day

const REFRESH_MS = 300000;               // travel times and forecast
const CAMERA_REFRESH_MS = 60000;         // camera images, forecast view only
const SLOT_MS = 300000;                  // one 5-min sample

// ── State ───────────────────────────────────────────────────────────────────
let currentView = null;          // 'home' | 'forecast'; null until the first route()
let currentCheckpoint = 'woodlands';
let cameraTimer = null;
let trafficCache = { data: null, fetchedAt: 0, inFlight: null };
let forecastView = null;         // { curve, dirs: { sg|jb: { originAt, points: [{ at, mins, base, model, lead }] } } }
let baselineDay = null;          // { sg|jb: [{ t, mins }] } typical day, 24 h
let baselineAt = 0;

// ── Small helpers ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// "HH:MM" in Singapore time, from epoch ms or an ISO string.
function sgtClock(ts) {
    const t = typeof ts === 'number' ? ts : Date.parse(ts);
    if (Number.isNaN(t)) return '—';
    return new Date(t).toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', timeZone: SGT });
}

// "HH:MM:SS" in Singapore time.
function sgtTime(ts) {
    return new Date(ts).toLocaleTimeString('en-GB', { hour12: false, timeZone: SGT });
}

// Singapore wall clock "YYYY-MM-DDTHH:MM:SS", the backend's date_time format.
function sgtNowParam() {
    return new Date().toLocaleString('sv-SE', { timeZone: SGT, hour12: false }).replace(' ', 'T');
}

// Feed timestamps are Singapore wall clock with no offset ("2026-09-06T14:30").
// A timestamp that carries an explicit offset is parsed as-is.
function parseSgt(ts) {
    if (typeof ts !== 'string') return null;
    if (/(Z|[+-]\d{2}:?\d{2})$/.test(ts)) {
        const t = Date.parse(ts);
        return Number.isNaN(t) ? null : t;
    }
    const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/.exec(ts);
    if (!m) return null;
    return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 8, +m[5], +(m[6] || 0)); // SGT = UTC+8
}

function mean(values) {
    return values.reduce((a, b) => a + b, 0) / values.length;
}

// Whole minutes. Every "now" figure (summary, forecast cards, chart footer) uses this,
// so they agree with each other and with the newest Google Maps sample.
function fmtMin(m) {
    return Math.round(m).toString();
}

// Congestion level for a travel time, judged on the rounded value so the colour
// always matches the digits on screen.
function congestionLevel(mins) {
    if (mins == null) return { text: 'No data', cls: 'nodata' };
    const m = Math.round(mins);
    if (m > LEVEL_DANGER_MIN) return { text: 'Heavy', cls: 'danger' };
    if (m >= LEVEL_WARN_MIN) return { text: 'Building up', cls: 'warning' };
    return { text: 'Smooth', cls: 'normal' };
}

async function fetchJson(url, timeoutMs, errors = {}) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
        const res = await fetch(url, { signal: ctl.signal });
        if (!res.ok) throw new Error(errors[res.status] || `HTTP ${res.status}`);
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

// Disable a refresh button while its task runs, plus a short cooldown.
function bindRefresh(id, task) {
    const btn = $(id);
    if (!btn) return;
    btn.addEventListener('click', () => {
        btn.disabled = true;
        Promise.resolve(task()).finally(() => setTimeout(() => { btn.disabled = false; }, 3000));
    });
}

// ── Traffic feed (Google Maps travel times) ─────────────────────────────────
// One fetch serves the summary cards, the trend chart and the forecast cards' "Now".
async function loadTrafficData(force = false) {
    if (!force && trafficCache.data && Date.now() - trafficCache.fetchedAt < 60000) return trafficCache.data;
    if (trafficCache.inFlight) return trafficCache.inFlight; // collapse concurrent callers

    trafficCache.inFlight = (async () => {
        try {
            const res = await fetch(TRAFFIC_API);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            trafficCache.data = await res.json();
            trafficCache.fetchedAt = Date.now();
            return trafficCache.data;
        } finally {
            trafficCache.inFlight = null;
        }
    })();
    return trafficCache.inFlight;
}

// Samples for one direction as [{ t (epoch ms), mins }], oldest first.
function seriesPoints(data, name) {
    return (data?.series?.[name]?.points || [])
        .map(p => ({ t: parseSgt(p[0]), mins: p[1] == null ? null : p[1] / 60 }))
        .filter(p => p.t != null && p.mins != null)
        .sort((a, b) => a.t - b.t);
}

// ── Summary cards ───────────────────────────────────────────────────────────
const TREND_TEXT = { up: 'Rising', down: 'Easing', steady: 'Steady' };

function macroTrend(pts, anchor) {
    const recentStart = anchor - TREND_RECENT_MIN * 60000;
    const priorStart = recentStart - TREND_PRIOR_MIN * 60000;
    const recent = pts.filter(p => p.t > recentStart && p.t <= anchor).map(p => p.mins);
    const prior = pts.filter(p => p.t > priorStart && p.t <= recentStart).map(p => p.mins);
    if (recent.length < TREND_MIN_POINTS || prior.length < TREND_MIN_POINTS) return { dir: 'steady', pct: 0 };
    const pct = ((mean(recent) - mean(prior)) / mean(prior)) * 100;
    return { dir: pct > TREND_STABLE_PCT ? 'up' : pct < -TREND_STABLE_PCT ? 'down' : 'steady', pct };
}

// Card = minutes (newest sample), colour (congestion level), arrow (macro trend).
function renderSummaryCard(d, mins, trend) {
    const card = $(`row-${d.key}`);
    const trendEl = $(`${d.key}-trend`);
    const level = congestionLevel(mins);

    $(`${d.key}-time`).textContent = mins == null ? '—' : fmtMin(mins);
    card.classList.remove('normal', 'warning', 'danger', 'nodata');
    card.classList.add(level.cls);

    const dir = trend ? trend.dir : 'steady';
    trendEl.className = `summary-trend ${dir}`;
    trendEl.querySelector('.trend-text').textContent = mins == null ? 'No data' : TREND_TEXT[dir];
    trendEl.title = mins == null ? '' : `Last 30 min vs the hour before: ${trend.pct >= 0 ? '+' : ''}${trend.pct.toFixed(0)}%`;

    const said = mins == null ? 'no data' : `${fmtMin(mins)} minutes, ${level.text.toLowerCase()}, ${TREND_TEXT[dir].toLowerCase()}`;
    card.setAttribute('aria-label', `${d.label}: ${said}. Open the forecast.`);
}

async function updateSummary() {
    const updated = $('last-updated');
    try {
        const data = await loadTrafficData();
        const pts = Object.fromEntries(DIRECTIONS.map(d => [d.key, seriesPoints(data, d.series)]));
        const latestTs = Math.max(...DIRECTIONS.map(d => pts[d.key].at(-1)?.t || 0));
        if (!latestTs) throw new Error('no usable samples in feed');

        // A stale feed is judged at its own newest sample rather than at the wall clock.
        const nowMs = Date.now();
        const stale = nowMs - latestTs > STALE_AFTER_MIN * 60000;
        const anchor = stale ? latestTs : nowMs;

        DIRECTIONS.forEach(d => {
            const latest = pts[d.key].filter(p => p.t <= anchor).at(-1);
            if (!latest || anchor - latest.t > NO_DATA_AFTER_MIN * 60000) renderSummaryCard(d, null);
            else renderSummaryCard(d, latest.mins, macroTrend(pts[d.key], anchor));
        });

        const age = Math.round((nowMs - latestTs) / 60000);
        updated.textContent = `${sgtClock(latestTs)} SGT${stale ? ` (${age} min old)` : ''}`;
    } catch (err) {
        console.error('Traffic summary error:', err);
        DIRECTIONS.forEach(d => renderSummaryCard(d, null));
        updated.textContent = 'Feed unavailable';
    }
}

// ── Trend chart (forecast view) ─────────────────────────────────────────────
// The last 24 hours of samples for both directions (a rolling window, so it is not empty just
// after midnight), plus the forecast drawn as a dashed extension in extra 5-min slots after
// the newest observation.
async function renderTrendChart() {
    const area = $('lc-chart-area');
    const showMessage = msg => {
        area.innerHTML = `<div class="lc-loading">${msg}</div>`;
        $('lc-footer').hidden = true;
    };

    let data;
    try {
        data = await loadTrafficData();
    } catch (err) {
        console.error('Trend chart error:', err);
        showMessage(`Could not load travel times: ${err.message}`);
        return;
    }

    // Points are keyed by their full timestamp: a clock time such as 00:40 can occur twice in 24 h.
    const cutoff = Date.now() - 24 * 3600 * 1000;
    const series = DIRECTIONS.map(d => {
        const pts = (data.series?.[d.series]?.points || [])
            .filter(p => p[1] != null && parseSgt(p[0]) >= cutoff)
            .map(p => ({ key: p[0], label: p[0].slice(11, 16), mins: p[1] / 60 }));
        return { d, pts, map: Object.fromEntries(pts.map(p => [p.key, p.mins])), fc: {} };
    });
    if (series.every(s => !s.pts.length)) { showMessage('No data in the last 24 hours yet.'); return; }

    // x axis: one slot per observed timestamp, then one per forecast slot.
    const keys = [...new Set(series.flatMap(s => s.pts.map(p => p.key)))].sort();
    const labels = keys.map(k => k.slice(11, 16));
    const n = keys.length;
    const lastEpoch = parseSgt(keys[n - 1]);
    let maxSlot = 0;
    series.forEach(s => {
        (forecastView?.dirs[s.d.key].points || []).forEach(p => {
            const slot = Math.round((p.at - lastEpoch) / SLOT_MS);
            if (slot >= 1 && slot <= FORECAST_MAX_SLOTS) {
                s.fc[slot] = p;
                maxSlot = Math.max(maxSlot, slot);
            }
        });
    });
    const hasFc = maxSlot > 0;
    const axis = labels.concat(Array.from({ length: maxSlot }, (_, k) => sgtClock(lastEpoch + (k + 1) * SLOT_MS)));
    const N = axis.length;
    const fcIdxs = [...new Set(series.flatMap(s => Object.keys(s.fc).map(Number)))]
        .sort((a, b) => a - b).map(slot => n - 1 + slot);

    // Geometry. A narrower viewBox keeps the labels legible on phones.
    const W = window.innerWidth < 640 ? 440 : 800, H = 240;
    const padL = 62, padR = 20, padT = 16, padB = 36;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const values = series.flatMap(s => s.pts.map(p => p.mins).concat(Object.values(s.fc).map(p => p.mins)));
    const pad = Math.max(1, (Math.max(...values) - Math.min(...values)) * 0.15);
    const minVal = Math.max(0, Math.min(...values) - pad);
    const maxVal = Math.max(...values) + pad;
    const xOf = i => padL + (i / Math.max(N - 1, 1)) * plotW;
    const yOf = v => padT + plotH - ((v - minVal) / (maxVal - minVal || 1)) * plotH;
    const xy = (x, y) => `${x.toFixed(1)},${y.toFixed(1)}`;

    // Dots shrink as the window fills up and disappear past ~120 points.
    const dotR = n > 120 ? 0 : n > 60 ? 1.8 : n > 30 ? 2.6 : 3.5;
    const dotStroke = dotR >= 2.6 ? 1.5 : 1;
    const lineW = n > 120 ? 1.8 : 2.5;

    function seriesSvg({ d, map, fc }) {
        const pts = keys.map((k, i) => map[k] != null ? [xOf(i), yOf(map[k])] : null).filter(Boolean);
        if (!pts.length) return '';
        const [lx, ly] = pts[pts.length - 1];
        const base = (padT + plotH).toFixed(1);
        const area = `M ${pts[0][0].toFixed(1)},${base} L ${pts.map(p => xy(...p)).join(' L ')} L ${lx.toFixed(1)},${base} Z`;
        const dots = dotR ? pts.map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${dotR}" fill="${d.color}" stroke="#070b13" stroke-width="${dotStroke}"/>`).join('') : '';
        const slots = Object.keys(fc).map(Number).sort((a, b) => a - b);
        const fcLine = slots.length
            ? `<polyline class="lc-fc-line ${d.key}" points="${[xy(lx, ly)].concat(slots.map(s => xy(xOf(n - 1 + s), yOf(fc[s].mins)))).join(' ')}"/>`
            : '';
        return `
            <path d="${area}" fill="url(#grad-${d.key})"/>
            <polyline points="${pts.map(p => xy(...p)).join(' ')}" fill="none" stroke="${d.color}" stroke-width="${lineW}" stroke-linejoin="round" stroke-linecap="round"/>
            ${dots}${fcLine}
            <circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="6" fill="${d.color}" opacity="0.35">
                <animate attributeName="r" values="6;13;6" dur="2s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.35;0;0.35" dur="2s" repeatCount="indefinite"/>
            </circle>
            <circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="5" fill="${d.color}" stroke="#070b13" stroke-width="1.5"/>`;
    }

    let grid = '';
    for (let s = 0; s <= 4; s++) {
        const v = minVal + (s / 4) * (maxVal - minVal);
        const y = yOf(v);
        grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="lc-grid"/>
                 <text x="${padL - 6}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="lc-axis">${v.toFixed(0)}</text>`;
    }

    // About six x labels, kept clear of the final label at the end of the forecast.
    const step = N <= 8 ? 1 : Math.ceil(N / 6);
    const xLabels = axis.map((lbl, i) => {
        const isEnd = i === N - 1;
        if (!isEnd && (i % step !== 0 || (hasFc && N - 1 - i < step))) return '';
        return `<text class="lc-axis${hasFc && isEnd ? ' lc-fc-xlabel' : ''}" x="${xOf(i).toFixed(1)}" y="${H - 6}" text-anchor="middle">${lbl}</text>`;
    }).join('');

    const fcZone = hasFc ? `
        <rect class="lc-fc-zone" x="${xOf(n - 1).toFixed(1)}" y="${padT}" width="${(xOf(N - 1) - xOf(n - 1)).toFixed(1)}" height="${plotH}"/>
        <text class="lc-fc-tag" x="${xOf(N - 1).toFixed(1)}" y="${padT + 11}" text-anchor="end">FORECAST</text>` : '';
    const midY = padT + plotH / 2;

    area.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="lc-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>${DIRECTIONS.map(d => `
            <linearGradient id="grad-${d.key}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="${d.color}" stop-opacity="0.25"/>
                <stop offset="100%" stop-color="${d.color}" stop-opacity="0"/>
            </linearGradient>`).join('')}
        </defs>
        ${grid}${fcZone}
        <text x="16" y="${midY}" transform="rotate(-90 16 ${midY})" text-anchor="middle" class="lc-axis-title">Travel time (min)</text>
        ${series.map(seriesSvg).join('')}
        ${xLabels}
        <line class="lc-crosshair" x1="0" y1="${padT}" x2="0" y2="${padT + plotH}" style="display:none"/>
        ${DIRECTIONS.map(d => `<circle class="lc-hover-dot" data-key="${d.key}" r="6" stroke="${d.color}" style="display:none"/>`).join('')}
        <rect class="lc-hit" x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent"/>
    </svg>`;

    // Hover / touch: crosshair, a ring on each series, and a tooltip.
    const svg = area.querySelector('svg');
    const cross = svg.querySelector('.lc-crosshair');
    const rings = Object.fromEntries([...svg.querySelectorAll('.lc-hover-dot')].map(c => [c.dataset.key, c]));
    const tip = document.createElement('div');
    tip.className = 'lc-tooltip';
    tip.hidden = true;
    area.appendChild(tip);

    function showAt(evt) {
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        const x = pt.matrixTransform(svg.getScreenCTM().inverse()).x;
        let i = Math.max(0, Math.min(N - 1, Math.round(((x - padL) / plotW) * (N - 1))));
        // Past the newest observation, snap to the nearest forecast point.
        if (hasFc && i > n - 1) i = fcIdxs.reduce((best, k) => Math.abs(k - i) < Math.abs(best - i) ? k : best, fcIdxs[0]);
        const isFc = hasFc && i > n - 1;
        const cx = xOf(i).toFixed(1);
        cross.setAttribute('x1', cx);
        cross.setAttribute('x2', cx);
        cross.style.display = '';

        const rows = [];
        series.forEach(({ d, map, fc }) => {
            const v = isFc ? fc[i - (n - 1)]?.mins : map[keys[i]];
            const ring = rings[d.key];
            if (v == null) { ring.style.display = 'none'; return; }
            ring.setAttribute('cx', cx);
            ring.setAttribute('cy', yOf(v).toFixed(1));
            ring.style.display = '';
            rows.push(`<div class="lc-tt-row"><span class="lc-dot ${d.key}"></span>${d.label}<strong>${v.toFixed(1)} min${isFc ? ' (forecast)' : ''}</strong></div>`);
        });
        if (!rows.length) { tip.hidden = true; return; }
        tip.innerHTML = `<div class="lc-tt-time">${axis[i]} SGT${isFc ? ' · forecast' : ''}</div>${rows.join('')}`;
        tip.hidden = false;

        // Place the tooltip beside the crosshair, flipping left near the right edge.
        const areaRect = area.getBoundingClientRect();
        const svgRect = svg.getBoundingClientRect();
        const scale = svgRect.width / W;
        let left = (svgRect.left - areaRect.left) + xOf(i) * scale + 14;
        if (left + tip.offsetWidth > areaRect.width) left -= tip.offsetWidth + 28;
        tip.style.left = `${Math.max(4, left)}px`;
        tip.style.top = `${(svgRect.top - areaRect.top) + padT * scale}px`;
    }
    function hideAll() {
        cross.style.display = 'none';
        Object.values(rings).forEach(r => { r.style.display = 'none'; });
        tip.hidden = true;
    }
    const hit = svg.querySelector('.lc-hit');
    hit.addEventListener('mousemove', showAt);
    hit.addEventListener('mouseleave', hideAll);
    hit.addEventListener('touchmove', e => { showAt(e.touches[0]); e.preventDefault(); }, { passive: false });
    hit.addEventListener('touchend', hideAll);

    // Header badge, forecast legend and footer.
    if (data.updated_at_sgt) $('lc-updated').textContent = `Updated ${data.updated_at_sgt.slice(11, 16)} SGT`;
    const legendFc = $('lc-legend-fc');
    legendFc.hidden = !hasFc;
    const span = Math.round(maxSlot * 5 / 30) * 30; // nearest half hour, e.g. "5 h" not "5.1 h"
    legendFc.textContent = `Forecast (next ${span < 60 ? `${span} min` : `${span / 60} h`})`;

    series.forEach(({ d, pts, fc }) => {
        const el = $(`lc-stat-${d.key}`);
        const latest = pts.at(-1);
        const next = fc[Math.min(...Object.keys(fc).map(Number))];
        el.innerHTML = latest
            ? `<span class="lc-dot ${d.key}"></span><strong>${d.label}</strong>&nbsp;Latest: <strong>${fmtMin(latest.mins)} min</strong> at ${latest.label}`
              + (next ? ` &rarr; <strong>${next.mins.toFixed(1)}</strong> forecast at ${sgtClock(next.at)}` : '')
            : '';
    });
    $('lc-footer').hidden = false;
}

// ── Forecast vs typical day (forecast view, Causeway) ───────────────────────
// `?curve=forecast` gives a forecast every 30 min from one origin; `?baseline=profile`
// is the calendar-profile "typical day", not a forecast. Only the curve is shown: no
// single 30-min point in its place.
const FORECAST_ERRORS = {
    400: 'Forecast request was rejected.',
    502: 'Forecast query failed on the server.',
    503: 'Forecast paused: no recent data for this direction.'
};

// The curve becomes { curve, dirs: { sg|jb: { originAt, points } } }.
function normalizeCurve(data) {
    const dirs = {};
    DIRECTIONS.forEach(d => {
        const src = data.directions?.[d.apiKey];
        dirs[d.key] = {
            originAt: Date.parse(src?.origin_ts),
            points: (src?.points || [])
                .filter(p => p.model !== 'profile' && p.forecast_min != null)
                .map(p => ({ at: Date.parse(p.forecast_for), mins: p.forecast_min, base: p.baseline_min, model: p.model, lead: p.lead_min }))
                .filter(p => Number.isFinite(p.at))
        };
    });
    return { curve: true, dirs };
}

async function loadBaselineDay() {
    if (baselineDay && Date.now() - baselineAt < BASELINE_REFRESH_MS) return;
    try {
        const data = await fetchJson(`${FORECAST_API}?baseline=profile&hours=24`, 30000, FORECAST_ERRORS);
        baselineDay = Object.fromEntries(DIRECTIONS.map(d => [d.key,
            (data.directions?.[d.apiKey] || [])
                .map(b => ({ t: Date.parse(b.bin_start), mins: b.forecast_min }))
                .filter(b => Number.isFinite(b.t) && b.mins != null)]));
        baselineAt = Date.now();
    } catch (err) {
        console.error('Typical-day fetch error:', err); // the chart falls back to the curve's own baseline_min
    }
}

// Small chart: the forecast curve against the typical day, from the forecast origin.
function forecastCompare(d, view, spanH) {
    const pts = view.points;
    const t0 = Number.isFinite(view.originAt) ? view.originAt : pts[0].at - 1800000;
    const t1 = t0 + spanH * 3600000;
    const bins = baselineDay?.[d.key]?.filter(b => b.t >= t0 && b.t <= t1) || [];
    const base = bins.length > 1 ? bins : pts.filter(p => p.base != null).map(p => ({ t: p.at, mins: p.base }));
    const fcPts = pts.filter(p => p.at <= t1).map(p => ({ t: p.at, mins: p.mins }));

    const vals = fcPts.concat(base).map(p => p.mins);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = Math.max(1, (hi - lo) * 0.12);
    const W = 420, H = 150, padL = 30, padR = 58, padT = 8, padB = 22;
    const x = t => padL + ((t - t0) / (t1 - t0)) * (W - padL - padR);
    const y = v => padT + (H - padT - padB) - ((v - (lo - pad)) / ((hi - lo + 2 * pad) || 1)) * (H - padT - padB);
    const poly = arr => arr.map(p => `${x(p.t).toFixed(1)},${y(p.mins).toFixed(1)}`).join(' ');

    const tickEvery = (spanH <= 6 ? 1 : 6) * 3600000;
    let ticks = '';
    for (let t = Math.ceil(t0 / tickEvery) * tickEvery; t <= t1; t += tickEvery) {
        ticks += `<line class="fc-tick" x1="${x(t).toFixed(1)}" y1="${padT}" x2="${x(t).toFixed(1)}" y2="${H - padB}"/>
                  <text class="fc-axis" x="${x(t).toFixed(1)}" y="${H - 6}" text-anchor="middle">${sgtClock(t)}</text>`;
    }
    const yLabel = v => `<text class="fc-axis" x="${padL - 4}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`;
    const last = fcPts[fcPts.length - 1];
    const dots = fcPts.slice(0, -1).map(p => `<circle cx="${x(p.t).toFixed(1)}" cy="${y(p.mins).toFixed(1)}" r="2.4" fill="${d.color}"/>`).join('');

    // Label each line at its right-hand end, pushed apart if the ends are close.
    const baseEnd = base.length > 1 ? base[base.length - 1] : null;
    let fy = y(last.mins), by = baseEnd ? y(baseEnd.mins) : null;
    if (by != null && Math.abs(fy - by) < 14) {
        const gap = (14 - Math.abs(fy - by)) / 2;
        if (fy <= by) { fy -= gap; by += gap; } else { fy += gap; by -= gap; }
    }
    const labels = `<text class="fc-line-label" x="${(x(last.t) + 8).toFixed(1)}" y="${(fy + 3).toFixed(1)}" fill="${d.color}">Forecast</text>`
        + (baseEnd ? `<text class="fc-line-label fc-line-label-base" x="${(x(baseEnd.t) + 8).toFixed(1)}" y="${(by + 3).toFixed(1)}">Typical</text>` : '');

    return `<svg viewBox="0 0 ${W} ${H}" class="fc-compare" role="img" aria-label="${d.label} forecast against the typical day">
        ${ticks}${yLabel(lo)}${yLabel(hi)}
        ${baseEnd ? `<polyline class="fc-base-line" points="${poly(base)}"/>` : ''}
        <polyline class="fc-fc-line" stroke="${d.color}" points="${poly(fcPts)}"/>
        ${dots}<circle cx="${x(last.t).toFixed(1)}" cy="${y(last.mins).toFixed(1)}" r="4" fill="${d.color}" stroke="#070b13" stroke-width="1.5"/>
        ${labels}
    </svg>`;
}

function renderForecastCards() {
    if (!forecastView) return;
    const traffic = trafficCache.data;
    const spanH = baselineDay ? COMPARE_HOURS : FORECAST_HOURS; // no typical day => the curve's own 5 h

    $('fc-cards').innerHTML = DIRECTIONS.map(d => {
        const view = forecastView.dirs[d.key];
        const f = view.points[0]; // nearest forecast point
        const head = `<div class="fc-card-head"><span class="lc-dot ${d.key}"></span>${d.label}</div>`;
        if (!f) {
            return `<div class="fc-card">${head}<div class="fc-value">—</div><div class="fc-when">no forecast available</div></div>`;
        }
        const now = traffic ? seriesPoints(traffic, d.series).at(-1)?.mins : null;
        const nowLine = now == null ? ''
            : `<div class="fc-now">Now <strong>${fmtMin(now)}</strong> min (${f.mins - now >= 0 ? '+' : ''}${(f.mins - now).toFixed(1)})</div>`;
        const lead = f.lead != null ? ` · ${Math.round(f.lead)} min ahead` : '';
        return `
            <div class="fc-card">
                ${head}
                <div class="fc-main">
                    <span class="fc-value lv-text ${congestionLevel(f.mins).cls}">${f.mins.toFixed(1)}</span>
                    <span class="fc-when">min at ${sgtClock(f.at)} SGT${lead}</span>
                </div>
                ${nowLine}
                ${forecastCompare(d, view, spanH)}
            </div>`;
    }).join('');

    const origin = DIRECTIONS.map(d => forecastView.dirs[d.key].originAt).filter(Number.isFinite).sort().pop();
    $('fc-updated').textContent = origin ? `Based on ${sgtClock(origin)} SGT data` : 'Updated';
    $('fc-meta').textContent = 'Model — ' + DIRECTIONS.map(d =>
        `${d.label}: ${[...new Set(forecastView.dirs[d.key].points.map(p => p.model))].join(', ') || '—'}`).join(' · ');
}

function setForecastStatus(text) {
    const el = $('fc-status');
    el.textContent = text;
    el.hidden = !text;
}

let forecastSeq = 0; // a newer fetchForecast (refresh button, timer) supersedes an older one
let pendingTimers = [];

const PENDING_IDS = ['lc-pending', 'fc-pending']; // one notice on each forecast chart

function mmss(ms) {
    const sec = Math.ceil(ms / 1000);
    return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
}

// "Model is retraining" notice with a two-minute countdown, on both charts. It appears only
// once the request has run PENDING_NOTICE_DELAY_MS, so a warm answer does not flash it.
function startPendingNotice() {
    stopPendingNotice();
    const started = Date.now();
    const tick = () => {
        const left = RETRAIN_COUNTDOWN_MS - (Date.now() - started);
        const html = left > 0
            ? '<span class="fc-spinner" aria-hidden="true"></span><div><strong>Model is retraining on the latest data. Processing, please wait…</strong>'
              + `<br>This may take up to 2 minutes. If it does not load, hit Refresh Forecast.</div><span class="fc-countdown" aria-label="time left">${mmss(left)}</span>`
            : '<span class="fc-spinner" aria-hidden="true"></span><div><strong>Still processing, taking longer than usual…</strong>'
              + '<br>If it does not load soon, hit Refresh Forecast.</div>';
        PENDING_IDS.forEach(id => { const el = $(id); if (el) { el.innerHTML = html; el.hidden = false; } });
        setForecastStatus('');
    };
    pendingTimers.push(setTimeout(() => {
        tick();
        pendingTimers.push(setInterval(tick, 1000));
    }, PENDING_NOTICE_DELAY_MS));
}

function stopPendingNotice() {
    pendingTimers.forEach(t => { clearTimeout(t); clearInterval(t); });
    pendingTimers = [];
    PENDING_IDS.forEach(id => { const el = $(id); if (el) el.hidden = true; });
}

async function fetchForecast() {
    const seq = ++forecastSeq;
    const superseded = () => seq !== forecastSeq;
    if (!forecastView) setForecastStatus('Fetching forecast…');
    startPendingNotice();

    let view = null, failure = null;
    try {
        view = normalizeCurve(await fetchJson(`${FORECAST_API}?curve=forecast&hours=${FORECAST_HOURS}`, CURVE_TIMEOUT_MS, FORECAST_ERRORS));
    } catch (err) {
        failure = err;
    }
    if (superseded()) return;
    stopPendingNotice();

    if (view && DIRECTIONS.some(d => view.dirs[d.key].points.length)) {
        forecastView = view;
        await loadBaselineDay();
        renderTrendChart(); // redraw with the dashed forecast extension
        try { await loadTrafficData(); } catch (_) { /* "Now" on the cards is optional */ }
        setForecastStatus('');
        renderForecastCards();
        fetchAdvice(); // after the curve, so forecast-api is warm when the advisor reads it
        return;
    }

    const err = failure || new Error(FORECAST_ERRORS[503]);
    const msg = err.name === 'AbortError' ? 'Forecast service timed out.'
        : err instanceof TypeError ? 'Forecast service unreachable.' : err.message;
    console.error('Forecast fetch error:', err);
    const hadForecast = forecastView != null;
    forecastView = null;
    if (hadForecast) renderTrendChart(); // drop the stale dashed extension
    $('fc-cards').innerHTML = '';
    $('fc-updated').textContent = 'Unavailable';
    $('fc-meta').textContent = '';
    setForecastStatus(`Forecast unavailable: ${msg} Hit Refresh Forecast to try again.`);
}

// ── Commute buddy (forecast-api ?advice=commute, forecast view) ───────────
// forecast-api reads the same cached 5 h curve, works out the best time to leave in code, and
// asks an LLM for a one-line insight and tips. Times and minutes on the card come from `facts`; only the sentences
// come from the LLM. If the service is missing or fails, the card stays hidden and nothing else changes.
const ADVISOR_TIMEOUT_MS = 90000;
let adviceSeq = 0;

function esc(text) {
    return String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function shortDate(iso) {
    const [y, m, d] = iso.split('-').map(Number);
    return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString('en-SG', { day: 'numeric', month: 'short', timeZone: 'UTC' });
}

// Calendar chips, from the service's facts (the holiday CSV), never from the LLM text.
function adviceChips(cal) { // short calendar chips for the Commute Buddy card
    const chips = [];
    const t = cal?.today;
    if (!t) return chips;
    if (t.sg_public_holiday.length) chips.push(`SG public holiday: ${t.sg_public_holiday.join(', ')}`);
    if (t.my_public_holiday.length) chips.push(`MY public holiday: ${t.my_public_holiday.join(', ')}`);
    if (t.eve_of_sg_public_holiday) chips.push('Eve of a SG public holiday');
    if (t.eve_of_my_public_holiday) chips.push('Eve of a MY public holiday');
    if (t.last_day_of_sg_long_weekend) chips.push('Last day of a SG long weekend');
    else if (t.sg_long_weekend) chips.push('SG long weekend');
    [['sg', 'SG'], ['my', 'MY']].forEach(([k, label]) => {
        const v = cal[`${k}_school_holidays`] || '';
        let m;
        if ((m = /^on now, \S+ to (\S+)/.exec(v))) chips.push(`${label} school holidays until ${shortDate(m[1])}`);
        else if ((m = /^start (\S+) \(in (\d+) days\)/.exec(v))) chips.push(`${label} school holidays start ${shortDate(m[1])} (in ${m[2]} days)`);
    });
    (cal.upcoming_public_holidays || []).slice(0, 2).forEach(h =>
        chips.push(`${h.country === 'Singapore' ? 'SG' : 'MY'} ${h.names.join(', ') || 'public holiday'} on ${shortDate(h.date)}`));
    return chips;
}

// ── Commute buddy clock ──
// A 12-hour dial in Singapore time. The ring runs from now to the last forecast slot, each sector
// coloured by the forecast trip time; the floating cyan arc is the best window to leave. The hands
// tick every second in the browser; the ring is redrawn each minute so passed slots drop off.
const CLOCK_C = 120, CLOCK_R_IN = 80, CLOCK_R_OUT = 102, CLOCK_R_ARC = 112, CLOCK_R_NUM = 64;
const HEAT_STOPS = [[25, '#4dd9a3'], [38, '#f5c66a'], [50, '#f2608f']]; // clear -> busy -> jammed (minutes)
const SGT_OFFSET_MIN = 480;
let adviceData = null;
let adviceTab = 'SG_TO_MY';
let clockTimer = null;
let clockMinute = null;

const sgtParts = ms => { const d = new Date(ms + SGT_OFFSET_MIN * 60000); return { h: d.getUTCHours(), m: d.getUTCMinutes(), s: d.getUTCSeconds() }; };
function clock12(ms) {
    const { h, m } = sgtParts(ms);
    return { text: `${(h + 11) % 12 + 1}:${String(m).padStart(2, '0')}`, ampm: h < 12 ? 'AM' : 'PM' };
}
const fmt12 = ms => { const c = clock12(ms); return `${c.text} ${c.ampm}`; };
function fmtRange(a, b) {
    const x = clock12(a), y = clock12(b);
    return x.ampm === y.ampm ? `${x.text}–${y.text} ${y.ampm}` : `${x.text} ${x.ampm}–${y.text} ${y.ampm}`;
}

function heatColour(v) {
    const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    if (v <= HEAT_STOPS[0][0]) return HEAT_STOPS[0][1];
    for (let i = 1; i < HEAT_STOPS.length; i++) if (v <= HEAT_STOPS[i][0]) {
        const [a, ca] = HEAT_STOPS[i - 1], [b, cb] = HEAT_STOPS[i], t = (v - a) / (b - a), A = hex(ca), B = hex(cb);
        return `rgb(${A.map((c, k) => Math.round(c + (B[k] - c) * t)).join(',')})`;
    }
    return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

function clockSvg(f, key, nowMs) {
    const C = CLOCK_C;
    const nowDial = (() => { const { h, m, s } = sgtParts(nowMs); return (h % 12) * 60 + m + s / 60; })();
    const dial = ms => nowDial + (ms - nowMs) / 60000; // dial minutes, continuous past 12
    const ang = min => (min / 720) * 2 * Math.PI - Math.PI / 2;
    const pt = (r, a) => `${(C + r * Math.cos(a)).toFixed(2)},${(C + r * Math.sin(a)).toFixed(2)}`;
    const wedge = (m0, m1, r0, r1) => {
        const big = m1 - m0 > 360 ? 1 : 0;
        return `M${pt(r1, ang(m0))} A${r1},${r1} 0 ${big} 1 ${pt(r1, ang(m1))} L${pt(r0, ang(m1))} A${r0},${r0} 0 ${big} 0 ${pt(r0, ang(m0))} Z`;
    };
    const arc = (m0, m1, r) => `M${pt(r, ang(m0))} A${r},${r} 0 ${m1 - m0 > 360 ? 1 : 0} 1 ${pt(r, ang(m1))}`;

    // leave-home times (the service subtracts the drive to the Causeway); a slot owns +/-15 min
    const samples = f.forecast.map(p => ({ t: Date.parse(p.at_iso), v: p.crossing_min })).filter(p => p.t + 15 * 60000 > nowMs);
    let svg = `<svg viewBox="0 0 240 240" class="cb-clock" aria-hidden="true">
        <defs><filter id="cb-glow-${key}" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="1.6" result="b"/>
        <feComponentTransfer in="b" result="g"><feFuncA type="linear" slope="0.6"/></feComponentTransfer>
        <feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        <radialGradient id="cb-face-${key}"><stop offset="0" stop-color="#1b2740"/><stop offset="1" stop-color="#111a2b"/></radialGradient></defs>
        <circle cx="${C}" cy="${C}" r="${CLOCK_R_OUT + 16}" fill="url(#cb-face-${key})" stroke="rgba(0,242,254,0.18)"/>
        <path d="${wedge(0, 719.9, CLOCK_R_IN, CLOCK_R_OUT)}" fill="rgba(255,255,255,0.06)"/><g filter="url(#cb-glow-${key})">`;
    samples.forEach((p, i) => {
        const m0 = Math.max(dial(nowMs), i === 0 ? dial(p.t) - 15 : dial((samples[i - 1].t + p.t) / 2));
        const m1 = i === samples.length - 1 ? dial(p.t) + 15 : dial((p.t + samples[i + 1].t) / 2);
        if (m1 - m0 > 1.6) svg += `<path d="${wedge(m0 + 0.8, m1 - 0.8, CLOCK_R_IN, CLOCK_R_OUT)}" fill="${heatColour(p.v)}" opacity="0.9"/>`;
    });
    svg += '</g>';
    for (let h = 1; h <= 12; h++) {
        const a = ang(h * 60), major = h % 3 === 0;
        const x = C + CLOCK_R_NUM * Math.cos(a), y = C + CLOCK_R_NUM * Math.sin(a);
        svg += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle" dominant-baseline="central" class="cb-num${major ? ' major' : ''}">${h}</text>`;
    }
    for (let k = 0; k < 60; k++) if (k % 5) {
        const a = (k / 60) * 2 * Math.PI - Math.PI / 2;
        svg += `<line x1="${pt(CLOCK_R_IN - 4, a).replace(',', '" y1="')}" x2="${pt(CLOCK_R_IN - 6.5, a).replace(',', '" y2="')}" class="cb-tick"/>`;
    }
    leaveArcs(f, nowMs).forEach(([s0, s1, fromNow]) => {
        const a0 = fromNow ? dial(nowMs) : dial(s0) - 6, a1 = Math.max(dial(s1) + 6, a0 + 12);
        svg += `<path d="${arc(a0, a1, CLOCK_R_ARC)}" class="cb-arc-halo"/><path d="${arc(a0, a1, CLOCK_R_ARC)}" class="cb-arc"/>`;
    });
    const n = ang(dial(nowMs));
    svg += `<line x1="${pt(CLOCK_R_IN - 2, n).replace(',', '" y1="')}" x2="${pt(CLOCK_R_OUT + 2, n).replace(',', '" y2="')}" class="cb-now"/>`;
    svg += `<g class="cb-hand-h"><line x1="${C}" y1="${C + 8}" x2="${C}" y2="${C - 44}"/></g>
        <g class="cb-hand-m"><line x1="${C}" y1="${C + 10}" x2="${C}" y2="${C - 68}"/></g>
        <g class="cb-hand-s"><line x1="${C}" y1="${C + 14}" x2="${C}" y2="${C - 72}"/></g>
        <circle cx="${C}" cy="${C}" r="5.5" fill="#ffffff"/><circle cx="${C}" cy="${C}" r="2.4" fill="#00f2fe"/></svg>`;
    return svg;
}

// forecast-api commute_advisor.decide: the earliest leave-home span of at least an hour whose
// slots are all in the fastest quarter of the next 5 h (else the quickest hour). A response
// without `decision` (an older forecast-api) falls back to the plain best window.
function leaveArcs(f, nowMs) { // [[startMs, endMs, startsNow]]
    const d = f.decision;
    if (d) return [[Math.max(Date.parse(d.window_start_iso), nowMs), Date.parse(d.window_end_iso), Date.parse(d.window_start_iso) <= nowMs]];
    const b = f.best;
    return [[b.leave_now ? nowMs : Date.parse(b.window_start_iso), Date.parse(b.window_end_iso), b.leave_now]];
}

function leaveText(f, nowMs) {
    const d = f.decision;
    if (d) {
        const ws = Date.parse(d.window_start_iso), we = Date.parse(d.window_end_iso);
        return ws <= nowMs ? `Now – ${fmt12(we)}` : fmtRange(ws, we); // under "Best time to leave"
    }
    const b = f.best, ws = Date.parse(b.window_start_iso), we = Date.parse(b.window_end_iso);
    if (b.leave_now) return we - ws < 60000 || we <= nowMs ? 'Leave now' : `Leave now, by ${fmt12(we)}`;
    return we - ws < 60000 ? `Leave at ${fmt12(ws)}` : `Leave ${fmtRange(ws, we)}`;
}

function vsChip(vs) {
    if (vs == null) return '';
    if (Math.abs(vs) < 2) return '<span class="cb-vs ok">● About usual</span>'; // same 2-min band as the insight
    const n = Math.round(Math.abs(vs));
    return vs > 0 ? `<span class="cb-vs up">▲ ${n} min slower than usual</span>` : `<span class="cb-vs ok">▼ ${n} min faster than usual</span>`;
}

function renderClocks() {
    if (!adviceData) return;
    const nowMs = Date.now();
    const dirs = adviceData.facts?.directions || {};
    const nowText = fmt12(nowMs);
    $('cb-tiles').innerHTML = DIRECTIONS.map(d => {
        const f = dirs[d.apiKey];
        if (!f) return '';
        return `<div class="cb-tile${d.apiKey === adviceTab ? ' active' : ''}" data-dir="${d.apiKey}">
            <div class="cb-dir"><span class="lc-dot ${d.key}"></span>${d.label}</div>
            <div class="cb-body">
                <div class="cb-clock-wrap">${clockSvg(f, d.key, nowMs)}<div class="cb-clock-cap">Now <b>${nowText}</b></div></div>
                <div class="cb-side">
                    <div class="cb-label">Best time to leave</div>
                    <div class="cb-verb">${esc(leaveText(f, nowMs))}</div>
                    <div class="cb-chips"><span class="cb-trip"><b>${Math.round((f.decision || f.best).crossing_min)}</b> min expected</span>${vsChip(f.vs_typical_avg_min)}</div>
                </div>
            </div>
        </div>`;
    }).join('');
    tickClocks();
}

function tickClocks() {
    const nowMs = Date.now();
    const { h, m, s } = sgtParts(nowMs);
    if (clockMinute !== null && clockMinute !== m) { clockMinute = m; renderClocks(); return; } // ring and caption move on
    clockMinute = m;
    const rot = deg => `rotate(${deg.toFixed(2)} ${CLOCK_C} ${CLOCK_C})`;
    document.querySelectorAll('#advice-card .cb-hand-h').forEach(g => g.setAttribute('transform', rot(((h % 12) + m / 60) * 30)));
    document.querySelectorAll('#advice-card .cb-hand-m').forEach(g => g.setAttribute('transform', rot((m + s / 60) * 6)));
    document.querySelectorAll('#advice-card .cb-hand-s').forEach(g => g.setAttribute('transform', rot(s * 6)));
}

function setAdviceTab(dir) {
    adviceTab = dir;
    document.querySelectorAll('#cb-tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.dir === dir)));
    document.querySelectorAll('#cb-tiles .cb-tile').forEach(t => t.classList.toggle('active', t.dataset.dir === dir));
}

function renderAdvice(data) {
    adviceData = data;
    const a = data.advice;
    $('cb-insight-text').textContent = a?.insight || ''; // one line: which direction differs from usual, and the likely why
    $('cb-headline').hidden = !a?.insight;
    renderClocks();
    const chips = adviceChips(data.facts?.calendar).map(c => `<span class="cb-cal">${esc(c)}</span>`).join('');
    $('cb-cal').innerHTML = chips;
    $('cb-tips').innerHTML = (a?.tips || []).map(t => `<li>${esc(t)}</li>`).join('');
    $('cb-tips').hidden = !a?.tips?.length;
    $('cb-content').hidden = false;
    $('advice-card').hidden = false;
    if (!clockTimer) clockTimer = setInterval(tickClocks, 1000);
}

function setAdviceStatus(text, isError) {
    $('advice-status').textContent = text;
    $('cb-status-row').hidden = !text;
    $('cb-status-row').classList.toggle('error', !!isError);
    $('cb-retry').hidden = !isError;
}

// The card shows once the forecast has loaded: "asking" while it waits, the clocks when the
// advice arrives, or a short "couldn't reach" message with Retry. The forecast above never depends on it.
async function fetchAdvice() {
    const seq = ++adviceSeq;
    const card = $('advice-card');
    if (!card) return;
    card.hidden = false;
    if (!adviceData) setAdviceStatus('Asking your Commute Buddy…', false);
    $('cb-retry').disabled = true;
    try {
        const data = await fetchJson(ADVISOR_API, ADVISOR_TIMEOUT_MS);
        if (seq !== adviceSeq) return;
        if (!Object.keys(data.facts?.directions || {}).length) throw new Error('no advice in the response');
        setAdviceStatus('', false);
        renderAdvice(data);
    } catch (err) {
        if (seq !== adviceSeq) return;
        console.error('Commute advice error:', err);
        setAdviceStatus(adviceData
            ? 'Couldn\u2019t refresh your Commute Buddy; showing the last advice. Retry, or refresh the page later.'
            : 'Couldn\u2019t reach your Commute Buddy right now. Retry, or refresh the page later.', true);
    } finally {
        if (seq === adviceSeq) $('cb-retry').disabled = false;
    }
}

// ── LTA cameras (forecast view) ─────────────────────────────────────────────
async function fetchLiveCameras() {
    const grid = $('live-cam-grid');
    const status = $('cam-last-fetched');
    const ids = CHECKPOINTS[currentCheckpoint].cameras;

    grid.innerHTML = '<div class="cam-state"><div class="cam-spinner"></div><p>Fetching live feeds from LTA…</p></div>';
    status.textContent = 'Fetching…';
    try {
        const res = await fetch(LTA_API_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        renderCameraGrid(json.items[0].cameras.filter(c => ids.includes(c.camera_id)));
        status.textContent = `Last updated: ${sgtTime(Date.now())} SGT`;
    } catch (err) {
        console.error('Camera fetch error:', err);
        grid.innerHTML = `<div class="cam-state"><span class="cam-error-icon">⚠️</span><p>Could not load camera feeds. ${err.message}</p></div>`;
        status.textContent = 'Feed unavailable';
    }
}

function renderCameraGrid(cameras) {
    const grid = $('live-cam-grid');
    grid.innerHTML = '';
    cameras.forEach(cam => {
        const tile = document.createElement('div');
        tile.className = 'cam-tile';
        tile.setAttribute('role', 'button');
        tile.setAttribute('tabindex', '0');
        tile.setAttribute('aria-label', `Camera ${cam.camera_id}, open full size`);
        tile.innerHTML = `
            <div class="cam-tile-overlay">
                <span class="cam-tile-id">CAM ${cam.camera_id}</span>
                <span class="cam-tile-rec">● LIVE</span>
            </div>
            <img class="cam-tile-img" src="${cam.image}" alt="CAM ${cam.camera_id} traffic feed">
            <div class="cam-tile-footer"><span class="cam-tile-time">${sgtTime(Date.parse(cam.timestamp))} SGT</span></div>`;
        tile.addEventListener('click', () => openCamModal(cam));
        tile.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') openCamModal(cam); });
        grid.appendChild(tile);
    });
}

// Full-size lightbox. Closes on the button, a click outside, or Escape.
function openCamModal(cam) {
    $('cam-modal')?.remove();
    const label = `CAM ${cam.camera_id}`;
    const modal = document.createElement('div');
    modal.id = 'cam-modal';
    modal.className = 'cam-modal-backdrop';
    modal.innerHTML = `
        <div class="cam-modal-box" role="dialog" aria-modal="true" aria-label="${label} camera feed">
            <div class="cam-modal-header">
                <span class="cam-modal-title">📷 ${label}</span>
                <button class="cam-modal-close" aria-label="Close camera feed">&times;</button>
            </div>
            <img class="cam-modal-img" src="${cam.image}" alt="${label} full resolution">
            <div class="cam-modal-meta">
                <span>Camera ID: ${cam.camera_id}</span>
                <span>Captured at: ${sgtTime(Date.parse(cam.timestamp))} SGT</span>
                <span>Resolution: ${cam.image_metadata?.width || '—'} &times; ${cam.image_metadata?.height || '—'}</span>
            </div>
        </div>`;

    const onKey = e => { if (e.key === 'Escape') close(); };
    function close() {
        document.removeEventListener('keydown', onKey);
        modal.classList.remove('visible');
        setTimeout(() => modal.remove(), 300);
    }
    modal.querySelector('.cam-modal-close').addEventListener('click', close);
    modal.addEventListener('click', e => { if (e.target === modal) close(); });
    document.addEventListener('keydown', onKey);

    document.body.appendChild(modal);
    requestAnimationFrame(() => modal.classList.add('visible'));
}

// ── AI vehicle detection (summary view) ─────────────────────────────────────
// swiftbackend returns the annotated JPEG as the body and the per-direction
// counts as response headers, so one request gives both. Frames are scored by
// the in-container YOLO26s (`model=local`, unbilled; the first call after a cold
// start is a little slower). `dirs` lists the directions a camera tracks: on 2702,
// traffic to Singapore is already past the checkpoint, so only To Johor is counted.
const AI_MODEL = 'local';
const AI_PANES = [
    { camera: '2701', dirs: ['sg', 'jb'] },
    { camera: '2702', dirs: ['sg'] }
];

// Queue-depth bands from camdetect (CONGESTION_BANDS): a count in one frame, not flow.
const AI_CONGESTION_CLASS = {
    'Free Flow': 'normal',
    'Quarter Way': 'normal',
    'Half Way': 'warning',
    'Back to Back': 'danger'
};

function readHeaderInt(headers, name) {
    const n = parseInt(headers.get(name), 10);
    return Number.isFinite(n) ? n : null;
}

// Missing headers (older service, or not exposed over CORS) read as "unavailable",
// never as zero.
function readDetection(headers) {
    const dirs = DIRECTIONS.map(d => ({
        ...d,
        count: readHeaderInt(headers, `X-Vehicle-Count-${d.aiKey}`),
        congestion: headers.get(`X-Congestion-${d.aiKey}`)
    }));
    const total = readHeaderInt(headers, 'X-Vehicle-Count');
    const unknown = readHeaderInt(headers, 'X-Vehicle-Count-Unknown');
    const attributed = dirs.reduce((sum, d) => sum + (d.count || 0), 0);
    const supported = dirs.some(d => d.count != null);
    return {
        dirs, total, unknown, supported,
        // Vehicles found but none attributed: the camera has no dividing line configured.
        uncalibrated: supported && total > 0 && attributed === 0 && unknown === total,
        frameTime: headers.get('X-Frame-Datetime')
    };
}

function ensureDetectionPanes() {
    const wrap = $('ai-compare');
    if (wrap.dataset.ready) return;
    wrap.innerHTML = AI_PANES.map(p => `
        <div class="ai-pane">
            <div class="ai-pane-head">
                <span class="ai-pane-title">CAM ${p.camera} <span class="ai-pane-note" id="ai-time-${p.camera}"></span></span>
                <span class="ai-pane-count" id="ai-count-${p.camera}"></span>
            </div>
            <div class="ai-detection-body">
                <p class="ai-detection-status" id="ai-status-${p.camera}">Fetching AI detection result…</p>
                <img id="ai-img-${p.camera}" class="ai-detection-img" alt="CAM ${p.camera} vehicle detections"
                     role="button" tabindex="0" aria-label="CAM ${p.camera} vehicle detections, open zoom view" hidden>
                <span class="ai-zoom-hint" aria-hidden="true">🔍 Click to zoom</span>
            </div>
            <div class="ai-dir-grid" id="ai-dirs-${p.camera}"></div>
        </div>`).join('');
    AI_PANES.forEach(p => {
        const img = $(`ai-img-${p.camera}`);
        const open = () => {
            if (img.hidden || !img.src) return;
            const time = $(`ai-time-${p.camera}`).textContent;
            openZoomViewer(img.src, `🤖 CAM ${p.camera} · AI detection${time ? ` · ${time}` : ''}`, img.alt);
        };
        img.addEventListener('click', open);
        img.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    });
    wrap.dataset.ready = '1';
}

// ── Zoom viewer (AI detection frames) ───────────────────────────────────────
// Full-window lightbox. The frame opens fitted to the window, enlarged past its
// own size when the window is bigger. Wheel or pinch zooms about the pointer, drag
// pans, double-click toggles fit / 100 % (one image pixel per screen pixel).
// Zoom stops at 4x the fitted size or 4x real pixels, whichever is larger.
// Keys: + / - zoom, 0 fit, 1 real size, Esc close.
const ZOOM_MAX = 4;
const ZOOM_STEP = 1.5;

function openZoomViewer(src, title, alt) {
    $('zoom-modal')?.remove();
    const modal = document.createElement('div');
    modal.id = 'zoom-modal';
    modal.className = 'cam-modal-backdrop zoom-backdrop';
    modal.innerHTML = `
        <div class="zoom-box" role="dialog" aria-modal="true" aria-label="${alt}, zoom view">
            <div class="cam-modal-header">
                <span class="cam-modal-title">${title}</span>
                <button class="cam-modal-close" aria-label="Close zoom view">&times;</button>
            </div>
            <div class="zoom-stage">
                <img class="zoom-img" src="${src}" alt="${alt}" draggable="false">
            </div>
            <div class="zoom-bar">
                <div class="zoom-controls">
                    <button type="button" class="zoom-btn" data-zoom="out" aria-label="Zoom out">&minus;</button>
                    <span class="zoom-level" aria-live="polite">—</span>
                    <button type="button" class="zoom-btn" data-zoom="in" aria-label="Zoom in">+</button>
                    <button type="button" class="zoom-btn zoom-btn-text" data-zoom="fit">Fit</button>
                    <button type="button" class="zoom-btn zoom-btn-text" data-zoom="real">100%</button>
                </div>
                <span class="zoom-help">Scroll or pinch to zoom · drag to pan · double-click for 100%</span>
                <span class="zoom-res"></span>
            </div>
        </div>`;

    const stage = modal.querySelector('.zoom-stage');
    const img = modal.querySelector('.zoom-img');
    const levelEl = modal.querySelector('.zoom-level');
    // scale = screen px per image px; (tx, ty) = image top-left in stage px.
    const v = { scale: 1, fit: 1, tx: 0, ty: 0, atFit: true };
    const pointers = new Map();
    let pinch = null;

    const maxScale = () => Math.max(v.fit * ZOOM_MAX, ZOOM_MAX);

    function clampPan() {
        const w = img.naturalWidth * v.scale, h = img.naturalHeight * v.scale;
        const sw = stage.clientWidth, sh = stage.clientHeight;
        v.tx = w <= sw ? (sw - w) / 2 : Math.min(0, Math.max(sw - w, v.tx));
        v.ty = h <= sh ? (sh - h) / 2 : Math.min(0, Math.max(sh - h, v.ty));
    }

    function apply() {
        clampPan();
        img.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`;
        levelEl.textContent = `${Math.round(v.scale * 100)}%`;
        stage.classList.toggle('pannable',
            img.naturalWidth * v.scale > stage.clientWidth + 1 || img.naturalHeight * v.scale > stage.clientHeight + 1);
    }

    function fit() {
        v.fit = Math.min(stage.clientWidth / img.naturalWidth, stage.clientHeight / img.naturalHeight) || 1;
        v.scale = v.fit;
        v.atFit = true;
        apply();
    }

    // Zoom to `scale`, keeping the image point under stage point (px, py) fixed.
    function zoomTo(scale, px = stage.clientWidth / 2, py = stage.clientHeight / 2) {
        const next = Math.min(maxScale(), Math.max(Math.min(v.fit, 1), scale));
        v.tx = px - (px - v.tx) * (next / v.scale);
        v.ty = py - (py - v.ty) * (next / v.scale);
        v.scale = next;
        v.atFit = Math.abs(next - v.fit) < 1e-3;
        apply();
    }

    const stagePoint = e => {
        const r = stage.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
    };

    stage.addEventListener('wheel', e => {
        e.preventDefault();
        zoomTo(v.scale * Math.exp(-e.deltaY * 0.0015), ...stagePoint(e));
    }, { passive: false });

    stage.addEventListener('dblclick', e => {
        // At fit, jump to real pixels; if the window already shows the frame at
        // or above real size, jump to 2.5x fit instead so the click still zooms.
        if (v.atFit) zoomTo(v.fit < 1 ? 1 : v.fit * 2.5, ...stagePoint(e));
        else fit();
    });

    stage.addEventListener('pointerdown', e => {
        stage.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId, stagePoint(e));
        if (pointers.size === 2) {
            const [a, b] = [...pointers.values()];
            pinch = { dist: Math.hypot(a[0] - b[0], a[1] - b[1]), scale: v.scale };
        }
        stage.classList.add('dragging');
    });

    stage.addEventListener('pointermove', e => {
        if (!pointers.has(e.pointerId)) return;
        const prev = pointers.get(e.pointerId);
        const cur = stagePoint(e);
        pointers.set(e.pointerId, cur);
        if (pointers.size === 2 && pinch) {
            const [a, b] = [...pointers.values()];
            const dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
            zoomTo(pinch.scale * dist / pinch.dist, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
        } else if (pointers.size === 1) {
            v.tx += cur[0] - prev[0];
            v.ty += cur[1] - prev[1];
            apply();
        }
    });

    const release = e => {
        pointers.delete(e.pointerId);
        if (pointers.size < 2) pinch = null;
        if (!pointers.size) stage.classList.remove('dragging');
    };
    stage.addEventListener('pointerup', release);
    stage.addEventListener('pointercancel', release);

    modal.querySelector('.zoom-controls').addEventListener('click', e => {
        const action = e.target.closest('[data-zoom]')?.dataset.zoom;
        if (action === 'in') zoomTo(v.scale * ZOOM_STEP);
        else if (action === 'out') zoomTo(v.scale / ZOOM_STEP);
        else if (action === 'fit') fit();
        else if (action === 'real') zoomTo(1);
    });

    const onResize = () => {
        const wasFit = v.atFit;
        v.fit = Math.min(stage.clientWidth / img.naturalWidth, stage.clientHeight / img.naturalHeight) || 1;
        if (wasFit) fit(); else zoomTo(v.scale);
    };
    const onKey = e => {
        if (e.key === 'Escape') close();
        else if (e.key === '+' || e.key === '=') zoomTo(v.scale * ZOOM_STEP);
        else if (e.key === '-') zoomTo(v.scale / ZOOM_STEP);
        else if (e.key === '0') fit();
        else if (e.key === '1') zoomTo(1);
    };
    function close() {
        document.removeEventListener('keydown', onKey);
        window.removeEventListener('resize', onResize);
        document.body.classList.remove('modal-open');
        modal.classList.remove('visible');
        setTimeout(() => modal.remove(), 300);
    }
    modal.querySelector('.cam-modal-close').addEventListener('click', close);
    modal.addEventListener('click', e => { if (e.target === modal) close(); });
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', onResize);

    const ready = () => {
        modal.querySelector('.zoom-res').textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
        fit();
        img.classList.add('ready');
    };
    document.body.appendChild(modal);
    document.body.classList.add('modal-open');
    if (img.complete && img.naturalWidth) ready(); else img.addEventListener('load', ready, { once: true });
    requestAnimationFrame(() => modal.classList.add('visible'));
    modal.querySelector('.cam-modal-close').focus();
}

function renderDetection(box, det, pane) {
    if (!det.supported) { box.innerHTML = '<p class="ai-dir-note">Direction split unavailable from this service.</p>'; return; }
    if (det.uncalibrated) { box.innerHTML = '<p class="ai-dir-note">No dividing line calibrated for this camera — vehicles counted, not split by direction.</p>'; return; }

    const cards = det.dirs.filter(d => pane.dirs.includes(d.key)).map(d => {
        const level = d.congestion || 'No Data';
        return `
            <div class="ai-dir-card">
                <span class="ai-dir-label"><span class="lc-dot ${d.key}"></span>${d.label}</span>
                <span class="ai-dir-count">${d.count ?? '—'}</span>
                <span class="ai-dir-unit">vehicles in frame</span>
                <span class="status-badge ${AI_CONGESTION_CLASS[level] || 'warning'}">${level}</span>
            </div>`;
    }).join('');
    // Unattributed vehicles are real detections outside the counting zone.
    const unknownNote = det.unknown
        ? `<p class="ai-dir-note">${det.unknown} of ${det.total} not attributed to a direction (outside the counting zone).</p>` : '';
    const untrackedNote = pane.dirs.includes('jb') ? ''
        : '<p class="ai-dir-note">Traffic to Singapore on this camera is past the checkpoint, so it is not tracked.</p>';
    box.innerHTML = cards + unknownNote + untrackedNote;
}

// swiftbackend answers errors as JSON {"error": "..."}. Turn the status into a
// sentence that says which side failed: LTA's camera feed, or our model.
const LTA_FEED_ERRORS = {
    'Traffic camera API request failed': 'its traffic camera service did not respond',
    'Failed to download source image': 'the camera image could not be downloaded',
    'Source frame is not a valid image': 'the camera image was damaged or empty'
};

async function detectionErrorMessage(res, pane, dateTime) {
    let detail = '';
    try { detail = (await res.json()).error || ''; } catch { /* not JSON */ }
    const time = dateTime.slice(11, 16);
    if (res.status === 404) {
        return `No camera image from LTA: the data.gov.sg traffic camera feed has no frame for CAM ${pane.camera} at ${time} SGT. ` +
            'This happens when LTA\'s feed is down or missing cameras. Try Refresh Detection in a few minutes.';
    }
    if (res.status === 502 && LTA_FEED_ERRORS[detail]) {
        return `Could not get the CAM ${pane.camera} image from LTA (data.gov.sg): ${LTA_FEED_ERRORS[detail]}. Try again in a few minutes.`;
    }
    if (res.status === 502 || res.status === 503) {
        return `The AI model could not process the CAM ${pane.camera} frame (${detail || `HTTP ${res.status}`}). Try again shortly.`;
    }
    return `AI detection unavailable: ${detail || `HTTP ${res.status}`}`;
}

async function fetchPaneDetection(pane, dateTime) {
    const img = $(`ai-img-${pane.camera}`);
    const status = $(`ai-status-${pane.camera}`);
    const dirsBox = $(`ai-dirs-${pane.camera}`);
    status.textContent = 'Fetching AI detection result…';
    img.hidden = true;
    dirsBox.innerHTML = '';
    $(`ai-count-${pane.camera}`).textContent = '';
    $(`ai-time-${pane.camera}`).textContent = '';

    try {
        // format=directional colours the boxes by direction; counts ride on the headers.
        // line=1 draws the dividing line that splits the two directions (off by default).
        const url = new URL(BACKEND_URL);
        url.searchParams.set('format', 'directional');
        url.searchParams.set('line', '1');
        url.searchParams.set('camera_id', pane.camera);
        url.searchParams.set('model', AI_MODEL);
        url.searchParams.set('date_time', dateTime);
        let res;
        try {
            res = await fetch(url);
        } catch {
            throw new Error('Could not reach the AI detection service. Check your connection and try again.');
        }
        if (!res.ok) throw new Error(await detectionErrorMessage(res, pane, dateTime));

        const det = readDetection(res.headers);
        const objectUrl = URL.createObjectURL(await res.blob());
        const previous = img.dataset.objectUrl;
        img.onload = () => {
            img.hidden = false;
            status.textContent = '';
            if (previous) URL.revokeObjectURL(previous); // release the previous frame
        };
        img.onerror = () => { status.textContent = 'Could not render detection image.'; };
        img.dataset.objectUrl = objectUrl;
        img.src = objectUrl;

        renderDetection(dirsBox, det, pane);
        // A camera that tracks one direction counts only that direction, matching the boxes drawn.
        const shown = pane.dirs.length < DIRECTIONS.length && det.supported && !det.uncalibrated
            ? det.dirs.filter(d => pane.dirs.includes(d.key)).reduce((sum, d) => sum + (d.count || 0), 0)
            : det.total;
        if (shown != null) $(`ai-count-${pane.camera}`).textContent = `${shown} vehicles`;
        $(`ai-time-${pane.camera}`).textContent = `${(det.frameTime || dateTime).slice(11, 19)} SGT`;
    } catch (err) {
        status.textContent = err.message;
        console.error(`Detection error (CAM ${pane.camera}):`, err);
        throw err;
    }
}

async function fetchDetections() {
    ensureDetectionPanes();
    const dateTime = sgtNowParam(); // one timestamp, so both cameras show the same moment
    const results = await Promise.allSettled(AI_PANES.map(p => fetchPaneDetection(p, dateTime)));
    $('ai-detection-timestamp').textContent = results.some(r => r.status === 'fulfilled')
        ? `Local model (YOLO26s) · requested ${dateTime.slice(11, 19)} SGT` : '';
}

// ── Views and routing ───────────────────────────────────────────────────────
function parseRoute() {
    // "#live" was this view's earlier name; old links still land here.
    const m = /^#(?:forecast|live)(?:\/(woodlands|tuas))?$/.exec(location.hash);
    return m ? { view: 'forecast', checkpoint: m[1] || 'woodlands' } : { view: 'home', checkpoint: 'woodlands' };
}

function applyCheckpoint(cp) {
    currentCheckpoint = cp;
    document.querySelectorAll('.seg-btn').forEach(b => {
        const on = b.dataset.checkpoint === cp;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', String(on));
    });
    // Panels for one checkpoint only: trend + forecast (Causeway), notice (Tuas).
    document.querySelectorAll('#view-forecast [data-only]').forEach(el => { el.hidden = el.dataset.only !== cp; });
    $('cam-title').textContent = `LTA Cameras — ${CHECKPOINTS[cp].name}`;
}

function route() {
    const { view, checkpoint } = parseRoute();
    const viewChanged = view !== currentView;
    const cpChanged = checkpoint !== currentCheckpoint;
    currentView = view;

    document.querySelectorAll('.view').forEach(el => { el.hidden = el.dataset.view !== view; });
    document.querySelectorAll('[data-view-link]').forEach(a => a.classList.toggle('active', a.dataset.viewLink === view));
    if (viewChanged) window.scrollTo(0, 0);

    // Cameras load and refresh only while the forecast view is open, to save mobile data.
    if (view !== 'forecast') {
        clearInterval(cameraTimer);
        cameraTimer = null;
        return;
    }
    applyCheckpoint(checkpoint);
    if (viewChanged || cpChanged) fetchLiveCameras();
    if (!cameraTimer) cameraTimer = setInterval(fetchLiveCameras, CAMERA_REFRESH_MS);
}

// ── Start ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateSummary();
    renderTrendChart();
    fetchForecast();
    fetchDetections();

    window.addEventListener('hashchange', route);
    route();

    setInterval(() => {
        updateSummary();
        renderTrendChart();
        fetchForecast();
    }, REFRESH_MS);

    // Causeway | Tuas switcher. The hash carries the choice, so Back works.
    document.querySelectorAll('.seg-btn').forEach(btn => btn.addEventListener('click', () => {
        location.hash = btn.dataset.checkpoint === 'woodlands' ? '#forecast' : `#forecast/${btn.dataset.checkpoint}`;
    }));

    bindRefresh('refresh-dashboard', async () => {
        $('last-updated').textContent = 'Updating…';
        try { await loadTrafficData(true); } catch (_) { /* updateSummary reports the failure */ }
        await Promise.all([updateSummary(), renderTrendChart()]);
        renderForecastCards(); // "Now" on the forecast cards follows the fresh sample
    });
    bindRefresh('refresh-forecast', fetchForecast);
    document.querySelectorAll('#cb-tabs button').forEach(b => b.addEventListener('click', () => setAdviceTab(b.dataset.dir)));
    $('cb-retry').addEventListener('click', fetchAdvice);
    bindRefresh('refresh-cameras', fetchLiveCameras);
    bindRefresh('refresh-ai-detection', fetchDetections);
});
