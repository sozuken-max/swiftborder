// SwiftBorder front end: Woodlands Causeway travel times, forecast and live cameras.
//
// Two views on one page, picked by the URL hash:
//   "#"           summary: one card per direction + forecast vs typical day
//   "#live"       Causeway: trend chart, LTA cameras, AI vehicle detection
//   "#live/tuas"  Tuas: LTA cameras only (no travel-time feed for Tuas)
//
// Data sources (all public, read from the browser):
//   traffic-24h.json  Google Maps Distance Matrix travel times, written every 5 min
//   forecast-api      forecast curve and typical-day profile (docs/runbooks/forecast-api.md)
//   data.gov.sg       LTA traffic camera images
//   swiftbackend      annotated camera frame + per-direction vehicle counts

// ── Endpoints ──────────────────────────────────────────────────────────────
const TRAFFIC_API = 'https://storage.googleapis.com/swiftborder-public/traffic-24h.json';
const FORECAST_API = 'https://forecast-api-1095552466513.asia-southeast1.run.app/';
const LTA_API_URL = 'https://api.data.gov.sg/v1/transport/traffic-images';
const BACKEND_URL = 'https://swiftbackend-1095552466513.europe-west1.run.app/';

const SGT = 'Asia/Singapore';

// ── Directions ─────────────────────────────────────────────────────────────
// `series` is the key in traffic-24h.json, `apiKey` the key in forecast-api,
// `aiKey` the direction suffix on swiftbackend's count headers. `color` must
// match --dir-sg / --dir-jb in style.css: blue and violet, kept clear of the
// green / amber / red congestion colours so a hue means one thing.
const DIRECTIONS = [
    { key: 'sg', label: 'To Johor', series: 'mandai_to_shell_jb', apiKey: 'SG_TO_MY', aiKey: 'SG-MY', color: '#5b9dff' },
    { key: 'jb', label: 'To Singapore', series: 'jb_to_woodlands', apiKey: 'MY_TO_SG', aiKey: 'MY-SG', color: '#c58aff' }
];

// ── Checkpoints and cameras (data.gov.sg camera IDs) ───────────────────────
const CHECKPOINTS = {
    woodlands: { name: 'Woodlands Causeway', cameras: ['2701', '2702', '2703', '2704', '2705', '2706'] },
    tuas: { name: 'Tuas Second Link', cameras: ['4703', '4712', '4713', '4714', '4716'] }
};

// ── Tuning ─────────────────────────────────────────────────────────────────
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
const FORECAST_TIMEOUT_MS = 60000;       // the first curve call can take ~20 s while models fit
const BASELINE_REFRESH_MS = 3600000;     // the typical-day profile is refitted once per day

const REFRESH_MS = 300000;               // travel times and forecast
const CAMERA_REFRESH_MS = 60000;         // camera images, live view only
const SLOT_MS = 300000;                  // one 5-min sample

// ── State ──────────────────────────────────────────────────────────────────
let currentView = null;          // 'home' | 'live'; null until the first route()
let currentCheckpoint = 'woodlands';
let cameraTimer = null;
let detectionLoaded = false;     // AI detection runs on the first Causeway live visit
let trafficCache = { data: null, fetchedAt: 0, inFlight: null };
let forecastView = null;         // { curve, dirs: { sg|jb: { originAt, points: [{ at, mins, base, model, lead }] } } }
let baselineDay = null;          // { sg|jb: [{ t, mins }] } typical day, 24 h
let baselineAt = 0;

const CHEVRON = '<svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>';

// ── Small helpers ──────────────────────────────────────────────────────────
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

// ── Traffic feed (Google Maps travel times) ────────────────────────────────
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

// ── Summary cards ──────────────────────────────────────────────────────────
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
    card.setAttribute('aria-label', `${d.label}: ${said}. Open live cameras and trend.`);
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

// ── Trend chart (live view) ────────────────────────────────────────────────
// Today's samples for both directions, plus the forecast drawn as a dashed
// extension in extra 5-min slots after the newest observation.
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

    const today = new Date().toLocaleDateString('en-CA', { timeZone: SGT });
    const series = DIRECTIONS.map(d => {
        const pts = (data.series?.[d.series]?.points || [])
            .filter(p => p[0].startsWith(today) && p[1] != null)
            .map(p => ({ label: p[0].slice(11, 16), mins: p[1] / 60 }));
        return { d, pts, map: Object.fromEntries(pts.map(p => [p.label, p.mins])), fc: {} };
    });
    if (series.every(s => !s.pts.length)) { showMessage('No data available for today yet.'); return; }

    // x axis: one slot per observed label, then one per forecast slot.
    const labels = [...new Set(series.flatMap(s => s.pts.map(p => p.label)))].sort();
    const n = labels.length;
    const lastEpoch = parseSgt(`${today}T${labels[n - 1]}`);
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

    // Dots shrink as the day fills up and disappear past ~120 points.
    const dotR = n > 120 ? 0 : n > 60 ? 1.8 : n > 30 ? 2.6 : 3.5;
    const dotStroke = dotR >= 2.6 ? 1.5 : 1;
    const lineW = n > 120 ? 1.8 : 2.5;

    function seriesSvg({ d, map, fc }) {
        const pts = labels.map((l, i) => map[l] != null ? [xOf(i), yOf(map[l])] : null).filter(Boolean);
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
            const v = isFc ? fc[i - (n - 1)]?.mins : map[axis[i]];
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

// ── Forecast vs typical day (summary view) ─────────────────────────────────
// `?curve=forecast` gives a forecast every 30 min from one origin; `?baseline=profile`
// is the calendar-profile "typical day", not a forecast. If the curve fails, the
// plain endpoint's single 30-min forecast is shown instead.
const FORECAST_ERRORS = {
    400: 'Forecast request was rejected.',
    502: 'Forecast query failed on the server.',
    503: 'Forecast paused: no recent data for this direction.'
};

// Both response shapes become { curve, dirs: { sg|jb: { originAt, points } } }.
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

function normalizeServed(data) {
    const dirs = {};
    DIRECTIONS.forEach(d => {
        const f = data.directions?.[d.apiKey];
        const at = Date.parse(f?.forecast_for);
        dirs[d.key] = {
            originAt: Date.parse(f?.origin_ts),
            points: f?.forecast_min != null && Number.isFinite(at)
                ? [{ at, mins: f.forecast_min, base: null, model: f.model, lead: f.lead_min }] : []
        };
    });
    return { curve: false, dirs };
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
        const head = `<div class="fc-card-head"><span class="lc-dot ${d.key}"></span>${d.label}<span class="summary-go" aria-hidden="true">${CHEVRON}</span></div>`;
        if (!f) {
            return `<a class="fc-card" href="#live">${head}<div class="fc-value">—</div><div class="fc-when">no forecast available</div></a>`;
        }
        const now = traffic ? seriesPoints(traffic, d.series).at(-1)?.mins : null;
        const nowLine = now == null ? ''
            : `<div class="fc-now">Now <strong>${fmtMin(now)}</strong> min (${f.mins - now >= 0 ? '+' : ''}${(f.mins - now).toFixed(1)})</div>`;
        const lead = f.lead != null ? ` · ${Math.round(f.lead)} min ahead` : '';
        return `
            <a class="fc-card" href="#live" aria-label="${d.label}: open live cameras and trend">
                ${head}
                <div class="fc-main">
                    <span class="fc-value lv-text ${congestionLevel(f.mins).cls}">${f.mins.toFixed(1)}</span>
                    <span class="fc-when">min at ${sgtClock(f.at)} SGT${lead}</span>
                </div>
                ${nowLine}
                ${forecastCompare(d, view, spanH)}
            </a>`;
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

async function fetchForecast() {
    setForecastStatus('Fetching forecast…');

    let view = null, failure = null;
    try {
        view = normalizeCurve(await fetchJson(`${FORECAST_API}?curve=forecast&hours=${FORECAST_HOURS}`, FORECAST_TIMEOUT_MS, FORECAST_ERRORS));
    } catch (err) {
        failure = err; // the curve fits several models and can be slow; try the cheap 30-min forecast
        try {
            view = normalizeServed(await fetchJson(FORECAST_API, FORECAST_TIMEOUT_MS, FORECAST_ERRORS));
        } catch (err2) {
            failure = err2;
        }
    }

    if (!view || !DIRECTIONS.some(d => view.dirs[d.key].points.length)) {
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
        setForecastStatus(`Forecast unavailable: ${msg}`);
        return;
    }

    forecastView = view;
    if (view.curve) await loadBaselineDay();
    renderTrendChart(); // redraw with the dashed forecast extension
    try { await loadTrafficData(); } catch (_) { /* "Now" on the cards is optional */ }
    setForecastStatus(view.curve ? '' : 'Showing the 30-minute forecast only; the longer forecast did not load.');
    renderForecastCards();
}

// ── LTA cameras (live view) ────────────────────────────────────────────────
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

// ── AI vehicle detection (live view, Causeway) ─────────────────────────────
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
                <img id="ai-img-${p.camera}" class="ai-detection-img" alt="CAM ${p.camera} vehicle detections" hidden>
            </div>
            <div class="ai-dir-grid" id="ai-dirs-${p.camera}"></div>
        </div>`).join('');
    wrap.dataset.ready = '1';
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
        const url = new URL(BACKEND_URL);
        url.searchParams.set('format', 'directional');
        url.searchParams.set('camera_id', pane.camera);
        url.searchParams.set('model', AI_MODEL);
        url.searchParams.set('date_time', dateTime);
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

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
        status.textContent = `AI detection unavailable: ${err.message}`;
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

// ── Views and routing ──────────────────────────────────────────────────────
function parseRoute() {
    const m = /^#live(?:\/(woodlands|tuas))?$/.exec(location.hash);
    return m ? { view: 'live', checkpoint: m[1] || 'woodlands' } : { view: 'home', checkpoint: 'woodlands' };
}

function applyCheckpoint(cp) {
    currentCheckpoint = cp;
    document.querySelectorAll('.seg-btn').forEach(b => {
        const on = b.dataset.checkpoint === cp;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', String(on));
    });
    // Panels for one checkpoint only: trend + AI detection (Causeway), notice (Tuas).
    document.querySelectorAll('#view-live [data-only]').forEach(el => { el.hidden = el.dataset.only !== cp; });
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

    // Cameras load and refresh only while the live view is open, to save mobile data.
    if (view !== 'live') {
        clearInterval(cameraTimer);
        cameraTimer = null;
        return;
    }
    applyCheckpoint(checkpoint);
    if (viewChanged || cpChanged) fetchLiveCameras();
    if (!cameraTimer) cameraTimer = setInterval(fetchLiveCameras, CAMERA_REFRESH_MS);
    if (checkpoint === 'woodlands' && !detectionLoaded) {
        detectionLoaded = true;
        fetchDetections();
    }
}

// ── Start ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateSummary();
    renderTrendChart();
    fetchForecast();

    window.addEventListener('hashchange', route);
    route();

    setInterval(() => {
        updateSummary();
        renderTrendChart();
        fetchForecast();
    }, REFRESH_MS);

    // Causeway | Tuas switcher. The hash carries the choice, so Back works.
    document.querySelectorAll('.seg-btn').forEach(btn => btn.addEventListener('click', () => {
        location.hash = btn.dataset.checkpoint === 'woodlands' ? '#live' : `#live/${btn.dataset.checkpoint}`;
    }));

    bindRefresh('refresh-dashboard', async () => {
        $('last-updated').textContent = 'Updating…';
        try { await loadTrafficData(true); } catch (_) { /* updateSummary reports the failure */ }
        await Promise.all([updateSummary(), renderTrendChart()]);
        renderForecastCards(); // "Now" on the forecast cards follows the fresh sample
    });
    bindRefresh('refresh-forecast', fetchForecast);
    bindRefresh('refresh-cameras', fetchLiveCameras);
    bindRefresh('refresh-ai-detection', fetchDetections);
});
