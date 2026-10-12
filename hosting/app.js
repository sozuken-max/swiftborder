// ── Camera IDs per checkpoint (from data.gov.sg LTA API) ─────────────────
const CHECKPOINT_CAMERAS = {
    woodlands: ['2701', '2702', '2703', '2704', '2705', '2706'],
    tuas: ['4703', '4712', '4713', '4714', '4716']
};

// ── Checkpoint Metadata ────────────────────────────────────────────────────
// Live travel-time series only exist for Woodlands. Tuas has cameras only,
// so the summary is Woodlands-only and Tuas appears only in the live view.
const checkpointMeta = {
    woodlands: { name: "Woodlands Causeway" },
    tuas: { name: "Tuas Second Link" }
};

// Direction config — keys map to the series names in traffic-24h.json.
// Direction colours for charts; must match --dir-sg / --dir-jb in style.css. Blue and
// violet stay clear of the green / amber / red congestion colours.
const DIR_COLOR = { sg: '#5b9dff', jb: '#c58aff' };

const DIRECTIONS = [
    { key: 'sg', label: 'To Johor', series: 'mandai_to_shell_jb', cardId: 'row-sg', timeId: 'sg-time', trendId: 'sg-trend' },
    { key: 'jb', label: 'To Singapore', series: 'jb_to_woodlands', cardId: 'row-jb', timeId: 'jb-time', trendId: 'jb-trend' }
];

// ── Trend / status tuning ──────────────────────────────────────────────────
const TREND_WINDOW_MIN = 60;   // no sample in this trailing window => the card shows no data
const STALE_AFTER_MIN = 20;    // newest sample older than this => label the card stale
// Congestion colour: fixed minute cut-offs, the same for both directions, applied
// to the number shown (whole minutes): under 30 green, 30-50 amber, over 50 red.
const LEVEL_WARN_MIN = 30;
const LEVEL_DANGER_MIN = 50;
// Summary-card trend is macro: the last 30 min against the 60 min before it,
// so one noisy 5-min sample cannot flip the arrow.
const MACRO_RECENT_MIN = 30;
const MACRO_PRIOR_MIN = 60;
const MACRO_MIN_POINTS = 3;     // per half; fewer => steady
const MACRO_STABLE_PCT = 10;

// ── Global State ───────────────────────────────────────────────────────────
let currentCheckpoint = 'woodlands';
let currentView = 'home';      // 'home' (summary) or 'live' (cameras + trend)
let cameraRefreshTimer = null;
let detectionLoaded = false;   // AI detection runs on the first visit to the Causeway live view
let chartRefreshTimer = null;
let trafficCache = { data: null, fetchedAt: 0 }; // shared by the transit card and the chart
// Forecast drawn as a dashed extension of the chart: { SG_TO_MY: [{ at, mins, status }], MY_TO_SG: [...] }.
// Null until forecast-api answers; the chart draws from traffic-24h.json alone in the meantime.
let chartForecast = null;

const LTA_API_URL = 'https://api.data.gov.sg/v1/transport/traffic-images';
const BACKEND_URL = 'https://swiftbackend-1095552466513.europe-west1.run.app/';
const TRAFFIC_API = 'https://storage.googleapis.com/swiftborder-public/traffic-24h.json';
// Cloud Run service endpoint (docs/runbooks/forecast-api.md). Public, CORS open.
const FORECAST_API = 'https://forecast-api-1095552466513.asia-southeast1.run.app/';
// Model cards (not fetched on this page): FORECAST_API + '?list=cards'  (same body as '?view=cards')

// ── Backend AI Detection Image ─────────────────────────────────────────────
// The detection service exposes a direction split for every frame it scores.
// One request carries both: the annotated JPEG as the body, and the counts as
// response headers — so reading the split costs no extra inference call.
// Only cameras with a calibrated dividing line return a usable split (2701 and 2702).

// Service labels travel direction SG-MY / MY-SG; the UI says SG -> JB / JB -> SG.
const AI_DIRECTIONS = [
    { key: 'sg', label: 'To Johor', cls: 'sg', countHeader: 'X-Vehicle-Count-SG-MY', congestionHeader: 'X-Congestion-SG-MY' },
    { key: 'jb', label: 'To Singapore', cls: 'jb', countHeader: 'X-Vehicle-Count-MY-SG', congestionHeader: 'X-Congestion-MY-SG' }
];

// Congestion here is queue depth — banded from a direction's vehicle count in
// the frame (camdetect CONGESTION_BANDS) — not throughput. A single frame cannot measure flow.
const AI_CONGESTION_CLASS = {
    'Free Flow': 'normal',
    'Quarter Way': 'normal',
    'Half Way': 'warning',
    'Back to Back': 'danger'
};

function readHeaderInt(headers, name) {
    const raw = headers.get(name);
    if (raw == null) return null;
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : null;
}

// Headers are absent when the service predates the directional change, or when
// the browser cannot see them (no Access-Control-Expose-Headers). Both cases
// degrade to "unavailable" rather than rendering zeros as if they were real.
function readDetectionSummary(headers) {
    const dirs = AI_DIRECTIONS.map(d => ({
        ...d,
        count: readHeaderInt(headers, d.countHeader),
        congestion: headers.get(d.congestionHeader)
    }));

    const supported = dirs.some(d => d.count != null);
    const total = readHeaderInt(headers, 'X-Vehicle-Count');
    const unknown = readHeaderInt(headers, 'X-Vehicle-Count-Unknown');
    const attributed = dirs.reduce((sum, d) => sum + (d.count || 0), 0);

    // Every vehicle unattributed, with at least one detected, means this camera
    // has no dividing line configured — not that both carriageways are empty.
    const uncalibrated = supported && total > 0 && attributed === 0 && unknown === total;

    return { supported, uncalibrated, dirs, total, unknown, attributed, frameTime: headers.get('X-Frame-Datetime') };
}

// One pane per camera, in display order, each scored by the in-container YOLO26s (`model=local`,
// unbilled; the first call after a cold start is ~1-2 s slower). The Roboflow workflows (`v4`/`v6`)
// stay on the backend but the page no longer calls them. `dirs` lists the AI_DIRECTIONS keys a
// pane shows: on 2702, JB -> SG is past the last checkpoint (free traffic into Singapore), so only
// SG -> JB is tracked there (camdetect `tracked`).
const AI_DETECTION_MODEL = 'local';
const AI_PANES = [
    { key: '2701', camera: '2701', label: 'CAM 2701', dirs: ['sg', 'jb'] },
    { key: '2702', camera: '2702', label: 'CAM 2702', dirs: ['sg'] }
];

function ensureDetectionPanes() {
    const wrap = document.getElementById('ai-compare');
    if (!wrap || wrap.dataset.ready) return wrap;
    wrap.innerHTML = AI_PANES.map(p => `
        <div class="ai-pane" data-camera="${p.camera}">
            <div class="ai-pane-head">
                <span class="ai-pane-title">${p.label} <span class="ai-pane-note" id="ai-time-${p.key}"></span></span>
                <span class="ai-pane-count" id="ai-count-${p.key}"></span>
            </div>
            <div class="ai-detection-body">
                <p class="ai-detection-status" id="ai-status-${p.key}">Fetching AI detection result…</p>
                <img id="ai-img-${p.key}" class="ai-detection-img" alt="${p.label} vehicle detections" style="display:none;" />
            </div>
            <div class="ai-dir-grid" id="ai-dirs-${p.key}"></div>
        </div>`).join('');
    wrap.dataset.ready = '1';
    return wrap;
}

function renderDetectionSummary(box, summary, pane) {
    if (!box) return;

    if (!summary.supported) {
        box.innerHTML = '<p class="ai-dir-note">Direction split unavailable from this service.</p>';
        return;
    }
    if (summary.uncalibrated) {
        box.innerHTML = '<p class="ai-dir-note">No dividing line calibrated for this camera — vehicles counted, not split by direction.</p>';
        return;
    }

    const cards = summary.dirs.filter(d => pane.dirs.includes(d.key)).map(d => {
        const count = d.count == null ? '—' : d.count;
        const level = d.congestion || 'No Data';
        const cls = AI_CONGESTION_CLASS[level] || 'warning';
        return `
            <div class="ai-dir-card ${d.cls}">
                <span class="ai-dir-label"><span class="lc-dot ${d.cls}"></span>${d.label}</span>
                <span class="ai-dir-count">${count}</span>
                <span class="ai-dir-unit">vehicles in frame</span>
                <span class="status-indicator-badge ${cls}">${level}</span>
            </div>`;
    }).join('');

    // Unattributed vehicles are real detections outside the dividing line's
    // span. Showing them keeps the two counts honest against the total.
    const unknownNote = summary.unknown
        ? `<p class="ai-dir-note">${summary.unknown} of ${summary.total} not attributed to a direction (outside the counting zone).</p>`
        : '';

    const untrackedNote = pane.dirs.includes('jb')
        ? ''
        : '<p class="ai-dir-note">Traffic to Singapore on this camera is past the checkpoint (free traffic into Singapore), so it is not tracked.</p>';

    box.innerHTML = cards + unknownNote + untrackedNote;
}

// Singapore wall-clock "YYYY-MM-DDTHH:MM:SS", the backend's date_time format.
function sgtNowParam() {
    return new Date().toLocaleString('sv-SE', { timeZone: 'Asia/Singapore', hour12: false }).replace(' ', 'T');
}

async function fetchPaneDetection(pane, dateTime) {
    const imgEl = document.getElementById(`ai-img-${pane.key}`);
    const statusEl = document.getElementById(`ai-status-${pane.key}`);
    const dirsEl = document.getElementById(`ai-dirs-${pane.key}`);
    const countEl = document.getElementById(`ai-count-${pane.key}`);
    const timeEl = document.getElementById(`ai-time-${pane.key}`);

    if (statusEl) statusEl.textContent = 'Fetching AI detection result…';
    if (imgEl) imgEl.style.display = 'none';
    if (dirsEl) dirsEl.innerHTML = '';
    if (countEl) countEl.textContent = '';
    if (timeEl) timeEl.textContent = '';

    try {
        // format=directional draws the dividing line and colours the boxes by
        // direction. The counts ride on the headers either way.
        const url = new URL(BACKEND_URL);
        url.searchParams.set('format', 'directional');
        url.searchParams.set('camera_id', pane.camera);
        url.searchParams.set('model', AI_DETECTION_MODEL);
        url.searchParams.set('date_time', dateTime);

        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const summary = readDetectionSummary(res.headers);
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);

        // Release the previous frame's blob before swapping in the new one.
        const previous = imgEl.dataset.objectUrl;
        imgEl.onload = () => {
            imgEl.style.display = 'block';
            if (statusEl) statusEl.textContent = '';
            if (previous) URL.revokeObjectURL(previous);
        };
        imgEl.onerror = () => { if (statusEl) statusEl.textContent = 'Could not render detection image.'; };
        imgEl.dataset.objectUrl = objectUrl;
        imgEl.src = objectUrl;

        renderDetectionSummary(dirsEl, summary, pane);
        // A pane that tracks one direction counts only that direction, matching the boxes drawn.
        const shown = pane.dirs.length < AI_DIRECTIONS.length && summary.supported && !summary.uncalibrated
            ? summary.dirs.filter(d => pane.dirs.includes(d.key)).reduce((sum, d) => sum + (d.count || 0), 0)
            : summary.total;
        if (countEl && shown != null) countEl.textContent = `${shown} vehicles`;
        // Prefer the frame's own capture time over the client clock.
        if (timeEl) timeEl.textContent = `${(summary.frameTime || dateTime).slice(11, 19)} SGT`;
        return summary;
    } catch (err) {
        if (statusEl) statusEl.textContent = `AI detection unavailable: ${err.message}`;
        console.error(`Backend image fetch error (CAM ${pane.camera}):`, err);
        throw err;
    }
}

async function fetchBackendMessage() {
    const bar = document.getElementById('backend-message');
    if (bar) bar.style.display = 'none';
    if (!ensureDetectionPanes()) return;

    const timestampEl = document.getElementById('ai-detection-timestamp');
    const dateTime = sgtNowParam(); // one timestamp, so both cameras show the same moment
    const results = await Promise.allSettled(AI_PANES.map(p => fetchPaneDetection(p, dateTime)));

    if (timestampEl) {
        const ok = results.some(r => r.status === 'fulfilled');
        timestampEl.textContent = ok ? `Local model (YOLO26s) · requested ${dateTime.slice(11, 19)} SGT` : '';
    }
}

// ── Live Camera Fetching ───────────────────────────────────────────────────
async function fetchLiveCameras() {
    const grid = document.getElementById('live-cam-grid');
    const lastFetched = document.getElementById('cam-last-fetched');
    const allowedIds = CHECKPOINT_CAMERAS[currentCheckpoint];

    if (grid) {
        grid.innerHTML = `
            <div class="cam-loading-state">
                <div class="cam-spinner"></div>
                <p>Fetching live feeds from LTA...</p>
            </div>`;
    }
    if (lastFetched) lastFetched.textContent = 'Fetching...';

    try {
        const res = await fetch(LTA_API_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const allCameras = json.items[0].cameras;
        const cameras = allCameras.filter(c => allowedIds.includes(c.camera_id));
        renderCameraGrid(cameras);
        const timeStr = new Date().toLocaleTimeString('en-GB', { hour12: false, timeZone: 'Asia/Singapore' });
        if (lastFetched) lastFetched.textContent = `Last updated: ${timeStr} SGT`;
    } catch (err) {
        console.error('Camera fetch error:', err);
        if (grid) {
            grid.innerHTML = `
                <div class="cam-error-state">
                    <span class="cam-error-icon">⚠️</span>
                    <p>Could not load camera feeds. ${err.message}</p>
                </div>`;
        }
        if (lastFetched) lastFetched.textContent = 'Feed unavailable';
    }
}

function renderCameraGrid(cameras) {
    const grid = document.getElementById('live-cam-grid');
    if (!grid) return;
    grid.innerHTML = '';

    cameras.forEach(cam => {
        const timeStr = new Date(cam.timestamp).toLocaleTimeString('en-GB', { hour12: false, timeZone: 'Asia/Singapore' });
        const tile = document.createElement('div');
        tile.className = 'cam-tile';
        tile.setAttribute('role', 'button');
        tile.setAttribute('tabindex', '0');
        tile.setAttribute('aria-label', `Camera ${cam.camera_id}`);
        tile.innerHTML = `
            <div class="cam-tile-overlay">
                <span class="cam-tile-id">CAM ${cam.camera_id}</span>
                <span class="cam-tile-rec">● LIVE</span>
            </div>
            <img
                class="cam-tile-img"
                src="${cam.image}"
                alt="CAM ${cam.camera_id} traffic feed"
                loading="eager"
            />
            <div class="cam-tile-footer">
                <span class="cam-tile-label">CAM ${cam.camera_id}</span>
                <span class="cam-tile-time">${timeStr} SGT</span>
            </div>
            <div class="cam-debug-url">
                🔗 <a href="${cam.image}" target="_blank" rel="noopener">${cam.image}</a>
            </div>
        `;
        tile.addEventListener('click', () => openCamModal(cam, `CAM ${cam.camera_id}`));
        tile.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') openCamModal(cam, `CAM ${cam.camera_id}`); });
        grid.appendChild(tile);
    });
}

// ── Camera Lightbox ────────────────────────────────────────────────────────
function openCamModal(cam, label) {
    const existing = document.getElementById('cam-modal');
    if (existing) existing.remove();

    const camTime = new Date(cam.timestamp).toLocaleTimeString('en-GB', { hour12: false, timeZone: 'Asia/Singapore' });
    const modal = document.createElement('div');
    modal.id = 'cam-modal';
    modal.className = 'cam-modal-backdrop';
    modal.innerHTML = `
        <div class="cam-modal-box" role="dialog" aria-modal="true" aria-label="${label} camera feed">
            <div class="cam-modal-header">
                <span class="cam-modal-title">📷 ${label}</span>
                <button class="cam-modal-close" id="cam-modal-close" aria-label="Close camera feed">&times;</button>
            </div>
            <img class="cam-modal-img" src="${cam.image}" alt="${label} full resolution" />
            <div class="cam-modal-meta">
                <span>Camera ID: ${cam.camera_id}</span>
                <span>Captured at: ${camTime} SGT</span>
                <span>Resolution: ${cam.image_metadata?.width || '—'} &times; ${cam.image_metadata?.height || '—'}</span>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    requestAnimationFrame(() => modal.classList.add('visible'));

    modal.querySelector('#cam-modal-close').addEventListener('click', () => {
        modal.classList.remove('visible');
        setTimeout(() => modal.remove(), 300);
    });
    modal.addEventListener('click', e => {
        if (e.target === modal) { modal.classList.remove('visible'); setTimeout(() => modal.remove(), 300); }
    });
    document.addEventListener('keydown', function escHandler(e) {
        if (e.key === 'Escape') {
            modal.classList.remove('visible');
            setTimeout(() => modal.remove(), 300);
            document.removeEventListener('keydown', escHandler);
        }
    });
}

// ── Traffic Feed (shared) ──────────────────────────────────────────────────
// One fetch serves both the transit card and the congestion chart.
async function loadTrafficData(force = false) {
    if (!force && trafficCache.data && Date.now() - trafficCache.fetchedAt < 60000) {
        return trafficCache.data;
    }
    // Collapse concurrent callers onto a single request.
    if (trafficCache.inFlight) return trafficCache.inFlight;

    trafficCache.inFlight = (async () => {
        try {
            const res = await fetch(TRAFFIC_API);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            trafficCache.data = data;
            trafficCache.fetchedAt = Date.now();
            return data;
        } finally {
            trafficCache.inFlight = null;
        }
    })();
    return trafficCache.inFlight;
}

// Timestamps in the feed are Singapore wall-clock with no offset
// ("2026-09-06T14:30:00"). Anything carrying an explicit offset is parsed as-is.
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

function seriesPoints(data, name) {
    const raw = data?.series?.[name]?.points || [];
    return raw
        .map(p => ({ t: parseSgt(p[0]), mins: p[1] == null ? null : p[1] / 60 }))
        .filter(p => p.t != null && p.mins != null)
        .sort((a, b) => a.t - b.t);
}

function mean(values) {
    return values.reduce((a, b) => a + b, 0) / values.length;
}

// Colour for a travel time, on the rounded value so the colour matches the digits shown.
function congestionLevel(mins) {
    if (mins == null) return { text: 'No Data', cls: 'nodata' };
    const m = Math.round(mins);
    if (m > LEVEL_DANGER_MIN) return { text: 'Heavy Congestion', cls: 'danger' };
    if (m >= LEVEL_WARN_MIN) return { text: 'Building Up', cls: 'warning' };
    return { text: 'Moving Smoothly', cls: 'normal' };
}

// ── Dashboard UI ───────────────────────────────────────────────────────────
// Whole minutes, the way the summary quotes Google's travel time. Every "now"
// figure on the page (summary, forecast cards, chart footer) goes through this,
// so they agree with each other and with the newest sample in traffic-24h.json.
function fmtMin(m) {
    return Math.round(m).toString();
}

// The summary headline is the newest Google Maps sample for each direction:
// duration_in_traffic from the Distance Matrix fetcher, as written to traffic-24h.json.
// Status and trend use the trailing hour so one noisy sample does not flip them.
function macroTrend(pts, anchor) {
    const recentStart = anchor - MACRO_RECENT_MIN * 60000;
    const priorStart = recentStart - MACRO_PRIOR_MIN * 60000;
    const recent = pts.filter(p => p.t > recentStart && p.t <= anchor).map(p => p.mins);
    const prior = pts.filter(p => p.t > priorStart && p.t <= recentStart).map(p => p.mins);
    if (recent.length < MACRO_MIN_POINTS || prior.length < MACRO_MIN_POINTS) return { dir: 'steady', pct: 0 };
    const pct = ((mean(recent) - mean(prior)) / mean(prior)) * 100;
    return { dir: pct > MACRO_STABLE_PCT ? 'up' : pct < -MACRO_STABLE_PCT ? 'down' : 'steady', pct };
}

const TREND_TEXT = { up: 'Rising', down: 'Easing', steady: 'Steady' };

function renderSummaryCard(d, { mins, level, trend }) {
    const card = document.getElementById(d.cardId);
    const timeEl = document.getElementById(d.timeId);
    const trendEl = document.getElementById(d.trendId);
    if (timeEl) timeEl.textContent = mins == null ? '—' : fmtMin(mins);
    if (card) {
        card.classList.remove('normal', 'warning', 'danger', 'nodata');
        card.classList.add(level ? level.cls : 'nodata');
        const what = mins == null ? 'no data' : `${fmtMin(mins)} minutes, ${level.text.toLowerCase()}, ${TREND_TEXT[trend.dir].toLowerCase()}`;
        card.setAttribute('aria-label', `${d.label}: ${what}. Open live cameras and trend.`);
    }
    if (trendEl) {
        const dir = trend ? trend.dir : 'steady';
        trendEl.className = `summary-trend ${dir}`;
        trendEl.querySelector('.trend-text').textContent = mins == null ? 'No data' : TREND_TEXT[dir];
        trendEl.title = trend && mins != null ? `Last 30 min vs the hour before: ${trend.pct >= 0 ? '+' : ''}${trend.pct.toFixed(0)}%` : '';
    }
}

// The summary headline is the newest Google Maps sample for each direction:
// duration_in_traffic from the Distance Matrix fetcher, as written to traffic-24h.json.
// Colour is that same number against the fixed cut-offs; the arrow is the macro trend.
async function updateDashboardUI() {
    const lastUpdated = document.getElementById('last-updated');

    try {
        const data = await loadTrafficData();
        const byKey = {};
        DIRECTIONS.forEach(d => { byKey[d.key] = seriesPoints(data, d.series); });

        const latestTs = Math.max(...Object.values(byKey).map(p => (p.length ? p[p.length - 1].t : 0)));
        if (!latestTs) throw new Error('no usable samples in feed');

        const nowMs = Date.now();
        const stale = nowMs - latestTs > STALE_AFTER_MIN * 60000;
        const anchor = stale ? latestTs : nowMs; // stale feed => last hour of available data
        const windowStart = anchor - TREND_WINDOW_MIN * 60000;

        DIRECTIONS.forEach(d => {
            const latest = byKey[d.key].filter(p => p.t <= anchor).pop();
            if (!latest || latest.t < windowStart) { renderSummaryCard(d, { mins: null }); return; }
            renderSummaryCard(d, { mins: latest.mins, level: congestionLevel(latest.mins), trend: macroTrend(byKey[d.key], anchor) });
        });

        if (lastUpdated) {
            const ageMin = Math.round((nowMs - latestTs) / 60000);
            lastUpdated.textContent = stale ? `${sgtClock(latestTs)} SGT (${ageMin} min old)` : `${sgtClock(latestTs)} SGT`;
        }
    } catch (err) {
        console.error('Traffic summary error:', err);
        DIRECTIONS.forEach(d => renderSummaryCard(d, { mins: null }));
        if (lastUpdated) lastUpdated.textContent = 'Feed unavailable';
    }
}

// ── Live Travel Time Chart ─────────────────────────────────────────────────
async function fetchAndRenderCongestionChart() {
    const loadingEl = document.getElementById('lc-loading');
    const chartArea = document.getElementById('lc-chart-area');
    const updatedEl = document.getElementById('lc-updated');
    const footerEl = document.getElementById('lc-footer');
    const statSg = document.getElementById('lc-stat-sg');
    const statJb = document.getElementById('lc-stat-jb');
    if (!chartArea) return;

    try {
        const data = await loadTrafficData();

        const todayStr = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Singapore' });
        const toPoints = (arr) => (arr || [])
            .filter(p => p[0].startsWith(todayStr) && p[1] != null)
            .map(p => ({ label: p[0].slice(11, 16), mins: p[1] / 60 }));

        const sgPts = toPoints(data.series.mandai_to_shell_jb?.points);
        const jbPts = toPoints(data.series.jb_to_woodlands?.points);

        if (sgPts.length === 0 && jbPts.length === 0) {
            if (loadingEl) { loadingEl.style.display = ''; loadingEl.textContent = 'No data available for today yet.'; }
            return;
        }
        if (loadingEl) loadingEl.style.display = 'none';

        const allLabels = [...new Set([...sgPts.map(p => p.label), ...jbPts.map(p => p.label)])].sort();
        const sgMap = Object.fromEntries(sgPts.map(p => [p.label, p.mins]));
        const jbMap = Object.fromEntries(jbPts.map(p => [p.label, p.mins]));

        // ── Forecast extension ─────────────────────────────────────────────
        // Extra x slots (5 min each) are appended after the last observed label.
        // Forecast points at or before the newest observation are ignored.
        const clockOf = ms => new Date(ms).toLocaleTimeString('en-GB',
            { hour12: false, hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Singapore' });
        const lastEpoch = parseSgt(`${todayStr}T${allLabels[allLabels.length - 1]}`);
        const fcSeries = {};
        let maxSlot = 0;
        FORECAST_DIRECTIONS.forEach(d => {
            const bySlot = {};
            (chartForecast?.[d.key] || []).forEach(p => {
                const slot = Math.round((p.at - lastEpoch) / 300000);
                if (lastEpoch != null && slot >= 1 && slot <= FORECAST_MAX_SLOTS) {
                    bySlot[slot] = p;
                    maxSlot = Math.max(maxSlot, slot);
                }
            });
            fcSeries[d.key] = bySlot;
        });
        const fc = maxSlot > 0;
        const fcSg = fcSeries.SG_TO_MY, fcJb = fcSeries.MY_TO_SG;
        const axisLabels = allLabels.concat(Array.from({ length: maxSlot },
            (_, k) => clockOf(lastEpoch + (k + 1) * 300000)));
        const N = axisLabels.length;
        const fcIdx = N - 1;
        // Chart indices that carry a forecast point (slot s sits at index n - 1 + s).
        const fcIdxs = [...new Set([...Object.keys(fcSg), ...Object.keys(fcJb)].map(Number))]
            .sort((x, y) => x - y).map(sl => allLabels.length - 1 + sl);

        const allMins = [...sgPts.map(p => p.mins), ...jbPts.map(p => p.mins),
            ...Object.values(fcSg).map(p => p.mins), ...Object.values(fcJb).map(p => p.mins)];
        const pad = Math.max(1, (Math.max(...allMins) - Math.min(...allMins)) * 0.15);
        const minVal = Math.max(0, Math.min(...allMins) - pad);
        const maxVal = Math.max(...allMins) + pad;

        const W = window.innerWidth < 640 ? 440 : 800, H = 240; // narrower viewBox keeps labels legible on phones
        const padL = 62, padR = 20, padT = 16, padB = 36;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;
        const n = allLabels.length;

        // Point density: shrink dots as the day fills up, drop them entirely
        // past ~120 points where they would overlap into a solid band.
        const dotR = n > 120 ? 0 : n > 60 ? 1.8 : n > 30 ? 2.6 : 3.5;
        const dotStroke = dotR >= 2.6 ? 1.5 : 1;
        const lineW = n > 120 ? 1.8 : 2.5;

        function xOf(i) { return padL + (i / Math.max(N - 1, 1)) * plotW; }
        function yOf(v) { return padT + plotH - ((v - minVal) / (maxVal - minVal || 1)) * plotH; }

        function polyline(map) {
            return allLabels
                .map((l, i) => map[l] != null ? `${xOf(i).toFixed(1)},${yOf(map[l]).toFixed(1)}` : null)
                .filter(Boolean).join(' ');
        }

        function areaPath(map) {
            const pts = allLabels.map((l, i) => map[l] != null ? [xOf(i), yOf(map[l])] : null).filter(Boolean);
            if (!pts.length) return '';
            const line = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L ');
            const first = pts[0], last = pts[pts.length - 1];
            return `M ${first[0].toFixed(1)},${(padT + plotH).toFixed(1)} L ${line} L ${last[0].toFixed(1)},${(padT + plotH).toFixed(1)} Z`;
        }

        // Pulsing marker on the most recent point of a series.
        function lastDot(map, color) {
            for (let i = allLabels.length - 1; i >= 0; i--) {
                const v = map[allLabels[i]];
                if (v == null) continue;
                const cx = xOf(i).toFixed(1), cy = yOf(v).toFixed(1);
                return `
                    <circle cx="${cx}" cy="${cy}" r="6" fill="${color}" opacity="0.35">
                        <animate attributeName="r" values="6;13;6" dur="2s" repeatCount="indefinite"/>
                        <animate attributeName="opacity" values="0.35;0;0.35" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="${cx}" cy="${cy}" r="5" fill="${color}" stroke="#070b13" stroke-width="1.5"/>`;
            }
            return '';
        }

        // Dashed line from a series' last observation through its forecast points.
        function fcExtension(map, bySlot, cls) {
            const slots = Object.keys(bySlot).map(Number).sort((x, y) => x - y);
            if (!slots.length) return '';
            let li = -1;
            for (let i = n - 1; i >= 0; i--) { if (map[allLabels[i]] != null) { li = i; break; } }
            if (li < 0) return '';
            const pts = [[xOf(li), yOf(map[allLabels[li]])]]
                .concat(slots.map(sl => [xOf(n - 1 + sl), yOf(bySlot[sl].mins)]));
            return `<polyline class="lc-fc-line ${cls}" points="${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')}"/>`;
        }
        const fcZone = fc ? `
            <rect class="lc-fc-zone" x="${xOf(n - 1).toFixed(1)}" y="${padT}" width="${(xOf(fcIdx) - xOf(n - 1)).toFixed(1)}" height="${plotH}"/>
            <text class="lc-fc-tag" x="${xOf(fcIdx).toFixed(1)}" y="${padT + 11}" text-anchor="end">FORECAST</text>` : '';

        const ySteps = 4;
        let gridLines = '';
        for (let s = 0; s <= ySteps; s++) {
            const v = minVal + (s / ySteps) * (maxVal - minVal);
            const y = yOf(v).toFixed(1);
            gridLines += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
            gridLines += `<text x="${padL - 6}" y="${parseFloat(y) + 4}" text-anchor="end" font-size="10" fill="rgba(148,163,184,0.7)">${v.toFixed(0)}</text>`;
        }

        let xLabels = '';
        const step = N <= 8 ? 1 : Math.ceil(N / 6);
        axisLabels.forEach((lbl, i) => {
            const isEnd = i === N - 1;
            // Keep stepped labels clear of the final (forecast) label.
            if (!isEnd && (i % step !== 0 || (fc && N - 1 - i < step))) return;
            const cls = fc && isEnd ? ' class="lc-fc-xlabel"' : '';
            xLabels += `<text${cls} x="${xOf(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="rgba(148,163,184,0.7)">${lbl}</text>`;
        });

        const sgLine = polyline(sgMap);
        const jbLine = polyline(jbMap);
        const axisMidY = padT + plotH / 2;

        const svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" class="lc-svg">
            <defs>
                <linearGradient id="grad-sg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="${DIR_COLOR.sg}" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="${DIR_COLOR.sg}" stop-opacity="0"/>
                </linearGradient>
                <linearGradient id="grad-jb" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="${DIR_COLOR.jb}" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="${DIR_COLOR.jb}" stop-opacity="0"/>
                </linearGradient>
            </defs>
            ${gridLines}
            ${fcZone}
            <text x="16" y="${axisMidY}" transform="rotate(-90 16 ${axisMidY})"
                  text-anchor="middle" font-size="11" fill="rgba(148,163,184,0.85)"
                  letter-spacing="0.5">Travel time (min)</text>
            <path d="${areaPath(sgMap)}" fill="url(#grad-sg)"/>
            <path d="${areaPath(jbMap)}" fill="url(#grad-jb)"/>
            ${sgLine ? `<polyline points="${sgLine}" fill="none" stroke="${DIR_COLOR.sg}" stroke-width="${lineW}" stroke-linejoin="round" stroke-linecap="round"/>` : ''}
            ${jbLine ? `<polyline points="${jbLine}" fill="none" stroke="${DIR_COLOR.jb}" stroke-width="${lineW}" stroke-linejoin="round" stroke-linecap="round"/>` : ''}
            ${dotR ? allLabels.map((l, i) => sgMap[l] != null ? `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(sgMap[l]).toFixed(1)}" r="${dotR}" fill="${DIR_COLOR.sg}" stroke="#070b13" stroke-width="${dotStroke}"/>` : '').join('') : ''}
            ${dotR ? allLabels.map((l, i) => jbMap[l] != null ? `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(jbMap[l]).toFixed(1)}" r="${dotR}" fill="${DIR_COLOR.jb}" stroke="#070b13" stroke-width="${dotStroke}"/>` : '').join('') : ''}
            ${fcExtension(sgMap, fcSg, 'sg')}
            ${fcExtension(jbMap, fcJb, 'jb')}
            ${lastDot(sgMap, DIR_COLOR.sg)}
            ${lastDot(jbMap, DIR_COLOR.jb)}
            ${xLabels}
            <line class="lc-crosshair" x1="0" y1="${padT}" x2="0" y2="${padT + plotH}"
                  stroke="rgba(148,163,184,0.55)" stroke-width="1" stroke-dasharray="4 4" style="display:none"/>
            <circle class="lc-hover-dot lc-hd-sg" r="6" fill="none" stroke="${DIR_COLOR.sg}" stroke-width="2" style="display:none"/>
            <circle class="lc-hover-dot lc-hd-jb" r="6" fill="none" stroke="${DIR_COLOR.jb}" stroke-width="2" style="display:none"/>
            <rect class="lc-hit" x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent"/>
        </svg>`;

        chartArea.innerHTML = svg;

        // ── Hover crosshair + tooltip ──────────────────────────────────────
        const svgEl = chartArea.querySelector('svg');
        const cross = svgEl.querySelector('.lc-crosshair');
        const hdSg = svgEl.querySelector('.lc-hd-sg');
        const hdJb = svgEl.querySelector('.lc-hd-jb');
        const hit = svgEl.querySelector('.lc-hit');

        const tip = document.createElement('div');
        tip.className = 'lc-tooltip';
        tip.style.display = 'none';
        chartArea.appendChild(tip);

        function toSvgX(evt) {
            const pt = svgEl.createSVGPoint();
            pt.x = evt.clientX;
            pt.y = evt.clientY;
            return pt.matrixTransform(svgEl.getScreenCTM().inverse()).x;
        }

        function showAt(evt) {
            const x = toSvgX(evt);
            let i = Math.round(((x - padL) / plotW) * (N - 1));
            i = Math.max(0, Math.min(N - 1, i));
            // Past the last observation, snap to the nearest forecast point.
            if (fc && i > n - 1) i = fcIdxs.reduce((best, k) => Math.abs(k - i) < Math.abs(best - i) ? k : best, fcIdxs[0]);
            const label = axisLabels[i];
            const isFc = fc && i > n - 1;
            const cx = xOf(i);

            cross.setAttribute('x1', cx.toFixed(1));
            cross.setAttribute('x2', cx.toFixed(1));
            cross.style.display = '';

            const rows = [];
            [[sgMap, fcSg, hdSg, 'To Johor', 'sg'], [jbMap, fcJb, hdJb, 'To Singapore', 'jb']].forEach(([map, fcVal, dot, name, cls]) => {
                // Slots past the last observation carry no data, except the forecast point.
                const fp = isFc ? fcVal[i - (n - 1)] : null;
                const v = isFc ? (fp ? fp.mins : null) : (i < n ? map[label] : null);
                if (v == null) { dot.style.display = 'none'; return; }
                dot.setAttribute('cx', cx.toFixed(1));
                dot.setAttribute('cy', yOf(v).toFixed(1));
                dot.style.display = '';
                rows.push(`<div class="lc-tt-row"><span class="lc-dot ${cls}"></span>${name}<strong>${v.toFixed(1)} min${isFc ? ' (forecast)' : ''}</strong></div>`);
            });

            if (!rows.length) { tip.style.display = 'none'; return; }
            tip.innerHTML = `<div class="lc-tt-time">${label} SGT${isFc ? ' · forecast' : ''}</div>${rows.join('')}`;
            tip.style.display = 'block';

            const areaRect = chartArea.getBoundingClientRect();
            const svgRect = svgEl.getBoundingClientRect();
            const scale = svgRect.width / W;
            let left = (svgRect.left - areaRect.left) + cx * scale + 14;
            if (left + tip.offsetWidth > areaRect.width) left -= tip.offsetWidth + 28;
            tip.style.left = `${Math.max(4, left)}px`;
            tip.style.top = `${(svgRect.top - areaRect.top) + padT * scale}px`;
        }

        function hideAll() {
            cross.style.display = 'none';
            hdSg.style.display = 'none';
            hdJb.style.display = 'none';
            tip.style.display = 'none';
        }

        hit.addEventListener('mousemove', showAt);
        hit.addEventListener('mouseleave', hideAll);
        hit.addEventListener('touchmove', e => { showAt(e.touches[0]); e.preventDefault(); }, { passive: false });
        hit.addEventListener('touchend', hideAll);

        if (updatedEl && data.updated_at_sgt) {
            updatedEl.textContent = `Updated ${data.updated_at_sgt.slice(11, 16)} SGT`;
        }

        const legendFc = document.getElementById('lc-legend-fc');
        if (legendFc) {
            legendFc.style.display = fc ? '' : 'none';
            const span = Math.round(maxSlot * 5 / 30) * 30; // nearest half hour, e.g. "5 h" not "5.1 h"
            legendFc.textContent = `Forecast (next ${span < 60 ? `${span} min` : `${span / 60} h`})`;
        }
        // Footer quotes the nearest forecast point only; the chart carries the rest.
        const firstFc = bySlotFirst => { const k = Object.keys(bySlotFirst).map(Number).sort((x, y) => x - y)[0]; return k == null ? null : bySlotFirst[k]; };
        const nextSg = firstFc(fcSg), nextJb = firstFc(fcJb);

        if (footerEl) footerEl.style.display = 'flex';
        const latestSg = sgPts[sgPts.length - 1];
        const latestJb = jbPts[jbPts.length - 1];
        if (statSg && latestSg) {
            statSg.innerHTML = `<span class="lc-dot sg"></span><strong>To Johor</strong>&nbsp;Latest: <strong>${fmtMin(latestSg.mins)} min</strong> at ${latestSg.label}${nextSg ? ` &rarr; <strong>${nextSg.mins.toFixed(1)}</strong> forecast at ${clockOf(nextSg.at)}` : ''}`;
        }
        if (statJb && latestJb) {
            statJb.innerHTML = `<span class="lc-dot jb"></span><strong>To Singapore</strong>&nbsp;Latest: <strong>${fmtMin(latestJb.mins)} min</strong> at ${latestJb.label}${nextJb ? ` &rarr; <strong>${nextJb.mins.toFixed(1)}</strong> forecast at ${clockOf(nextJb.at)}` : ''}`;
        }

    } catch (err) {
        console.error('Congestion chart fetch error:', err);
        if (loadingEl) { loadingEl.style.display = ''; loadingEl.textContent = `Could not load congestion data: ${err.message}`; }
    }
}

// ── Forecast Widget ────────────────────────────────────────────────────────
// forecast-api (docs/runbooks/forecast-api.md). `?curve=forecast` returns a forecast
// every 30 min from one origin. Only 30 min is evaluated; later points are exploratory,
// and the model beat the typical-day baseline through about 5.5 h in the study (not
// confirmed on Run B). So the page asks for 5 h, which stays inside the model's range.
// `?baseline=profile` is the calendar-profile "typical day", not a forecast. The
// comparison chart always spans the full day: the 5 h forecast over the 24 h typical day.
const FORECAST_HOURS = 5;
const COMPARE_HOURS = 24;            // forecast-vs-typical chart span
const FORECAST_MAX_SLOTS = 66;        // 5.5 h in 5-min chart slots
const FORECAST_TIMEOUT_MS = 60000;    // the first curve call can take ~20 s while the models fit
const BASELINE_REFRESH_MS = 3600000;  // the profile is refitted once per SGT day
const FORECAST_DIRECTIONS = [
    { key: 'SG_TO_MY', label: 'To Johor', cls: 'sg', color: DIR_COLOR.sg, series: 'mandai_to_shell_jb' },
    { key: 'MY_TO_SG', label: 'To Singapore', cls: 'jb', color: DIR_COLOR.jb, series: 'jb_to_woodlands' }
];

const FORECAST_ERRORS = {
    400: 'Forecast request was rejected.',
    502: 'Forecast query failed on the server.',
    503: 'Forecast paused: no recent data for this direction.'
};

let forecastView = null;      // { dirs: { KEY: { points, originAt } }, curve }
let baselineDay = null;       // { KEY: [{ t, mins }] }
let baselineAt = 0;

function sgtClock(iso) {
    const t = typeof iso === 'number' ? iso : Date.parse(iso);
    if (Number.isNaN(t)) return '—';
    return new Date(t).toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Singapore' });
}

function setForecastStatus(text) {
    const el = document.getElementById('fc-status');
    if (!el) return;
    el.textContent = text;
    el.style.display = text ? '' : 'none';
}

async function fetchJson(url, timeoutMs) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
        const res = await fetch(url, { signal: ctl.signal });
        if (!res.ok) throw new Error(FORECAST_ERRORS[res.status] || `HTTP ${res.status}`);
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

// Both response shapes become { points: [{ at, mins, base, status, model, lead }], originAt }.
function normalizeCurve(data) {
    const dirs = {};
    FORECAST_DIRECTIONS.forEach(d => {
        const src = data.directions?.[d.key];
        const points = (src?.points || [])
            .filter(p => p.model !== 'profile' && p.forecast_min != null)
            .map(p => ({ at: Date.parse(p.forecast_for), mins: p.forecast_min, base: p.baseline_min,
                         status: p.status, model: p.model, lead: p.lead_min }))
            .filter(p => Number.isFinite(p.at));
        dirs[d.key] = { points, originAt: Date.parse(src?.origin_ts) };
    });
    return { dirs, curve: true };
}

function normalizeServed(data) {
    const dirs = {};
    FORECAST_DIRECTIONS.forEach(d => {
        const f = data.directions?.[d.key];
        const at = Date.parse(f?.forecast_for);
        dirs[d.key] = {
            points: f && f.forecast_min != null && Number.isFinite(at)
                ? [{ at, mins: f.forecast_min, base: null, status: 'evaluated', model: f.model, lead: f.lead_min }] : [],
            originAt: Date.parse(f?.origin_ts)
        };
    });
    return { dirs, curve: false };
}

async function loadBaselineDay() {
    if (baselineDay && Date.now() - baselineAt < BASELINE_REFRESH_MS) return;
    try {
        const data = await fetchJson(`${FORECAST_API}?baseline=profile&hours=24`, 30000);
        const out = {};
        FORECAST_DIRECTIONS.forEach(d => {
            out[d.key] = (data.directions?.[d.key] || [])
                .map(b => ({ t: Date.parse(b.bin_start), mins: b.forecast_min }))
                .filter(b => Number.isFinite(b.t) && b.mins != null);
        });
        baselineDay = out;
        baselineAt = Date.now();
    } catch (err) {
        console.error('Baseline fetch error:', err); // the chart falls back to the curve's own baseline_min (5 h)
    }
}

// Forecast against the typical day, from the forecast origin out to spanH hours.
function forecastCompare(d, view, spanH) {
    const pts = view.points;
    if (!pts.length) return '';
    const t0 = Number.isFinite(view.originAt) ? view.originAt : pts[0].at - 1800000;
    const t1 = t0 + Math.max(spanH, FORECAST_HOURS) * 3600000;
    const baseBins = baselineDay?.[d.key]?.filter(b => b.t >= t0 && b.t <= t1)
        || [];
    const base = baseBins.length > 1 ? baseBins
        : pts.filter(p => p.base != null).map(p => ({ t: p.at, mins: p.base }));
    const fcPts = pts.filter(p => p.at <= t1);

    const vals = fcPts.map(p => p.mins).concat(base.map(b => b.mins));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = Math.max(1, (hi - lo) * 0.12);
    const W = 420, H = 150, padL = 30, padR = 58, padT = 8, padB = 22;
    const x = t => padL + ((t - t0) / (t1 - t0)) * (W - padL - padR);
    const y = v => padT + (H - padT - padB) - ((v - (lo - pad)) / ((hi + pad) - (lo - pad) || 1)) * (H - padT - padB);
    const poly = arr => arr.map(p => `${x(p.t ?? p.at).toFixed(1)},${y(p.mins).toFixed(1)}`).join(' ');

    const tickEvery = (spanH <= 6 ? 1 : 6) * 3600000; // 6-hourly ticks on the full day
    let ticks = '';
    for (let t = Math.ceil(t0 / tickEvery) * tickEvery; t <= t1; t += tickEvery) {
        ticks += `<line class="fc-tick" x1="${x(t).toFixed(1)}" y1="${padT}" x2="${x(t).toFixed(1)}" y2="${H - padB}"/>
                  <text class="fc-axis" x="${x(t).toFixed(1)}" y="${H - 6}" text-anchor="middle">${sgtClock(t)}</text>`;
    }
    const yLab = v => `<text class="fc-axis" x="${padL - 4}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`;
    const last = fcPts[fcPts.length - 1];
    const dots = fcPts.slice(0, -1).map(p => `<circle cx="${x(p.at).toFixed(1)}" cy="${y(p.mins).toFixed(1)}" r="2.4" fill="${d.color}"/>`).join('');

    // Label each line at its right-hand end; push the labels apart if the ends are close.
    const baseEnd = base.length > 1 ? base[base.length - 1] : null;
    let fy = y(last.mins), by = baseEnd ? y(baseEnd.mins) : null;
    if (by != null && Math.abs(fy - by) < 14) {
        const gap = (14 - Math.abs(fy - by)) / 2;
        if (fy <= by) { fy -= gap; by += gap; } else { fy += gap; by -= gap; }
    }
    const lx = x(last.at) + 8;
    const endLabels = `<text class="fc-line-label" x="${lx.toFixed(1)}" y="${(fy + 3).toFixed(1)}" fill="${d.color}">Forecast</text>`
        + (baseEnd ? `<text class="fc-line-label fc-line-label-base" x="${(x(baseEnd.t) + 8).toFixed(1)}" y="${(by + 3).toFixed(1)}">Typical</text>` : '');

    return `<svg viewBox="0 0 ${W} ${H}" class="fc-compare" role="img"
                 aria-label="${d.label} forecast against the typical day">
        ${ticks}${yLab(lo)}${yLab(hi)}
        ${base.length > 1 ? `<polyline class="fc-base-line" points="${poly(base)}"/>` : ''}
        <polyline class="fc-fc-line" stroke="${d.color}" points="${poly(fcPts)}"/>
        ${dots}<circle cx="${x(last.at).toFixed(1)}" cy="${y(last.mins).toFixed(1)}" r="4" fill="${d.color}" stroke="#070b13" stroke-width="1.5"/>
        ${endLabels}
    </svg>`;
}

function renderForecastCards() {
    const cards = document.getElementById('fc-cards');
    if (!cards || !forecastView) return;
    const traffic = trafficCache.data;

    // Without the 24 h typical day the chart can only cover the curve's own 5 h.
    const spanH = baselineDay ? COMPARE_HOURS : FORECAST_HOURS;

    cards.innerHTML = FORECAST_DIRECTIONS.map(d => {
        const view = forecastView.dirs[d.key];
        const f = view.points[0];
        if (!f) {
            return `
                <div class="fc-card ${d.cls}">
                    <div class="fc-card-head"><span class="lc-dot ${d.cls}"></span>${d.label}</div>
                    <div class="fc-value">—</div>
                    <div class="fc-when">no forecast available</div>
                </div>`;
        }
        const obs = traffic ? seriesPoints(traffic, d.series) : [];
        const now = obs.length ? obs[obs.length - 1].mins : null;
        const delta = now == null ? null : f.mins - now;

        // Same cut-offs as the summary cards.
        const level = congestionLevel(f.mins);

        const nowLine = delta == null ? '' :
            `<div class="fc-now">Now <strong>${fmtMin(now)}</strong> min <span class="fc-delta">(${delta >= 0 ? '+' : ''}${delta.toFixed(1)})</span></div>`;
        const lead = f.lead != null ? ` · ${Math.round(f.lead)} min ahead` : '';

        return `
            <a class="fc-card ${d.cls}" href="#live" aria-label="${d.label}: open live cameras and trend">
                <div class="fc-card-head"><span class="lc-dot ${d.cls}"></span>${d.label}<span class="summary-go fc-go" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg></span></div>
                <div class="fc-main">
                    <span class="fc-value lv-text ${level.cls}">${f.mins.toFixed(1)}</span>
                    <span class="fc-when">min at ${sgtClock(f.at)} SGT${lead}</span>
                </div>
                ${nowLine}
                ${forecastCompare(d, view, spanH)}
            </a>`;
    }).join('');

    const origin = FORECAST_DIRECTIONS.map(d => forecastView.dirs[d.key].originAt).filter(Number.isFinite).sort().pop();
    const updatedEl = document.getElementById('fc-updated');
    if (updatedEl) updatedEl.textContent = origin ? `Based on ${sgtClock(origin)} SGT data` : 'Updated';
    const metaEl = document.getElementById('fc-meta');
    if (metaEl) {
        const models = FORECAST_DIRECTIONS.map(d => {
            const m = [...new Set(forecastView.dirs[d.key].points.map(p => p.model))];
            return `${d.label}: ${m.join(', ') || '—'}`;
        }).join(' · ');
        metaEl.textContent = `Model — ${models}`;
    }
}

async function fetchForecast() {
    const cards = document.getElementById('fc-cards');
    const updatedEl = document.getElementById('fc-updated');
    const metaEl = document.getElementById('fc-meta');
    if (!cards) return;

    setForecastStatus('Fetching forecast…');

    let view = null;
    let failure = null;
    try {
        view = normalizeCurve(await fetchJson(`${FORECAST_API}?curve=forecast&hours=${FORECAST_HOURS}`, FORECAST_TIMEOUT_MS));
    } catch (err) {
        // The curve fits several models and can be slow; fall back to the cheap 30-min forecast.
        failure = err;
        try {
            view = normalizeServed(await fetchJson(FORECAST_API, FORECAST_TIMEOUT_MS));
        } catch (err2) {
            failure = err2;
        }
    }

    if (!view || !FORECAST_DIRECTIONS.some(d => view.dirs[d.key].points.length)) {
        const err = failure || new Error(FORECAST_ERRORS[503]);
        const msg = err.name === 'AbortError' ? 'Forecast service timed out.'
            : err instanceof TypeError ? 'Forecast service unreachable.'
                : err.message;
        if (chartForecast) { chartForecast = null; fetchAndRenderCongestionChart(); } // drop a stale extension
        forecastView = null;
        cards.innerHTML = '';
        setForecastStatus(`Forecast unavailable: ${msg}`);
        if (updatedEl) updatedEl.textContent = 'Unavailable';
        if (metaEl) metaEl.textContent = '';
        console.error('Forecast fetch error:', err);
        return;
    }

    forecastView = view;
    if (view.curve) await loadBaselineDay();

    // Hand the forecast line to the chart, which redraws with the dashed extension.
    chartForecast = {};
    FORECAST_DIRECTIONS.forEach(d => {
        chartForecast[d.key] = view.dirs[d.key].points.map(p => ({ at: p.at, mins: p.mins, status: p.status }));
    });
    fetchAndRenderCongestionChart();

    try { await loadTrafficData(); } catch (_) { /* "now" is optional */ }
    setForecastStatus(view.curve ? '' : 'Showing the 30-minute forecast only; the longer forecast did not load.');
    renderForecastCards();
}

// ── Views ──────────────────────────────────────────────────────────────────
// Two views on one page, picked by the URL hash: "#" is the summary, "#live" the
// cameras and trend for the Causeway, "#live/tuas" the Tuas cameras.
function parseRoute() {
    const m = /^#live(?:\/(woodlands|tuas))?$/.exec(location.hash);
    return m ? { view: 'live', checkpoint: m[1] || 'woodlands' } : { view: 'home', checkpoint: 'woodlands' };
}

function applyCheckpoint(cp) {
    currentCheckpoint = cp;
    document.querySelectorAll('.seg-btn').forEach(b => {
        const on = b.dataset.checkpoint === cp;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    // Panels that exist for one checkpoint only (trend + AI detection: Causeway; notice: Tuas).
    document.querySelectorAll('#view-live [data-only]').forEach(el => {
        el.hidden = el.dataset.only !== cp;
    });
    const title = document.getElementById('cam-title');
    if (title) title.textContent = `LTA Cameras — ${checkpointMeta[cp].name}`;
}

function startCameraRefresh() {
    if (cameraRefreshTimer) return;
    cameraRefreshTimer = setInterval(fetchLiveCameras, 60000);
}

function stopCameraRefresh() {
    clearInterval(cameraRefreshTimer);
    cameraRefreshTimer = null;
}

function route() {
    const { view, checkpoint } = parseRoute();
    const viewChanged = view !== currentView;
    const cpChanged = checkpoint !== currentCheckpoint;
    currentView = view;

    document.querySelectorAll('.view').forEach(el => { el.hidden = el.dataset.view !== view; });
    document.querySelectorAll('[data-view-link]').forEach(a =>
        a.classList.toggle('active', a.dataset.viewLink === view));
    if (viewChanged) window.scrollTo(0, 0);

    if (view !== 'live') { stopCameraRefresh(); return; }

    applyCheckpoint(checkpoint);
    // Cameras load only while the live view is open, which saves mobile data on the summary.
    if (viewChanged || cpChanged || !document.querySelector('#live-cam-grid .cam-tile')) fetchLiveCameras();
    startCameraRefresh();
    if (checkpoint === 'woodlands' && !detectionLoaded) {
        detectionLoaded = true;
        fetchBackendMessage();
    }
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateDashboardUI();
    fetchAndRenderCongestionChart();
    fetchForecast();

    window.addEventListener('hashchange', route);
    currentView = null; // force the first route() to treat the view as new
    route();

    chartRefreshTimer = setInterval(() => {
        fetchAndRenderCongestionChart();
        updateDashboardUI();
        fetchForecast();
    }, 300000);

    const refreshCamerasBtn = document.getElementById('refresh-cameras');
    if (refreshCamerasBtn) {
        refreshCamerasBtn.addEventListener('click', () => {
            refreshCamerasBtn.disabled = true;
            fetchLiveCameras().finally(() => setTimeout(() => { refreshCamerasBtn.disabled = false; }, 3000));
        });
    }

    const refreshAiBtn = document.getElementById('refresh-ai-detection');
    if (refreshAiBtn) {
        refreshAiBtn.addEventListener('click', () => {
            refreshAiBtn.disabled = true;
            fetchBackendMessage().finally(() => setTimeout(() => { refreshAiBtn.disabled = false; }, 3000));
        });
    }

    const refreshForecastBtn = document.getElementById('refresh-forecast');
    if (refreshForecastBtn) {
        refreshForecastBtn.addEventListener('click', () => {
            refreshForecastBtn.disabled = true;
            fetchForecast().finally(() => setTimeout(() => { refreshForecastBtn.disabled = false; }, 3000));
        });
    }

    // Causeway | Tuas switcher in the live view. The hash carries the choice, so Back works.
    document.querySelectorAll('.seg-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const cp = btn.dataset.checkpoint;
            location.hash = cp === 'woodlands' ? '#live' : `#live/${cp}`;
        });
    });

    const refreshBtn = document.getElementById('refresh-dashboard');
    const lastUpdated = document.getElementById('last-updated');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            refreshBtn.disabled = true;
            if (lastUpdated) lastUpdated.textContent = 'Updating...';
            try {
                await loadTrafficData(true); // bypass the 60s cache
                await Promise.all([updateDashboardUI(), fetchAndRenderCongestionChart()]);
                renderForecastCards(); // "Now" on the forecast cards follows the fresh sample
            } catch (err) {
                console.error('Dashboard refresh error:', err);
                await updateDashboardUI();
            } finally {
                setTimeout(() => { refreshBtn.disabled = false; }, 3000);
            }
        });
    }
});
