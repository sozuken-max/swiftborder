// ── Camera IDs per checkpoint (from data.gov.sg LTA API) ─────────────────
const CHECKPOINT_CAMERAS = {
    woodlands: ['2701', '2702', '2703', '2704', '2705', '2706'],
    tuas: ['4703', '4712', '4713', '4714', '4716']
};

// ── Checkpoint Metadata ────────────────────────────────────────────────────
// Live travel-time series only exist for Woodlands. Tuas has cameras only.
const checkpointMeta = {
    woodlands: { name: "Woodlands Causeway", hasTransitFeed: true },
    tuas: { name: "Tuas Second Link", hasTransitFeed: false }
};

// Direction config — keys map to the series names in traffic-24h.json.
const DIRECTIONS = [
    { key: 'sg', series: 'mandai_to_shell_jb', timeId: 'sg-time', subId: 'sg-sub', trendId: 'sg-trend', statusId: 'sg-status' },
    { key: 'jb', series: 'jb_to_woodlands', timeId: 'jb-time', subId: 'jb-sub', trendId: 'jb-trend', statusId: 'jb-status' }
];

// ── Trend / status tuning ──────────────────────────────────────────────────
const TREND_WINDOW_MIN = 60;   // trailing window, minutes
const RECENT_POINTS = 3;       // "now" = mean of the last N samples
const TREND_STABLE_PCT = 10;   // recent mean within +/-10% of the 1h mean => stable
const TREND_MIN_POINTS = 6;    // fewer samples than this => report stable, limited data
const STALE_AFTER_MIN = 20;    // newest sample older than this => label the card stale
const BASELINE_PCTILE = 0.20;  // 20th percentile of the 24h series = free-flow reference
const STATUS_WARN_RATIO = 1.25; // recent mean / baseline
const STATUS_DANGER_RATIO = 1.75;

// ── Toll Fees ──────────────────────────────────────────────────────────────
const tollFees = {
    woodlands: {
        car: { sg_to_my: { sg: 0, my: 2.50 }, my_to_sg: { sg: 0.80, my: 0 } },
        motorcycle: { sg_to_my: { sg: 0, my: 0 }, my_to_sg: { sg: 0, my: 0 } },
        van: { sg_to_my: { sg: 0, my: 3.80 }, my_to_sg: { sg: 1.20, my: 0 } },
        bus: { sg_to_my: { sg: 0, my: 5.00 }, my_to_sg: { sg: 1.60, my: 0 } }
    },
    tuas: {
        car: { sg_to_my: { sg: 2.10, my: 2.50 }, my_to_sg: { sg: 2.10, my: 0 } },
        motorcycle: { sg_to_my: { sg: 0.50, my: 0 }, my_to_sg: { sg: 0.50, my: 0 } },
        van: { sg_to_my: { sg: 3.10, my: 3.80 }, my_to_sg: { sg: 3.10, my: 0 } },
        bus: { sg_to_my: { sg: 4.20, my: 5.00 }, my_to_sg: { sg: 4.20, my: 0 } }
    }
};

const MYR_PER_SGD = 3.25; // approximate; update periodically

// ── Global State ───────────────────────────────────────────────────────────
let currentCheckpoint = 'woodlands';
let cameraRefreshTimer = null;
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
// Only cameras with a calibrated dividing line return a usable split; 2701 is
// the only one today, so the panel is pinned to it explicitly rather than
// relying on the service's default.
const AI_DETECTION_CAMERA = '2701';

// Service labels travel direction SG-MY / MY-SG; the UI says SG -> JB / JB -> SG.
const AI_DIRECTIONS = [
    { key: 'sg', label: 'SG → JB', cls: 'sg', countHeader: 'X-Vehicle-Count-SG-MY', congestionHeader: 'X-Congestion-SG-MY' },
    { key: 'jb', label: 'JB → SG', cls: 'jb', countHeader: 'X-Vehicle-Count-MY-SG', congestionHeader: 'X-Congestion-MY-SG' }
];

// Congestion here is queue depth — how far up the frame a direction's
// detections reach — not throughput. A single frame cannot measure flow.
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

// Side-by-side comparison: the same LTA frame (pinned by date_time) scored by two
// Roboflow workflow versions. `model` is a backend allowlist (camdetect WORKFLOW_VERSIONS).
// Each pane is one billed Roboflow call.
const AI_MODELS = [
    { key: 'v6', label: 'Model v6', note: 'current' },
    { key: 'v4', label: 'Model v4', note: 'previous' }
];

function ensureModelPanes() {
    const wrap = document.getElementById('ai-compare');
    if (!wrap || wrap.dataset.ready) return wrap;
    wrap.innerHTML = AI_MODELS.map(m => `
        <div class="ai-pane" data-model="${m.key}">
            <div class="ai-pane-head">
                <span class="ai-pane-title">${m.label} <span class="ai-pane-note">${m.note}</span></span>
                <span class="ai-pane-count" id="ai-count-${m.key}"></span>
            </div>
            <div class="ai-detection-body">
                <p class="ai-detection-status" id="ai-status-${m.key}">Fetching AI detection result…</p>
                <img id="ai-img-${m.key}" class="ai-detection-img" alt="${m.label} detections at the checkpoint" style="display:none;" />
            </div>
            <div class="ai-dir-grid" id="ai-dirs-${m.key}"></div>
        </div>`).join('');
    wrap.dataset.ready = '1';
    return wrap;
}

function renderDetectionSummary(box, summary) {
    if (!box) return;

    if (!summary.supported) {
        box.innerHTML = '<p class="ai-dir-note">Direction split unavailable from this service.</p>';
        return;
    }
    if (summary.uncalibrated) {
        box.innerHTML = '<p class="ai-dir-note">No dividing line calibrated for this camera — vehicles counted, not split by direction.</p>';
        return;
    }

    const cards = summary.dirs.map(d => {
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

    box.innerHTML = cards + unknownNote;
}

// Singapore wall-clock "YYYY-MM-DDTHH:MM:SS", the backend's date_time format.
function sgtNowParam() {
    return new Date().toLocaleString('sv-SE', { timeZone: 'Asia/Singapore', hour12: false }).replace(' ', 'T');
}

async function fetchModelDetection(model, dateTime) {
    const imgEl = document.getElementById(`ai-img-${model.key}`);
    const statusEl = document.getElementById(`ai-status-${model.key}`);
    const dirsEl = document.getElementById(`ai-dirs-${model.key}`);
    const countEl = document.getElementById(`ai-count-${model.key}`);

    if (statusEl) statusEl.textContent = 'Fetching AI detection result…';
    if (imgEl) imgEl.style.display = 'none';
    if (dirsEl) dirsEl.innerHTML = '';
    if (countEl) countEl.textContent = '';

    try {
        // format=directional draws the dividing line and colours the boxes by
        // direction. The counts ride on the headers either way.
        const url = new URL(BACKEND_URL);
        url.searchParams.set('format', 'directional');
        url.searchParams.set('camera_id', AI_DETECTION_CAMERA);
        url.searchParams.set('model', model.key);
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

        renderDetectionSummary(dirsEl, summary);
        if (countEl && summary.total != null) countEl.textContent = `${summary.total} vehicles`;
        return summary;
    } catch (err) {
        if (statusEl) statusEl.textContent = `AI detection unavailable: ${err.message}`;
        console.error(`Backend image fetch error (${model.key}):`, err);
        throw err;
    }
}

async function fetchBackendMessage() {
    const bar = document.getElementById('backend-message');
    if (bar) bar.style.display = 'none';
    if (!ensureModelPanes()) return;

    const timestampEl = document.getElementById('ai-detection-timestamp');
    const dateTime = sgtNowParam(); // one timestamp, so both models score the same frame
    const results = await Promise.allSettled(AI_MODELS.map(m => fetchModelDetection(m, dateTime)));

    // Prefer the frame's own capture time over the client clock.
    const ok = results.find(r => r.status === 'fulfilled');
    if (timestampEl) {
        const frameTime = ok?.value?.frameTime;
        const stamp = frameTime ? `${frameTime.slice(11, 19)} SGT` : `${dateTime.slice(11, 19)} SGT`;
        timestampEl.textContent = ok ? `CAM ${AI_DETECTION_CAMERA} · Frame: ${stamp} · same frame for both models` : '';
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
        const timeStr = new Date().toLocaleTimeString('en-GB', { hour12: false });
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
        const timeStr = new Date(cam.timestamp).toLocaleTimeString('en-GB', { hour12: false });
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

    const camTime = new Date(cam.timestamp).toLocaleTimeString('en-GB', { hour12: false });
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

function percentile(values, q) {
    if (!values.length) return null;
    const s = [...values].sort((a, b) => a - b);
    const pos = q * (s.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - lo); // linear interpolation
}

function mean(values) {
    return values.reduce((a, b) => a + b, 0) / values.length;
}

// Current conditions for one direction over the trailing window.
// Trend = mean of the most recent RECENT_POINTS samples vs the mean of the
// whole window. Outside +/- TREND_STABLE_PCT it is increasing / decreasing.
function analyseDirection(allPts, windowStart, anchor) {
    const win = allPts.filter(p => p.t >= windowStart && p.t <= anchor);
    if (!win.length) return { ok: false };

    const mins = win.map(p => p.mins);
    const windowMean = mean(mins);
    const recent = mins.slice(-RECENT_POINTS);
    const recentMean = mean(recent);

    const pct = windowMean > 0 ? ((recentMean - windowMean) / windowMean) * 100 : 0;
    const thin = win.length < TREND_MIN_POINTS;
    const trend = thin ? 'stable'
        : pct > TREND_STABLE_PCT ? 'up'
            : pct < -TREND_STABLE_PCT ? 'down'
                : 'stable';

    return {
        ok: true,
        p25: percentile(mins, 0.25),
        p75: percentile(mins, 0.75),
        median: percentile(mins, 0.50),
        recentMean,
        recentCount: recent.length,
        windowMean,
        count: win.length,
        latestAt: win[win.length - 1].t,
        pct,
        trend,
        thin,
        // 24h free-flow reference for this direction, used for the congestion label
        baseline: percentile(allPts.map(p => p.mins), BASELINE_PCTILE)
    };
}

// Congestion label is per direction: current level against that direction's
// own 24h free-flow baseline. JB -> SG runs far slower than SG -> JB, so a
// single shared threshold would mislabel one of them.
function congestionLevel(r) {
    if (!r.ok || !r.baseline) return { text: 'No Data', cls: 'warning' };
    const ratio = r.recentMean / r.baseline;
    if (ratio >= STATUS_DANGER_RATIO) return { text: 'Heavy Congestion', cls: 'danger', ratio };
    if (ratio >= STATUS_WARN_RATIO) return { text: 'Building Up', cls: 'warning', ratio };
    return { text: 'Moving Smoothly', cls: 'normal', ratio };
}

function renderTrend(el, r) {
    if (!el) return;
    if (!r.ok) { el.textContent = '- no data'; el.className = 'time-trend stable'; return; }
    if (r.thin) { el.textContent = '▬ Stable (limited data)'; el.className = 'time-trend stable'; return; }
    const sign = r.pct >= 0 ? '+' : '';
    const pctTxt = `${sign}${r.pct.toFixed(0)}% vs 1h avg`;
    if (r.trend === 'up') { el.textContent = `▲ Increasing ${pctTxt}`; el.className = 'time-trend up'; }
    else if (r.trend === 'down') { el.textContent = `▼ Decreasing ${pctTxt}`; el.className = 'time-trend down'; }
    else { el.textContent = `▬ Stable ${pctTxt}`; el.className = 'time-trend stable'; }
}

// ── Dashboard UI ───────────────────────────────────────────────────────────
function setDirStatus(el, level) {
    if (!el) return;
    el.textContent = level.text;
    el.className = `status-indicator-badge ${level.cls}`;
}

async function updateDashboardUI() {
    const rows = document.getElementById('transit-breakdowns');
    const unavailable = document.getElementById('transit-unavailable');
    const lastUpdated = document.getElementById('last-updated');
    const meta = checkpointMeta[currentCheckpoint];

    // Tuas has no travel-time feed - say so rather than inventing numbers.
    if (!meta.hasTransitFeed) {
        if (rows) rows.style.display = 'none';
        if (unavailable) {
            unavailable.style.display = 'block';
            unavailable.textContent = `Live travel-time data is not available for ${meta.name}. Camera feeds below are live.`;
        }
        if (lastUpdated) lastUpdated.textContent = '-';
        return;
    }

    if (rows) rows.style.display = '';
    if (unavailable) unavailable.style.display = 'none';

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
            const r = analyseDirection(byKey[d.key], windowStart, anchor);
            const timeEl = document.getElementById(d.timeId);
            const subEl = document.getElementById(d.subId);
            const trendEl = document.getElementById(d.trendId);
            const statusEl = document.getElementById(d.statusId);

            if (!r.ok) {
                if (timeEl) timeEl.textContent = '-';
                if (subEl) subEl.textContent = 'No samples in the last hour';
                setDirStatus(statusEl, { text: 'No Data', cls: 'warning' });
                renderTrend(trendEl, r);
                return;
            }

            // Typical range for the last hour: interquartile band, not the raw spread.
            if (timeEl) timeEl.textContent = `${r.p25.toFixed(0)} - ${r.p75.toFixed(0)} min`;
            if (subEl) {
                subEl.textContent = `1h median ${r.median.toFixed(1)} · now ${r.recentMean.toFixed(1)} · ${r.count} pts`;
            }
            setDirStatus(statusEl, congestionLevel(r));
            renderTrend(trendEl, r);
        });

        if (lastUpdated) {
            const t = new Date(latestTs).toLocaleTimeString('en-GB', { hour12: false, timeZone: 'Asia/Singapore' });
            const ageMin = Math.round((nowMs - latestTs) / 60000);
            lastUpdated.textContent = stale ? `${t} SGT (${ageMin} min old)` : `${t} SGT`;
        }
    } catch (err) {
        console.error('Traffic summary error:', err);
        DIRECTIONS.forEach(d => {
            const timeEl = document.getElementById(d.timeId);
            const subEl = document.getElementById(d.subId);
            if (timeEl) timeEl.textContent = '-';
            if (subEl) subEl.textContent = 'Live data unavailable';
            setDirStatus(document.getElementById(d.statusId), { text: 'Unavailable', cls: 'danger' });
            renderTrend(document.getElementById(d.trendId), { ok: false });
        });
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

        const W = 800, H = 240;
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
                    <stop offset="0%" stop-color="#00f2fe" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="#00f2fe" stop-opacity="0"/>
                </linearGradient>
                <linearGradient id="grad-jb" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#ff9f1c" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="#ff9f1c" stop-opacity="0"/>
                </linearGradient>
            </defs>
            ${gridLines}
            ${fcZone}
            <text x="16" y="${axisMidY}" transform="rotate(-90 16 ${axisMidY})"
                  text-anchor="middle" font-size="11" fill="rgba(148,163,184,0.85)"
                  letter-spacing="0.5">Travel time (min)</text>
            <path d="${areaPath(sgMap)}" fill="url(#grad-sg)"/>
            <path d="${areaPath(jbMap)}" fill="url(#grad-jb)"/>
            ${sgLine ? `<polyline points="${sgLine}" fill="none" stroke="#00f2fe" stroke-width="${lineW}" stroke-linejoin="round" stroke-linecap="round"/>` : ''}
            ${jbLine ? `<polyline points="${jbLine}" fill="none" stroke="#ff9f1c" stroke-width="${lineW}" stroke-linejoin="round" stroke-linecap="round"/>` : ''}
            ${dotR ? allLabels.map((l, i) => sgMap[l] != null ? `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(sgMap[l]).toFixed(1)}" r="${dotR}" fill="#00f2fe" stroke="#070b13" stroke-width="${dotStroke}"/>` : '').join('') : ''}
            ${dotR ? allLabels.map((l, i) => jbMap[l] != null ? `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(jbMap[l]).toFixed(1)}" r="${dotR}" fill="#ff9f1c" stroke="#070b13" stroke-width="${dotStroke}"/>` : '').join('') : ''}
            ${fcExtension(sgMap, fcSg, 'sg')}
            ${fcExtension(jbMap, fcJb, 'jb')}
            ${lastDot(sgMap, '#00f2fe')}
            ${lastDot(jbMap, '#ff9f1c')}
            ${xLabels}
            <line class="lc-crosshair" x1="0" y1="${padT}" x2="0" y2="${padT + plotH}"
                  stroke="rgba(148,163,184,0.55)" stroke-width="1" stroke-dasharray="4 4" style="display:none"/>
            <circle class="lc-hover-dot lc-hd-sg" r="6" fill="none" stroke="#00f2fe" stroke-width="2" style="display:none"/>
            <circle class="lc-hover-dot lc-hd-jb" r="6" fill="none" stroke="#ff9f1c" stroke-width="2" style="display:none"/>
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
            [[sgMap, fcSg, hdSg, 'SG → JB', 'sg'], [jbMap, fcJb, hdJb, 'JB → SG', 'jb']].forEach(([map, fcVal, dot, name, cls]) => {
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
            statSg.innerHTML = `<span class="lc-dot sg"></span><strong>SG → JB</strong>&nbsp;Latest: <strong>${latestSg.mins.toFixed(1)} min</strong> at ${latestSg.label}${nextSg ? ` &rarr; <strong>${nextSg.mins.toFixed(1)}</strong> forecast at ${clockOf(nextSg.at)}` : ''}`;
        }
        if (statJb && latestJb) {
            statJb.innerHTML = `<span class="lc-dot jb"></span><strong>JB → SG</strong>&nbsp;Latest: <strong>${latestJb.mins.toFixed(1)} min</strong> at ${latestJb.label}${nextJb ? ` &rarr; <strong>${nextJb.mins.toFixed(1)}</strong> forecast at ${clockOf(nextJb.at)}` : ''}`;
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
// `?baseline=profile` is the calendar-profile "typical day", not a forecast.
const FORECAST_HOURS = 5;
const FORECAST_MAX_SLOTS = 66;        // 5.5 h in 5-min chart slots
const FORECAST_TIMEOUT_MS = 60000;    // the first curve call can take ~20 s while the models fit
const BASELINE_REFRESH_MS = 3600000;  // the profile is refitted once per SGT day
const FORECAST_DIRECTIONS = [
    { key: 'SG_TO_MY', label: 'SG → JB', cls: 'sg', color: '#00f2fe', series: 'mandai_to_shell_jb' },
    { key: 'MY_TO_SG', label: 'JB → SG', cls: 'jb', color: '#ff9f1c', series: 'jb_to_woodlands' }
];

const FORECAST_ERRORS = {
    400: 'Forecast request was rejected.',
    502: 'Forecast query failed on the server.',
    503: 'Forecast paused: no recent data for this direction.'
};

let forecastView = null;      // { dirs: { KEY: { points, originAt } }, curve }
let baselineDay = null;       // { KEY: [{ t, mins }] }
let baselineAt = 0;
let forecastSpanH = FORECAST_HOURS;

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
        console.error('Baseline fetch error:', err); // the curve's own baseline_min still draws the 5 h view
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

    const tickEvery = (spanH <= 6 ? 1 : 6) * 3600000;
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

        // Label the forecast against this direction's own 24h free-flow level.
        const baseline = obs.length ? percentile(obs.map(p => p.mins), BASELINE_PCTILE) : null;
        const level = congestionLevel({ ok: baseline != null, baseline, recentMean: f.mins });

        // Same stable band as the Traffic Summary card.
        const pct = now > 0 ? (delta / now) * 100 : 0;
        const trend = delta == null ? null
            : pct > TREND_STABLE_PCT ? { cls: 'up', txt: `▲ Rising ${pct.toFixed(0)}%` }
                : pct < -TREND_STABLE_PCT ? { cls: 'down', txt: `▼ Easing ${Math.abs(pct).toFixed(0)}%` }
                    : { cls: 'stable', txt: '▬ Steady' };

        const nowLine = delta == null ? '' :
            `<div class="fc-now">Now <strong>${now.toFixed(1)}</strong> min <span class="fc-delta">(${delta >= 0 ? '+' : ''}${delta.toFixed(1)})</span></div>`;
        const lead = f.lead != null ? ` · ${Math.round(f.lead)} min ahead` : '';

        return `
            <div class="fc-card ${d.cls}">
                <div class="fc-card-head"><span class="lc-dot ${d.cls}"></span>${d.label}</div>
                <div class="fc-main">
                    <span class="fc-value">${f.mins.toFixed(1)}</span>
                    <span class="fc-when">min at ${sgtClock(f.at)} SGT${lead}</span>
                </div>
                ${nowLine}
                <div class="fc-badges">
                    <span class="status-indicator-badge ${level.cls}">${level.text}</span>
                    ${trend ? `<span class="time-trend ${trend.cls}">${trend.txt} vs now</span>` : ''}
                </div>
                ${forecastCompare(d, view, forecastSpanH)}
            </div>`;
    }).join('');

    const withBaseline = forecastView.curve && FORECAST_DIRECTIONS.some(d => forecastView.dirs[d.key].points.length > 1);
    const range = document.getElementById('fc-range');
    if (range) range.style.display = withBaseline && baselineDay ? '' : 'none';
    document.querySelectorAll('#fc-range button').forEach(b =>
        b.classList.toggle('active', +b.dataset.hours === forecastSpanH));

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

// ── Toll Calculator ────────────────────────────────────────────────────────
function updateTollCalculator() {
    const calcRoute = document.getElementById('calc-route');
    const calcVehicle = document.getElementById('calc-vehicle');
    const fuelLevel = document.getElementById('fuel-level');
    const tankGroup = document.getElementById('tank-group');
    const sgTollText = document.getElementById('sg-toll');
    const myTollText = document.getElementById('my-toll');
    const totalCostText = document.getElementById('total-cost');
    const complianceAlert = document.getElementById('compliance-alert');
    const directionEl = document.querySelector('input[name="direction"]:checked');
    if (!calcRoute || !calcVehicle || !directionEl) return;

    const route = calcRoute.value;
    const vehicle = calcVehicle.value;
    const direction = directionEl.value.replace(/-/g, '_'); // HTML uses hyphens; tollFees uses underscores
    const fuelVal = fuelLevel ? parseInt(fuelLevel.value) : 100;

    if (tankGroup) tankGroup.style.display = direction === 'my_to_sg' ? 'none' : 'flex';

    const fees = tollFees[route][vehicle][direction];
    const sgToll = fees.sg;
    const myToll = fees.my;
    const totalSgd = sgToll + (myToll / MYR_PER_SGD);

    if (sgTollText) sgTollText.textContent = `SGD ${sgToll.toFixed(2)}`;
    if (myTollText) myTollText.textContent = `MYR ${myToll.toFixed(2)}`;
    if (totalCostText) totalCostText.textContent = `SGD ${totalSgd.toFixed(2)}`;

    if (complianceAlert) {
        if (direction === 'sg_to_my') {
            complianceAlert.style.display = 'flex';
            if (fuelVal >= 75) {
                complianceAlert.className = 'compliance-alert compliant';
                complianceAlert.innerHTML = `
                    <div class="alert-icon">✅</div>
                    <div class="alert-content">
                        <p class="alert-title">Fuel Gauge Compliant (${fuelVal}%)</p>
                        <p class="alert-desc">Your fuel level is above 75%. You comply with the Singapore 3/4 tank rule.</p>
                    </div>`;
            } else {
                complianceAlert.className = 'compliance-alert warning';
                complianceAlert.innerHTML = `
                    <div class="alert-icon">⚠️</div>
                    <div class="alert-content">
                        <p class="alert-title">Non-Compliant Alert (${fuelVal}%)</p>
                        <p class="alert-desc">Your tank is below 75%. Leaving Singapore with less than 3/4 tank is an offense (fine up to SGD 500).</p>
                    </div>`;
            }
        } else {
            complianceAlert.style.display = 'none';
        }
    }
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateDashboardUI();
    fetchAndRenderCongestionChart();
    fetchForecast();
    updateTollCalculator();
    fetchBackendMessage();
    fetchLiveCameras();

    cameraRefreshTimer = setInterval(fetchLiveCameras, 60000);
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

    document.querySelectorAll('#fc-range button').forEach(btn => {
        btn.addEventListener('click', () => {
            forecastSpanH = +btn.dataset.hours;
            renderForecastCards();
        });
    });

    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentCheckpoint = btn.dataset.checkpoint;
            const calcRoute = document.getElementById('calc-route');
            if (calcRoute) calcRoute.value = currentCheckpoint;
            updateDashboardUI();
            updateTollCalculator();
            fetchLiveCameras();
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
            } catch (err) {
                console.error('Dashboard refresh error:', err);
                await updateDashboardUI();
            } finally {
                setTimeout(() => { refreshBtn.disabled = false; }, 3000);
            }
        });
    }

    document.getElementsByName('direction').forEach(input => input.addEventListener('change', updateTollCalculator));
    const calcRoute = document.getElementById('calc-route');
    const calcVehicle = document.getElementById('calc-vehicle');
    const fuelLevel = document.getElementById('fuel-level');
    if (calcRoute) calcRoute.addEventListener('change', updateTollCalculator);
    if (calcVehicle) calcVehicle.addEventListener('change', updateTollCalculator);
    if (fuelLevel) fuelLevel.addEventListener('input', updateTollCalculator);
});
