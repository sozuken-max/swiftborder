# Delta analysis: forecast widget in `hosting/app.js` and `hosting/index.html`

Baseline: the `app.js` and `index.html` uploaded on 2026-10-03 (the live Hosting release, per ADR 0002).
Result: `hosting/app.js` and `hosting/index.html` on branch `claude/vigilant-cerf-eexe6f`.

## Summary

| File | Added | Removed | Changed |
| --- | --- | --- | --- |
| `app.js` | 106 | 0 | 0 |
| `index.html` | 21 | 0 | 0 |
| `style.css` | not part of this change (no new classes; the card reuses existing ones) | | |

No existing function was modified. `fetchBackendMessage`, `BACKEND_URL` and the AI detection code are untouched.

## `app.js` changes

| # | Location | Change |
| --- | --- | --- |
| 1 | After `TRAFFIC_API` | New constant `FORECAST_API` (`https://forecast-api-1095552466513.asia-southeast1.run.app/`) |
| 2 | Before "Toll Calculator" | New block: `FORECAST_DIRECTIONS`, `FORECAST_ERRORS`, `sgtClock`, `setForecastStatus`, `fetchForecast` |
| 3 | `DOMContentLoaded` | `fetchForecast()` on page load |
| 4 | `chartRefreshTimer` (5 min) | `fetchForecast()` added to the tick |
| 5 | `DOMContentLoaded` | Click handler for `#refresh-forecast`, same 3 s disable pattern as the other refresh buttons |

### Reused, not modified

- `loadTrafficData` (shared cache) supplies the latest observed value, so there is no extra request for it.
- `seriesPoints` reads that value for each direction.

### Data contract used

`GET FORECAST_API` with no parameters returns the `served` model for both directions (runbook: `docs/runbooks/forecast-api.md`).
Fields read: `directions.{SG_TO_MY,MY_TO_SG}.{forecast_min, forecast_for, origin_ts, model}`.

### Error handling

| Condition | UI |
| --- | --- |
| 401 / 403 | "Forecast service requires authentication and is not public yet." |
| 502 | "Forecast query failed on the server." |
| 503 | "No recent data for this direction yet." |
| `TypeError` (CORS block or network) | "Forecast service unreachable (blocked by CORS or not public)." |
| Direction missing in response | Card shows "no forecast available" |

## `index.html` changes

One new card, `#forecast-widget`, inside `#predictions` directly after the Live Congestion chart card.
Element ids: `fc-updated`, `fc-status`, `fc-cards`, `fc-meta`, `refresh-forecast`.
Reuses existing classes: `live-chart-card`, `dashboard-card`, `lc-header`, `lc-title-group`, `lc-badge`, `ai-dir-grid`, `ai-dir-card`, `ai-dir-note`, `cam-grid-footer`, `refresh-btn`.

## Risks and open items

1. **Service is private.** The runbook says callers need `roles/run.invoker` and a browser cannot attach an identity token, so the widget shows the 401/403 message until the service is public.
2. **CORS.** The runbook does not say the service allows `https://swiftborder-92b45.web.app` or `.firebaseapp.com`.
3. **Honesty label.** The card states the value forecasts the Google Maps estimate, not a measured crossing time. No MAE or "beats Google" claim is made.
4. **`served` model is interim.** `lin_bq[frozen]` for SG to JB and `persistence` for JB to SG until the frozen run on or after 20 Oct.
5. **Not run end to end.** `node --check` passes on `app.js`. The widget has not been run against the live service or with your `style.css`.
6. **ADR drift.** ADR 0001 says the cards do not call `forecastapi/`; that stops being true on deploy.
7. **ADR 0002.** `hosting/` still lacks `style.css`, `firebase.json` and `.firebaserc`. Deploy stays manual.

## Unrelated: AI Vehicle Detection 404

Not caused by this change. On 2026-10-03 ~22:17 SGT, `swiftbackend` returned `404 {"error": "No image found for camera 2701 ..."}`, and the LTA feed listed only cameras 2702 and 2704 of the six Woodlands cameras.

## Full diff: `app.js`

```diff
@@ -55,4 +55,8 @@ const LTA_API_URL = 'https://api.data.gov.sg/v1/transport/traffic-images';
 const BACKEND_URL = 'https://swiftbackend-1095552466513.europe-west1.run.app/';
 const TRAFFIC_API = 'https://storage.googleapis.com/swiftborder-public/traffic-24h.json';
+// Cloud Run service endpoint (docs/runbooks/forecast-api.md). The service is private, so a
+// browser can only call it once it is public (roles/run.invoker for allUsers) and allows
+// this origin via CORS.
+const FORECAST_API = 'https://forecast-api-1095552466513.asia-southeast1.run.app/';
 
 // ── Backend AI Detection Image ─────────────────────────────────────────────
@@ -757,4 +761,96 @@ async function fetchAndRenderCongestionChart() {
 }
 
+// ── 30-min Forecast Widget ─────────────────────────────────────────────────
+// Reads forecast-api with no parameters, which returns the `served` model for
+// both directions. Responses are cached server-side for 5 minutes, so this
+// shares the chart's 5-minute timer. The value forecasts Google Maps'
+// duration_in_traffic, not a measured crossing time.
+const FORECAST_DIRECTIONS = [
+    { key: 'SG_TO_MY', label: 'SG → JB', cls: 'sg', series: 'mandai_to_shell_jb' },
+    { key: 'MY_TO_SG', label: 'JB → SG', cls: 'jb', series: 'jb_to_woodlands' }
+];
+
+const FORECAST_ERRORS = {
+    401: 'Forecast service requires authentication and is not public yet.',
+    403: 'Forecast service requires authentication and is not public yet.',
+    502: 'Forecast query failed on the server.',
+    503: 'No recent data for this direction yet.'
+};
+
+function sgtClock(iso) {
+    const t = Date.parse(iso);
+    if (Number.isNaN(t)) return '—';
+    return new Date(t).toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Singapore' });
+}
+
+function setForecastStatus(text) {
+    const el = document.getElementById('fc-status');
+    if (!el) return;
+    el.textContent = text;
+    el.style.display = text ? '' : 'none';
+}
+
+async function fetchForecast() {
+    const cards = document.getElementById('fc-cards');
+    const updatedEl = document.getElementById('fc-updated');
+    const metaEl = document.getElementById('fc-meta');
+    if (!cards) return;
+
+    setForecastStatus('Fetching forecast…');
+
+    try {
+        const res = await fetch(FORECAST_API);
+        if (!res.ok) throw new Error(FORECAST_ERRORS[res.status] || `HTTP ${res.status}`);
+        const data = await res.json();
+        const dirs = data.directions || {};
+
+        // Latest observed value per direction, from the feed the chart already loaded.
+        let traffic = null;
+        try { traffic = await loadTrafficData(); } catch (_) { /* "now" is optional */ }
+
+        cards.innerHTML = FORECAST_DIRECTIONS.map(d => {
+            const f = dirs[d.key];
+            if (!f || f.forecast_min == null) {
+                return `
+                    <div class="ai-dir-card ${d.cls}">
+                        <span class="ai-dir-label"><span class="lc-dot ${d.cls}"></span>${d.label}</span>
+                        <span class="ai-dir-count">—</span>
+                        <span class="ai-dir-unit">no forecast available</span>
+                    </div>`;
+            }
+            const pts = traffic ? seriesPoints(traffic, d.series) : [];
+            const now = pts.length ? pts[pts.length - 1].mins : null;
+            const delta = now == null ? null : f.forecast_min - now;
+            const nowTxt = delta == null
+                ? ''
+                : ` · now ${now.toFixed(1)} (${delta >= 0 ? '+' : ''}${delta.toFixed(1)})`;
+            return `
+                <div class="ai-dir-card ${d.cls}">
+                    <span class="ai-dir-label"><span class="lc-dot ${d.cls}"></span>${d.label}</span>
+                    <span class="ai-dir-count">${f.forecast_min.toFixed(1)}</span>
+                    <span class="ai-dir-unit">min at ${sgtClock(f.forecast_for)} SGT${nowTxt}</span>
+                </div>`;
+        }).join('');
+
+        setForecastStatus('');
+        const origin = FORECAST_DIRECTIONS.map(d => dirs[d.key]?.origin_ts).filter(Boolean).sort().pop();
+        if (updatedEl) updatedEl.textContent = origin ? `Based on ${sgtClock(origin)} SGT data` : 'Updated';
+        if (metaEl) {
+            const models = FORECAST_DIRECTIONS.map(d => `${d.label}: ${dirs[d.key]?.model || '—'}`).join(' · ');
+            metaEl.textContent = `Model — ${models}`;
+        }
+    } catch (err) {
+        // A CORS block surfaces as a bare TypeError with no status, so name it.
+        const msg = err instanceof TypeError
+            ? 'Forecast service unreachable (blocked by CORS or not public).'
+            : err.message;
+        cards.innerHTML = '';
+        setForecastStatus(`Forecast unavailable: ${msg}`);
+        if (updatedEl) updatedEl.textContent = 'Unavailable';
+        if (metaEl) metaEl.textContent = '';
+        console.error('Forecast fetch error:', err);
+    }
+}
+
 // ── Toll Calculator ────────────────────────────────────────────────────────
 function updateTollCalculator() {
@@ -816,4 +912,5 @@ document.addEventListener('DOMContentLoaded', () => {
     updateDashboardUI();
     fetchAndRenderCongestionChart();
+    fetchForecast();
     updateTollCalculator();
     fetchBackendMessage();
@@ -824,4 +921,5 @@ document.addEventListener('DOMContentLoaded', () => {
         fetchAndRenderCongestionChart();
         updateDashboardUI();
+        fetchForecast();
     }, 300000);
 
@@ -842,4 +940,12 @@ document.addEventListener('DOMContentLoaded', () => {
     }
 
+    const refreshForecastBtn = document.getElementById('refresh-forecast');
+    if (refreshForecastBtn) {
+        refreshForecastBtn.addEventListener('click', () => {
+            refreshForecastBtn.disabled = true;
+            fetchForecast().finally(() => setTimeout(() => { refreshForecastBtn.disabled = false; }, 3000));
+        });
+    }
+
     const tabBtns = document.querySelectorAll('.tab-btn');
     tabBtns.forEach(btn => {
```

## Full diff: `index.html`

```diff
@@ -183,4 +183,25 @@
                 </div>
             </div>
+
+            <!-- 30-minute forecast (forecast-api) -->
+            <div class="live-chart-card dashboard-card" id="forecast-widget" style="margin-top: 1.5rem;">
+                <div class="lc-header">
+                    <div class="lc-title-group">
+                        <h3>Woodlands Causeway &mdash; 30-min Forecast (min)</h3>
+                        <span class="lc-badge" id="fc-updated">Loading&hellip;</span>
+                    </div>
+                </div>
+
+                <p class="ai-dir-note" id="fc-status">Fetching forecast&hellip;</p>
+                <div class="ai-dir-grid" id="fc-cards"></div>
+
+                <p class="ai-dir-note">Forecast of the Google Maps travel-time estimate 30 minutes ahead, not a
+                    measured crossing time.</p>
+
+                <div class="cam-grid-footer">
+                    <span id="fc-meta"></span>
+                    <button class="refresh-btn" id="refresh-forecast">&#128260; Refresh Forecast</button>
+                </div>
+            </div>
         </div>
     </section>
```

## Revision 2 (2026-10-04): forecast curve, baseline, public API

The first revision above called `forecast-api` with no parameters and drew one 30-minute value. The API has since gained a forecast curve and a typical-day baseline ([runbook](runbooks/forecast-api.md)), and the service is public. `hosting/` now does the following.

| Area | Change |
| --- | --- |
| Live Congestion chart | Forecast only: each direction continues as a dashed line for the next 5 h (`?curve=forecast&hours=5`, 10 points, all model points; the runbook's model range ends at 5.5 h). No baseline lines on this chart. Hover snaps to the nearest forecast point and says "exploratory" beyond 30 min. |
| Forecast card | Renamed "Forecast vs Typical Day". Hero value is the first (30-min, evaluated) point with `lead_min`. Each direction plots the forecast against the calendar-profile baseline, with a "Next 5 h / Full day" toggle (`?baseline=profile&hours=24`, refreshed hourly). |
| Failure handling | 60 s timeout on the curve call. If it fails, the page falls back to the 30-minute `served` forecast and says so. If both fail, the chart returns to observed data only. The old 401/403 "not public" and CORS messages are gone. |
| Page copy | By product decision the page carries no accuracy caveats: a single line, "Based on Google Maps travel-time estimates", and the baseline is labelled "Typical for this day and time". The runbook's status (only 30 min evaluated, later points exploratory, not confirmed on Run B) is unchanged and still applies to any claim made elsewhere. |

Cost to keep in mind: the first `hours=5` curve call fits about nine models per instance per SGT day, then is cached for 5 minutes. A cold call can take around 20 s.
