# Serve Woodlands travel times from the public JSON file

## Status

Proposed

## Context

The Firebase client (`C:\Users\vi_ci\Downloads\app.js`, outside this repo) already has three external reads:

| Function | Call | What the UI does with it |
| --- | --- | --- |
| `fetchBackendMessage` | `GET` `BACKEND_URL` (`swiftbackend`) with `format=directional` and `camera_id=2701` | Annotated JPEG body plus `X-Vehicle-Count-*` / `X-Congestion-*` headers for camera 2701 |
| `fetchLiveCameras` | `GET` `https://api.data.gov.sg/v1/transport/traffic-images` | Filters the LTA camera list to the active checkpoint and draws the grid |
| `loadTrafficData` | `GET` `TRAFFIC_API` = `https://storage.googleapis.com/swiftborder-public/traffic-24h.json` | One JSON payload for both Woodlands directions. `updateDashboardUI` and `fetchAndRenderCongestionChart` both call it |

`loadTrafficData` keeps a 60-second memory cache and collapses concurrent callers onto `trafficCache.inFlight`. On load, `updateDashboardUI` and `fetchAndRenderCongestionChart` therefore share one HTTP request. The chart timer (`chartRefreshTimer`, 300000 ms) calls both again; the memory cache has expired, and `inFlight` still makes that tick a single download. The refresh button calls `loadTrafficData(true)` once, then both consumers hit the fresh cache.

There is no per-direction HTTP call. `DIRECTIONS` names the series `mandai_to_shell_jb` (SG to JB) and `jb_to_woodlands` (JB to SG). `seriesPoints` reads `data.series[name].points`, uses `p[0]` as an SGT timestamp and `p[1] / 60` as minutes, and ignores a third element. The chart also reads `data.updated_at_sgt`. Tuas sets `hasTransitFeed: false` and does not fetch this file.

Observed object at **2026-10-03 10:45 SGT** (18,478 bytes, `Cache-Control: public, max-age=300`):

```json
{
  "point_format": ["datetime_sgt", "duration_in_traffic_sec", "speed_kmh"],
  "window_hours": 24,
  "updated_at_sgt": "2026-10-03T10:45:03",
  "series": {
    "mandai_to_shell_jb": {
      "name": "Mandai Rd to Shell after Woodlands checkpoint",
      "direction": "SG_TO_MY",
      "points": [["2026-10-03T10:45", 2528, 10.11]]
    },
    "jb_to_woodlands": {
      "name": "JB approach to Woodlands (post-checkpoint)",
      "direction": "MY_TO_SG",
      "points": [["2026-10-03T10:45", 1503, 15.9]]
    }
  }
}
```

Each series had **288** points (24 hours at 5 minutes). `p[1]` is `duration_in_traffic_sec`. `p[2]` is `speed_kmh`. The UI never reads `point_format`, `window_hours`, `name`, `direction`, or `speed_kmh`.

Cloud Scheduler job `Gmap-Woodlands` is `*/5 * * * *` in `Asia/Singapore`, state `ENABLED`, HTTP GET to Cloud Run `gmap-woodlands-fetcher` (`asia-southeast1`, revision `gmap-woodlands-fetcher-00004-msb`). The container source is not in this repo. This pass did not read it and did not print env values; the service has an env var named `GOOGLE_MAPS_API_KEY`. The object's `Last-Modified` stepped from 10:40:05 to 10:45:05 SGT. Scheduler `status` came back empty, so `lastAttemptTime` was not read.

BigQuery, same morning, read-only:

- `causeway.travel_times` (location `US`, partitioned on `observed_date_sgt`, clustered on `route_id`). Table metadata ~10:48 SGT: **15,760** rows, **2,742,224** logical bytes, streaming buffer estimated 4 rows.
- A query of `observed_date_sgt` from the previous Singapore date, `status = 'OK'`, both Woodlands `route_id`s, processed **32,032** bytes (dry run) and returned **418** rows per route with `MAX(observed_at)` **2026-10-03 02:45:03 UTC** (10:45:03 SGT), the same instant as `updated_at_sgt`.
- `traffic_prediction.v_forecast_recent` is a view. Its checked-in SQL runs `ML.PREDICT` on `lin_h30`. This pass listed the view and did not execute it. `app.js` does not read `forecast_30min_min`.

Bucket `swiftborder-public` CORS allows `GET` and `HEAD` from `https://swiftborder-92b45.web.app`, `https://swiftborder-92b45.firebaseapp.com`, and `http://localhost:5000`. A GET with `Origin: https://swiftborder-92b45.web.app` returned `Access-Control-Allow-Origin` for that origin. That hosting URL returned HTTP 200. A simple `fetch` of the JSON sends no custom headers, so the browser does not need a preflight. OPTIONS for `http://localhost:5000` returned 200 with `Access-Control-Allow-Methods: GET,HEAD`.

`camdetect` `detect` stays the camera call. Its body is a JPEG (or detection JSON). Mixing travel-time JSON into that response would change the image contract `fetchBackendMessage` reads.

## Decision

The callable travel-time API is the public object the client already fetches:

`GET https://storage.googleapis.com/swiftborder-public/traffic-24h.json`

No new Cloud Run handler **for these cards**. No request-time BigQuery query on the browser tick. (`forecastapi/` is a separate, private endpoint for the 30-minute forecast, which this file does not contain; the cards do not call it.)

| | Public GCS object | Direct BigQuery on each refresh |
| --- | --- | --- |
| Freshness | File and `travel_times` both moved at 10:45:03 SGT. A query at request time cannot see a Maps sample newer than the last fetcher run. | Same 5-minute sample. The streaming buffer is the in-flight insert from that run, not a finer cadence. |
| Latency | 18,478 bytes in `asia-southeast1`, reusable for 300 seconds. Not benchmarked here. | Query job runs in `US`. Adds job startup on top of the object GET. Not benchmarked here. |
| Cost | One object GET. `max-age=300` matches the scheduler, so a refresh inside five minutes can be served from cache. | The pruned aggregate above processed 32,032 bytes. On-demand billing still rounds each query up to 10 MiB, once per browser that misses cache. |
| CORS | Already allows the live Hosting origins and the Firebase emulator. | Needs a Cloud Run service with the `camdetect` CORS habit. Nothing like that is deployed for this payload. |
| Payload | One body, both directions, which is what `seriesPoints` and the chart read. | Would rebuild that same JSON, or return `v_forecast_recent` fields the UI ignores. |

Direct BigQuery is the path to add later only for a value this file does not contain. The concrete case is `v_forecast_recent` (`forecast_30min_min`, `serving_model`): a 30-minute forecast of Maps `duration_in_traffic`, `lin_h30` or persistence. That view runs `ML.PREDICT`. If the UI grows a control that reads it, a server should cache the result for at least the 5-minute fetcher interval. The browser should not start that query on every tick. That handler is specified in [../runbooks/forecast-api.md](../runbooks/forecast-api.md) and is not deployed.

## Consequences

- Chad leaves `TRAFFIC_API` on `https://storage.googleapis.com/swiftborder-public/traffic-24h.json`. No URL edit in `app.js`. `BACKEND_URL` stays the camera `detect` service.
- A new Hosting origin needs a bucket CORS entry. This pass did not change CORS.
- This pass did not add a handler, did not deploy, and did not run a new pytest module. `detect` is unchanged.
- `app.js` stays in Downloads. Client recommendations, not implemented here:
  - Keep `chartRefreshTimer` at 5 minutes. A faster poll re-downloads a file the fetcher has not replaced (`max-age=300` may hide that, and the 60-second memory cache would not).
  - Keep one `loadTrafficData` call for both series. Do not split directions into two fetches.
  - `fetchLiveCameras` has no cache. It runs on load, every 60 seconds, on `#refresh-cameras`, and on every checkpoint tab click, and each call downloads the full LTA list before filtering. Cache that JSON for about 60 seconds and filter `CHECKPOINT_CAMERAS` locally.
  - `fetchBackendMessage` runs on load and on `#refresh-ai-detection` only. Leave it off the 60-second camera timer. It is a billed Roboflow call and an image response, so it cannot replace `loadTrafficData`.

## What this pass did not verify

- The fetcher container's source, so the write to the object and to `travel_times` is inferred from Chad's description plus the matching 10:45:03 timestamps, not from code in git.
- Scheduler `lastAttemptTime`.
- A fresh `gcloud services list` of Firebase APIs. Hosting did respond at `https://swiftborder-92b45.web.app`. Re-checked in [0002](0002-firebase-hosting-source.md): both Firebase APIs are disabled on project `swiftborder`.
- End-user latency numbers.
- A live `ML.PREDICT` from `v_forecast_recent`.

## Related

- [0002-firebase-hosting-source.md](0002-firebase-hosting-source.md): where the Hosting client lives, and that its deploy stays manual.
