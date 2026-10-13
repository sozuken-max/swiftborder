# gmapfetcher — Cloud Run `gmap-woodlands-fetcher`

Source of the Cloud Run service `gmap-woodlands-fetcher` (`asia-southeast1`), copied from the
console **Source** tab on 2026-10-13. Node.js 22, function entry point `fetchTravelTimes`.

Every run (Cloud Scheduler `Gmap-Woodlands`, every 5 minutes):

1. Calls the Google Maps Distance Matrix API (`traffic_model: best_guess`) for two Woodlands routes:
   `mandai_to_shell_jb` (`SG_TO_MY`) and `jb_to_woodlands` (`MY_TO_SG`).
2. Streams one row per route into BigQuery `causeway.travel_times` (failed calls are kept with a
   non-`OK` `status`).
3. Rebuilds the last-24 h snapshot `gs://swiftborder-public/traffic-24h.json` that `hosting/` reads.
   A snapshot failure does not fail the run.

## Configuration (names only; values live in Cloud Run)

| Env var | Default |
| --- | --- |
| `GOOGLE_MAPS_API_KEY` | none, required |
| `BQ_DATASET` / `BQ_TABLE` | `causeway` / `travel_times` |
| `BQ_LOCATION` | `US` |
| `SNAPSHOT_BUCKET` / `SNAPSHOT_OBJECT` | `swiftborder-public` / `traffic-24h.json` |

## Deploy

Cloud Build trigger `rmgpgab-gmap-woodlands-fetcher-asia-southeast1-sozuken-max-szsd`
(`33e4b77e-dbce-426e-a701-de234ec16d60`, global), created on 2026-10-13 by the Cloud Run
**Connect to repo** flow. Push to `^main$` with included files `gmapfetcher/**`; inline config:
Google Cloud buildpacks on `gmapfetcher/`, function target `fetchTravelTimes`, then
`gcloud run services update` with the new image. Env vars and other service settings carry over.
There is no test step yet. `engines.node` in `package.json` pins Node.js 22; without it the
buildpack picks the latest Node.

First build `9f1684b1-e1d3-4a94-9e16-082aa3009bf9` (commit `5ed2f4d`) succeeded on
2026-10-13 12:25 SGT; the 12:30 and 12:35 SGT runs wrote `OK` rows for both routes and
refreshed `traffic-24h.json`.

Do not edit the code in the console **Source** tab: the next build from `main` overwrites it.

Run locally: `npm install && GOOGLE_MAPS_API_KEY=... npm start`, then `curl localhost:8080`
(this writes to the real BigQuery table and bucket under your ADC credentials).
