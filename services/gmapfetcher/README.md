# gmapfetcher — Cloud Run `gmap-woodlands-fetcher`

Source of the Cloud Run service `gmap-woodlands-fetcher` (`asia-southeast1`), copied from the
console **Source** tab on 2026-10-13. Node.js 22, function entry point `fetchTravelTimes`.

Every run (Cloud Scheduler `Gmap-Woodlands`; every 5 minutes until 2026-10-13, then every 10; this code
expects every 20, set by `FETCH_INTERVAL_MIN`):

1. Calls the Google Maps Distance Matrix API (`traffic_model: best_guess`) for two Woodlands routes:
   `mandai_to_shell_jb` (`SG_TO_MY`) and `jb_to_woodlands` (`MY_TO_SG`).
2. Streams one row per route into BigQuery `causeway.travel_times` (failed calls are kept with a
   non-`OK` `status`), plus the carried-forward rows below.
3. Rebuilds the last-24 h snapshot `gs://swiftborder-public/traffic-24h.json` that `hosting/` reads.
   A snapshot failure does not fail the run.

If `TOMTOM_ENABLED` is on, the same run does steps 1-3 a second time with TomTom (below), into its own
table and snapshot.

## TomTom (trial, runs beside Google)

TomTom's [Routing API](https://developer.tomtom.com/routing-api/documentation/tomtom-maps/calculate-route)
`calculateRoute` (`traffic=true`, `travelMode=car`, `departAt=now`) is called for the same two routes
and coordinates. `GMAPS_ENABLED` and `TOMTOM_ENABLED` switch each provider on or off on its own; the
defaults (Google on, TomTom off) keep the service exactly as before. A provider whose call or insert
fails does not stop the other; the run then returns 500.

- **Table:** `causeway.travel_times_tomtom` (`TOMTOM_BQ_TABLE`), same columns as `travel_times` plus
  `provider` (`'tomtom'`). It is not `travel_times` on purpose: `v_bins_10min` averages every `OK` row
  in a route's 10-minute bin, so shared rows would silently blend the two providers into
  `v_training_set`, the BQML models and `forecast-api`. Google rows are unchanged and have no `provider`
  column (absent/null = Google). To compare, `UNION ALL` the two tables on `route_id` and the 10-minute bin.
- **Columns:** `duration_in_traffic_sec` = `travelTimeInSeconds`; `duration_sec` =
  `noTrafficTravelTimeInSeconds` (free flow). Google's `duration` is a typical-traffic time, so
  `congestion_ratio` is **not** comparable across providers; compare `duration_in_traffic_sec`.
  Failed calls are kept as `HTTP_<code>`, `NO_ROUTE` or `ERROR` rows. Carried-forward rows work as for Google.
- **Snapshot:** `gs://swiftborder-public/traffic-24h-tomtom.json` (`TOMTOM_SNAPSHOT_OBJECT`), same format.
  The website still reads only `traffic-24h.json`.
- **Same road as Google:** `jb_to_woodlands` sends `vehicleHeading=198`. Without it TomTom started on the
  northbound carriageway and U-turned (8,453 m vs Google's 6,639 m); with it, 6,610 m on 2026-10-30.
  TomTom rows for that route written before 1.4.1 took the longer road (`distance_m` about 8,450): drop
  them from comparisons. `mandai_to_shell_jb` already matched (6,980 m vs 7,046 m) and has no heading.
- **Not measured:** whether TomTom's times track Google's, or the crossing, on these routes. Run both
  until a comparison exists before pointing anything downstream at TomTom.

One-time setup before turning it on (the table must exist, or every TomTom insert fails):

```bash
bq show --schema --format=json swiftborder:causeway.travel_times \
  | jq '. + [{"name":"provider","type":"STRING","mode":"NULLABLE"}]' > /tmp/tt_tomtom_schema.json
bq mk --table --location=US --time_partitioning_field=observed_date_sgt --time_partitioning_type=DAY \
  --clustering_fields=route_id swiftborder:causeway.travel_times_tomtom /tmp/tt_tomtom_schema.json
```

Keep the key in Secret Manager and expose it to the service as `TOMTOM_API_KEY`. The runtime identity is
the default compute service account (see [inventory](../../docs/inventory.md)):

```bash
printf '%s' "$TOMTOM_KEY" | gcloud secrets create tomtom-api-key --project=swiftborder --data-file=-
gcloud secrets add-iam-policy-binding tomtom-api-key --project=swiftborder \
  --member=serviceAccount:1095552466513-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
gcloud run services update gmap-woodlands-fetcher --project=swiftborder --region=asia-southeast1 \
  --update-secrets=TOMTOM_API_KEY=tomtom-api-key:latest --update-env-vars=TOMTOM_ENABLED=true
```

Turn either provider off with `--update-env-vars=GMAPS_ENABLED=false` (or `TOMTOM_ENABLED=false`). Each
change makes a new revision; a misspelled value fails that revision at startup and the old one keeps
serving. The Cloud Build deploy (`services update --image`) leaves env vars and secrets alone.

## 20-minute fetch, 10-minute bins, 5-minute snapshot

The fetcher runs every `FETCH_INTERVAL_MIN` minutes (default 20, a multiple of 10) to stay within the
Distance Matrix free tier. Everything downstream still expects **one row per route in every 10-minute
bin**: `v_bins_10min`, the positional `LAG`/`LEAD` in `v_training_set`, and the time-based lags and
labels in `forecast-api` (`y_30` is three bins ahead, so with 20-minute rows only it would always land on
an empty bin and no 30-minute model could be fitted).

So each run at T also writes, for every `OK` route, a **carried-forward** copy of the same reading at
T+10 (T+10 and T+20 at 30 minutes; none at 10). It is a normal `status = 'OK'` row with
`error_message = 'carried_forward from <T>'`. It uses only the reading at T, so no bin ever holds a later
observation, and it is in BigQuery before its bin opens. A failed call is not carried, so a failed run
leaves its bins empty, as before. To drop carried rows from a query:
`NOT STARTS_WITH(IFNULL(error_message, ''), 'carried_forward')`.

Caveat for evaluation: from the switch on, half the 10-minute bins (and half the `y_30` labels) are
copies of the previous bin. Persistence is exactly right on those bins, so MAE measured on data after
the switch is not comparable with the 5/10-minute period unless carried rows are filtered out.

The JSON snapshot skips carried rows and keeps the website's 5-minute grid: between two real points
whose gap is within 1 minute of 10, 15 or 20 minutes (up to `FETCH_INTERVAL_MIN`), `snapshot.js` adds
straight-line points every 5 minutes (duration rounded to whole seconds, speed to 2 dp). Each point is
`[datetime_sgt, duration_in_traffic_sec, speed_kmh, interpolated]`; the newest point is always real.
5-minute history and longer gaps are left as they are.

Interpolated points are deliberately **not** written to BigQuery: a 12:05 point is computed from the
12:20 reading, so it would leak a later observation into an earlier bin and change that bin after
`forecast-api` had already read it.

Tests: `npm test` (Node's built-in runner, no install needed for `test/`).

## Configuration (names only; values live in Cloud Run)

| Env var | Default |
| --- | --- |
| `GOOGLE_MAPS_API_KEY` | none, required while `GMAPS_ENABLED` |
| `TOMTOM_API_KEY` | none, required while `TOMTOM_ENABLED` (Secret Manager `tomtom-api-key`) |
| `GMAPS_ENABLED` / `TOMTOM_ENABLED` | `true` / `false` (`true/false`, `1/0`, `yes/no`, `on/off`) |
| `TOMTOM_BQ_TABLE` | `travel_times_tomtom` (must differ from `BQ_TABLE`) |
| `TOMTOM_SNAPSHOT_OBJECT` | `traffic-24h-tomtom.json` (must differ from `SNAPSHOT_OBJECT`) |
| `FETCH_INTERVAL_MIN` | `20` (must match the scheduler) |
| `BQ_DATASET` / `BQ_TABLE` | `causeway` / `travel_times` |
| `BQ_LOCATION` | `US` |
| `SNAPSHOT_BUCKET` / `SNAPSHOT_OBJECT` | `swiftborder-public` / `traffic-24h.json` |

## Deploy

Cloud Build trigger `rmgpgab-gmap-woodlands-fetcher-asia-southeast1-sozuken-max-szsd`
(`33e4b77e-dbce-426e-a701-de234ec16d60`, global), created on 2026-10-13 and switched on 2026-10-17 to
[`cloudbuild.yaml`](cloudbuild.yaml). Push to `^main$` with included files `services/gmapfetcher/**`
(ignoring `README.md` and `test/**`):
step `Test` runs `node --test`, `Buildpack` builds `--path=services/gmapfetcher` with function target
`fetchTravelTimes`, and `Deploy` runs `gcloud run services update` with the new image. Env vars and
other service settings carry over. `engines.node` in `package.json` pins Node.js 22; without it the
buildpack picks the latest Node.

First build `9f1684b1-e1d3-4a94-9e16-082aa3009bf9` (commit `5ed2f4d`) succeeded on
2026-10-13 12:25 SGT; the 12:30 and 12:35 SGT runs wrote `OK` rows for both routes and
refreshed `traffic-24h.json`.

Do not edit the code in the console **Source** tab: the next build from `main` overwrites it.

Run locally: `npm install && GOOGLE_MAPS_API_KEY=... npm start`, then `curl localhost:8080`
(this writes to the real BigQuery table and bucket under your ADC credentials).
