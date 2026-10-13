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

There is no Cloud Build trigger for this folder yet; the live service is still deployed from the
console. To deploy this folder by hand (keeps existing env vars and secrets):

```bash
gcloud run deploy gmap-woodlands-fetcher --project=swiftborder --region=asia-southeast1 \
  --source=gmapfetcher --function=fetchTravelTimes --base-image=nodejs22
```

Run locally: `npm install && GOOGLE_MAPS_API_KEY=... npm start`, then `curl localhost:8080`
(this writes to the real BigQuery table and bucket under your ADC credentials).
