# GCP inventory (dated copy)

**Observed:** 2026-10-01 ~00:40 SGT (2026-09-30 16:38 UTC), read-only `gcloud` / `bq` queries of project `swiftborder`, plus model, view and horizon checks from the 2026-09-30 ~21:35 SGT pass. **One change since:** the weather tables were appended on 2026-10-01 ~01:50 SGT (requested by the team; see `rainfall` / `weatherforecast` below). No other resources or settings were changed. Camera and image date ranges, and `model_registry`, were re-queried read-only on 2026-10-01 ~03:30 SGT. **2026-10-03 ~10:48 SGT** re-checked `causeway.travel_times` metadata, `traffic-24h.json`, bucket CORS, Hosting, and the Maps scheduler (see those sections). **2026-10-03 ~11:20 SGT** re-checked `traffic_prediction` models, `model_registry`, and `v_forecast_recent`, and listed Cloud Run services (see those sections). No forecast HTTP service exists. When this file and the project disagree, the project wins; re-query and update this file.

## Project metadata

- Display name: **Swiftborder**
- Project ID: `swiftborder`
- Project number: `1095552466513`

## BigQuery

Seven datasets: `cam2701`, `cam2702`, `causeway`, `rainfall`, `traffic_images`, `traffic_prediction`, `weatherforecast`.

Locations: `cam2701`, `cam2702`, `rainfall`, `traffic_images`, `weatherforecast` are `asia-southeast1`. `causeway` and `traffic_prediction` are `US`. A query cannot join across the two locations, which is why the weather/camera join is done offline in `eval/`.

### `cam2701`

- **Table `Cam2701`** -- last modified **2026-07-18 10:54 SGT**; **298,137 rows**. One row per detection (`filename`, `date`, `time`, `label`, `direction`, `confidence`, boxes). **Frames dated 2026-03-13 to 2026-04-22 (33 days, 3,513 frames)**, queried 2026-10-01. No overlap with `causeway.travel_times`. Not receiving new rows.
- **View `v_congestion_index_10min`** -- 10-minute vehicle counts from `Cam2701`. Directions `to_Woodlands` / `to_JB` map to `MY_TO_SG` / `SG_TO_MY`. **Not referenced by `v_training_set`.** Its per-frame counts come from detections only, so frames with zero vehicles are dropped. 5,631 rows, bins 2026-03-13 05:10 to 2026-04-22 03:30 UTC.

### `cam2702`

- **Table `Cam2702`** -- last modified **2026-07-18 11:06 SGT**; **233,463 rows**. Same schema plus `camera_id`; frames dated 2026-03-13 to 2026-04-22 (33 days, 3,507 frames). No congestion view.

### `causeway`

- **Partitioned table `travel_times`** -- location `US`; **14,368 rows** (7,184 per direction), `observed_at` 2026-09-05 17:53 UTC to 2026-09-30 16:35 UTC, as queried 2026-10-01. Partitioned on `observed_date_sgt`, clustered on `route_id`. Maps Distance Matrix logging (**LIVE**). Mean `duration_in_traffic_sec`: **26.0 min** (`MY_TO_SG`), **25.4 min** (`SG_TO_MY`) on that 2026-10-01 snapshot. On **2026-10-03 ~10:48 SGT** table metadata showed **15,760 rows** and **2,742,224** logical bytes (streaming buffer estimated 4 rows). A read-only query of `observed_date_sgt` from the previous Singapore date, `status = 'OK'`, both Woodlands routes, returned **418** rows per route and `MAX(observed_at)` **2026-10-03 02:45:03 UTC**. The means above were not recomputed.
- Routes: `jb_to_woodlands` = `MY_TO_SG`, `mandai_to_shell_jb` = `SG_TO_MY` (1:1).

### `rainfall` / `weatherforecast`

Both tables are Woodlands-only, timestamps in UTC. Until 2026-10-01 they held 2025-09-01 to 2026-08-31 SGT (last modified 3 Sep). On **2026-10-01 ~01:50 SGT** the rows for **1–30 Sep 2026 SGT** were appended with [`Causeway/load_bigquery.py`](../Causeway/load_bigquery.py) from complete data.gov.sg day files (verified after load: no duplicate timestamps).

- **`rainfall.rainfall`** -- **103,922 rows** (+8,640), station `Woodlands Centre Road` (S210), 5-minute readings, 2025-08-31 16:00 to 2026-09-30 15:55 UTC. The appended rows first went in as `Woodlands Centre` (data.gov.sg's current name for S210); a one-off `UPDATE` restricted to those 8,640 appended rows restored `Woodlands Centre Road` so the view keeps one station. The loader now pins that name.
- **`weatherforecast.weatherforecast`** -- **24,482 rows** (+1,876), area Woodlands, one row per issue, to 2026-09-30 15:30 UTC. New nullable column **`update_timestamp`** (data.gov.sg acquisition time) is set on the appended rows and NULL on older rows.
- **Snapshots** (expire 2026-10-31): `rainfall.rainfall_snapshot_20261001`, `weatherforecast.weatherforecast_snapshot_20261001` hold the pre-load tables. Revert with `CREATE OR REPLACE TABLE <table> CLONE <snapshot>`.
- **No scheduled loader** keeps these tables current; re-run the Causeway fetchers and `load_bigquery.py --execute` (it refuses overlapping ranges).
- **View `v_weather_features_10min`** -- Woodlands forecast flags plus Woodlands Centre Road rainfall on 10-minute bins; now returns bins to 2026-09-30. It bins forecasts by **issue** time, not valid period. **Not referenced by `v_training_set`.**

### `traffic_images`

- **`metadata`** -- **368,905 rows**; last modified **2026-09-13 04:29 SGT**. Six cameras (2701, 2702, 2704, 4703, 4712, 4713), about 61.5k frames each, `capture_timestamp` 2025-06-30 23:55 to **2026-09-11 23:55**, images in `gs://sg-lta-traffic-cameras/`. Camera 2701 has 132–144 frames per day on 5–11 Sep (every ~10 min). Frames only; no detections are stored for them.
- **`labels`** -- **0 rows** (labelling lives in Roboflow).
- **`backfill_checkpoint`** -- 63,074 rows (2026-09-13 04:29 SGT). **`backfill_failures`** -- 0 rows.

### `traffic_prediction`

- **`model_registry`** -- 2 rows, decided 2026-09-12 on the 11–12 Sep window: `SG_TO_MY` -> `lin_h30`; `MY_TO_SG` -> `persistence`. Snapshot: [model_registry_reference.sql](../sql/bigquery/traffic_prediction/model_registry_reference.sql).
- **Models (BigQuery ML)**, trained once, never retrained:
  - `lin_h30` -- `LINEAR_REGRESSION`, created **2026-09-12 06:15 UTC**.
  - `xgb_h30` -- `BOOSTED_TREE_REGRESSOR`, created **2026-09-12 06:19 UTC**; early stop at iteration 28; adds `tod_block`. Not used in `v_forecast_recent`.
  - Both: `dataSplitMethod = CUSTOM`, `dataSplitColumn = is_val` (11–12 Sep), label `y_30`. BQML-internal eval MAE 3.04 (lin) / 2.87 (xgb) on that split; these are not harness results. Every date from **13 Sep** is out-of-sample for both.
- **`v_bins_10min`**, **`v_training_set`** (last modified 2026-09-12 13:58 SGT), **`v_forecast_recent`** (2026-09-12 14:16 SGT) -- live SQL matches [sql/](../sql/) (2026-09-30 check). `v_training_set` uses Maps columns only; `LEAD(bin_ts, 3)` was exactly 30 min on every row (0 of 3,572 per direction differ, 2026-09-30).
- `v_forecast_recent` serves **30 minutes** ahead: `lin_h30` where the registry says so, otherwise persistence.
- **2026-10-03 ~11:20 SGT** (read-only): `bq ls --models` still only `lin_h30` and `xgb_h30`. `model_registry` still the two rows above. `INFORMATION_SCHEMA.VIEWS` for `v_forecast_recent` still matches the checked-in SQL (`ML.PREDICT` on `lin_h30`, `INTERVAL 30 MINUTE`). `bq show --model` has one training run each and no version list. Resource `creationTime`: `lin_h30` `2026-09-12T06:15:33.163Z`, `xgb_h30` `2026-09-12T06:19:18.783Z`. `gcloud run services list` returned only `gmap-woodlands-fetcher` and `swiftbackend`.

## Cloud Run / Scheduler / Storage

### Cloud Run

- `gmap-woodlands-fetcher` -- `asia-southeast1`, revision `gmap-woodlands-fetcher-00004-msb`, ingress `all`, last deployed 2026-09-05 18:14 UTC.
- `swiftbackend` -- `europe-west1`, latest ready revision **`swiftbackend-00023-f8g`** (commit `e49b232`, Cloud Build `33a6d0e5`, created 2026-09-26 07:10 UTC). Function target `detect`.
  - **Exposure:** `run.googleapis.com/invoker-iam-disabled: true` with an empty IAM policy, and ingress `all`: **anyone can call it**. `ALLOWED_ORIGIN` is not set, so CORS is `*`. Each call can trigger a billed Roboflow inference.
  - **Env var names:** `ROBOFLOW_API_KEY` (a plain env var, not Secret Manager) and `CACHE_BUCKET` (set, but not read by `camdetect/main.py`; no code in git writes `swiftborder-frame-cache`).
  - Recorded as a known risk; no change made (see [findings.md](findings.md)).
- Job `traffic-backfill` -- **Cloud Run Job** (not a dataset), `asia-southeast1`. Three executions on 2026-09-12: two failed, then one succeeded (completed 2026-09-12 19:53 UTC / 13 Sep 03:53 SGT). None since.

### Cloud Build

- Trigger `76bbca35-c1b4-4836-9f34-d7adda53ea17` (`rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc`). GitHub `sozuken-max/swiftborder`, push to `^main$`. Config is **inline on the trigger**. Step `Test` (`python:3.11`) runs `pytest` in `camdetect/` before buildpacks deploy `swiftbackend`.
- `includedFiles`: `camdetect/main.py`, `camdetect/requirements.txt`, `camdetect/Dockerfile`, `camdetect/cloudbuild.yaml`, `camdetect/*.{yaml,yml,json,toml}`. Tests, `pytest.ini`, `requirements-dev.txt` and `README.md` do not start a build.
- Latest builds: `33a6d0e5` (2026-09-26 07:07 UTC, SUCCESS, `e49b232`), `8ca80327` (06:31 UTC, SUCCESS, `cf1c228`).
- The Maps fetcher, backfill job, views and models are not in this trigger. A GitHub Actions test workflow ([.github/workflows/tests.yml](../.github/workflows/tests.yml)) is committed on branch `eval-integrity-overhaul` (PR #2), not yet on `main`; it runs tests only and deploys nothing.
- **2026-10-03:** a second trigger, `forecast-api` (`949ff029-31c9-4521-8ff4-d0e8d16dfa25`), GitHub `sozuken-max/swiftborder`, push to `^main$`, config `forecastapi/cloudbuild.yaml`. `includedFiles` is `forecastapi/**` and `forecastapi/cloudbuild.yaml`. `ignoredFiles` is `camdetect/**`. It has not run. Cloud Run still lists only `gmap-woodlands-fetcher` and `swiftbackend`. Runtime identity `forecast-api@swiftborder.iam.gserviceaccount.com` has `roles/bigquery.jobUser` and `roles/bigquery.dataViewer` on project `swiftborder` only. See [runbooks/forecast-api.md](runbooks/forecast-api.md).

### Cloud Scheduler

- `Gmap-Woodlands` -- `*/5 * * * *` `Asia/Singapore` -> `gmap-woodlands-fetcher`. **ENABLED** on 2026-10-03; last attempt recorded here is still 2026-09-30 16:35 UTC because this describe returned an empty `status` and no newer `lastAttemptTime`. The public object below was rewritten at 10:40:05 and 10:45:05 SGT.

### Cloud Storage (six buckets)

| Bucket | Location |
| --- | --- |
| `sg-lta-traffic-cameras` | `asia-southeast1` |
| `swiftborder-frame-cache` | `europe-west1` |
| `run-sources-swiftborder-asia-southeast1` | `asia-southeast1` |
| `run-sources-swiftborder-europe-west1` | `europe-west1` |
| `swiftborder-public` | `asia-southeast1` |
| `swiftborder_cloudbuild` | `US` (underscore) |

Public object `gs://swiftborder-public/traffic-24h.json` (`https://storage.googleapis.com/swiftborder-public/traffic-24h.json`). Observed **2026-10-03 10:45 SGT**: **18,478** bytes, `Cache-Control: public, max-age=300`, `updated_at_sgt` `2026-10-03T10:45:03`, **288** points on each of `mandai_to_shell_jb` (`SG_TO_MY`) and `jb_to_woodlands` (`MY_TO_SG`). Point triple: SGT minute timestamp, `duration_in_traffic_sec`, `speed_kmh`. A metadata describe at **10:55 SGT** showed size **18,480** and `update_time` `2026-10-03T02:55:06Z` (the 10:45 body was not re-parsed). Bucket CORS allows `GET` and `HEAD` from `https://swiftborder-92b45.web.app`, `https://swiftborder-92b45.firebaseapp.com`, and `http://localhost:5000`. Serve decision: [adr/0001-firebase-client-api-calls.md](adr/0001-firebase-client-api-calls.md).

### Firebase / Hosting

Checked **2026-10-03 10:57 SGT**. `https://swiftborder-92b45.web.app` and `https://swiftborder-92b45.firebaseapp.com` returned HTTP 200. `/`, `/app.js`, and `/style.css` share `Last-Modified: Sat, 19 Sep 2026 05:54:38 GMT` (15,989, 42,084, and 44,139 bytes). Response `Vary` includes `x-fh-requested-host`. Firebase Management API and Firebase Hosting API are disabled on GCP project `swiftborder` (`gcloud services list --enabled` returned no `firebase*` rows; both REST calls returned `SERVICE_DISABLED`). `gcloud projects describe swiftborder-92b45` returned permission denied for the active account, so that id is not confirmed as a project this account can read. No Cloud Build trigger and no GitHub workflow deploys this site. Decision: [adr/0002-firebase-hosting-source.md](adr/0002-firebase-hosting-source.md).

## Querying this project from a Windows dev box

If `gcloud` fails with `No module named 'six'`, stale `CLOUDSDK_*` variables point at a broken SDK copy. Per session:

```powershell
$root = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk"
$env:CLOUDSDK_PYTHON = "$root\platform\bundledpython\python.exe"
Remove-Item env:CLOUDSDK_ROOT_DIR, env:CLOUDSDK_PYTHON_ARGS, env:CLOUDSDK_GSUTIL_PYTHON -ErrorAction SilentlyContinue
& "$root\bin\gcloud.cmd" run services list --project swiftborder
```

Always pass `--project swiftborder` (the local default project may differ).

## How to use this file

Evidence appendix for the final report. Interpretation and the claims register: [findings.md](findings.md). Methods and result tables: [evaluation.md](evaluation.md). Design: [architecture.md](architecture.md). Repo changes: [CHANGELOG.md](../CHANGELOG.md).
