# GCP inventory (dated copy)

Dated copy of GCP project `swiftborder`. When this file and the project disagree, the project wins; re-query and update this file. Older blocks below are history. They keep the revisions and sentences that were true at their own timestamps.

## Update: 2026-10-10, approximately 10:50 SGT (read-only)

- **Cloud Run:**
  - `forecast-api` serves revision `forecast-api-00012-zal`, commit `3bd5b6b`, from Cloud Build `c4143c3f`. `?curve=forecast` returns four points at 30–120 min, and `?list=cards` returns 200.
  - `swiftbackend` served `swiftbackend-00033-mh9` (commit `f9895ec`) at 10:50 SGT. Later the same day it served `swiftbackend-00035-5dg` (commit `f72b8b2`, adds `model=local`, 2 GiB memory). Both images are built from `camdetect/Dockerfile`. Its env var names are `ROBOFLOW_API_KEY` and `CACHE_BUCKET`, with no `ROBOFLOW_WORKFLOW_ID`, so the code default (workflow v6) is live. `camdetect/main.py` does not read `CACHE_BUCKET`. The invoker IAM check is disabled on both services.
- **Triggers:** both now build from checked-in files: `forecast-api` from `forecastapi/cloudbuild.yaml`, and `76bbca35` from `camdetect/cloudbuild.yaml`. The camera trigger ran twice each for `63aa50b` and `f9895ec`, about 4 minutes apart; all those builds succeeded.
- **Hosting:** the live `app.js` differs from `hosting/app.js` (62,054 vs 61,557 characters; the live copy still says "exploratory"). The page has not been redeployed from git.

## Latest verification: 2026-10-04, approximately 14:40-14:45 SGT

Read-only. No cloud resources or settings were changed. Repo checkout for the doc pass: local `main` `7bd94e369ea1b1760881f2e90f71d4ea93f8da02`. This pass did not re-query `model_registry`, view SQL, `traffic_images` row counts, weather row counts, Hosting file bytes, bucket CORS, or service IAM policies. Those stay at the dates in the sections below.

- **Cloud Run (100% traffic on the latest ready revision):** `forecast-api` `forecast-api-00009-hax` (`asia-southeast1`); `swiftbackend` `swiftbackend-00024-86t` (`europe-west1`); `gmap-woodlands-fetcher` `gmap-woodlands-fetcher-00004-msb` (`asia-southeast1`); `weather-backfill-tmp` `weather-backfill-tmp-00004-tjj` (`asia-southeast1`, revision created 2026-10-04T02:38:10Z). The temporary weather service's implementation was not inspected. Earlier notes that it was `weather-backfill-tmp-00002-4zt` match the 2026-10-03 night listing, not this one.
- **Forecast serving:** `forecast-api` revision `forecast-api-00009-hax` is labelled commit `7bd94e369ea1b1760881f2e90f71d4ea93f8da02`, Cloud Build `d039ebc6-bb3d-457c-b548-b638d457ce1f` (SUCCESS, 2026-10-04T06:00:46Z, trigger `forecast-api`). Ingress `all`. `run.googleapis.com/invoker-iam-disabled` is `true`.
- **Forecast smoke:** an anonymous `GET ?model=served` with the Hosting `Origin` returned **HTTP 200**, `Access-Control-Allow-Origin: *`, `source: local model`, `status: evaluated`, `horizon_min: 30`, `SG_TO_MY: lin_bq[frozen]`, `MY_TO_SG: persistence`, and `commit` equal to that same sha (`generated_at` 2026-10-04T06:42:17Z). The checked-in Cloud Build smoke test accepts **200 or 503** for `model=served` (503 is a data condition: no servable origin). It does not expect 403. Statements that the smoke test still expects anonymous 403 are historical (2026-10-03 night).
- **Camera serving:** `swiftbackend-00024-86t` is labelled commit `41c06e8c56e1fdb4c3a9ad87775f4f66f361efee`, Cloud Build `e44ca468-af36-48e5-bcde-e41cf443d8e0` (SUCCESS, 2026-10-03T09:00:22Z). No detection request was made.
- **Ingestion:** `Gmap-Woodlands` is ENABLED, `*/5 * * * *`, `Asia/Singapore`, `lastAttemptTime` `2026-10-04T06:40:04Z`. `causeway.travel_times`: **8,217 rows per route (16,434 total)**, both `jb_to_woodlands` and `mandai_to_shell_jb`, earliest `2026-09-05 17:53:30 UTC`, latest `2026-10-04 06:40:04 UTC`. Gap and horizon-mismatch checks were not repeated.
- **Camera detections:** `cam2701.Cam2701` **298,137** rows, `date` **2026-03-13 to 2026-04-22**; `cam2702.Cam2702` **233,463** rows, same date span. No September detection rows in either table.
- **BigQuery:** seven datasets (`cam2701`, `cam2702`, `causeway`, `rainfall`, `traffic_images`, `traffic_prediction`, `weatherforecast`). `bq ls --models traffic_prediction` still lists only `lin_h30` and `xgb_h30`. The HTTP handler does not call them.
- **Build:** triggers `forecast-api` (`949ff029-31c9-4521-8ff4-d0e8d16dfa25`) and the camera trigger `76bbca35-c1b4-4836-9f34-d7adda53ea17` both exist. The forecast trigger has run; its latest build in this list is the SUCCESS above. Sentences that it has never run, or that Cloud Run lists only the fetcher and `swiftbackend`, describe the 2026-10-03 morning listing.
- **Jobs and storage:** Cloud Run Job `traffic-backfill` last execution still succeeded at `2026-09-12T19:53:47Z`. Seven buckets: `sg-lta-traffic-cameras`, `swiftborder-frame-cache`, `run-sources-swiftborder-asia-southeast1`, `run-sources-swiftborder-europe-west1`, `swiftborder-public`, `swiftborder_cloudbuild`, `swiftborder_asia-southeast1_cloudbuild`.

Queries: `gcloud run services list`, `services describe` / `revisions describe` for the three revisions named above, `gcloud scheduler jobs describe Gmap-Woodlands`, `gcloud builds triggers list`, `gcloud builds list --limit=8`, `gcloud run jobs list`, `gcloud storage buckets list`; `bq ls`, `bq ls --models`, and read-only aggregates on `Cam2701`, `Cam2702`, and `travel_times`; anonymous HTTP GET of `?model=served`. `traffic-backfill` remains a Cloud Run Job.

**Not part of that query (repo only, this branch):** `forecastapi/` adds `model` on each direction and each curve point, and `components` only on blends. Revision `forecast-api-00009-hax` does not serve that shape until this merges and trigger `forecast-api` runs. `hosting/` (`index.html`, `app.js`, `style.css`) is in git. Neither fact is a new row count or a new revision name.

## Historical: 2026-10-04 approximately 00:35-01:05 SGT

Superseded by the latest verification above where they disagree (in particular `forecast-api-00002-zaj`).

**Camera tables and forecast access, 2026-10-04 approximately 00:45-01:05 SGT** (read-only `bq show`, aggregates, Cloud Audit Logs). No settings changed.
- **`cam2701.Cam2701`:** 298,137 rows; 3,513 frames on 33 days, 2026-03-13 to 2026-04-22; last modified 2026-07-18. Classes `car`, `motorcycle`, `bus`, `truck` and `pedestrian`. Directions `to_JB`, `to_Woodlands` and `none`. Minimum confidence 0.12. View `v_congestion_index_10min`.
- **`cam2702.Cam2702`:** 233,463 rows; 3,507 frames on the same 33 days; last modified 2026-07-18. `camera_id` 2702. No pedestrian rows, no `none` direction, minimum confidence 0.10, no view.
- **Common to both:** neither table records a model or workflow version.
- **Live detections:** none are written anywhere. `swiftbackend` (`camdetect/main.py`) has no BigQuery or Cloud Storage client.
- **`forecast-api` access (that hour):** revision `forecast-api-00002-zaj`; an anonymous GET returned 200 with CORS `*`. `run.googleapis.com/invoker-iam-disabled: true` had been set by an `UpdateService` call from the project owner at 2026-10-03 14:27:44 UTC.

**Diagram refresh, 2026-10-04 approximately 00:35 SGT:** a Cloud Run and bucket listing that hour returned four service revisions, including `forecast-api-00002-zaj` and `weather-backfill-tmp-00002-4zt`, and seven buckets. An anonymous forecast GET returned `source: local model`, `horizon_min: 30`, `SG_TO_MY: lin_bq[frozen]`, and `MY_TO_SG: persistence`. That refresh did not re-count BigQuery tables.

## Historical verification: 2026-10-03, approximately 23:35-23:40 SGT

Read-only final-report readiness review. Superseded by the 2026-10-04 14:40 SGT block. No cloud resources or settings were changed during this review.

- **Cloud Run (that night):** four services: `forecast-api` (`forecast-api-00002-zaj`), `gmap-woodlands-fetcher` (`gmap-woodlands-fetcher-00004-msb`), `swiftbackend` (`swiftbackend-00024-86t`), and `weather-backfill-tmp` (`weather-backfill-tmp-00002-4zt`). The temporary weather service's implementation and purpose were not inspected. The 2026-10-04 afternoon listing has newer forecast and weather revisions.
- **Forecast serving (that night):** `forecast-api` sent 100% of traffic to revision `forecast-api-00002-zaj`, commit `ba0b844fb8731ae88e0589bdfc211a75c3639440`, successful Cloud Build `2cc5bb51-60ae-404e-b41d-606a5065c15d`. A live `model=served` response identified `source: local model`, `SG_TO_MY: lin_bq[frozen]`, `MY_TO_SG: persistence`, and `horizon_min: 30`. Runtime identity was `forecast-api@swiftborder.iam.gserviceaccount.com`. The old BQML registry still existed but was not this HTTP handler's serving implementation.
- **Forecast access (that night):** `run.googleapis.com/invoker-iam-disabled` was `true`, ingress was `all`, and an anonymous request with the Hosting origin returned HTTP 200 and `Access-Control-Allow-Origin: *`. An empty service IAM policy does not imply private access when the invoker check is disabled. The note that the checked-in smoke test still expected anonymous HTTP 403 was true of the tree at that review and is not true of local `main` `7bd94e3`.
- **Forecast timing evidence:** one response was generated at `2026-10-03T15:32:49.24402Z` using origin `2026-10-03T15:30:00Z`, before that ten-minute bin closed. Its target was `2026-10-03T16:00:00Z`. See [final-report-readiness.md](final-report-readiness.md) for the evaluation/serving mismatch.
- **Camera serving:** latest ready revision `swiftbackend-00024-86t` carries commit `41c06e8c56e1fdb4c3a9ad87775f4f66f361efee`; invoker check remains disabled. No billed detection request was made in this review.
- **Ingestion:** `Gmap-Woodlands` is ENABLED, schedule `*/5 * * * *`, last attempt `2026-10-03T15:35:04.017863Z`. A `travel_times` aggregate returned **8,036 rows per direction (16,072 total)**, earliest `2026-09-05 17:53:30 UTC`, latest `2026-10-03 15:35:04 UTC`. A separate bin query found zero single-bin gaps (`gap_min = 20`) and zero non-null three-row targets differing from 30 minutes, in either direction. This is a snapshot, not a guarantee for later data.
- **BigQuery:** seven datasets remain. `traffic_prediction` lists `lin_h30` and `xgb_h30`; registry choices remain `SG_TO_MY: lin_h30`, `MY_TO_SG: persistence`. The retrieved `v_training_set` definition remains Maps-only, with row-based LAG/LEAD. `traffic_images` metadata counts: `metadata` 368,905; `labels` 0; `backfill_checkpoint` 63,074; `backfill_failures` 0.
- **Build:** both the `forecast-api` trigger (file `forecastapi/cloudbuild.yaml`) and the camera trigger exist, both matching `^main$`. The forecast trigger has now run successfully. Earlier statements that it has never run or that the test workflow is only on an unmerged branch are historical.
- **Storage:** **seven** buckets are now listed: `sg-lta-traffic-cameras`, `swiftborder-frame-cache`, `run-sources-swiftborder-asia-southeast1`, `run-sources-swiftborder-europe-west1`, `swiftborder-public`, `swiftborder_cloudbuild`, and `swiftborder_asia-southeast1_cloudbuild`. The six-bucket diagram rule is stale; update it before regenerating assets.
- **Hosting:** the public `app.js` now contains `FORECAST_API` pointing at `forecast-api-1095552466513.asia-southeast1.run.app`; `fetchForecast()` calls it and passes results to forecast cards/chart rendering. The API was independently confirmed anonymously accessible with CORS. The full browser interaction was not exercised. Hosting source remains absent from this checkout. Its comments still describe the API as private and are stale.

Queries used: `gcloud run services list/describe/get-iam-policy`, `gcloud builds triggers list`, `gcloud builds list`, `gcloud scheduler jobs list`, `gcloud run jobs list`, `gcloud storage buckets list`; `bq ls`, `bq ls --models`, `bq show`, and read-only aggregates over `travel_times`, `v_bins_10min`, `model_registry`, and `traffic_images.__TABLES__`; HTTP GET of the public Hosting script and forecast endpoint. `traffic-backfill` remains a Cloud Run Job.

## Earlier observations (historical; latest verification above takes precedence)

**Observed (historical):** 2026-10-01 ~00:40 SGT (2026-09-30 16:38 UTC), read-only `gcloud` / `bq` queries of project `swiftborder`, plus model, view and horizon checks from the 2026-09-30 ~21:35 SGT pass. **One change since that pass:** the weather tables were appended on 2026-10-01 ~01:50 SGT (requested by the team; see `rainfall` / `weatherforecast` below). Camera and image date ranges, and `model_registry`, were re-queried read-only on 2026-10-01 ~03:30 SGT. **2026-10-03 ~10:48 SGT** re-checked `causeway.travel_times` metadata, `traffic-24h.json`, bucket CORS, Hosting, and the Maps scheduler (see those sections). **2026-10-03 ~11:20 SGT** re-checked `traffic_prediction` models, `model_registry`, and `v_forecast_recent`, and listed Cloud Run services (see those sections). That 11:20 list did not yet include a forecast HTTP service. Later the same day `forecast-api` was deployed; see the latest verification.

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
- **2026-10-03 ~11:20 SGT** (read-only, historical): `bq ls --models` still only `lin_h30` and `xgb_h30`. `model_registry` still the two rows above. `INFORMATION_SCHEMA.VIEWS` for `v_forecast_recent` still matches the checked-in SQL (`ML.PREDICT` on `lin_h30`, `INTERVAL 30 MINUTE`). `bq show --model` has one training run each and no version list. Resource `creationTime`: `lin_h30` `2026-09-12T06:15:33.163Z`, `xgb_h30` `2026-09-12T06:19:18.783Z`. `gcloud run services list` at that minute returned only `gmap-woodlands-fetcher` and `swiftbackend`. `forecast-api` exists in later listings.

## Cloud Run / Scheduler / Storage

### Cloud Run

Revisions in this subsection are the 2026-10-01 snapshot unless a dated line says otherwise. Current revisions are in the latest verification.

- `gmap-woodlands-fetcher` -- `asia-southeast1`, revision `gmap-woodlands-fetcher-00004-msb`, ingress `all`, last deployed 2026-09-05 18:14 UTC. Still the latest ready revision on 2026-10-04.
- `swiftbackend` -- `europe-west1`. **2026-10-01 snapshot:** latest ready revision `swiftbackend-00023-f8g` (commit `e49b232`, Cloud Build `33a6d0e5`, created 2026-09-26 07:10 UTC). **Later:** `swiftbackend-00024-86t` (see latest verification). Function target `detect`.
  - **Exposure:** `run.googleapis.com/invoker-iam-disabled: true` with an empty IAM policy, and ingress `all`: **anyone can call it**. `ALLOWED_ORIGIN` is not set, so CORS is `*`. Each call can trigger a billed Roboflow inference.
  - **Env var names:** `ROBOFLOW_API_KEY` (a plain env var, not Secret Manager) and `CACHE_BUCKET` (set, but not read by `camdetect/main.py`; no code in git writes `swiftborder-frame-cache`).
  - Recorded as a known risk; no change made (see [findings.md](findings.md)).
- Job `traffic-backfill` -- **Cloud Run Job** (not a dataset), `asia-southeast1`. Three executions on 2026-09-12: two failed, then one succeeded (completed 2026-09-12 19:53 UTC / 13 Sep 03:53 SGT). None since.

### Cloud Build

- Trigger `76bbca35-c1b4-4836-9f34-d7adda53ea17` (`rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc`). GitHub `sozuken-max/swiftborder`, push to `^main$`. Config is **inline on the trigger**. Step `Test` (`python:3.11`) runs `pytest` in `camdetect/` before buildpacks deploy `swiftbackend`.
- `includedFiles`: `camdetect/main.py`, `camdetect/requirements.txt`, `camdetect/Dockerfile`, `camdetect/cloudbuild.yaml`, `camdetect/*.{yaml,yml,json,toml}`. Tests, `pytest.ini`, `requirements-dev.txt` and `README.md` do not start a build.
- Latest builds: `33a6d0e5` (2026-09-26 07:07 UTC, SUCCESS, `e49b232`), `8ca80327` (06:31 UTC, SUCCESS, `cf1c228`).
- The Maps fetcher, backfill job, views and models are not in this trigger. A GitHub Actions test workflow ([.github/workflows/tests.yml](../.github/workflows/tests.yml)) is committed on branch `eval-integrity-overhaul` (PR #2), not yet on `main`; it runs tests only and deploys nothing.
- **2026-10-03 morning (historical):** a second trigger, `forecast-api` (`949ff029-31c9-4521-8ff4-d0e8d16dfa25`), GitHub `sozuken-max/swiftborder`, push to `^main$`, config `forecastapi/cloudbuild.yaml`. `includedFiles` is `forecastapi/**` and `forecastapi/cloudbuild.yaml`. `ignoredFiles` is `camdetect/**`. At the time this paragraph was written the trigger had not run, and Cloud Run listed only `gmap-woodlands-fetcher` and `swiftbackend`. Both statements are obsolete: see the latest verification. Runtime identity `forecast-api@swiftborder.iam.gserviceaccount.com` has `roles/bigquery.jobUser` on project `swiftborder`, and `roles/bigquery.dataViewer` on datasets `traffic_prediction` and `causeway` only (narrowed 2026-10-03 from project-wide `dataViewer`; the views read `causeway.travel_times` and `traffic_prediction.model_registry`; those IAM bindings were not re-queried on 2026-10-04). The trigger itself builds as the default Compute Engine service account. See [runbooks/forecast-api.md](runbooks/forecast-api.md).

### Cloud Scheduler

- `Gmap-Woodlands` -- `*/5 * * * *` `Asia/Singapore` -> `gmap-woodlands-fetcher`. **ENABLED** on 2026-10-03; last attempt recorded here is still 2026-09-30 16:35 UTC because this describe returned an empty `status` and no newer `lastAttemptTime`. The public object below was rewritten at 10:40:05 and 10:45:05 SGT.

### Cloud Storage (six buckets on the 2026-10-01 listing; seven from 2026-10-03, reconfirmed 2026-10-04)

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
