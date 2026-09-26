# GCP inventory snapshot

**Source of truth:** GCP project `swiftborder`. This file is a dated copy of a read-only query, not the authority. If it disagrees with the project, the project wins; update this file.

**Observed:** 2026-09-26, around 13:40 SGT, via read-only `bq` and `gcloud` (`--project=swiftborder`). An earlier Console pass the same morning is superseded where the two differ. **No resources or settings were changed.** Firebase Hosting was not re-checked in the afternoon pass.

## Project metadata

- Display name: **Swiftborder**
- Project ID: `swiftborder`
- Project number: `1095552466513`

## BigQuery

Seven datasets in `swiftborder`: `cam2701`, `cam2702`, `causeway`, `rainfall`, `traffic_images`, `traffic_prediction`, and `weatherforecast`.

Locations: `cam2701`, `cam2702`, `rainfall`, `traffic_images`, and `weatherforecast` are `asia-southeast1`. `causeway` and `traffic_prediction` are `US`.

### `cam2701`

- **Table `Cam2701`** -- last modified **2026-07-18 10:54 SGT**; **298,137 rows**. Detection-style schema (filename, date/time, night/brightness, class/label, direction, confidence, bbox fields). Not receiving new rows.
- **View `v_congestion_index_10min`** -- 10-minute vehicle counts from `Cam2701` only. Directions `to_Woodlands` / `to_JB` map to `MY_TO_SG` / `SG_TO_MY`. **Not referenced by `traffic_prediction.v_training_set`.**

### `cam2702`

- **Table `Cam2702`** -- last modified **2026-07-18 11:06 SGT**; **233,463 rows**. Same detection schema family (plus `camera_id`). No congestion view.

### `causeway`

- **Partitioned table `travel_times`** -- created **2026-09-06**, last modified **2026-09-26 13:11 SGT**; location `US`; **11,790 rows**. Partitioned on `observed_date_sgt`, clustered on `route_id`. Maps Distance Matrix logging (**LIVE**).
- Coverage at check time: `MY_TO_SG` and `SG_TO_MY`, **5,900 rows each**, `observed_at` from **2026-09-05 17:53:30 UTC** through **2026-09-26 05:35:04 UTC** (13:35 SGT). Mean `duration_in_traffic_sec` about **26.0 min** (`MY_TO_SG`) and **25.5 min** (`SG_TO_MY`).

### `rainfall`

- **Table `rainfall`** -- last modified **2026-09-03 21:13 SGT**; **95,282 rows** (`timestamp`, `station_id`, `value_mm`).

### `traffic_images`

- **`backfill_checkpoint`** -- **63,074 rows**; last modified **2026-09-13 04:29 SGT**.
- **`backfill_failures`** -- **0 rows** (last modified 2026-07-19 11:25 SGT).
- **`labels`** -- **0 rows** (labelling lives in Roboflow; last modified 2026-07-19 00:00 SGT).
- **`metadata`** -- **368,905 rows**; last modified **2026-09-13 04:29 SGT**.

### `traffic_prediction`

- **`model_registry`** -- **2 rows**, last modified **2026-09-12 14:16 SGT** (`direction`, `serving_model`, reason, decided_on, test_window). Row contents were not re-read in this pass.
- **Models (BigQuery ML):**
  - `lin_h30` -- `LINEAR_REGRESSION`, created **2026-09-12**.
  - `xgb_h30` -- `BOOSTED_TREE_REGRESSOR`, created **2026-09-12**.
- **`v_bins_10min`** -- 10-minute averages of `travel_times` where `status = 'OK'`.
- **`v_training_set`** -- lags, rolling means, slope, time-of-day, weekend, and peak flags from those bins. Targets `y_30` and `y_60` (lead 3 and 6 bins). **Maps columns only.** No join to weather or camera views.
- **`v_forecast_recent`** -- rows from `v_training_set` in the last 24 hours with `lag_60` present. `ML.PREDICT` on `lin_h30`. Output column `forecast_30min_min` is `lin_h30` when `model_registry.serving_model = 'lin_h30'`, otherwise persistence (`y_persistence`). `xgb_h30` is not called. `y_60` is not selected.

### `weatherforecast`

- **Table `weatherforecast`** -- **22,606 rows**; last modified **2026-09-03 21:18 SGT**.
- **View `v_weather_features_10min`** -- Woodlands forecast text (`rain` / `heavy` flags) plus Woodlands Centre Road rainfall, on 10-minute bins. **Not referenced by `v_training_set`.**

## Cloud Run / Scheduler / Storage

### Cloud Run

- `gmap-woodlands-fetcher` -- Cloud Run service, `asia-southeast1`; last transition **2026-09-05 18:14 UTC** (6 Sep SGT). Cloud Functions API is disabled on this project; the service is reached by the scheduler URL below.
- `swiftbackend` -- Cloud Run service, `europe-west1`; last transition **2026-09-20 08:20 UTC**. Image built from commit `10500b3` (current `main` at check time), path `camdetect`, function target `detect`.
- Job `traffic-backfill` -- `asia-southeast1`; latest execution **succeeded**, completed **2026-09-12 19:53 UTC** (13 Sep 03:53 SGT). This is a **Cloud Run Job**, not a BigQuery dataset.

### Cloud Build

- Trigger `76bbca35-c1b4-4836-9f34-d7adda53ea17` (`rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc`), created **2026-08-29**. GitHub `sozuken-max/swiftborder`, push to `^main$`. The build config is **inline on the trigger**, not a file in git. It builds `camdetect` with Cloud Native Buildpacks and deploys `swiftbackend` in `europe-west1`. No test step.
- **Path filter (set 2026-09-26 via `gcloud builds triggers import`):** `includedFiles` is code and config under `camdetect/` only (`*.py`, `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `Dockerfile`, `cloudbuild.yaml`, `*.yaml`, `*.yml`, `*.json`, `*.toml`). `README.md` and paths outside `camdetect/` do not start this build.
- The Maps fetcher, backfill job, views, and BQML models are not in this trigger. No GitHub Actions workflows are in the repo.

### Cloud Scheduler

- `Gmap-Woodlands` -- `*/5 * * * *` in `Asia/Singapore` -> `gmap-woodlands-fetcher`. State **ENABLED** at check time. Last attempt **2026-09-26 05:30 UTC** (13:30 SGT).

### Cloud Storage (six buckets observed)

| Bucket | Location |
| --- | --- |
| `sg-lta-traffic-cameras` | `asia-southeast1` |
| `swiftborder-frame-cache` | `europe-west1` |
| `run-sources-swiftborder-asia-southeast1` | `asia-southeast1` |
| `run-sources-swiftborder-europe-west1` | `europe-west1` |
| `swiftborder-public` | `asia-southeast1` |
| `swiftborder_cloudbuild` | `US` |

### Firebase / Hosting

Morning Console pass: Firebase showed a setup / terms prompt; **no Hosting resources inventoried**. Not re-checked at 13:40. Treat Firebase as **planned**, not live.

## How to use this file

This is the evidence appendix for the final report. Interpretation and the claims register are in [findings.md](findings.md). Metric definitions and empty result tables are in [evaluation.md](evaluation.md). Design drawings are in [architecture.md](architecture.md).
