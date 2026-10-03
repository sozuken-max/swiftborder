# Forecast API

HTTP read of the Woodlands **30-minute** forecast of Google Maps `duration_in_traffic`. The service code is `forecastapi/`. **Cloud Run service `forecast-api` is not deployed.** A separate Cloud Build trigger deploys it only on a push to `main` that changes `forecastapi/`. This branch does not deploy it. Nothing here created a BigQuery object, and trigger `76bbca35` is unchanged.

A teammate can run it locally and read the contract here. Deploy is the pipeline below, not a second copy of `swiftbackend`.

## What it returns

`GET /?list=models` returns the catalog. `model` is not required. `list` wins if both are set. The list does not start a BigQuery job.

`GET` with `model` returns JSON for one logical model, and only when that model is callable:

| Field | Meaning |
| --- | --- |
| `model` | The model id you asked for |
| `version` | The version that was used. If you omit `version`, this is the default below |
| `horizon_min` | Always `30` until a verified model supports something else |
| `generated_at` | When this process ran the query (a cache hit repeats that timestamp) |
| `source` | `ML.PREDICT`, `bigquery view`, or `fixture` (tests only) |
| `label` | `maps_duration_in_traffic_min` |
| `directions` | Per direction: `forecast_min`, `forecast_for` (the timestamp the value applies to), `origin_ts` (the feature bin, 30 minutes earlier) |

`forecast_min` is a forecast of Maps `duration_in_traffic`, in minutes. It is not a measured crossing time, and this API does not report an error against that label.

Example shape (values are illustrative; a live call fills them from BigQuery):

```json
{
  "model": "lin_h30",
  "version": "2026-09-12T06:15:33.163Z",
  "horizon_min": 30,
  "generated_at": "2026-10-03T03:20:00Z",
  "source": "ML.PREDICT",
  "label": "maps_duration_in_traffic_min",
  "directions": {
    "SG_TO_MY": {
      "forecast_min": 41.2,
      "forecast_for": "2026-10-03T04:00:00Z",
      "origin_ts": "2026-10-03T03:30:00Z"
    },
    "MY_TO_SG": {
      "forecast_min": 25.0,
      "forecast_for": "2026-10-03T04:00:00Z",
      "origin_ts": "2026-10-03T03:30:00Z"
    }
  }
}
```

## What it does not return

- Not the camera JPEG or the detection JSON from Cloud Run `swiftbackend` (`camdetect`, function `detect`).
- Not `gs://swiftborder-public/traffic-24h.json`. That file is the Maps history the Firebase UI already charts. This API does not replace it.
- Not a 24-hour product forecast. No model here is scored at 24 hours. Models scored at 60 minutes are listed and are not callable. A forecast request for one of them is HTTP 400, not a number. A served model still rejects `horizon_min` other than 30.
- Not a claim that the forecast beat Google or persistence. Scored comparisons live in [evaluation.md](../evaluation.md) and only where that page cites a harness run. This API does not copy those errors into the JSON.

The live serve view `traffic_prediction.v_forecast_recent` still chooses per direction: `lin_h30` for `SG_TO_MY`, `persistence` for `MY_TO_SG`. This API does not change that view. If you pass `model=lin_h30`, both directions are that model's `ML.PREDICT` output, including `MY_TO_SG`, where the view would have used persistence.

## Parameters

Query string, or a JSON body on `GET`. A body value wins unless it is missing, null, or `""`, in which case the query string is used (same rule as `camdetect`).

| Parameter | Required | Default | Values |
| --- | --- | --- | --- |
| `list` | no | none | `models` lists the catalog and does not forecast |
| `model` | yes, unless `list=models` | none | any `id` from the catalog; only `production` and `api` ids return a forecast |
| `version` | no | the version on that catalog row | one string per model; anything else is 400 |
| `direction` | no | `both` | `SG_TO_MY`, `MY_TO_SG`, `both`. A model that is only scored on one direction rejects the other |
| `horizon_min` | no | that model's `horizon_min` | the horizon that model is scored at. Served models are 30 only |

| `model` | Default `version` | How the default was chosen | `source` | Query |
| --- | --- | --- | --- | --- |
| `served` | `2026-09-12` | `model_registry.decided_on`. The per-direction mix the live system serves (`lin_h30` for `SG_TO_MY`, persistence for `MY_TO_SG`) | `bigquery view` | latest row per direction of `v_forecast_recent` |
| `lin_h30` | `2026-09-12T06:15:33.163Z` | Model resource `creationTime`. This is the model `v_forecast_recent` passes to `ML.PREDICT`, with no `VERSION` clause | `ML.PREDICT` | `ML.PREDICT` on `traffic_prediction.lin_h30`, latest 24h bin per direction |
| `xgb_h30` | `2026-09-12T06:19:18.783Z` | Model resource `creationTime`. Trained, not called by the view | `ML.PREDICT` | `ML.PREDICT` on `traffic_prediction.xgb_h30` |
| `persistence` | `2026-09-12` | `model_registry.decided_on`. Registry serving model for `MY_TO_SG`. Not a BigQuery ML model | `bigquery view` | `y_persistence` on `v_training_set` (the column the view uses when `serving_model` is not `lin_h30`) |

There is no single default `model`. `model` is required unless `list=models`. For the forecast the live system shows, use `model=served`.

Callable today (a forecast request runs the SQL already in `forecastapi/main.py`):

| `model` | `deploy_state` | `callable` |
| --- | --- | --- |
| `served` | `production` | true. What `v_forecast_recent` publishes. Use this for anything traveller-facing |
| `lin_h30` | `api` | true. Direct query, both directions. The registry does not serve it for `MY_TO_SG`, where it is worse than persistence |
| `xgb_h30` | `api` | true. Trained, not served by the view. `ML.PREDICT` on the BigQuery model can |
| `persistence` | `api` | true. Last Maps bin mean (`y_persistence`), not a model file |

`api` means "this endpoint can query it", not "the live system serves it".

`GET /?list=models` is the full catalog. Shortened shape (every row also has `family`, `version`, `horizon_min`, `directions`, and `callable`):

```json
{
  "models": [
    {"id": "served", "deploy_state": "production"},
    {"id": "lin_h30", "deploy_state": "api"},
    {"id": "xgb_h30", "deploy_state": "api"},
    {"id": "persistence", "deploy_state": "api"},
    {"id": "xgb[maps]", "deploy_state": "artifact"},
    {"id": "XGB (sklearn)", "deploy_state": "artifact"},
    {"id": "lstm", "deploy_state": "artifact"},
    {"id": "rules", "deploy_state": "artifact"}
  ]
}
```

`xgb_to_fuzzy` is `fuzzy_traffic.CANDIDATES` key `xgb` (the 60-minute XGBoost forecast turned into a traffic level). `rules` is the learned fuzzy rule base. `XGB (sklearn)` is the offline 60-minute model in `timeseries_xgb.py`, route `jb_to_woodlands` only (`MY_TO_SG`).

`bq show --model` has no version list. Each of `lin_h30` and `xgb_h30` has one training run. The SQL in `forecastapi/main.py` names the model and does not add a `VERSION` clause, matching the view. A new training run needs a human to refresh the allow-list (and the SQL, if BigQuery then has a real version to pin) before this API will accept it.

## Errors

| Status | When |
| --- | --- |
| 400 | Missing `model`, unknown model, model not deployed, unknown version, bad `direction`, `horizon_min` outside that model's scored horizon, `list` other than `models` |
| 405 | Any method other than `GET` or `OPTIONS` |
| 502 | The query failed. The body is `{"error": "Forecast query failed"}`. SQL and stack traces stay in the process log |
| 503 | The query succeeded and did not return every requested direction. The body is `{"error": "No recent forecast for the requested direction"}`. A smoke test that hits a data gap should see 503, not a failed deploy |

## Cache

Successful responses are cached **in this process** for 300 seconds. The key is `(model, version, direction, horizon_min)`. A second request with the same key does not start a BigQuery job. `generated_at` stays at the time of the query. A failure is not cached.

The cache is per Cloud Run instance. Two instances can both miss and both run a job. The response also sends `Cache-Control: private, max-age=300`, so a browser can skip a repeat call. Do not poll faster than every 5 minutes. A faster poll re-bills BigQuery once the server cache expires, and it cannot see a newer Maps sample than the last fetcher run.

`lin_h30` and `xgb_h30` run `ML.PREDICT` over the last 24 hours of `v_training_set`. `persistence` reads that view and does not call `ML.PREDICT`. This pass did not execute either query, so it does not state bytes billed. On-demand BigQuery still rounds each job up. A cache miss per browser, per model, is a job.

## CORS and public invocation

`ALLOWED_ORIGIN` defaults to `*`, the same env default as `camdetect`. That is wide open. It is not a decision to publish this service the way `swiftbackend` is published.

`swiftbackend` has invoker IAM disabled, ingress open to all, and `ALLOWED_ORIGIN` unset, and it holds a Roboflow key. This service has no Maps key and no Roboflow key. BigQuery uses Application Default Credentials. A public URL would still let anyone who finds it start `ML.PREDICT` jobs on every cache miss.

`ALLOWED_ORIGIN` is one value, matching `camdetect`. The Firebase Hosting origins observed for the public JSON bucket are `https://swiftborder-92b45.web.app`, `https://swiftborder-92b45.firebaseapp.com`, and `http://localhost:5000`. One setting cannot name all three. `*` allows all of them and every other site.

Browser `fetch` from Hosting cannot attach a Cloud Run identity token by itself. A private service is the safer deploy. A public service is a separate approval: set one origin (or accept `*` and the cost), and pass `--allow-unauthenticated` only in that approval. Do not reuse `swiftbackend`.

## How model ids and versions were discovered

Read-only, project `swiftborder`, 2026-10-03 ~11:20 SGT. Dataset location is `US`. These commands do not create or replace anything.

```text
bq ls --models --project_id=swiftborder traffic_prediction
bq show --model --format=prettyjson --project_id=swiftborder swiftborder:traffic_prediction.lin_h30
bq show --model --format=prettyjson --project_id=swiftborder swiftborder:traffic_prediction.xgb_h30
```

`bq ls --models` returned only `lin_h30` (`LINEAR_REGRESSION`, created 12 Sep 14:15 local / 06:15 UTC) and `xgb_h30` (`BOOSTED_TREE_REGRESSOR`, created 12 Sep 14:19 local / 06:19 UTC). `creationTime` in the show JSON is `2026-09-12T06:15:33.163Z` and `2026-09-12T06:19:18.783Z`. Each show has one `trainingRuns` entry and no version field. Training-run `startTime` (not the API version string) was `2026-09-12T06:15:16.096Z` for `lin_h30` and `2026-09-12T06:10:34.468Z` for `xgb_h30`.

```text
bq query --nouse_legacy_sql --location=US --project_id=swiftborder --maximum_bytes_billed=10485760 "SELECT direction, serving_model, CAST(decided_on AS STRING) AS decided_on, test_window FROM `swiftborder.traffic_prediction.model_registry` ORDER BY direction"
```

| direction | serving_model | decided_on | test_window |
| --- | --- | --- | --- |
| MY_TO_SG | persistence | 2026-09-12 | 2026-09-11..12 |
| SG_TO_MY | lin_h30 | 2026-09-12 | 2026-09-11..12 |

```text
bq query --nouse_legacy_sql --location=US --project_id=swiftborder --maximum_bytes_billed=10485760 "SELECT SUBSTR(view_definition, 1, 2000) AS view_definition FROM `swiftborder.traffic_prediction.INFORMATION_SCHEMA.VIEWS` WHERE table_name = 'v_forecast_recent'"
```

The view still runs `ML.PREDICT(MODEL swiftborder.traffic_prediction.lin_h30, ...)` and then `CASE serving_model WHEN 'lin_h30'`. It does not name `xgb_h30`. Horizon in the view is `INTERVAL 30 MINUTE`. Checked-in SQL: [sql/bigquery/traffic_prediction/v_forecast_recent.sql](../../sql/bigquery/traffic_prediction/v_forecast_recent.sql).

`gcloud run services list --project=swiftborder` returned `gmap-woodlands-fetcher` (`asia-southeast1`) and `swiftbackend` (`europe-west1`) only.

Refresh the allow-list in `forecastapi/main.py` (`MODELS`) from these commands before adding a model or a version. Tests use a fake query; they do not prove the allow-list still matches BigQuery.

## Local run

From the repo root, PowerShell:

```text
py -3.11 -m pip install -r forecastapi/requirements-dev.txt
py -3.11 -m functions_framework --target forecast --source forecastapi/main.py --port 8080
```

The process uses Application Default Credentials for BigQuery (`gcloud auth application-default login` as a principal that can run jobs and read `traffic_prediction`). It does not read `GOOGLE_MAPS_API_KEY` or `ROBOFLOW_API_KEY`.

List every model:

```text
curl.exe "http://127.0.0.1:8080/?list=models"
```

Default version of the model the live view predicts:

```text
curl.exe "http://127.0.0.1:8080/?model=lin_h30"
```

Explicit model and version:

```text
curl.exe "http://127.0.0.1:8080/?model=xgb_h30&version=2026-09-12T06:19:18.783Z"
```

One direction (registry choice for Malaysia to Singapore):

```text
curl.exe "http://127.0.0.1:8080/?model=persistence&direction=MY_TO_SG"
```

`curl.exe` is the Windows curl. Git Bash `curl` is the same URLs.

## Tests

No GCP calls. From the repo root:

```text
py -3.11 -m pip install -r forecastapi/requirements-dev.txt
scripts\run_tests.ps1 -Suite forecastapi
```

Or only this file, from `forecastapi/`:

```text
py -3.11 -m pytest -q
```

GitHub Actions job `pytest` includes the `forecastapi` suite. That workflow does not deploy.

## How to deploy a model

This is how an artifact gets a version, not how the HTTP service is shipped. The service pipeline is [Deploy](#deploy). Do not add `forecastapi/` to Cloud Build trigger `76bbca35`. That trigger deploys `swiftbackend` from `camdetect/`.

One Cloud Run service, many versions. The `version` query parameter selects the artifact object name. The default version for a model is the catalog row (the one marked `served` when a forecast is allowed). Do not create a Cloud Run service per model.

### BigQuery ML

`lin_h30` and `xgb_h30` stay in BigQuery in project `swiftborder`, dataset `traffic_prediction`. The API's served path is `ML.PREDICT` (and, for the live mix, `v_forecast_recent`, which calls `lin_h30` and then may substitute persistence). Do not put these models in Cloud Run and do not export them to a bucket for this service.

A new version is `CREATE OR REPLACE MODEL` only after a human approves that write. Then update the allow-list version string in `forecastapi/main.py` to the new resource `creationTime` from `bq show --model`. Until that edit, the API returns 400 for any other version string. The SQL does not add a `VERSION` clause, because the live view does not and no second version object was found.

`xgb_h30` is already in BigQuery, so this API can call `ML.PREDICT` on it. The view still does not. Changing the view is a separate approved write.

### persistence

`persistence` is the last Maps `duration_in_traffic` in the current bin (`v_training_set.y_persistence`), which is what `model_registry` selects for `MY_TO_SG`. It is not a trained artifact and it does not need a bucket or a second service. The handler already reads that column. It is listed `callable: true`. A unit test supplies a fake row and does not call BigQuery.

### sklearn, XGBoost, ensembles, fuzzy

Daily-refit ridge and XGBoost (`ridge[maps]`, `xgb[maps]`, and the other `ridge[...]` / `xgb[...]` / `ensemble[...]` ids), `ensemble_mean`, `mean[models]`, `stack`, `select`, `fuzzy stack`, the 60-minute `XGB (sklearn)`, and the fuzzy level models (`rules`, `xgb_to_fuzzy`) are not in BigQuery. The promoted run `20261003T035807Z_offline-bqml-joined-deep-fuzzy-ensemble` scored the ones marked `artifact`. `ridge[maps+weather+camera]`, `xgb[maps+weather+camera]`, and `ensemble[maps+weather+camera]` are `code-only`: the feature set is in `eval/joined.py` and that run did not emit them as candidates. None of these is loaded in production. A forecast request returns 400 `model is not deployed`.

When a human approves serving one of them, ship a versioned joblib (or an equivalent pickle the service already depends on) in a GCS bucket in project `swiftborder`. This change does not create that bucket and does not name one. Object name: `{model}/{version}`, and the `version` query parameter is that object name's version segment. The default service is the same Cloud Run service `forecast-api` in `asia-southeast1`, separate from `swiftbackend`. Do not load TensorFlow in that service. There is no Random Forest forecast in the catalog. A later one would use this same joblib path, not a new service.

Fuzzy ids predict a traffic level (light, moderate, heavy), not a duration in minutes. Serving them is still this service, and the response shape would have to be agreed before `callable` is set true. Do not return a made-up minute value.

### Deep models

Do not deploy `lstm`, `gru`, `transformer`, or `transformer_raw` now. [ADR 0003](../adr/0003-deep-training-and-feature-matrix.md) (Accepted) says train them locally. On the short split they are insufficient-data and worse than the offline XGBoost. They are listed so a caller can see that, with `deploy_state` `artifact` and `callable` false.

After a promote step, a versioned SavedModel can go in the same bucket. Run it on a separate Cloud Run revision or a second service so the default `forecast-api` image stays small and does not import TensorFlow. No GPU. Set minimum instances to 0. That service does not exist.

### What is not a model

`eval/camera_forecast.py` learns a Fourier profile of camera 2701 queue depth. `joined.py` uses it as a feature (`camfc`, `mpfc`). It is not a row in this catalog.

## Deploy

The service is still **not deployed**. The first deploy happens when the trigger below runs on `main`. Do not run `gcloud builds submit` or `gcloud run deploy` from this branch.

Config: [forecastapi/cloudbuild.yaml](../../forecastapi/cloudbuild.yaml). It matches the `swiftbackend` shape (pytest, then a buildpack image, then Cloud Run) and it is a different trigger.

| | |
| --- | --- |
| Service | Cloud Run `forecast-api`, `asia-southeast1`, function target `forecast`, minimum instances 0, no GPU |
| Image | `asia-southeast1-docker.pkg.dev/swiftborder/cloud-run-source-deploy` (the repository already in that region). No keys in the image |
| Identity | `forecast-api@swiftborder.iam.gserviceaccount.com`. `roles/bigquery.jobUser` on project `swiftborder`, and `roles/bigquery.dataViewer` on datasets `traffic_prediction` and `causeway` only (narrowed 2026-10-03 from project-wide `dataViewer`; the views read `causeway.travel_times` and `traffic_prediction.model_registry`). Application Default Credentials. No `ROBOFLOW_API_KEY`, no `GOOGLE_MAPS_API_KEY` |
| Env | `BQ_PROJECT=swiftborder` only. Each deploy sets that list and does not carry a key forward |
| Auth | `--no-allow-unauthenticated`. A public URL is a separate approval. Do not copy `swiftbackend` |
| GitHub Actions | [tests.yml](../../.github/workflows/tests.yml) runs the `forecastapi` suite. It has no deploy credentials |

**What a merge to `main` does.** If the commit changes a file under `includedFiles` (`forecastapi/**`, including `forecastapi/cloudbuild.yaml`) and that file is not ignored, Cloud Build installs `forecastapi/requirements-dev.txt`, runs `python -m pytest forecastapi`, builds the image, and deploys `forecast-api`. A failing test stops the deploy.

**What it does not do.** It does not deploy `swiftbackend`. It does not edit trigger `76bbca35` (`includedFiles` there stay the `camdetect/` runtime paths). `ignoredFiles` is `camdetect/**`, so a camera-only commit does not start this build. The branch filter is `^main$`, so a push of `eval-integrity-overhaul` does not deploy. It does not load a joblib, a SavedModel, or TensorFlow, and it does not attach a GPU.

BigQuery jobs still run in `US` because dataset `traffic_prediction` is in `US`.

### Smoke check

After the first green build on `main`, the URL exists. Until then this command has nothing to call.

```text
gcloud run services describe forecast-api --project=swiftborder --region=asia-southeast1 --format="value(status.url)"
$token = gcloud auth print-identity-token
curl.exe -H "Authorization: Bearer $token" "https://REPLACE_ME.asia-southeast1.run.app/?list=models"
curl.exe -H "Authorization: Bearer $token" "https://REPLACE_ME.asia-southeast1.run.app/?model=lin_h30&direction=SG_TO_MY"
```

Expect HTTP 200 on the list, and on the forecast: `model` `lin_h30`, `version` `2026-09-12T06:15:33.163Z`, `horizon_min` 30, `source` `ML.PREDICT`, and `directions.SG_TO_MY.forecast_min`. A second forecast within five minutes should repeat `generated_at`. `GET /?model=lstm` should be HTTP 400. A callable model with no row for the requested direction in the query window returns HTTP 503 (`No recent forecast for the requested direction`), which is a data gap rather than a failed deploy.

### Rollback

Send all traffic to the previous revision. Do not delete `Gmap-Woodlands` and do not touch `swiftbackend`.

```text
gcloud run revisions list --service=forecast-api --region=asia-southeast1 --project=swiftborder
gcloud run services update-traffic forecast-api --region=asia-southeast1 --project=swiftborder --to-revisions=PREVIOUS_REVISION=100
```

To take the API off the internet without removing the image, deploy is not required: `gcloud run services delete forecast-api --region=asia-southeast1 --project=swiftborder` removes the service only. The trigger will recreate it on the next `main` change under `forecastapi/`.

### Trigger

Created 2026-10-03. It has not built anything, and Cloud Run still has no `forecast-api` service.

| | |
| --- | --- |
| Name | `forecast-api` |
| Id | `949ff029-31c9-4521-8ff4-d0e8d16dfa25` |
| GitHub | `sozuken-max/swiftborder`, push branch `^main$` |
| Config | `forecastapi/cloudbuild.yaml` |
| `includedFiles` | `forecastapi/**`, `forecastapi/cloudbuild.yaml` |
| `ignoredFiles` | `camdetect/**` |
| Build identity | the same Cloud Build service account as trigger `76bbca35` |

Re-check with `gcloud builds triggers describe 949ff029-31c9-4521-8ff4-d0e8d16dfa25 --project=swiftborder --region=global`. Do not add a pull-request pattern and do not point this trigger at `eval-integrity-overhaul`. A merge of that branch to `main` is what first deploys the service.
