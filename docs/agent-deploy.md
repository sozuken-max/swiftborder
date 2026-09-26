# Agent instructions: bring a local deploy into the repo

**Audience:** a coding agent working for a teammate who deployed SwiftBorder from their own machine (Cloud Run, Cloud Build, Scheduler, BigQuery).  
**Project:** `swiftborder` (`1095552466513`). That project is the source of truth. This repo is not.  
**As of the last query:** 26 September 2026, ~13:40 SGT.

Read [AGENTS.md](../AGENTS.md) first. Do not invent services, env vars, or metrics. Do not commit API keys.

---

## What is already in git

| Path | What it is | What it is not |
| --- | --- | --- |
| `camdetect/main.py` | HTTP function `detect`. Resolves an LTA frame, calls a Roboflow workflow, counts vehicles against a dividing line. | A BigQuery writer. It does not insert into `Cam2701` or `Cam2702`. |
| `camdetect/requirements.txt` | `functions-framework`, `requests`, `Pillow`, `google-cloud-storage` | `google-cloud-storage` is unused. There is no `Dockerfile` and no `cloudbuild.yaml`. |
| `Causeway/*.py` | One-shot CSV downloads of NEA rainfall and the 2-hour forecast from data.gov.sg. Output goes to `Causeway/data/`, which is gitignored. | The BigQuery loaders for `rainfall.rainfall` and `weatherforecast.weatherforecast`. Those tables are not written by these scripts. |
| `docs/` | Report drafts, inventory, roadmap | The Maps fetcher, the backfill job, or the forecast SQL |

Direction names in `camdetect` are `SG-MY` and `MY-SG`. The BigQuery congestion view maps `to_JB` / `to_Woodlands` onto `SG_TO_MY` / `MY_TO_SG`. Any join has to translate those names. Do not assume they already match.

The only dividing line in code is camera **2701**.

---

## What runs in the project and is not in git

Queried 26 Sep 2026. Re-query before you edit a service.

| Live resource | Region | In git? |
| --- | --- | --- |
| Cloud Run `swiftbackend` | `europe-west1` | Source is `camdetect/` on `main`. Deploy is an **inline** Cloud Build trigger (`76bbca35-c1b4-4836-9f34-d7adda53ea17`), push to `^main$`, buildpacks, function target `detect`. No test step. No `cloudbuild.yaml` in the repo. **Included files** are code and config under `camdetect/` only (`*.py`, requirements, `pytest.ini`, Dockerfile, Cloud Build and data files). `README.md` and the rest of the repo do not start this build. |
| Cloud Run `gmap-woodlands-fetcher` | `asia-southeast1` | No |
| Cloud Scheduler `Gmap-Woodlands` | `asia-southeast1` | No. It calls the fetcher every 5 minutes. |
| Cloud Run job `traffic-backfill` | `asia-southeast1` | No |
| Views and models under `traffic_prediction`, plus the weather and congestion views | BigQuery | No |

`swiftbackend` was last deployed 20 Sep 2026 from commit `10500b3`. If `main` has moved, compare the running image's commit label with `git rev-parse HEAD` before you change the pipeline.

---

## Rules

1. Export the running service. Do not rewrite it from memory and call the result production.
2. If the source that was deployed cannot be recovered from the teammate's machine or from Cloud Build source, stop. Write that in the service note. Do not fabricate a fetcher.
3. Secrets stay in Secret Manager or Cloud Run env configuration. Commit variable **names** only. `.env` is gitignored.
4. One service, one directory, one note. A push that only changes `camdetect/` must not redeploy the Maps fetcher.
5. Add a test step that can fail the build before the deploy step. The live `swiftbackend` trigger does not have one. Matching that omission is not the goal.
6. After the change, update [inventory.md](inventory.md) from a fresh query and add a line to the claims register in [findings.md](findings.md) only if what is deployed changed.

---

## Procedure

Work in `swiftborder`. The local `gcloud` default project may be something else. Pass `--project=swiftborder` on every command.

### 1. Identify the service you own

```bash
gcloud run services list --project=swiftborder --format="table(metadata.name,region,status.latestReadyRevisionName)"
gcloud run jobs list --project=swiftborder --format="table(metadata.name,region)"
gcloud builds triggers list --project=swiftborder --format="table(id,name,github.name)"
gcloud scheduler jobs list --project=swiftborder --location=asia-southeast1
```

### 2. Export the running spec

Replace `SERVICE` and `REGION`.

```bash
gcloud run services describe SERVICE --project=swiftborder --region=REGION --format=export > /tmp/SERVICE.yaml
gcloud run services describe SERVICE --project=swiftborder --region=REGION --format="yaml(spec.template.spec.containers[0].env,spec.template.metadata.annotations)"
```

For a job:

```bash
gcloud run jobs describe JOB --project=swiftborder --region=REGION --format=export > /tmp/JOB.yaml
```

From the export, record: container image, command, memory, timeout, service account, and env var **names**. Redact values that look like keys before the file is committed. Prefer not to commit the raw export. Commit a short note instead.

### 3. Recover the source

Try, in order:

1. The teammate's local directory that they last deployed.
2. `gcloud builds log BUILD_ID --project=swiftborder` and the Cloud Storage source object for that build (`run-sources-swiftborder-<region>`).
3. The image's commit label, then `git show COMMIT:path`.

Copy the source into a new top-level directory named after the service, for example `gmap-woodlands-fetcher/`. Include a `requirements.txt` or a `Dockerfile`, whichever the running build actually uses.

For `swiftbackend`, the source is already `camdetect/`. Do not create a second copy.

### 4. Add a Cloud Build file that matches the running deploy

Put `cloudbuild.yaml` next to that service, or one file at the repo root with a `dir` per step. The `swiftbackend` build observed on 26 Sep does this:

- Builder image `gcr.io/k8s-skaffold/pack`, buildpacks, `--path=camdetect`, env `GOOGLE_FUNCTION_TARGET=detect`
- Image `europe-west1-docker.pkg.dev/swiftborder/cloud-run-source-deploy/swiftborder/swiftbackend:$COMMIT_SHA`
- `gcloud run services update swiftbackend --region=europe-west1 --image=...`

Reproduce those values. Insert a test step **before** the image push. For `camdetect`, unit-test the dividing-line and direction functions with no network and no Roboflow key. A failing test must stop the deploy.

Point the existing trigger at the file instead of leaving the build inline:

```bash
gcloud builds triggers describe 76bbca35-c1b4-4836-9f34-d7adda53ea17 --project=swiftborder
```

Update that trigger only after the file builds the same image path and region. Do not create a second trigger that also deploys `swiftbackend`.

For a service that has no trigger, add one that runs only when its directory changes.

### 5. Document it in the same pull request

Create `docs/deploy/SERVICE.md` from the template below. Link it from [docs/README.md](README.md). If you learned the service does something the inventory does not say, update [inventory.md](inventory.md) from `gcloud` / `bq`, not from memory.

### 6. Open the pull request

State: which live resource you captured, the revision or commit you compared, which env names are required, and what you could not recover. Do not claim a new model score.

---

## Service note template

```markdown
# SERVICE

- GCP name:
- Kind: Cloud Run service | Cloud Run job | scheduled job
- Region:
- Source directory in this repo:
- Cloud Build file:
- Trigger: id, or "none yet"
- Scheduler: name and cron, or "none"
- Entry point / command:
- Env var names (no values):
- Secret Manager ids, if any:
- Compared to live revision: NAME at YYYY-MM-DD
- Not recovered:
```

---

## `camdetect` follow-ups that are feasible now

These do not require the missing fetcher source.

1. Add tests for `_classify_direction`, `_y_on_line`, and `_congestion_level` using the built-in 2701 line. No API key.
2. Drop `google-cloud-storage` from `requirements.txt` unless a new code path imports it.
3. Add `Causeway/requirements.txt` with `requests` so the CSV scripts install cleanly. Do not describe them as the BigQuery weather pipeline.
4. When a 2702 line is calibrated, add it under `DEFAULT_DIVIDING_LINES` in the same shape as 2701. Until then, 2702 counts stay `Unknown`.

## Code that is still missing after a successful export

- SQL for `v_bins_10min`, `v_training_set`, `v_forecast_recent`, `v_weather_features_10min`, `v_congestion_index_10min`, and the `CREATE MODEL` text for `lin_h30` and `xgb_h30`. Export with `bq show --view` and `bq show --model`. Commit the SQL. That is roadmap step 3.
- A checked-in evaluation script. That is roadmap step 2. CI should run unit tests, not a paid BigQuery training job, on every push.
