# Release checklist: PR #2 (`eval-integrity-overhaul`)

PR #2 touches four live things: a `swiftbackend` redeploy and a first `forecast-api` deploy (both on merge), the weather tables in BigQuery (already written), and the snapshots that allow reverting them (they expire). This page lists owners, checks and rollback for each. The redeploy, weather, and snapshot facts were re-queried read-only on 2026-10-01. The `forecast-api@` IAM row in the GCP changes table was checked on 2026-10-03. If the project disagrees, the project wins.

| Item | State | Owner | Deadline |
| --- | --- | --- | --- |
| `swiftbackend` credential / public access ([findings.md](findings.md#known-risk-public-swiftbackend-documented-not-changed)) | **Accepted as is** (team decision 2026-10-03): settings unchanged, not a merge blocker | — | Revisit if credit use looks abnormal |
| `swiftbackend` redeploy on merge | Pending merge | _team to assign_ | At merge |
| `forecast-api` first deploy on merge (trigger `949ff029`) | Pending merge | _team to assign_ | At merge |
| Weather append (1–30 Sep 2026) | Applied 2026-10-01 ~01:50 SGT | _team to assign_ | Retention decision before snapshots expire |
| Snapshots `*_snapshot_20261001` | Expire **2026-10-31 01:41 SGT** | same | 2026-10-30 |
| Deck PNGs `docs/images/*.png` | Stale; **handled separately from this PR** (team decision 2026-10-03). Regenerate per [diagram skill](../skills/diagram-image-generation/SKILL.md) | Implementer; Yingzhao signs off | Before the next deck |

Commands below assume the Windows gcloud setup in [inventory.md](inventory.md#querying-this-project-from-a-windows-dev-box) and `--project swiftborder`.

## 1. `swiftbackend` redeploy (on merge to `main`)

Merging changes `camdetect/main.py` and `camdetect/requirements.txt`, which starts Cloud Build trigger `76bbca35`: pytest, then a buildpacks deploy to Cloud Run `swiftbackend` (`europe-west1`). There is no staging service. The checks below are a post-deploy smoke test with a one-command rollback.

**Before merge**

1. Record the serving revision (on 2026-10-01: `swiftbackend-00023-f8g`, 100% of traffic):
   ```powershell
   gcloud run services describe swiftbackend --region europe-west1 --project swiftborder --format="value(status.traffic[0].revisionName,status.traffic[0].percent)"
   ```
2. Confirm `camdetect` tests pass on the merge commit (GitHub Actions `pytest (camdetect)`; Cloud Build runs them again).

**After the build succeeds** (`gcloud builds list --project swiftborder --limit 1`)

3. New revision serves 100% and keeps the same env var **names** (`ROBOFLOW_API_KEY`, `CACHE_BUCKET`) and ingress / IAM settings:
   ```powershell
   gcloud run services describe swiftbackend --region europe-west1 --project swiftborder --format="yaml(spec.template.spec.containers[0].env[].name,status.traffic,metadata.annotations)"
   ```
4. Smoke test. Steps a and b make no billed call; step c makes **one** Roboflow call.
   ```powershell
   $u = gcloud run services describe swiftbackend --region europe-west1 --project swiftborder --format="value(status.url)"
   curl.exe -s -o NUL -w "%{http_code}`n" -X OPTIONS "$u/"                                         # a. expect 204
   curl.exe -s -w "`n%{http_code}`n" "$u/?camera_id=2701&date_time=not-a-date&format=json"          # b. expect 400, generic message
   curl.exe -s "$u/?camera_id=2701&format=json"                                                     # c. expect 200 JSON with directions.sg_my / my_sg (count, congestion, extent)
   ```
   Also check that error responses carry no stack traces or upstream URLs, and that the logs show no new errors:
   ```powershell
   gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="swiftbackend" AND severity>=ERROR' --project swiftborder --freshness 30m --limit 20
   ```
5. **Rollback** (instant, traffic only; the new revision stays for inspection):
   ```powershell
   gcloud run services update-traffic swiftbackend --region europe-west1 --project swiftborder --to-revisions swiftbackend-00023-f8g=100
   ```
   Then revert the merge on `main` so the next push does not redeploy the same code.

The redeploy does not change authentication. The service stays public until item 3 is decided.

## 1b. `forecast-api` first deploy (on merge to `main`)

**Later correction (2026-10-04).** The checklist below is what PR #2 expected at first deploy: a private service, anonymous 403, and `xgb[maps]` not callable. Live `forecast-api` is public. The checked-in smoke test accepts anonymous 200 or 503 for `model=served`. Callable ids include `xgb[maps]`, `xgb[maps+prof]`, and `profile`. Current revision and smoke result: [inventory.md](inventory.md). Do not treat the curl comments below as the current contract.

The merge commit adds `forecastapi/**`, which starts Cloud Build trigger `forecast-api` (`949ff029`). The build:

1. runs the `forecastapi` tests and the harness equivalence test;
2. builds the image;
3. **creates** Cloud Run `forecast-api` in `asia-southeast1` (private, serving local models, ADR 0004);
4. runs a smoke test inside the build.

Merging therefore deploys two services. Later builds deploy a no-traffic candidate and move traffic only after the smoke test passes. The first deploy has no earlier revision, so it takes traffic directly. Full detail: [runbooks/forecast-api.md](runbooks/forecast-api.md).

**After the build succeeds.** The build's `Smoke` step already checked: no token gives 403, `list=models` 200, `model=served` 200 or 503, `model=lstm` 400. Then, by hand:

1. The service exists, is private and runs as `forecast-api@`:
   ```powershell
   gcloud run services describe forecast-api --region asia-southeast1 --project swiftborder --format="value(status.url,spec.template.spec.serviceAccountName)"
   gcloud run services get-iam-policy forecast-api --region asia-southeast1 --project swiftborder   # no allUsers
   ```
2. Smoke test with an identity token (each forecast call runs one small BigQuery job):
   ```powershell
   $u = gcloud run services describe forecast-api --region asia-southeast1 --project swiftborder --format="value(status.url)"
   $t = gcloud auth print-identity-token
   curl.exe -s -o NUL -w "%{http_code}`n" "$u/?list=models"                                    # expect 403 (no token)
   curl.exe -s -H "Authorization: Bearer $t" "$u/?list=models"                                 # expect 200, catalog
   curl.exe -s -H "Authorization: Bearer $t" "$u/?model=served"                                # expect 200, both directions; equals v_forecast_recent
   curl.exe -g -s -w "`n%{http_code}`n" -H "Authorization: Bearer $t" "$u/?model=xgb[maps]"     # expect 400 model is not deployed
   ```
   A 502 on `served` means the runtime identity cannot read a dataset the views use; check the dataset grants in [inventory.md](inventory.md).
3. **Rollback:** `gcloud run services delete forecast-api --region asia-southeast1 --project swiftborder` (service only; the image and trigger stay). Then revert `forecastapi/` on `main`, or disable trigger `949ff029`, so the next push does not recreate it.

## GCP changes made for this PR

All other access to project `swiftborder` was read-only.

| When (SGT) | Change | Revert |
| --- | --- | --- |
| 2026-10-01 01:50 | Weather append, `update_timestamp` column, snapshots (section 2) | Section 2 |
| 2026-10-03 (by the `ae5d034` author) | Cloud Build trigger `forecast-api` (`949ff029`, `^main$`, `forecastapi/**`, ignores `camdetect/**`); service account `forecast-api@` with project `bigquery.jobUser` and `bigquery.dataViewer` | `gcloud builds triggers delete 949ff029-31c9-4521-8ff4-d0e8d16dfa25`; `gcloud iam service-accounts delete forecast-api@swiftborder.iam.gserviceaccount.com` |
| 2026-10-03 | `forecast-api@`: removed project-wide `bigquery.dataViewer`; granted `dataViewer` on datasets `traffic_prediction` and `causeway` only | `REVOKE` the two dataset grants and re-add the project binding |
| 2026-10-03 | Build SA `1095552466513-compute@`: `roles/iam.serviceAccountOpenIdTokenCreator` on itself, so the `forecast-api` Smoke step can mint an ID token (follow-up to PR #4) | `gcloud iam service-accounts remove-iam-policy-binding 1095552466513-compute@developer.gserviceaccount.com --member serviceAccount:1095552466513-compute@developer.gserviceaccount.com --role roles/iam.serviceAccountOpenIdTokenCreator` (Smoke then fails again) |

Build identity: the trigger builds as the default Compute Engine service account, which usually holds broad project roles. **Accepted for this student project** (team decision 2026-10-03). A dedicated build identity (Cloud Run admin on the one service, `iam.serviceAccountUser` on `forecast-api@`, Artifact Registry writer, logs writer) is the narrower option if that changes.

## 2. Weather append: verify, keep or revert

**Verify** (read-only; expected values are from the 2026-10-01 load):

```sql
-- rainfall: 103,922 rows, one station name, no duplicate timestamps, last reading 2026-09-30 15:55 UTC
SELECT COUNT(*) n, COUNT(DISTINCT station_id) stations, COUNT(*) - COUNT(DISTINCT timestamp) dup_ts, MAX(timestamp) last_ts
FROM `swiftborder.rainfall.rainfall`;
-- forecast: 24,482 rows, 1,876 with update_timestamp, no duplicate issues, last issue 2026-09-30 15:30 UTC
SELECT COUNT(*) n, COUNTIF(update_timestamp IS NOT NULL) with_update, COUNT(*) - COUNT(DISTINCT issue_timestamp) dup_issue, MAX(issue_timestamp) last_issue
FROM `swiftborder.weatherforecast.weatherforecast`;
```

Run with `bq query --nouse_legacy_sql --location asia-southeast1 --project_id swiftborder`. The snapshots hold the pre-load tables (95,282 and 22,606 rows).

**Revert** (destructive; restores the tables to the snapshots and drops the `update_timestamp` column):

```sql
CREATE OR REPLACE TABLE `swiftborder.rainfall.rainfall` CLONE `swiftborder.rainfall.rainfall_snapshot_20261001`;
CREATE OR REPLACE TABLE `swiftborder.weatherforecast.weatherforecast` CLONE `swiftborder.weatherforecast.weatherforecast_snapshot_20261001`;
```

**Retention decision** (record here before 2026-10-30):

- [ ] **Keep the append, let snapshots expire.** Default if the verify queries match and nothing downstream broke. Nothing reads these tables on a schedule; `v_weather_features_10min` now returns bins to 30 Sep.
- [ ] **Keep the append, extend the snapshots** (for example to the final report on 31 Oct plus a week):
  ```powershell
  bq update --expiration 1209600 swiftborder:rainfall.rainfall_snapshot_20261001
  bq update --expiration 1209600 swiftborder:weatherforecast.weatherforecast_snapshot_20261001
  ```
- [ ] **Revert** with the commands above.

Decision: ______ Date: ______ By: ______

## 3. Credential and public access (accepted, unchanged)

**Team decision 2026-10-03:** keep `swiftbackend` as it is and merge without changing it. The steps below are recorded for later; none is planned.

`swiftbackend` accepts unauthenticated calls, allows any origin, and holds `ROBOFLOW_API_KEY` as a plain env var. This PR documents that and does not change it. The redeploy keeps the current settings. Each fix below changes production and needs the team's approval. The first two also need the Roboflow workspace owner.

1. Rotate the Roboflow key in the Roboflow workspace. Anyone who has read the service configuration has the current key.
2. Store the new key in Secret Manager and expose it to the service under the same variable name (`gcloud run services update swiftbackend --update-secrets ROBOFLOW_API_KEY=<secret>:latest`). No code change is needed.
3. Restrict callers: set `ALLOWED_ORIGIN`; require an invoker identity or an API key. This breaks the anonymous `curl` examples in the README and any public demo page, so decide the demo path first.

## 4. Merge order

1. Section 3 is accepted as is and does not block the merge. The PR stays as one PR (team decision 2026-10-03).
2. Merge PR #2, then work through section 1 steps 3–5 and section 1b.
3. Section 2 retention decision by 2026-10-30.
4. Regenerate the deck PNGs before the next presentation.
