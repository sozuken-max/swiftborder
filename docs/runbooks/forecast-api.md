# Runbook: `forecast-api`

A private HTTP read of the Woodlands 30-minute forecast of Google Maps `duration_in_traffic`. It is served from **local models** that the service fits itself ([ADR 0004](../adr/0004-serve-local-models.md)). BigQuery ML (`lin_h30`, `xgb_h30`, `v_forecast_recent`) is no longer called.

- **Code:** [`forecastapi/main.py`](../../forecastapi/main.py) and [`forecastapi/local_models.py`](../../forecastapi/local_models.py).
- **CI/CD:** [`forecastapi/cloudbuild.yaml`](../../forecastapi/cloudbuild.yaml), run by trigger `forecast-api` (`949ff029`) on a push to `main` that touches `forecastapi/**`.
- **Not** `swiftbackend`, and not trigger `76bbca35`.

What it is not:
- **Not a crossing-time measurement.** The label is Maps' own estimate.
- **Not a 24-hour forecast.** Only 30-minute models are callable.
- **Not the public `traffic-24h.json`** the Firebase cards read ([ADR 0001](../adr/0001-firebase-client-api-calls.md)).

## How a forecast is made

1. **Latest inputs.** The latest `v_training_set` row per direction: `lag_60` present, within the last 24 hours. This is one BigQuery read per request; responses are cached for 5 minutes.
2. **Model.** Each model is fitted in the process and cached until the SGT day changes. The rows and settings are those of the harness fold for that day ([`eval/joined.py`](../../eval/joined.py)):

| `model` | Harness column | Training rows | Settings |
| --- | --- | --- | --- |
| `xgb[maps]` | `xgb[maps]` | every label observed before 00:00 SGT today | XGBoost: 300 trees, learning rate 0.05, depth 4, seed 42 |
| `ridge[maps]` | `ridge[maps]` | same | median impute + indicators, standardise, ridge alpha 1 |
| `xgb_bq[daily]` | `xgb_bq[daily]` | same, rows with `lag_60` | BQML `xgb_h30` settings (28 trees, base 0.5, 5 seeds) |
| `lin_bq[daily]` | `lin_bq[daily]` | same, rows with `lag_60` | BQML `lin_h30` settings (ridge L2 0.1, standardised) |
| `xgb_bq[frozen]` | `xgb_bq[frozen]` | BQML's 12 Sep rows (1,406) | as `xgb_bq[daily]` |
| `lin_bq[frozen]` | `lin_bq[frozen]` | BQML's 12 Sep rows | as `lin_bq[daily]` |
| `persistence` | `persistence` | none | the latest bin mean (`y_persistence`) |
| `served` | — | — | per-direction choice in `SERVED_SELECTION` |

`eval/tests/test_forecastapi_models.py` asserts that the service and the harness use the same feature lists and settings, and that a service fit predicts exactly what the harness predicts on the same rows. It runs in GitHub Actions and in the Cloud Build `Test` step.

**`served` today** is an interim choice until the frozen run: `SG_TO_MY` → `lin_bq[frozen]`, `MY_TO_SG` → `persistence`. That is the 12 Sep registry rebuilt from local replicas. After the frozen run (on or after 20 Oct), set `SERVED_SELECTION` by the ADR 0004 rule: per direction, a model that beats persistence in Run A and is confirmed in Run B, both significant under run-wide Holm, otherwise persistence. Then change `SELECTION_ID` and push to `main`.

## Request

`GET` with a query string (or a JSON body; a body value wins unless it is missing, null or `""`).

| Parameter | Default | Values |
| --- | --- | --- |
| `list` | none | `models` returns the catalog and a description of these parameters |
| `model` | `served` | `served`, `persistence`, or a model id above, used for every requested direction; other catalog ids return 400 |
| `model_sg_to_my` | none | a callable id for `SG_TO_MY` only; overrides `model` there |
| `model_my_to_sg` | none | a callable id for `MY_TO_SG` only; overrides `model` there |
| `version` | current | must equal the current version, otherwise 400; not allowed with a per-direction override |
| `direction` | `both` | `SG_TO_MY`, `MY_TO_SG`, `both` |
| `horizon_min` | 30 | 30 only |

**Manual selection.**
- `?model=xgb[maps]` uses one model for both directions.
- `?model=xgb[maps]&model_my_to_sg=persistence` mixes models: the response says `"model": "custom"`, names the model in each direction, and its `version` lists each direction's model and version.
- An override for a direction that was not requested (`direction=MY_TO_SG&model_sg_to_my=...`) returns 400.
- With no parameters at all, the request returns `served`.

**Versions:**
- daily models: `labels_before=YYYY-MM-DDT00:00:00+08:00`;
- frozen replicas: `bqml-replica-2026-09-12`;
- `persistence`: `latest-bin`;
- `served`: the `SELECTION_ID`.

## Response

```json
{
  "model": "served", "version": "registry-2026-09-12-local-replica", "horizon_min": 30,
  "source": "local model", "label": "maps_duration_in_traffic_min", "commit": "<git sha>",
  "directions": {
    "SG_TO_MY": {"forecast_min": 34.6, "forecast_for": "...Z", "origin_ts": "...Z", "model": "lin_bq[frozen]"},
    "MY_TO_SG": {"forecast_min": 48.5, "forecast_for": "...Z", "origin_ts": "...Z", "model": "persistence"}
  },
  "model_meta": {"lin_bq[frozen]": {"eval_id": "lin_bq[frozen]", "version": "bqml-replica-2026-09-12", "training_rows": 1406, "seeds": []},
                 "persistence": {"eval_id": "persistence", "version": "latest-bin"}}
}
```

| Status | When |
| --- | --- |
| 400 | Unknown model (in `model` or an override), model not deployed, stale `version`, `version` with an override, an override for a direction not requested, bad `direction` or `horizon_min`, `list` other than `models` |
| 405 | Method other than `GET` / `OPTIONS` |
| 502 | BigQuery read or fit failed (`{"error": "Forecast query failed"}`; details stay in the log) |
| 503 | No recent row for a requested direction (a data gap, not a failed deploy) |

## Performance

A local run against live BigQuery on 2026-10-03 took:
- **Cold start:** about 13 s for the first `served` request (training read plus fits).
- **Each later request:** about 3 s, mostly the latest-row read; a repeat inside 5 minutes is served from the cache.

The fitted daily models are reused until 00:00 SGT. Memory is 1 GiB.

## CI/CD (`forecastapi/cloudbuild.yaml`)

1. **Test:** `pytest forecastapi`, then the harness equivalence test.
2. **Buildpack:** builds the image from `forecastapi/` (Python 3.11; all images pinned by digest).
3. **Deploy:** if the service exists, the new revision is deployed with **no traffic** and tag `candidate`. The first deploy takes traffic directly, because there is no earlier revision. The service is private (`--no-allow-unauthenticated`) and runs as `forecast-api@`. It sets `BQ_PROJECT` and `COMMIT_SHA`.
4. **Smoke:** against the candidate URL with an identity token, the build checks the cases below. Inside Cloud Build, `gcloud auth print-identity-token` fails and the metadata identity endpoint returns 404. The step therefore mints the token with the IAM Credentials `generateIdToken` API. That needs the build SA (`1095552466513-compute@`) to hold `roles/iam.serviceAccountOpenIdTokenCreator` on itself (granted 2026-10-03).
   - no token → 403;
   - `list=models` → 200 and lists `served`;
   - `model=served` → 200 with `"source": "local model"` (503 is accepted as a data gap);
   - `model=lstm` → 400.
5. **Promote:** traffic moves to the new revision and the `candidate` tag is removed.

If Test, Deploy or Smoke fails, traffic stays on the previous revision. GitHub Actions runs the same `forecastapi` suite, the equivalence test, and a static check of the build file (`scripts/tests/test_forecastapi_cloudbuild.py`) on every push and PR.

## Operations

```powershell
$u = gcloud run services describe forecast-api --region asia-southeast1 --project swiftborder --format="value(status.url)"
$t = gcloud auth print-identity-token
curl.exe -s -H "Authorization: Bearer $t" "$u/?model=served"
curl.exe -g -s -H "Authorization: Bearer $t" "$u/?model=xgb[maps]&direction=SG_TO_MY"
curl.exe -g -s -H "Authorization: Bearer $t" "$u/?model=xgb_bq[daily]&model_my_to_sg=persistence"
```

On 2026-10-03 a user token from `gcloud auth print-identity-token` got 401 from this service. If yours does too, call it through a service account you may impersonate (`--impersonate-service-account=<sa> --audiences=$u`). That needs `roles/iam.serviceAccountOpenIdTokenCreator` on that SA, and the SA needs invoke rights on the service.

- **Roll back:** `gcloud run services update-traffic forecast-api --region asia-southeast1 --project swiftborder --to-revisions <previous-revision>=100`.
- **Remove:** `gcloud run services delete forecast-api --region asia-southeast1 --project swiftborder`. Then disable trigger `949ff029` so the next push does not recreate it.
- **Access:** callers need `roles/run.invoker` on the service. A browser on Firebase Hosting cannot attach an identity token by itself; publishing the service is a separate decision.
- **Identity:** `forecast-api@swiftborder.iam.gserviceaccount.com`: BigQuery job user on the project and data viewer on `traffic_prediction` and `causeway` only. It holds no keys.
