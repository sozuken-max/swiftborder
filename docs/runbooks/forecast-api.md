# Runbook: `forecast-api`

A **public** HTTP read of the Woodlands 30-minute forecast (the invoker IAM check has been disabled since 2026-10-03, so the Firebase page can call it from the browser) of Google Maps `duration_in_traffic`. It is served from **local models** that the service fits itself ([ADR 0004](../adr/0004-serve-local-models.md)). BigQuery ML (`lin_h30`, `xgb_h30`, `v_forecast_recent`) is no longer called.

- **Code:** [`forecastapi/main.py`](../../forecastapi/main.py) and [`forecastapi/local_models.py`](../../forecastapi/local_models.py).
- **CI/CD:** [`forecastapi/cloudbuild.yaml`](../../forecastapi/cloudbuild.yaml), run by trigger `forecast-api` (`949ff029`) on a push to `main` that touches `forecastapi/**`.
- **Not** `swiftbackend`, and not trigger `76bbca35`.

What it is not:
- **Not a crossing-time measurement.** The label is Maps' own estimate.
- **Not an evaluated long-range forecast.** Only the 30-minute forecast is evaluated by the report harness. Horizons from 60 min to 24 h and the 30-minute forecast curve are exploratory (`status: exploratory`). Past 5.5 h the curve is the calendar-profile baseline. See [Exploratory horizons and the profile baseline](#exploratory-horizons-and-the-profile-baseline).
- **Not the public `traffic-24h.json`** the Firebase cards read ([ADR 0001](../adr/0001-firebase-client-api-calls.md)).

## How a forecast is made

1. **Latest inputs and the availability contract.** Each uncached request reads the last 3 hours of **closed** `v_bins_10min` bins. A bin `[t, t+10)` counts as closed once `t + 10 min + INGEST_GRACE` (60 s, env `INGEST_GRACE_SECONDS`) has passed; the harness treats a bin as known at `t + 10`. The features (lags, rolling means, `after_gap`) are built in Python with the harness's time-based rules (`local_models.features_from_bins`, a copy of `eval/features.maps_features`). They are not read from `v_training_set`, whose positional LAG/LEAD shift by a bin after a skipped bin. The origin is the newest closed bin. It is served only while its target bin `[t+30, t+40)` has not started, and only when no gap of more than 25 minutes lies in the six bins before it (`after_gap = 0`, as in the scored rows). Otherwise that direction answers **503**; a 200 never carries an expired forecast. With current ingestion the target is 9–19 minutes ahead of the request. One missed fetch still leaves an earlier origin servable, with less lead. Responses are cached for 5 minutes, and a cached answer is dropped as soon as its target bin starts.
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

**`served` today** is an interim choice until the frozen run: `SG_TO_MY` → `lin_bq[frozen]`, `MY_TO_SG` → `persistence`. That is the 12 Sep registry rebuilt from local replicas, not a weighted ensemble. The response keeps `"model": "served"` at the top level. Each direction's `model` is the id that produced that value (`lin_bq[frozen]` or `persistence`). After the frozen run (on or after 20 Oct), set `SERVED_SELECTION` by the ADR 0004 rule: per direction, a model that beats persistence in Run A and is confirmed in Run B, both significant under run-wide Holm, otherwise persistence. Then change `SELECTION_ID` and push to `main`.

## Request

`GET` with a query string (or a JSON body; a body value wins unless it is missing, null or `""`).

| Parameter | Default | Values |
| --- | --- | --- |
| `list` | none | `models` returns the catalog (callable and not) and a description of these parameters. `cards` returns the model-card array (same body as `view=cards`). `horizon-study` returns the exploratory study summary |
| `view` | none | `cards` returns the same JSON array as `list=cards` |
| `model` | `served` | A callable id, used for every requested direction. Callable: `served`, `persistence`, `profile`, `ridge[maps]`, `xgb[maps]`, `xgb[maps+prof]`, `lin_bq[daily]`, `xgb_bq[daily]`, `lin_bq[frozen]`, `xgb_bq[frozen]`. Other catalog ids (`lstm`, scored ensembles, and the rest with `deploy_state` `artifact` or `code-only`) return 400 `model is not deployed`. Unknown ids return 400 |
| `model_sg_to_my` | none | a callable id for `SG_TO_MY` only; overrides `model` there |
| `model_my_to_sg` | none | a callable id for `MY_TO_SG` only; overrides `model` there |
| `version` | current | must equal the current version, otherwise 400; not allowed with a per-direction override |
| `direction` | `both` | `SG_TO_MY`, `MY_TO_SG`, `both` |
| `horizon_min` | 30 | 30 for every model. Any multiple of 30 up to 1440 (exploratory) for `served`, `persistence`, `xgb[maps]`, `xgb[maps+prof]` and `profile` |
| `baseline` | none | `profile` returns the baseline curve instead of a forecast (with `hours`, 1–24, default 24) |
| `curve` | none | `forecast` returns forecasts every 30 minutes (with `hours`, 1-24, default 2); see [the forecast curve](#exploratory-horizons-and-the-profile-baseline). Model points stop at 5.5 h (generous end of the study). Later points are the profile |

**Manual selection.**
- `?model=xgb[maps]` uses one model for both directions.
- `?model=xgb[maps]&model_my_to_sg=persistence` mixes models: the response says `"model": "custom"`, names the model in each direction, and its `version` lists each direction's model and version.
- An override for a direction that was not requested (`direction=MY_TO_SG&model_sg_to_my=...`) returns 400.
- With no parameters at all, the request returns `served`.

**Versions:**
- daily models: `labels_before=YYYY-MM-DDT00:00:00+08:00`;
- frozen replicas: `bqml-replica-2026-09-12`;
- `persistence`: `latest-bin`;
- `served`: the `SELECTION_ID` at 30 min; `horizon-study-2026-10-04` (`EXPLORATORY_SELECTION_ID`) at other horizons;
- `profile`: `labels_before=...` (refitted each SGT day).

## Model cards

`GET /?list=cards` and `GET /?view=cards` return a JSON array. Each object has `id` (the `model` query value, except the two code-only cards below), `name`, `summary`, `kind` (`single` or `mix`), `callable`, `default_for` (or null), and `members` on a mix. A single model omits `members`. `callable` matches the allow-list above. `timesfm` and `fcm_mlp` are cards only: `Causeway/layer_b_timesfm.py` and `Causeway/layer_b_fcm_mlp.py` are not called by this service, and both summaries start with "Not served."

The Hosting chart's Typical line is `profile` (`?baseline=profile`), the calendar profile. `served` at 30 minutes is the registry choice, with no weights.

```json
[
  {
    "id": "served",
    "name": "Served default",
    "kind": "mix",
    "callable": true,
    "default_for": "request with no model parameter; public curve (?curve=forecast)",
    "members": ["lin_bq[frozen]", "persistence", "xgb[maps]", "xgb[maps+prof]", "profile"]
  }
]
```

## Response

The shape below is the code in this tree. Live revision `forecast-api-00009-hax` (commit `7bd94e3`) does not include per-point `model` or `components` until this merges and trigger `forecast-api` runs. A live GET is not proof of these fields.

```json
{
  "model": "served", "version": "registry-2026-09-12-local-replica", "horizon_min": 30,
  "source": "local model", "label": "maps_duration_in_traffic_min", "target_offset_min": [30, 40],
  "commit": "<git sha>",
  "directions": {
    "SG_TO_MY": {"forecast_min": 19.9, "model": "lin_bq[frozen]",
                 "origin_ts": "2026-10-03T16:40:00Z", "origin_closed_at": "2026-10-03T16:50:00Z",
                 "forecast_for": "2026-10-03T17:10:00Z", "forecast_window_end": "2026-10-03T17:20:00Z",
                 "observation_age_min": 10.8, "lead_min": 9.2},
    "MY_TO_SG": {"forecast_min": 35.5, "model": "persistence", "...": "same timing fields"}
  },
  "model_meta": {"lin_bq[frozen]": {"eval_id": "lin_bq[frozen]", "version": "bqml-replica-2026-09-12", "training_rows": 1406, "seeds": []},
                 "persistence": {"eval_id": "persistence", "version": "latest-bin"}}
}
```

Top-level `model` is the id the caller asked for. `directions.<d>.model`, and `model` on each `?curve=forecast` point, is the id that produced that value. Those differ for `served`, because the two directions use different models, and along the curve, because the id changes with lead time. A single-model value omits `components`. A blend (nothing in `served` or the forecast curve is a blend today) adds `components` on that direction or curve point: a list of `{model, weight}` whose weights sum to 1. The service copies those weights; it does not invent or rescale them. A blend value would look like `"model": "mean[models]"` with `"components"` naming `lin_bq[frozen]` and `persistence` at weight 0.5 each.

| Status | When |
| --- | --- |
| 400 | Unknown model (in `model` or an override), a catalog id that is not callable (`model is not deployed`), stale `version`, `version` with an override, an override for a direction not requested, bad `direction` or `horizon_min`, `list` other than `models` or `horizon-study` |
| 405 | Method other than `GET` / `OPTIONS` |
| 502 | BigQuery read or fit failed (`{"error": "Forecast query failed"}`; details stay in the log) |
| 503 | No servable origin for a requested direction: ingestion stale (the newest closed bin's target has started), or within an hour after a gap of more than 25 minutes. A data condition, not a failed deploy |

**Timing fields.** `forecast_for` / `forecast_window_end` bound the target bin; the forecast is its mean Maps duration. `origin_closed_at` is when the newest observation bin closed. `observation_age_min` and `lead_min` are measured when the response is sent, including from the cache. The example is a local run against live BigQuery at 2026-10-03 17:00:17 UTC: the 16:50 bin had not cleared its grace minute, so the origin was 16:40.

## Exploratory horizons and the profile baseline

The source is [docs/horizon-study.md](../horizon-study.md): an exploratory study on 13–30 Sep, not confirmed on Run B. The figures are exposed on purpose, labelled.

- **`status`** (top level):
  - `evaluated`: a 30-minute model of the report harness;
  - `exploratory`: any horizon other than 30, or `xgb[maps+prof]`;
  - `baseline`: `model=profile`.
  - `mixed`: the directions differ, e.g. `model=served&model_my_to_sg=profile`. Each direction also carries its own `status`, so read `directions.<d>.status`.
- **No profile history:** when a direction has no closed bin before the serving day (a new route, or data starting today), there is no profile.
  - A forecast that does not need the profile (persistence, `xgb[maps]`, the 30-minute models) still answers 200 with `baseline.directions.<d>.forecast_min: null`.
  - `model=profile`, `xgb[maps+prof]`, `?curve=forecast` and `?baseline=profile` answer 503 "No baseline history for the requested direction".
  - No response ever contains `NaN`: a non-finite number becomes a 502.
- **Curve metadata:** `model_meta` in a curve is keyed `<model>@<h>min`, one entry per fit, and each point names its entry in `model_meta_key`.
- **Serving-day rollover:** cached answers are keyed by the serving day, so the first request after 00:00 SGT uses the new fit and profile. Each request reads the clock once when it starts (`_serving_now`). A request that spans midnight therefore fits, predicts, keys its cache and writes its metadata on one serving day; only lead and expiry use the live clock.
- **Browser caching:** forecast and curve responses send `Cache-Control: private, max-age=N`. `N` is at most 300 and never past the first target or the origin's 30-minute expiry, so a browser cannot reuse an expired forecast. It is `no-store` once that time is under a second.
- **`served` at an exploratory horizon:** `xgb[maps]` up to 60 min, `xgb[maps+prof]` from 90 min (`EXPLORATORY_SELECTION`, version `horizon-study-2026-10-04`). At 30 min it stays `SERVED_SELECTION`.
- **Timing:** `target_offset_min` is `[h, h+10]` and `lead_min` is about `h − 10` to `h − 20` with current ingestion. The origin rules do not change with the horizon: the newest closed bin, at most 30 minutes old. A 24-hour request therefore still answers 503 when ingestion is stale.
- **`baseline`** (in every forecast response): the calendar profile at the same target bin, per direction. It carries `"status": "baseline"` and the label *"Baseline: typical for this day and time (calendar profile), not a forecast"*. It is a per-direction Fourier (K = 8) × weekend ridge, refitted each SGT day on every closed bin before 00:00 SGT, so it ignores current traffic.
- **`study`** (when the summary is bundled): the study MAE at this horizon for the model used, the profile and persistence, with `vs_profile` differences and decisions.
- **`?baseline=profile&hours=N`:** the profile for the next `N` hours in 10-minute bins (`bin_start`, `forecast_min`) per direction. It needs no latest-bin read.
- **`?list=horizon-study`:** `forecastapi/horizon_study.json`, written by `cd eval; python horizon_study.py --publish`. Re-run and commit it after the study changes; the service only reads it.

Example (local run against live BigQuery, 2026-10-04 about 01:01 UTC; origin 00:50 UTC):

| Request | Model | SG → JB | JB → SG | Profile baseline (SG → JB / JB → SG) | Study MAE (model / profile / persistence) |
| --- | --- | --- | --- | --- | --- |
| `horizon_min=120` | `xgb[maps+prof]` | 34.4 | 22.1 | 42.0 / 24.3 | 3.96 / 5.19 / 6.86 |
| `horizon_min=240` | `xgb[maps+prof]` | 26.4 | 31.9 | 32.1 / 30.8 | 4.49 / 5.21 / 9.88 |
| `horizon_min=1440` | `xgb[maps+prof]` | 24.8 | 25.2 | 28.4 / 26.2 | 4.57 / 5.00 / 5.83 |

**Forecast curve, `?curve=forecast&hours=N`** (the recommended call for a chart):
- Returns the forecast every 30 minutes from one origin, out to `N` hours (1–24, default **2**: four points at 30, 60, 90 and 120 min).
- **Each point** carries `horizon_min`, `forecast_for`, `forecast_window_end`, `forecast_min`, `baseline_min` (the profile at the same target), `model` (the id that produced that lead, not one id for the whole curve), `status`, `lead_min` and `study_mae_min`. A single-model point omits `components`.
- **Which model per point:**
  - 30 min uses the evaluated `served` choice;
  - 60 min uses `xgb[maps]`;
  - 90–330 min use `xgb[maps+prof]`;
  - after 330 min (5.5 h) the point **is** the profile baseline (`model: profile`, `status: baseline`), because in the study no model beat it consistently there (`CURVE_MODEL_MAX_MIN`). There are scattered uncorrected wins at 8–9 h and 19–23 h, and none survives Holm over the 336 comparisons.
- **Per direction:** `origin_ts`, `origin_closed_at` and `observation_age_min`.
- **Caching:** the curve is cached for 5 minutes and dropped when its first target starts.
- **Live check** (local run against live BigQuery, 2026-10-04 02:01 UTC, origin 01:50 UTC):
  - SG → JB: 33.0 / 36.3 / 36.2 / 31.7 min at 30 / 60 / 90 / 120 min ahead, with the baseline at 39.4 / 42.0 / 41.7 / 38.5.
  - The first call took 29 s: the training read, the profile and four fits. `hours=24` then took 10 s more for its extra fits.
- **Cost:** each 30-minute step is one daily fit, about 0.6–1 s locally. The default curve costs about 4 fits per instance per day, and `hours=24` costs 11, because the baseline points need none.

Why not model points every 30 minutes to 24 h: from 6 h on, the model points would be, within the study's uncertainty, the baseline under another name. A single horizon past 5.5 h can still be requested with `horizon_min` (exploratory).

**For the page (`hosting/app.js`):**
- Call `?curve=forecast` (or `&hours=4` for 8 points) and plot each point's `forecast_min` at `forecast_for`. This gives the three or more points 60–120 min ahead the page asked for, all from one origin and one request.
- Plot `?baseline=profile&hours=24` as a separately styled line labelled with `label`.
- Show `status` on exploratory values, and use `lead_min` rather than a fixed "30 minutes ahead".
- With `hours` above 5, draw the points where `model` is `profile` in the baseline style. The curve can step at the switch: in a live run on 2026-10-04, SG → JB went 23.5 → 27.3 between 5.5 h and 6 h. That step is the model letting go, not a predicted change.
- Past about 5.5 h, the study found the forecast no better than the baseline. The claims register still says so. The checked-in page does not: a product decision on the Hosting commits left one sourcing line ("Based on Google Maps travel-time estimates") and labels the baseline "Typical for this day and time". See [findings.md](../findings.md#claims-register).
- Each new horizon costs one daily fit per instance on its first request (about 4–5 s locally).

## Performance

A local run against live BigQuery on 2026-10-03 took:
- **Cold start:** about 13 s for the first `served` request (training read plus fits). On 2026-10-04, after the exploratory horizons and the profile were added, it was about 22 s.
- **Each later request:** about 3 s, mostly the latest-row read; a repeat inside 5 minutes is served from the cache. The first request at a new exploratory horizon took 4–6 s (one more fit).

The fitted daily models are reused until 00:00 SGT. Memory is 1 GiB.

## CI/CD (`forecastapi/cloudbuild.yaml`)

1. **Test:** `pytest forecastapi`, then the harness equivalence test.
2. **Buildpack:** builds the image from `forecastapi/` (Python 3.11; all images pinned by digest).
3. **Deploy:** if the service exists, the new revision is deployed with **no traffic** and tag `candidate`. The first deploy takes traffic directly, because there is no earlier revision. The service is **public** (`--no-invoker-iam-check`, matching the live setting since 2026-10-03), is capped at `_MAX_INSTANCES` (3), and runs as `forecast-api@`. It sets `BQ_PROJECT` and `COMMIT_SHA`.
4. **Smoke:** against the candidate URL, anonymously and with the Hosting `Origin`, as the browser calls it:
   - the service annotation `run.googleapis.com/invoker-iam-disabled` is `true`;
   - `list=models` → 200 with an `Access-Control-Allow-Origin` header, and lists `served`;
   - `model=served` → 200 with `"source": "local model"` and `lead_min` (503 is accepted as a data condition);
   - `model=lstm` → 400.

   The earlier token-based smoke test (403 without a token) contradicted the live access setting and was dropped. The build SA's `roles/iam.serviceAccountOpenIdTokenCreator` on itself, granted for it on 2026-10-03, is no longer used. It can be removed with the command in [release-pr2.md](../release-pr2.md#gcp-changes-made-for-this-pr).
5. **Promote:** traffic moves to the new revision and the `candidate` tag is removed.

If Test, Deploy or Smoke fails, traffic stays on the previous revision. GitHub Actions runs the same `forecastapi` suite, the equivalence test, and a static check of the build file (`scripts/tests/test_forecastapi_cloudbuild.py`) on every push and PR.

## Operations

```powershell
$u = gcloud run services describe forecast-api --region asia-southeast1 --project swiftborder --format="value(status.url)"
curl.exe -s "$u/?model=served"
curl.exe -g -s "$u/?model=xgb[maps]&direction=SG_TO_MY"
curl.exe -g -s "$u/?model=xgb_bq[daily]&model_my_to_sg=persistence"
```

- **Roll back:** `gcloud run services update-traffic forecast-api --region asia-southeast1 --project swiftborder --to-revisions <previous-revision>=100`.
- **Remove:** `gcloud run services delete forecast-api --region asia-southeast1 --project swiftborder`. Then disable trigger `949ff029` so the next push does not recreate it.
- **Access:** public by decision (invoker IAM check disabled; CORS `*`). An empty service IAM policy does not make it private. To make it private again, deploy with `--invoker-iam-check`. Then restore a token-based smoke test, and give the Firebase page a way to call the service, because a browser cannot attach an identity token. Risk note: [findings.md](../findings.md#known-risk-public-forecast-api-by-decision).
- **Identity:** `forecast-api@swiftborder.iam.gserviceaccount.com`: BigQuery job user on the project and data viewer on `traffic_prediction` and `causeway` only. It holds no keys.
