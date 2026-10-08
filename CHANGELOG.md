# Changelog

Repo and deploy changes that are not worth repeating in long-lived READMEs. For what is live in GCP, query project `swiftborder` and refresh [docs/inventory.md](docs/inventory.md).

## 2026-10-08

- `camdetect` dividing line: an end within 1% of frame height of the top or bottom edge snaps onto it and runs on along that edge to the frame side. On CAM 2701 the line leaves the bottom edge at x=176, so vehicles in the bottom-left corner (SG-MY side) were `Unknown`; they are now `SG-MY`. A box cut off by the bottom of the frame counts as just inside it. The JSON `dividing_line` now starts `[0, 1080], [176, 1080]`. No new score is claimed. Merging redeploys `swiftbackend`.
- `swiftbackend` builds from `camdetect/Dockerfile` (`python:3.11-slim`, the Test step's image, then `functions-framework --target=detect`) instead of buildpacks. Build d949348e, the first run of `camdetect/cloudbuild.yaml` after trigger `76bbca35` was switched to it in the Console, spent most of its 60+ s Buildpack step pulling builder `google-22` and lifecycle `0.21.18`, neither cached on the build machine, for about 8 s of build work. Python stays 3.11. Merging starts a build.
- `camdetect/cloudbuild.yaml`: the `swiftbackend` build in git. Build e78b62ff (inline trigger config) installed **Python 3.14.6** in the service image because `builder:latest` is now Ubuntu 24.04 and no version is set, while the `Test` step runs 3.11. The file pins builder `google-22` with `GOOGLE_PYTHON_VERSION=3.11.x`, tests in `mirror.gcr.io` `python:3.11-slim`, publishes from the buildpack step (no docker pull/push steps), and pins every image by digest (as `forecastapi/cloudbuild.yaml` does). No GCP change: trigger `76bbca35` still runs its inline config until it is pointed at this file. `includedFiles` matches `camdetect/cloudbuild.yaml`, so merging starts one build with the old inline config.

## 2026-10-07

- `camdetect` congestion label now bands each direction's count instead of its vertical spread: `Free Flow` <20, `Quarter Way` 20-39, `Half Way` 40-70, `Back to Back` >70. On 14 v6 frames of CAM 2701 (6-7 Oct) MY-SG spread was 0.61-0.68 at anything from 17 to 93 vehicles, and the old `Back to Back` cut (0.75) was out of reach, so an 82-vehicle queue read `Half Way`. The cuts were set by eye, not fitted; SG-MY never exceeded 12 in the sample, so its cuts are untested on congestion. Counts and `extent` are unchanged in the response. `congestion_ordinal` written by Layer A after the `swiftbackend` redeploy means count bands, not spread bands, so older rows are not comparable. Merging redeploys `swiftbackend`.
- The directional banner says "SG to MY" / "MY to SG" in plain text: the font in the `swiftbackend` image has no arrow glyph, so "→" rendered as an empty box.
- `camdetect` `format=directional` draws 1 px boxes with no per-box labels (in heavy traffic the labels hid the vehicles). The top-left banner is now one block per direction filled with that direction's box colour, so it doubles as the colour key. Counts, headers and JSON are unchanged.
- `camdetect` accepts `model=v4|v6` (fixed allowlist `WORKFLOW_VERSIONS`; any other value is 400 before an upstream call) and returns `X-Workflow-Id`. The default is still `ROBOFLOW_WORKFLOW_ID`. The page's AI Vehicle Detection panel calls both versions in parallel on the same frame (one shared `date_time`), so each page load or refresh makes two billed Roboflow calls instead of one. No score is claimed for either version.
- `camdetect` default `ROBOFLOW_WORKFLOW_ID` moves from `...-4-yolo26s-t1-logic` to `vehicle-detection-proejct-vvehicle-detection-proejct-6-yolo26s-t1-logic` (workflow version 6, as deployed in Roboflow; the workflow itself was not changed). A `ROBOFLOW_WORKFLOW_ID` env var on `swiftbackend`, if set, still wins over this default. The workflow's output shape was not checked against `_extract_predictions` here (the Roboflow tools were not available in this session). No new score is claimed. Merging this changes `camdetect/main.py`, so the `swiftbackend` trigger redeploys.

## 2026-10-05

- `timesfm`, `timesfm_calibrated` and `fcm_mlp` were rescored on the section 2 window (both directions, 13-30 Sep, 5,184 rows). The earlier 25 Sep fit and 26-30 Sep scores are not that comparison. Daily-refit 30-minute MAE: persistence 2.640, `xgb[maps]` 2.276, `timesfm` 2.340, `timesfm_calibrated` 2.310, `fcm_mlp` 2.230. Eighteen calendar days. None is a challenger versus `xgb[maps]` by the 0.5 min bar, so the served mix is unchanged and the forecast API does not call them. `timesfm_layer_a` and `fcm_mlp_layer_a` were not scored. No Cloud Run deploy. `lin_h30`, `xgb_h30`, `model_registry`, and `v_forecast_recent` were not changed.

## 2026-10-04

- Layer B point models were scored locally where that was possible. `fcm_mlp` (no camera) on the script's 26-30 Sep test split has five calendar days, so the harness decision is insufficient data; it is worse than `xgb[maps]` on those same 30-minute rows and stays out of the served mix. `timesfm` was not scored (`AI.FORECAST` was not run). `forecast-api` cards stay non-callable. No Cloud Run deploy. `lin_h30`, `xgb_h30`, `model_registry`, and `v_forecast_recent` were not changed.
- `GET /?list=cards` (same body as `?view=cards`) returns a JSON array of model cards for the frontend. `list=models` is unchanged. `timesfm` and `fcm_mlp` are cards only (`Not served`); the forecast API does not call those Causeway modules. No deploy.
- Forecast responses name the underlying model on each direction and each curve point. `components` appears only on a blend, as `{model, weight}` pairs whose weights sum to 1. That shape is in this tree. Live revision `forecast-api-00009-hax` does not serve it until this merges and the `forecast-api` trigger runs. `swiftbackend` is unchanged.
- `hosting/` is in git. Architecture, ADR 0002, and the claims register no longer say the client is absent. Deploy stays manual. The page copy, by product decision, drops the accuracy caveats; the claims register still treats 30 min as the only evaluated result, later curve points as exploratory, Holm skill only to 3 h, and 5.5 h as a generous end.
- Docs refreshed against local `main` `7bd94e3` and a read-only query of project `swiftborder` (about 14:40 SGT). No deploy and no cloud change. `forecast-api` is revision `forecast-api-00009-hax` at that commit; an anonymous `model=served` GET returned 200 (the smoke test accepts 200 or 503). `weather-backfill-tmp` is revision `00004-tjj`. Inventory's older "403", "trigger has not run", and `forecast-api-00002-zaj` lines are marked historical. Horizon wording stays exploratory: Holm keeps `xgb[maps+prof]` ahead of the profile only to 3 h; the public curve's 5.5 h model cutoff is the generous end, not a confirmed 24-hour forecast. Deck PNGs were not redrawn.
- Add two Layer B modules with offline tests in `Causeway/tests/`:
  - [Causeway/layer_b_timesfm.py](Causeway/layer_b_timesfm.py): TimesFM 2.5 through BigQuery `AI.FORECAST`, a ridge calibration of its residual and Layer A fusion.
  - [Causeway/layer_b_fcm_mlp.py](Causeway/layer_b_fcm_mlp.py): Fuzzy C-Means regimes feeding an MLP seed ensemble; results on 6–30 Sep in [Causeway/layer-b-fcm-mlp-results.md](Causeway/layer-b-fcm-mlp-results.md).
  - `Causeway/requirements.txt` adds the team pins `numpy==2.4.6` and `scikit-learn==1.9.1`, so the Causeway CI suite runs the new tests.
  - Neither module is deployed and no GCP settings changed. A deploy needs a `Causeway/Procfile` (see the module docstrings).

- `eval/horizon_study.py` and [docs/horizon-study.md](docs/horizon-study.md): an exploratory study of horizons from 30 min to 24 h on the report cache and the 13–30 Sep folds.
  - Candidates: persistence, same time yesterday, same time last week, a calendar-profile baseline, `xgb[maps]` and `xgb[maps+prof]`.
  - Result (30-minute grid): `xgb[maps+prof]` beats the profile and "same time last week" at every step to 5.5 h, or to 3 h / 1.5 h under Holm over all 336 comparisons. After that there is no consistent winner (about 5 min MAE).
  - The study also tests `xgb[maps+prof]` against "same time last week" (7 comparisons per horizon).
  - Not a report run. Findings item 12.
  - The study writes three charts by default (MAE by horizon, skill over the profile with joint CIs, example days), each with a CSV. `--publish` copies them to `docs/images/horizon-study/` and writes `forecastapi/horizon_study.json`.
  - `forecast-api` exposes the study, labelled. `horizon_min` takes 60 to 1440 for `served`, `persistence`, `xgb[maps]`, the new `xgb[maps+prof]` and the `profile` baseline. Exploratory responses carry `status: exploratory` and the study MAE.
  - Every forecast response carries the profile as `baseline` ("not a forecast"). `?baseline=profile&hours=N` returns its curve, and `?list=horizon-study` returns the study summary.
  - The 30-minute `served` selection is unchanged.
  - The study grid is now every 30 minutes (48 horizons). `?curve=forecast&hours=N` returns forecasts every 30 minutes from one origin (default 2 h, four points). On the 30-minute grid `xgb[maps+prof]` beats the profile at every step to 5.5 h, so model points run to 5.5 h; after that the points are the profile baseline, labelled. Each point carries its baseline value and study MAE.
  - `eval/tests/test_forecastapi_models.py` asserts the service equals the study at 2 h and 24 h.

- Address [docs/final-report-readiness.md](docs/final-report-readiness.md); the per-finding status is in its Response table.
  - `forecast-api` availability contract:
    - origins are closed bins only (60 s grace);
    - a direction whose target bin has started, or that lies within an hour after a >25 min gap, answers 503;
    - cached answers expire with their target;
    - responses add `origin_closed_at`, `forecast_window_end`, `observation_age_min`, `lead_min` and `target_offset_min`.
  - Features for serving and daily training are built from `v_bins_10min` with the harness's time-based rules (`local_models.features_from_bins`, tested equal to `eval/features.maps_features` with skipped bins). They no longer come from `v_training_set`'s positional LAG/LEAD.
  - `forecastapi/cloudbuild.yaml` matches the live public setting: `--no-invoker-iam-check`, `--max-instances=3`, and an anonymous smoke test with the Hosting origin. The token mint is removed, so the build SA's `serviceAccountOpenIdTokenCreator` grant is unused (revert command in `docs/release-pr2.md`; not revoked).
  - `eval/significance.py` pooled comparisons:
    - the gate counts shared calendar days;
    - the decision needs the joint calendar-day bootstrap CI;
    - a day-clustered p-value is reported (also in the run-wide multiplicity check).
  - A sensitivity re-run on the snapshot's cached data (`eval/runs/sensitivity-joint-day`, not promoted) left every 30-minute pooled decision unchanged and made the three fuzzy decisions "insufficient data" (6 shared days).
  - Roadmap C5/C8: "not significant" is "no improvement detected", not confirmation.
  - Docs re-queried `cam2701` / `cam2702`: a Mar–Apr detection batch with no model version; live camera calls are not stored.
  - README, findings, architecture (Mermaid; PNGs now stale on two labels), runbook, ADR 0004 amendment and inventory updated.
  - No GCP settings changed.

- Regenerate all four `docs/images/` deck PNGs with built-in image generation after rechecking live serving and storage. Update architecture/evaluation Mermaid and report embeds for the local forecast API, current Hosting connection, historical BQML, pending vision metrics and seven verified buckets. Record prompts and visual QA in `docs/diagrams/`; refresh the diagram skill and its Cursor mirror. No application code or cloud settings changed.

## 2026-10-03

- Add `docs/final-report-readiness.md`: repository, evaluation, live GCP and diagram review, with prioritized submission gaps. Refresh the inventory with the deployed local-model forecast API, public access, newer camera revision, seven buckets and current ingestion. All 315 fast tests and five slow TensorFlow tests passed. No cloud settings or application code changed in this review.

- PR #3 fixes two evaluation edge cases.
  - `promote_report_run.py` now refuses source and report directories that are equal or nested. Before the fix, `rmtree` on the report directory could delete the source run.
  - `camera_forecast.cross_validate` now predicts at `bin_ts + NOW_OFFSET`, the offset used for fitting and for feature generation.
  - On the cached camera history, CV MAE moves by at most 0.006 counts and K=8 is still chosen, so `eval/runs/report/` stays the citation target.

- Fix the `forecast-api` Smoke step. Build 363590b1 created the service (`forecast-api-00001-dz9`), then failed in Smoke: `gcloud auth print-identity-token` cannot mint an ID token in Cloud Build, and the metadata identity endpoint returns 404 there (2b76a1c3). Smoke now calls IAM Credentials `generateIdToken` for the build SA. GCP change: `1095552466513-compute@` holds `roles/iam.serviceAccountOpenIdTokenCreator` on itself. A smoke-only build against the live service (e5e280f6) passed every check; `model=served` returned `"source": "local model"`.

- Fix the first `forecast-api` build on `main` (a8600413 failed in Buildpack): `builder:latest` is Ubuntu 24 and ships only Python 3.13 / 3.14, so `GOOGLE_PYTHON_VERSION=3.11` could not resolve. The build now uses `builder:google-22` (pinned by digest) with `3.11.x`, matching the Python the tests and harness use. A build-only Cloud Build of the fixed config (e92cf936: Test + Buildpack, no deploy) succeeded with Python 3.11.x.

- Team decisions on the review: the default Compute Engine build identity for `forecast-api` is accepted; `swiftbackend` credential and public-access settings stay as they are and do not block the merge; the PR stays as one PR; deck PNGs are handled separately. Recorded in `docs/release-pr2.md`.

- Review of `ae5d034`: `forecastapi` catalog states renamed. `GET /?list=models` lists every catalog id. `served` (`production`) returns the `v_forecast_recent` registry mix; `lin_h30`, `xgb_h30` and `persistence` are `api` (direct query, not necessarily what is served); other ids are `artifact` or `code-only` and are not callable. `forecastapi/cloudbuild.yaml` pins all four images by digest. `forecast-api@` runtime identity narrowed in GCP: project-wide `bigquery.dataViewer` removed, `dataViewer` granted on `traffic_prediction` and `causeway` only. `docs/release-pr2.md` adds the `forecast-api` first-deploy checks and rollback and a table of every GCP change for this PR. Firebase wording aligned with the observed Hosting site. Report re-run and re-promoted from the tree at that point: `20261003T035807Z` from `b3b6562` (every metric and decision identical to the previous report). Later commits (`b9bfccc`, `bf2382e`) changed `eval/*.py`, so the snapshot is no longer promotable from the current tree; it stays the citation target.

- Evaluation freeze: reported rows must have a label or target time at or before 2026-10-19 23:59 SGT. Through the 31 Oct deliverables the remaining work is the frozen-window harness, the report, and the deck. Collectors stay up. Recorded in [docs/roadmap.md](docs/roadmap.md) and [docs/evaluation.md](docs/evaluation.md).
- `forecastapi/`: undeployed HTTP read of the 30-minute Maps `duration_in_traffic` forecast (`model` and `version` select an allow-list refreshed from BigQuery; 5-minute in-process cache). `GET /?list=models` lists every catalog id. An earlier wording of this bullet said only `lin_h30`, `xgb_h30`, and `persistence` were callable; the review bullet above is the current catalog. Runbook: [docs/runbooks/forecast-api.md](docs/runbooks/forecast-api.md). `forecastapi/cloudbuild.yaml` is a separate Cloud Build path (pytest, then Cloud Run `forecast-api` in `asia-southeast1` on `^main$` only). Not deployed from this branch. Trigger `76bbca35` is unchanged.
- [docs/adr/0001-firebase-client-api-calls.md](docs/adr/0001-firebase-client-api-calls.md): the Firebase travel-time cards keep reading public `traffic-24h.json`. That decision is unchanged, and `camdetect` `detect` is unchanged.
- [docs/adr/0002-firebase-hosting-source.md](docs/adr/0002-firebase-hosting-source.md): put the Hosting client in this repo once `index.html`, `style.css`, `app.js`, and a recovered `firebase.json` are together. Deploy stays manual. No Hosting workflow, and `app.js` was not copied.
- Offline XGB window ablation in `eval/timeseries_xgb.py` (Chad, day 2): (A) lags, (B) lags plus a causal first difference and a 60-minute rolling mean, (C) lags plus a per-window z-score Daubechies db2 wavelet at level 2 (level 3 optional; db4 is not the default). `use_dwt` stays off for the promoted report model. `python timeseries_xgb.py` scores the cache; below 10 day-blocks it prints point estimates only.
- [docs/adr/0003-deep-training-and-feature-matrix.md](docs/adr/0003-deep-training-and-feature-matrix.md): keep LSTM / GRU / Transformer training on the local CPU. The next matrix is trees on the full wavelet and camera-profile factorial; deep models only on the cells importance selects. No GCP job.
- `forecast-api` manual model selection: `model` (default `served`) for every direction, and `model_sg_to_my` / `model_my_to_sg` overrides per direction (answer `model: custom`, model named per direction, per-direction version). `list=models` describes the parameters.
- `forecast-api` serves local models (ADR 0004 accepted; frozen-run plan signed off).
  - `forecastapi/local_models.py` holds `xgb[maps]`, `ridge[maps]`, `xgb_bq` / `lin_bq` [daily] and [frozen]. They are fitted in the process once per SGT day from `v_training_set`, on the harness fold's rows.
  - `served` is a per-direction selection (interim: the 12 Sep registry rebuilt from local replicas).
  - Responses carry `model_meta` and `commit`. BigQuery ML ids and SQL are removed.
  - `eval/tests/test_forecastapi_models.py` asserts a service fit equals the harness prediction.
  - `forecastapi/cloudbuild.yaml`: test (including that check), build, no-traffic candidate deploy, smoke test (403 without a token, catalog, served, undeployed 400), then promote. It sets 1 GiB and `COMMIT_SHA`.
  - `scripts/tests/test_forecastapi_cloudbuild.py` statically checks the build file (PyYAML pinned for the repo suite).
  - Live read-only check on 2026-10-03: about 13 s cold, 3 s warm.
- BigQuery ML leaves the evaluation. `eval/bq_replica.py` replicates `lin_h30` / `xgb_h30` locally (same features, settings, 12 Sep rows; XGBoost `base_score` 0.5 as in BQML's XGBoost 0.9). `eval/bqml_parity.py` (`--bqml-parity`, promoted to `eval/runs/parity-bqml/`, run `20261003T065653Z`) shows both replicas within ±0.5 min of BQML (−0.080 and −0.045 min MAE). `joined.py` adds `lin_bq` / `xgb_bq` [frozen] and [daily] in a `replica` Holm family; `ensemble.py` builds a local-only pool when `--bqml` is absent; `--skip-offline`; `--ensemble` needs only `--joined`. Frozen-run plan revised (local only, claims C1–C8). `docs/adr/0004-serve-local-models.md` (proposed): `forecast-api` fits the selected local models once per SGT day in-process.
- `generate_comparison_plots.py --data-cutoff` (SGT): drops observations, rainfall and forecasts after the cutoff in every component; sets and checks the BQML window end and `--joined-end`; records `data_cutoff` in `run.json`. `promote_report_run.py --target report-confirm` for the October-only confirmation run; `replay_series.py` refuses both snapshots. Frozen-window plan (windows, claims C1–C7, 0.5-minute threshold, test groups, 60-minute design) proposed in `docs/roadmap.md` for team sign-off before 19 Oct.
- `README` wording on reproducibility corrected: a re-run with the cached export reproduces the snapshot numbers (three identical report runs); it is not done only because it would be a new run to re-promote. Replay folders are tracked by pattern (`replay-*/`).
- Figure writers emit a same-stem CSV beside each PNG. The cited snapshot in `eval/runs/report/` is unchanged; its series were replayed from `run.json` into `eval/runs/replay-20261003T035807Z/` with no refit. `offline/holdout-sample` was never stored. Older schema-v2 promotions replay from the `run.json` in the commit that promoted them; `20260926T073406Z` is schema v1 and is not replayable.

## 2026-10-01

Branch `eval-integrity-overhaul` (PR #2). Work plan and task status: `docs/plan-eval-integrity.md`.

### Evaluation (fixed)

- Offline XGB: the "persistence T-60" baseline was the value 115 min before the target; it is now the last observation at or before the forecast origin (60 min before the label; up to 30 min older when bins are missing). `duration_sec` was taken at the target row (leak); now at the origin. Labels were interpolated, bfilled and slew-limited before scoring; now raw. Backtest days overlapped training; now full days after the split. XGB and LSTM share one split. `regularized_series` removed; `score_forecast_days` signature changed.
- Significance: Diebold–Mariano test with Newey–West variance (lag at least one day of samples) and the Harvey–Leybourne–Newbold correction; overlapping day-block bootstrap within direction; Holm over two-sided p-values in one family per harness; decision "insufficient data" below 10 day-blocks; scipy required (no silent normal fallback). Effect on reported decisions: offline XGB vs persistence and per-direction weekend slices are "insufficient data"; `lin_h30` vs persistence across both directions is "not significant".

### Evaluation (added)

- `eval/layer_b.py` rewritten: fixed window from 2026-09-13 SGT, refuses models trained inside the window, gap filters, direction × time-of-day / day type / day-night slices, `ensemble_mean`, headline and slice Holm families. `--holdout-days` is rejected with an error.
- `eval/features.py` (causal feature table; parity with the live `v_training_set` recorded in `run.json`), `eval/joined.py` (rolling-origin joined experiment, Maps-typical baseline; forecasts joined by data.gov.sg acquisition time), `eval/layer_a.py` (+ fixtures), `eval/backfill_camera_counts.py` (budgeted, resumable, stops on quota).
- Run manifest schema v2 (`run_artifacts.py`: provenance, data hashes, windows, metrics, significance, validation). `promote_report_run.py` refuses invalid runs, dirty trees (unless `--allow-dirty`) and runs whose code hash differs from the tree.
- `eval/timeseries_transformer.py` (patch Transformer) and `eval/deep_forecast.py`: LSTM, GRU and patch Transformer (+ raw-target ablation) under one training protocol, three seeds, DM/Holm vs persistence and XGB (`--deep`). `build_recurrent_model` now respects `LSTMTrainConfig.seed`; `_require_keras()` no longer resets every model to seed 42.
- `eval/fuzzy_traffic.py`: light / moderate / heavy traffic level 60 min ahead, both directions; Ruspini trapezoid partition, learned fuzzy rule-based classifier (certainty factors, single winner), XGB → fuzzy-level hybrid; accuracy, macro-F1, severe errors, RPS; DM on 0/1 loss (`--fuzzy`).
- `eval/ensemble.py`: served (registry) forecast, equal mean, rolling convex LAD stack, rolling selection and fuzzy-gated stack over BQML and daily-refit 30-min models; error-by-regime diagnostic; optional 60-min XGB + deep pool (`--ensemble`). `layer_b.bqml_component`, `joined.joined_component` and `deep_forecast.deep_component` accept `keep=` to hand their out-of-sample rows on.
- `eval/camera_forecast.py`: Layer A queue forecast for Layer B. Cached read-only pull of `cam2701.v_congestion_index_10min`, partial-day filter, leave-days-out CV of Fourier × weekend ridge profiles, `camfc_now` / `camfc_30` / `camfc_delta`. `joined.py` adds `maps+camfc`, a per-fold Maps-profile control (`maps+mpfc`) and `maps+mpfc+camfc` in their own Holm family (`camfc`), and MAE by observed-change regime.
- Manifest components `deep`, `fuzzy`, `ensemble`; classification metric rows validated (`FUZZY_METRIC_KEYS`) and tabulated in run READMEs. Forest plot clips degenerate CIs.
- Review response: `run.json` → `multiplicity` repeats every decision with one run-wide Holm family (7 of 134 flip to not significant; documented as "family-level only"); `provenance` records OS, CPU and every installed distribution; `deep.config.tensorflow` records the TensorFlow build; CI job `pytest-slow` (manual / weekly) runs the TensorFlow tests. `docs/evaluation.md` explains the Holm family boundaries and practical vs statistical significance. `docs/release-pr2.md`: redeploy smoke test and rollback, weather-append verification, retention and revert, credential risk.
- `eval/runs/report/` promoted from `20260930T202959Z_offline-bqml-joined-deep-fuzzy-ensemble`, made from committed code `eb435dd` on a clean tree (metrics unchanged). `eval/runs/LATEST.json` is now a gitignored local pointer.
- `docs/deep-learning-assessment.md`: design and scored results of the deep models; kept as a comparison, not served.

### Testing and dependencies

- `scripts/run_tests.ps1` / `scripts/run_tests.sh` (all suites; `-Slow` / `--slow` adds TensorFlow tests; `-Coverage`), `.github/workflows/tests.yml` (fast suites on Python 3.11 for push and PR; no secrets, no deploy), `slow` pytest marker, `scripts/tests/test_docs.py` (links, banned strings, skill mirrors), `requirements-lock-py311.txt`.
- Every `requirements*.txt` pinned to exact versions. `camdetect/requirements.txt` pins `functions-framework`, `requests`, `Pillow`: **merging this to `main` redeploys `swiftbackend`.** DirectML requirements documented as Python 3.8–3.10 in their own venv.

### Ingest and detection

- `Causeway/`: shared `datagov.py`; only complete days are written (failed or partial days are retried), atomic writes, 5xx/429 retries with `Retry-After`, today skipped unless `--allow-partial`, repeated-token guard, optional `DATAGOV_API_KEY`, `.nodata` markers for empty days (retried). Forecast rows keep `update_timestamp`. Filters sort, de-duplicate and ignore partial files. New `load_bigquery.py`: manual append-only loader (dry run by default, refuses overlapping ranges, snapshots before appending).
- `camdetect/main.py` (not deployed until merge): `detect_frame()` core; `confidence=0` honoured; empty-string body values fall back to the query string; frame download status-checked and validated before the billed call; malformed predictions no longer crash image modes; `date_time` validated (400); generic error messages with server-side logging; clear 500 without a key; safe `DEFAULT_CONFIDENCE`; `directions.*.extent` in JSON. Handler tests added.

### BigQuery (write, requested by the team)

- Appended Woodlands weather for 1–30 Sep 2026 SGT: `rainfall.rainfall` +8,640 rows; `weatherforecast.weatherforecast` +1,876 rows and a new nullable column `update_timestamp`. Pre-load snapshots `*_snapshot_20261001` (expire 2026-10-31). A follow-up `UPDATE` limited to the 8,640 appended rows set `station_id` back to `Woodlands Centre Road`; the loader now pins that name.

### Documentation

- `docs/inventory.md` refreshed from a read-only query (1 Oct 00:40 SGT) plus the weather append: `swiftbackend` public exposure, plaintext key, unused `CACHE_BUCKET`, failed backfill runs, no Firebase APIs, model training dates and split, view parity, weather tables to 30 Sep, Windows gcloud note.
- `docs/evaluation.md` and `docs/findings.md` rewritten from the promoted `run.json` (security risk entry, claims register; sections 4–7: deep models, fuzzy traffic level, ensembles and hybrids, Layer A output as a Layer B input). Section 7b states what the camera results do and do not support (shared daily cycle, not a Distance Matrix substitute; visibility risk) and the overlap test on 6–11 Sep frames. Inventory camera facts corrected from a read-only query: detections cover 13 Mar–22 Apr 2026 (tables last modified 18 Jul), image frames run to 11 Sep, so no Layer A output overlaps the Maps label; README headline, results, MVP runbook, testing and techniques; roadmap, grading and agent-deploy updated; camera pilot handoff for the Roboflow key holder (`docs/handoff-camera-pilot.md`).
- Diagrams: bucket `swiftborder_cloudbuild` (underscore), frame-cache edges dashed ("writer not in git"), `eval/` block, Layer A scorer present, weather tables appended manually. PNGs renamed `architecture-high-level.png` / `architecture-detailed.png`; all four deck PNGs are stale until regenerated.
- Skills: mojibake and BOM removed (also in `AGENTS.md`); the diagram skill no longer instructs the hyphenated bucket; annotation panels vs topology rule reconciled; mirrors regenerated.

## 2026-09-26

### Added (LSTM and Windows TensorFlow)

- `eval/timeseries_lstm.py`: architectures (`lstm`, `stacked_lstm`, `bilstm`, `gru`, `residual_gated`, `bilstm_attention`), `train_lstm`, `tune_lstm_hyperparameters`, `compare_lstm_architectures`.
- `eval/train_lstm.py` CLI; `eval/check_tf_gpu.py`; `eval/requirements-tf-gpu-windows.txt` (DirectML for NVIDIA on native Windows).
- Teammate XGB pipeline: slew-rate cap, tuned `XGBTrainConfig`, optional DWT (default off); `eval/notebooks/reference_Test_Time_Series_Prediction.md`.

### Changed (Layer B evaluation)

- Offline scope documented as **JB→SG only** (`jb_to_woodlands`); SG→JB not in notebook; bidirectional **30 min** remains `layer_b.py`.
- `eval/timeseries_xgb.py`: teammate hyperparameters, slew limit fix, flexible datetime parsing.
- LSTM: `TF_DISABLE_CUDNN_RNN` for DirectML compatibility on Windows GPU.

### Added (Layer B evaluation)

- Offline path: `eval/timeseries_xgb.py`, `eval/timeseries_lstm.py`, notebook, canonical `travel_times` sync and cache docs under `eval/data/`.
- Paired significance: `eval/significance.py`; comparison plots `eval/plots.py`, `eval/generate_comparison_plots.py`.
- Per-run output under `eval/runs/<run_id>/`; **committed report snapshot** `eval/runs/report/` (promote via `eval/promote_report_run.py`).
- Tests for metrics, XGB/LSTM helpers, significance, plots, and run artifacts.

### Changed

- `eval/layer_b.py`: `bin_ts` alignment, `--significance`, `--plots`, run-folder output.
- Evaluation and findings: Layer B tables, significance protocol, plot analysis; offline 60 min vs BQML 30 min kept separate.
- Root README, `docs/README.md`, `eval/README.md`: point at `eval/runs/report/` for final-report figures and `run.json`.

### Added (notebook integration)

- `eval/notebooks/causeway_xgb_timeseries.ipynb` (sklearn XGB, 60-minute horizon, `jb_to_woodlands`).

### Documentation

- Comprehensive doc pass: align findings, roadmap, agent-deploy, and architecture with `eval/`, pytest-on-deploy, and inventory/CHANGELOG split. Refresh `travel_times` count (~11,830 rows) in inventory.

### Added

- `sql/` — BigQuery view DDL (`v_bins_10min`, `v_training_set`, `v_forecast_recent`, weather and congestion side views), BQML scripts (`bqml_lin_h30`, `bqml_xgb_h30`), and `model_registry` reference MERGE.
- `eval/layer_b.py` and offline metric tests (read-only BigQuery hold-out for persistence, `lin_h30`, `xgb_h30` at 30 minutes).
- `Causeway/README.md`, `requirements.txt`, offline tests for station filter; `zoneinfo` fallback on rainfall/forecast fetch scripts.
- `camdetect/tests/` and Cloud Build step `Test` (`python -m pytest` in `camdetect/`) before buildpack deploy.
- Documentation spine under `docs/` (architecture, evaluation, findings, roadmap, agent-deploy, inventory).

### Changed

- Documentation: repo/deploy history moved to this file; READMEs and report docs stay long-lived (current behavior). Dated GCP copies remain in `docs/inventory.md`.
- Root and report docs: Layer B wording matches Maps-only training and 30-minute serve; removed legacy proposal PNGs (`architecture-proposal.png`, `dataflow-target.png`).
- Cloud Build trigger `76bbca35-c1b4-4836-9f34-d7adda53ea17`: narrowed `includedFiles` to runtime paths (`main.py`, `requirements.txt`, Dockerfile, yaml/json/toml). Test-only paths and `camdetect/README.md` no longer start a deploy.
- `swiftbackend` revision `swiftbackend-00022-cv4` from commit `cf1c228` (camdetect tests and Py3.8 `zoneinfo` backport).

### Removed

- `google-cloud-storage` from `camdetect/requirements.txt` (unused in `camdetect/`).
