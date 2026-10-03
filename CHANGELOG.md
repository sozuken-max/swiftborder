# Changelog

Repo and deploy changes that are not worth repeating in long-lived READMEs. For what is live in GCP, query project `swiftborder` and refresh [docs/inventory.md](docs/inventory.md).

## 2026-10-03

- Team decisions on the review: the default Compute Engine build identity for `forecast-api` is accepted; `swiftbackend` credential and public-access settings stay as they are and do not block the merge; the PR stays as one PR; deck PNGs are handled separately. Recorded in `docs/release-pr2.md`.

- Review of `ae5d034`: `forecastapi` catalog states renamed. `GET /?list=models` lists every catalog id. `served` (`production`) returns the `v_forecast_recent` registry mix; `lin_h30`, `xgb_h30` and `persistence` are `api` (direct query, not necessarily what is served); other ids are `artifact` or `code-only` and are not callable. `forecastapi/cloudbuild.yaml` pins all four images by digest. `forecast-api@` runtime identity narrowed in GCP: project-wide `bigquery.dataViewer` removed, `dataViewer` granted on `traffic_prediction` and `causeway` only. `docs/release-pr2.md` adds the `forecast-api` first-deploy checks and rollback and a table of every GCP change for this PR. Firebase wording aligned with the observed Hosting site. Report re-run and re-promoted from the tree at that point: `20261003T035807Z` from `b3b6562` (every metric and decision identical to the previous report). Later commits (`b9bfccc`, `bf2382e`) changed `eval/*.py`, so the snapshot is no longer promotable from the current tree; it stays the citation target.

- Evaluation freeze: reported rows must have a label or target time at or before 2026-10-19 23:59 SGT. Through the 31 Oct deliverables the remaining work is the frozen-window harness, the report, and the deck. Collectors stay up. Recorded in [docs/roadmap.md](docs/roadmap.md) and [docs/evaluation.md](docs/evaluation.md).
- `forecastapi/`: undeployed HTTP read of the 30-minute Maps `duration_in_traffic` forecast (`model` and `version` select an allow-list refreshed from BigQuery; 5-minute in-process cache). `GET /?list=models` lists every catalog id. An earlier wording of this bullet said only `lin_h30`, `xgb_h30`, and `persistence` were callable; the review bullet above is the current catalog. Runbook: [docs/runbooks/forecast-api.md](docs/runbooks/forecast-api.md). `forecastapi/cloudbuild.yaml` is a separate Cloud Build path (pytest, then Cloud Run `forecast-api` in `asia-southeast1` on `^main$` only). Not deployed from this branch. Trigger `76bbca35` is unchanged.
- [docs/adr/0001-firebase-client-api-calls.md](docs/adr/0001-firebase-client-api-calls.md): the Firebase travel-time cards keep reading public `traffic-24h.json`. That decision is unchanged, and `camdetect` `detect` is unchanged.
- [docs/adr/0002-firebase-hosting-source.md](docs/adr/0002-firebase-hosting-source.md): put the Hosting client in this repo once `index.html`, `style.css`, `app.js`, and a recovered `firebase.json` are together. Deploy stays manual. No Hosting workflow, and `app.js` was not copied.
- Offline XGB window ablation in `eval/timeseries_xgb.py` (Chad, day 2): (A) lags, (B) lags plus a causal first difference and a 60-minute rolling mean, (C) lags plus a per-window z-score Daubechies db2 wavelet at level 2 (level 3 optional; db4 is not the default). `use_dwt` stays off for the promoted report model. `python timeseries_xgb.py` scores the cache; below 10 day-blocks it prints point estimates only.
- [docs/adr/0003-deep-training-and-feature-matrix.md](docs/adr/0003-deep-training-and-feature-matrix.md): keep LSTM / GRU / Transformer training on the local CPU. The next matrix is trees on the full wavelet and camera-profile factorial; deep models only on the cells importance selects. No GCP job.
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
