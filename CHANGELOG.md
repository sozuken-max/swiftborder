# Changelog

Repo and deploy changes that are not worth repeating in long-lived READMEs. For what is live in GCP, query project `swiftborder` and refresh [docs/inventory.md](docs/inventory.md).

## 2026-09-30

### Added (documentation)

- `docs/plan-eval-integrity.md`: work plan from the full-repo review and a read-only query of project `swiftborder` (2026-09-30 ~21:35 SGT). Linked from `docs/README.md` and `docs/roadmap.md`.

### Added (testing)

- `scripts/run_tests.ps1` / `scripts/run_tests.sh`: run `camdetect`, `Causeway`, `eval`, and repo doc checks; `-Slow` / `--slow` adds TensorFlow tests; `-Coverage` / `--coverage`.
- `.github/workflows/tests.yml`: fast suites on Python 3.11, push and PR. No secrets, no deploy.
- `slow` pytest marker in every `pytest.ini` (excluded by default); `eval/pytest.ini` (was missing).
- `scripts/tests/test_docs.py`: relative Markdown links must resolve.
- `requirements-lock-py311.txt`: full resolved test environment.

### Changed (dependencies)

- Every `requirements*.txt` pinned to exact versions (Python 3.11). `eval/requirements.txt` adds `scipy`; `eval/requirements-dev.txt` adds `PyWavelets`, `pytest-cov`, `Pillow`, `requests`. `camdetect/requirements.txt` pins `functions-framework`, `requests`, `Pillow`: **a push of this file to `main` redeploys `swiftbackend`.**
- `requirements-tf-gpu-windows.txt`: DirectML plugin pinned; documented as Python 3.8–3.10 in its own venv.

### Fixed (evaluation)

- Offline XGB: persistence baseline was the value 115 min before the target (labelled T-60); now the last observation exactly 60 min before. `duration_sec` was taken at the target row (leak); now at the origin. Labels were interpolated/bfilled/slew-limited before scoring; now raw. Backtest days overlapped training; now full days after the split. XGB and LSTM share one split. `regularized_series` removed; `score_forecast_days` signature changed.
- Significance: Diebold–Mariano (Newey–West, HLN), overlapping day-block bootstrap within direction, Holm correction, explicit decision rule; scipy required (no silent normal fallback).

### Added (evaluation)

- `eval/layer_b.py` rewritten: fixed window from 2026-09-13 SGT, model-training-time check, gap filters, direction × time-of-day / day type / day-night slices, `ensemble_mean`, headline and slice Holm families. `--holdout-days` removed.
- `eval/features.py` (causal joined feature table, parity with live `v_training_set`), `eval/joined.py` (rolling-origin joined experiment, Maps-typical baseline), `eval/layer_a.py` (+ fixtures), `eval/backfill_camera_counts.py` (budgeted, resumable, stops on quota).
- Run manifest schema v2 (`run_artifacts.py`: provenance, data hashes, windows, metrics, significance, validation); `promote_report_run.py` validates and refuses dirty runs unless `--allow-dirty`.
- `eval/runs/report/` re-promoted from `20260930T175104Z_offline-bqml-joined` (offline + BQML fixed window + joined, with parity against the live view), made from committed code `15f094b` on a clean tree.

### BigQuery (2026-10-01, requested write)

- Appended Woodlands weather for 1–30 Sep 2026 SGT: `rainfall.rainfall` +8,640 rows, `weatherforecast.weatherforecast` +1,876 rows with new nullable column `update_timestamp`. Pre-load snapshots `*_snapshot_20261001` (expire 2026-10-31). One follow-up `UPDATE` on the 8,640 appended rows corrected `station_id` to `Woodlands Centre Road`. New loader `Causeway/load_bigquery.py` (dry run by default, refuses overlaps, snapshots before append).
- Camera pilot documented as a handoff to the Roboflow key holder (`docs/handoff-camera-pilot.md`).

### Changed (after review, 2026-10-01)

- Significance: Holm over two-sided DM p-values in one family (family-wise error ≤ α for directional claims); HAC lag at least one day of samples; decision "insufficient data" below 10 bootstrap blocks. Offline hold-out (5 blocks) and weekend slices are now "insufficient data"; `lin_h30` vs persistence combined is now "not significant".
- Forecast join uses data.gov.sg acquisition time (`update_timestamp`, now fetched and kept by the Causeway scripts). Days with no data leave a `.nodata` marker and are retried. `camdetect`: an empty-string body value falls back to the query string. Promotion refuses a run whose code hash differs from the tree.

### Changed (ingest and detection)

- `Causeway/`: shared `datagov.py`; complete days only (failed or partial days are never written and are retried), atomic writes, 5xx/429 retries with `Retry-After`, today skipped unless `--allow-partial`, repeated-token guard, optional `DATAGOV_API_KEY`; filters sort, de-duplicate and ignore partial files. Backfilled rainfall and forecasts for 5–30 Sep (local, gitignored).
- `camdetect/main.py` (not deployed): `detect_frame()` core; `confidence=0` honoured; frame download status-checked and validated before the billed call; malformed predictions no longer crash image modes; `date_time` validated (400); generic error messages with server-side logging; clear 500 without a key; safe `DEFAULT_CONFIDENCE`; `directions.*.extent` in JSON. Handler tests added.

### Documentation

- `docs/inventory.md` refreshed (2026-10-01 read-only query): `swiftbackend` public exposure, plaintext key, unused `CACHE_BUCKET`, failed backfill runs, no Firebase APIs, model training dates and split, view parity, Windows gcloud note.
- `docs/evaluation.md` rewritten from the promoted `run.json`; `docs/findings.md` results, security risk entry and claims register; README headline, MVP runbook, testing and techniques; roadmap, grading, agent-deploy updated.
- Diagrams: bucket `swiftborder_cloudbuild` (underscore), frame-cache edges dashed ("writer not in git"), `eval/` block, Layer A scorer present. PNGs renamed `architecture-high-level.png` / `architecture-detailed.png`; all four deck PNGs are marked stale pending regeneration.
- Skills: mojibake and BOM removed (also in `AGENTS.md`); diagram skill no longer instructs the hyphenated bucket; annotation panels vs topology rule reconciled; mirrors regenerated. Repo test checks links, banned strings and mirror parity.
- `eval/runs/LATEST.json` is now a gitignored local pointer (tracked copy deleted).

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
