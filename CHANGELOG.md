# Changelog

Repo and deploy changes that are not worth repeating in long-lived READMEs. For what is live in GCP, query project `swiftborder` and refresh [docs/inventory.md](docs/inventory.md).

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
