# Changelog

Repo and deploy changes that are not worth repeating in long-lived READMEs. For what is live in GCP, query project `swiftborder` and refresh [docs/inventory.md](docs/inventory.md).

## 2026-09-26

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
