# SwiftBorder

Woodlands-only causeway crossing-time forecasting (NUS-ISS Practice Module, Group 3). The **live** system serves a **30-minute** forecast of the Google Maps duration series; a 24-hour horizon and <= 15 min MAE are **product targets**, not results.

| Layer | Job |
| --- | --- |
| **A - Vision** | LTA camera frames -> Roboflow YOLO (label / train / serve) -> per-direction vehicle counts (camera 2701) via Cloud Run `swiftbackend` |
| **B - Forecasting** | Maps Distance Matrix durations in BigQuery -> Cloud Run `forecast-api`, which fits local models in-process ([ADR 0004](docs/adr/0004-serve-local-models.md)): the local `lin_h30` replica for SG→MY, persistence for MY→SG. The target is the bin 30–40 min after the newest closed bin. The live fetch cadence is in [inventory.md](docs/inventory.md). BQML `lin_h30` / `xgb_h30` / `v_forecast_recent` remain as history |

**Principal risk:** queue counts are not crossing duration, and the label is Maps' own estimate. Every Layer B score is skill over persistence on the Maps series; there is no independent wait-time measurement.

## Results (13–30 Sep 2026, out-of-sample, frozen 17 Oct 2026)

All reported metrics are pinned to the **17 Oct 2026 final-report freeze** (`main` merge `199c2e5`; [final-report-readiness.md](docs/final-report-readiness.md)). 30-minute MAE on the Maps series, both directions: persistence 2.64 min, BQML-era served forecast (registry, now served via local replicas in `forecast-api`) 2.54, `lin_h30` 2.78, `xgb_h30` 2.49, ensemble 2.46, daily-refit XGBoost (`xgb[maps]`) 2.28, and an exploratory 50/50 `fcm_mlp` + `xgb[maps]` mix 2.19 ([docs/layer-b-mix-module.md](docs/layer-b-mix-module.md)). The ensemble is significantly better than persistence (`xgb_h30` too within its test family, not under a run-wide check); all 30-minute gains are under half a minute, so none is claimed as a product-relevant win; `lin_h30` (the model currently served for `SG_TO_MY`) is not significantly different from persistence across both directions. "Significant" means a Diebold–Mariano test (do two forecasts' errors differ more than chance?) with a Holm correction for testing several models at once, and confidence intervals from a bootstrap that resamples whole days so that correlated 5-minute errors are not counted as independent. Weather features gave no significant gain on this window. Stacking or gating in the original §6 combiners adds nothing over daily-refit `xgb[maps]`, which lowers MAE relative to the served registry forecast by about 0.27 min. At 60 minutes on one route, LSTM, GRU and a patch Transformer trail XGBoost. An exploratory horizon study (13-30 Sep, Maps duration, every 30 min to 24 h, not a report run) finds `xgb[maps+prof]` ahead of a calendar profile to 5.5 h on uncorrected steps, and only to 3 h under Holm over 336 comparisons. The public curve uses that 5.5 h mark as its generous end and draws the profile after it. That is not a confirmed 24-hour forecast ([docs/horizon-study.md](docs/horizon-study.md)). A fuzzy light / moderate / heavy forecast is best as a hybrid on point estimates (XGBoost defuzzified into levels, 0.785 accuracy vs 0.690 for persistence). Its 6-day hold-out is too short for a significance claim under the pooled-direction rule ([findings](docs/findings.md#pooled-significance-sensitivity-2026-10-04)). On the 15-frame camera-2701 `test` split ([eval/test-reports/](eval/test-reports/README.md) and [evaluation.md](docs/evaluation.md#layer-a--vision)), served `model=local` (`conf=0.20, overlap=0.60`) has car MAE 1.87 (1.73 at the test-swept `conf=0.30, overlap=off` operating point; notebook IoU-0.5 test-split AP50 is 0.562), whereas served Roboflow v6 has car MAE 4.40 (`eval/layer_a.py` directional crossing-line accuracy remains unscored). `cam2701.local_counts` overlaps the Maps window (6 Sep–4 Oct 2026). Those counts correlate with Maps duration and lower MAE in a restricted 4-feature contemporaneous-Maps XGBoost by about 0.083 min on a fold-level test, though both restricted trees remain worse than persistence and full `xgb[maps]`; that is not a substitute for Distance Matrix data ([evaluation.md §7b](docs/evaluation.md#7b-could-layer-a-output-replace-the-distance-matrix-data), [§7c](docs/evaluation.md#7c-observed-camera-counts--correlation-and-layer-b-ablation-evallayer_a_evalpy)). A queue forecast learned from the March–April detections lowers XGBoost 30-min MAE by 0.065 min inside its own Holm family only; the run-wide Holm check does not keep that XGBoost result. Ridge's improvement survives the run-wide check. Neither improves on a daily profile learned from Maps itself. Details, slices and caveats: [docs/evaluation.md](docs/evaluation.md); evidence bundle: [`eval/runs/report/`](eval/runs/report/).

## Techniques (Practice Module)

| Category | Where |
| --- | --- |
| Supervised learning | Roboflow-labelled YOLO26s/m; regression of future Maps duration (BQML, XGBoost, ridge, MLP); traffic-level classification; ResNet18 + BiGRU congestion sequence classifier ([eval/cnn_rnn_congestion/](eval/cnn_rnn_congestion/README.md)) |
| Unsupervised learning & fuzzy logic | Fuzzy C-Means clustering on historical traffic profiles (`eval/layer_b_fcm_mlp.py`); Mamdani light / moderate / heavy partition and learned Wang–Mendel fuzzy rule base (`eval/lb/fuzzy_traffic.py`) |
| Machine learning / deep learning | YOLO26s ONNX (`services/camdetect`); `lin_h30`, `xgb_h30` & local replicas (`services/forecastapi`); offline XGBoost, LSTM, GRU, TCN and patch Transformer ([deep-learning assessment](docs/deep-learning-assessment.md)) |
| Intelligent sensing | LTA camera frames -> directional occupancy with camera 2701 & 2702 dividing lines (`services/camdetect/main.py`) and `traffic-backfill` Cloud Run Job (`services/camdetect/backfill_local_counts.py`) |
| Hybrid / ensemble | Mean of `lin_h30` + `xgb_h30`; ridge + XGBoost; rolling LAD stack, rolling selection and fuzzy-gated stack; XGBoost forecast defuzzified into traffic levels; 50/50 `fcm_mlp` + `xgb[maps]` mix ([docs/layer-b-mix-module.md](docs/layer-b-mix-module.md)). All scored against single models |

## Run the MVP

1. **Directional counts (Layer A, live):** call `swiftbackend` for camera 2701. It is public and each call uses Roboflow credits (see [docs/findings.md](docs/findings.md#known-risk-public-swiftbackend-documented-not-changed)).
   ```bash
   curl "https://swiftbackend-1095552466513.europe-west1.run.app/?camera_id=2701&format=json"
   curl -o frame.jpg "https://swiftbackend-1095552466513.europe-west1.run.app/?camera_id=2701&format=directional"
   ```
2. **Forecast (Layer B, live, public):** `forecast-api` ([runbook](docs/runbooks/forecast-api.md)). A 503 means no servable origin bin (stale ingestion), not an outage of the service.
   ```bash
   # evaluated 30-minute forecast
   curl "https://forecast-api-1095552466513.asia-southeast1.run.app/?model=served"
   # forecast curve every 30 min, 30-120 min ahead (exploratory beyond 30 min; add &hours=24 for the full day,
   # where points after 5.5 h are the labelled calendar-profile baseline)
   curl "https://forecast-api-1095552466513.asia-southeast1.run.app/?curve=forecast"
   # the baseline curve on its own, and the study the exploratory figures come from
   curl "https://forecast-api-1095552466513.asia-southeast1.run.app/?baseline=profile&hours=24"
   curl "https://forecast-api-1095552466513.asia-southeast1.run.app/?list=horizon-study"
   ```
3. **Evaluation (read-only BigQuery queries):**
   ```bash
   cd eval
   pip install -r requirements-dev.txt
   python layer_b.py --window-end "2026-09-30 23:50"
   ```
   Pinning `--window-end` reproduces the report window; the full report run is in [eval/README.md](eval/README.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [`services/`](services/README.md) | Deployed Cloud Run endpoints and jobs: [`services/camdetect/`](services/camdetect/README.md) (`swiftbackend` + `traffic-backfill` job + local ONNX YOLO26s), [`services/forecastapi/`](services/forecastapi/README.md) (`forecast-api` local-model serving, [ADR 0004](docs/adr/0004-serve-local-models.md)), and [`services/gmapfetcher/`](services/gmapfetcher/README.md) (`gmap-woodlands-fetcher` Distance Matrix ingestion, Node 22) |
| [`hosting/`](hosting/index.html) | Firebase Hosting web client (`index.html`, `app.js`, `style.css`), deployed to `swiftborder-92b45` via `firebase.json`, `.firebaserc`, and [`.github/workflows/firebase-hosting.yml`](.github/workflows/firebase-hosting.yml) ([ADR 0002](docs/adr/0002-firebase-hosting-source.md)) |
| [`data/`](data/README.md) | BigQuery warehouse DDL and offline data ingestion: [`data/sql/`](data/sql/README.md) (views `v_bins_10min`, `v_training_set`, `v_forecast_recent`, `cam2701.local_counts`, and historical BQML definitions) and [`data/weather/`](data/weather/README.md) (NEA data.gov.sg rainfall/forecast fetchers and `load_bigquery.py`) |
| [`eval/`](eval/README.md) | Layer A and B harnesses (DM + Holm), FCM-MLP / TimesFM / mix modules, 15-frame detector benchmark ([`test-reports/`](eval/test-reports/README.md)), CNN-RNN study ([`cnn_rnn_congestion/`](eval/cnn_rnn_congestion/README.md)), and frozen evidence (`eval/runs/report/`, `layer_a_count/`, `layer_a_eval/`) |
| [`docs/`](docs/README.md) | Report spine (architecture, evaluation, findings, GCP inventory, final-report readiness, grading alignment), deep-learning & horizon studies, ADRs, runbooks, and planning archive |
| [`scripts/`](scripts/run_tests.ps1) & [`skills/`](skills/documentation-generation/SKILL.md) | Cross-suite test runners (`run_tests.ps1`, `run_tests.sh`), repository doc/diagram/Cloud Build gate tests (`scripts/tests/`), and agent review/documentation skills |

GCP project `swiftborder` is the source of truth for what is deployed; [docs/inventory.md](docs/inventory.md) is the dated copy. Weather tables can be appended manually with `data/weather/load_bigquery.py`.

## Testing

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r services/camdetect/requirements-dev.txt -r data/weather/requirements-dev.txt -r eval/requirements-dev.txt -r services/forecastapi/requirements-dev.txt -r scripts/requirements-dev.txt
scripts\run_tests.ps1            # all fast suites (camdetect, weather, eval, forecastapi, repo doc checks)
scripts\run_tests.ps1 -Slow      # + TensorFlow tests (needs eval/requirements-notebook.txt)
scripts\run_tests.ps1 -Coverage
cd services/gmapfetcher && npm test  # Node 22 unit tests for traffic-24h.json snapshot builder
```

On Linux/macOS use `scripts/run_tests.sh [--slow] [--coverage]`. GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs the fast suites on Python 3.11 for every push and PR; it deploys nothing. Dependencies are pinned per suite; the full resolved environment is [requirements-lock-py311.txt](requirements-lock-py311.txt).

## Documentation & Lecturer Review Guide

Start with the six core files below for the NUS-ISS Practice Module (Pattern Recognition Systems) final review; see [docs/README.md](docs/README.md) for the complete index:

| Review area | Document | What to verify |
| --- | --- | --- |
| **1. Grading alignment** | [docs/grading/nus-iss-practice-module.md](docs/grading/nus-iss-practice-module.md) | How the repository satisfies all four Practice Module technique categories (Intelligent Sensing, Supervised/Unsupervised Learning, ML/DL, Hybrid/Ensemble) and rubric deliverables |
| **2. Final-report readiness & freeze** | [docs/final-report-readiness.md](docs/final-report-readiness.md) | 17 Oct 2026 submission freeze at `199c2e5`, reproducibility audit, claim hygiene, and lecturer reading guide |
| **3. System architecture** | [docs/architecture.md](docs/architecture.md) | High-level & detailed GCP architecture, Layer A (`camdetect`) and Layer B (`forecastapi`, `gmapfetcher`, BigQuery) contracts, and active vs unused paths |
| **4. Empirical evaluation** | [docs/evaluation.md](docs/evaluation.md) | Frozen Evidence Index, Layer A 15-frame detector comparison + box AP50, Layer B 18-day rolling backtests, DM + Holm significance, fuzzy/deep/ensemble ablations, and Layer A → B correlation & ablation |
| **5. Findings & claims register** | [docs/findings.md](docs/findings.md) | Six numbered technical findings, explicit negative results, and the safe-claims register |
| **6. Live GCP inventory** | [docs/inventory.md](docs/inventory.md) | Dated read-only query snapshot of GCP project `swiftborder` (BigQuery tables/views/models, Cloud Run revisions, Cloud Build triggers, Cloud Scheduler, GCS) |
| **Supporting studies** | [docs/deep-learning-assessment.md](docs/deep-learning-assessment.md) · [docs/horizon-study.md](docs/horizon-study.md) · [eval/test-reports/](eval/test-reports/README.md) · [eval/cnn_rnn_congestion/](eval/cnn_rnn_congestion/README.md) · [docs/layer-b-mix-module.md](docs/layer-b-mix-module.md) | Sequence model trade-offs, 30 min – 24 h horizon skill curve, annotated YOLO test frames & GPU latency benchmark, CNN-RNN auxiliary vision study, and FCM-MLP + XGBoost 50/50 mix |
| **Full index & changelog** | [docs/README.md](docs/README.md) · [CHANGELOG.md](CHANGELOG.md) · [docs/agent-deploy.md](docs/agent-deploy.md) | Complete index including ADRs (`docs/adr/`), runbooks (`docs/runbooks/`), historical roadmap/plans, and repository history |

## Agents and grading

- [AGENTS.md](AGENTS.md) — harness-agnostic docs/review rules
- [Practice Module grading lens](docs/grading/nus-iss-practice-module.md)
- Skills: [documentation-generation](skills/documentation-generation/SKILL.md), [diagram-image-generation](skills/diagram-image-generation/SKILL.md), [architecture-code-review](skills/architecture-code-review/SKILL.md). Edit those, then mirror the body under `.cursor/skills/` and fix relative links (`../../` vs `../../../`).
