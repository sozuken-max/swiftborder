# Documentation Index

Start at the repository [README](../README.md) for a one-page overview. All reported metrics are pinned to the **17 Oct 2026 final-report freeze** (`main` merge `199c2e5`; see [final-report-readiness.md](final-report-readiness.md)).

## 1. Core Report Spine (Start Here — Lecturer Review)

A reviewer can read these six documents in order to verify system design, empirical methodology, negative/positive findings, live cloud state, and NUS-ISS Practice Module grading alignment:

| Section | Document | Summary |
| --- | --- | --- |
| **Grading & methods** | [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md) | NUS-ISS Practice Module (Pattern Recognition Systems) rubric mapping across Intelligent Sensing, Supervised/Unsupervised Learning, ML/DL, and Hybrid/Ensemble methods |
| **Readiness & freeze** | [final-report-readiness.md](final-report-readiness.md) | Final-report readiness review. **24 Oct 2026** assembly guide: nine-chapter structure, four research questions, canonical figures. Earlier sections are the 17 Oct freeze and claim audit |
| **Architecture** | [architecture.md](architecture.md) | High-level and detailed GCP architecture (`swiftborder`), Layer A (`camdetect`) and Layer B (`forecastapi`, `gmapfetcher`, BigQuery) contracts, and active vs unused paths |
| **Evaluation** | [evaluation.md](evaluation.md) | Frozen Evidence Index, Layer A 15-frame detector comparison + box AP50, Layer B 18-day rolling-origin backtests, Diebold–Mariano + Holm significance, deep/fuzzy/ensemble ablations, and Layer A → B correlation & ablation |
| **Findings & risks** | [findings.md](findings.md) | Six numbered technical findings, explicit negative results, risk register, and the safe-claims register |
| **GCP inventory** | [inventory.md](inventory.md) | Dated read-only query snapshot of GCP project `swiftborder` (BigQuery datasets/views/models, Cloud Run services/jobs, Cloud Build triggers, Cloud Scheduler, Cloud Storage) |

## 2. Technical Studies & Component Evidence Reports

| Topic | Document | Summary |
| --- | --- | --- |
| **Layer A detector benchmark (15 frames)** | [../eval/test-reports/README.md](../eval/test-reports/README.md) | Per-frame car/vehicle counts, box AP50 (IoU 0.50), WebP visual comparisons, and ONNX CPU/GPU speed benchmark on the Roboflow v6 `test` split ([raw frame manifest](../eval/test-reports/frames/README.md)) |
| **Layer A count rescoring & reproducibility** | [../eval/runs/layer_a_count/REPORT.md](../eval/runs/layer_a_count/REPORT.md) | Fixed served operating point (`conf=0.20, overlap=0.60`, car MAE **1.87**) vs threshold-swept artifact scores (`conf=0.30, overlap=off`, car MAE **1.73**) |
| **Layer A CNN-RNN congestion research** | [../eval/cnn_rnn_congestion/README.md](../eval/cnn_rnn_congestion/README.md) | Auxiliary ResNet18 + BiGRU + Bahdanau attention sequence classifier on camera 2701 crops (v8 Maps-supervision negative result; v10a human-label inter-rater study, macro-F1 **0.659**) |
| **Camera 2701 coverage map** | [camera-coverage/README.md](camera-coverage/README.md) | Where camera 2701 looks on the causeway, both checkpoints marked, and estimated distances from the edge of the view to each checkpoint (~0.4 km Woodlands, ~1.0 km BSI) |
| **Layer A → Layer B correlation & ablation** | [../eval/runs/layer_a_eval/report.md](../eval/runs/layer_a_eval/report.md) | 18-day correlation ($r = +0.462$ `SG_TO_MY`, $r = +0.618$ `MY_TO_SG`) and restricted 4-feature XGBoost ablation ($\Delta = -0.0827$ min, fold paired-$t$ $p = 0.0067$) |
| **Layer B horizon study (30 min – 24 h)** | [horizon-study.md](horizon-study.md) | Multi-horizon skill curve (`xgb[maps+prof]` vs persistence and weekday×time-of-day profile across 16 horizons; Holm-significant out to **3 h**) |
| **Layer B FCM-MLP & 50/50 XGBoost mix** | [layer-b-mix-module.md](layer-b-mix-module.md) | Fuzzy C-Means + MLP standalone ([results](layer-b-fcm-mlp-results.md), MAE **2.230 min**) and 50/50 ensemble with `xgb[maps]` (pooled MAE **2.186 min**, within $\pm 0.5$ min band) |
| **Deep learning assessment** | [deep-learning-assessment.md](deep-learning-assessment.md) | Why deep learning helps in Layer A (YOLO26s) while trees match or beat LSTM/GRU/TCN/Transformer on short univariate Layer B histories |

## 3. Runbooks, SQL, Component Guides & ADRs

| Topic | Document | Summary |
| --- | --- | --- |
| **Forecast API runbook** | [runbooks/forecast-api.md](runbooks/forecast-api.md) | Operating, verifying, and rolling back the `forecast-api` Cloud Run service ([component README](../services/forecastapi/README.md)) |
| **Camera detection service** | [../services/camdetect/README.md](../services/camdetect/README.md) | `swiftbackend` HTTP contract, `traffic-backfill` Cloud Run Job, dividing-line geometry (2701 & 2702), and local ONNX weights |
| **Maps ingestion service** | [../services/gmapfetcher/README.md](../services/gmapfetcher/README.md) | `gmap-woodlands-fetcher` Cloud Run service (`causeway.travel_times` + `traffic-24h.json` snapshot builder) |
| **Weather ingestion (offline)** | [../data/weather/README.md](../data/weather/README.md) | data.gov.sg rainfall and 2-hour forecast fetchers; manual `load_bigquery.py` append to `rainfall` / `weatherforecast` |
| **Evaluation harness guide** | [../eval/README.md](../eval/README.md) | CLI commands, data layout ([`eval/data/`](../eval/data/README.md)), frozen run artifacts ([`eval/runs/`](../eval/runs/README.md)), and exploratory notebooks ([`eval/notebooks/`](../eval/notebooks/README.md)) |
| **BigQuery SQL & DDL** | [../data/sql/README.md](../data/sql/README.md) | Checked-in BigQuery views, historical BQML model DDL, `model_registry`, and [re-export instructions](../data/sql/bigquery/EXPORT.md) |
| **Diagram sources & prompts** | [diagrams/README.md](diagrams/README.md) | Canonical Mermaid sources (`*.mmd`), Nano Banana 2 prompt specs, and `sync_mermaid.py` checker |
| **Local deploy to CI capture** | [agent-deploy.md](agent-deploy.md) | Playbook for capturing teammates' GCP deployments into git and wiring Cloud Build |
| **ADR 0001** | [adr/0001-firebase-client-api-calls.md](adr/0001-firebase-client-api-calls.md) | Browser-direct public `forecast-api` calls with CORS allowlist and cost caps |
| **ADR 0002** | [adr/0002-firebase-hosting-source.md](adr/0002-firebase-hosting-source.md) | Storing Firebase Hosting static source in `hosting/` (later wired to GitHub Actions via `firebase.json` and `.firebaserc`) |
| **ADR 0003** | [adr/0003-deep-training-and-feature-matrix.md](adr/0003-deep-training-and-feature-matrix.md) | Causal 10-minute feature matrix and multi-seed deep learning evaluation protocol |
| **ADR 0004** | [adr/0004-serve-local-models.md](adr/0004-serve-local-models.md) | Replacing live BQML `ML.PREDICT` with in-process local models (`lin_bq[frozen]`, `xgb_bq[frozen]`, persistence) in `forecast-api` |

## 4. Roadmap & Historical Planning Archive

These files record earlier sprint plans, PR checklists, and handoff notes. Where they mention a proposed 19 Oct data cutoff or future Runs A/B, the **17 Oct 2026 freeze** ([final-report-readiness.md](final-report-readiness.md)) supersedes those future tasks:

| Historical document | Context |
| --- | --- |
| [roadmap.md](roadmap.md) | Pre-freeze roadmap and workstream tracker (superseded on 17 Oct 2026 by the final-report freeze note at the top of the file) |
| [archive/plan-next-steps.md](archive/plan-next-steps.md) | Earlier two-run execution plan (superseded by [roadmap.md](roadmap.md) and the 17 Oct freeze) |
| [archive/plan-eval-integrity.md](archive/plan-eval-integrity.md) | 30 Sep – 1 Oct evaluation integrity overhaul checklist (completed) |
| [archive/handoff-camera-pilot.md](archive/handoff-camera-pilot.md) | 2 Oct camera 2701 backfill pilot handoff (superseded by the `traffic-backfill` Cloud Run Job on 4 Oct) |
| [archive/release-pr2.md](archive/release-pr2.md) | Release runbook for PR #2 (`eval-integrity-overhaul`, merged 3 Oct) |
| [archive/hosting-forecast-widget-delta.md](archive/hosting-forecast-widget-delta.md) | 3–4 Oct Firebase Hosting forecast widget delta note (superseded by 12 Oct `hosting/` updates) |
