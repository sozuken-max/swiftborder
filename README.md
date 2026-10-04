# SwiftBorder

Woodlands-only causeway crossing-time forecasting (NUS-ISS Practice Module, Group 3). The **live** system serves a **30-minute** forecast of the Google Maps duration series; a 24-hour horizon and <= 15 min MAE are **product targets**, not results.

| Layer | Job |
| --- | --- |
| **A - Vision** | LTA camera frames -> Roboflow YOLO (label / train / serve) -> per-direction vehicle counts (camera 2701) via Cloud Run `swiftbackend` |
| **B - Forecasting** | Maps Distance Matrix durations every 5 minutes in BigQuery -> Cloud Run `forecast-api`, which fits local models in-process ([ADR 0004](docs/adr/0004-serve-local-models.md)): the local `lin_h30` replica for SG→MY, persistence for MY→SG. The target is the bin 30–40 min after the newest closed bin. BQML `lin_h30` / `xgb_h30` / `v_forecast_recent` remain as history |

**Principal risk:** queue counts are not crossing duration, and the label is Maps' own estimate. Every Layer B score is skill over persistence on the Maps series; there is no independent wait-time measurement.

## Results (13–30 Sep 2026, out-of-sample)

These are the committed snapshot's numbers. They come from the BQML era, and `forecast-api` now serves local replicas instead; final numbers come from the frozen run ([roadmap](docs/roadmap.md)). 30-minute MAE on the Maps series, both directions: persistence 2.64 min, BQML-era served forecast (registry) 2.54, `lin_h30` 2.78, `xgb_h30` 2.49, ensemble 2.46, daily-refit XGBoost 2.28. The ensemble is significantly better than persistence (`xgb_h30` too within its test family, not under a run-wide check); all 30-minute gains are under half a minute, so none is claimed as a product-relevant win; `lin_h30` (the model currently served for some directions) is not significantly different from persistence across both directions. "Significant" means a Diebold–Mariano test (do two forecasts' errors differ more than chance?) with a Holm correction for testing several models at once, and confidence intervals from a bootstrap that resamples whole days so that correlated 5-minute errors are not counted as independent. Weather features gave no significant gain. Stacking or gating the models adds nothing over the daily-refit XGBoost, which would beat the served forecast by about 0.27 min. At 60 minutes on one route, LSTM, GRU and a patch Transformer trail XGBoost. An exploratory horizon study (every 30 min to 24 h, same folds, not a report run) finds current traffic beating a calendar-profile baseline to about 5.5 h (3 h under a strict Holm check), and no consistent winner after; the API shows those horizons labelled exploratory ([docs/horizon-study.md](docs/horizon-study.md)). A fuzzy light / moderate / heavy forecast is best as a hybrid on point estimates (XGBoost defuzzified into levels, 0.785 accuracy vs 0.690 for persistence). Its 6-day hold-out is too short for a significance claim under the pooled-direction rule ([findings](docs/findings.md#pooled-significance-sensitivity-2026-10-04)). No Layer A output overlaps the Maps window, so Layer B gets a queue forecast learned from the March–April camera detections instead. That forecast lowers XGBoost 30-min MAE by 0.065 min inside its own Holm family only; the run-wide Holm check does not keep the XGBoost result. Ridge's improvement survives the run-wide check. Neither beats a daily profile learned from Maps itself. Observed camera counts are still untested, and nothing yet supports using camera counts in place of Distance Matrix data ([evaluation.md §7b](docs/evaluation.md#7b-could-layer-a-output-replace-the-distance-matrix-data)). Layer A metrics are pending, so Layer A is an unvalidated prototype until an export is scored. Live camera calls are not stored, and `cam2701` / `cam2702` in BigQuery are a March–April detection batch. Details, slices and caveats: [docs/evaluation.md](docs/evaluation.md); evidence bundle: [`eval/runs/report/`](eval/runs/report/).

## Techniques (Practice Module)

| Category | Where |
| --- | --- |
| Supervised learning | Roboflow-labelled YOLO; regression of future Maps duration (BQML, XGBoost, ridge); traffic-level classification |
| Machine learning / deep learning | YOLO; `lin_h30`, `xgb_h30`; offline XGBoost; LSTM, GRU and a patch Transformer (scored, [deep-learning assessment](docs/deep-learning-assessment.md)) |
| Intelligent sensing | LTA frames -> directional occupancy with the camera 2701 dividing line |
| Hybrid / ensemble | Mean of `lin_h30` + `xgb_h30`; ridge + XGBoost; rolling LAD stack, rolling selection and fuzzy-gated stack; XGBoost forecast defuzzified into traffic levels. All scored against the single models |
| Fuzzy logic | Light / moderate / heavy partition and a learned fuzzy rule-based classifier |

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
| [`camdetect/`](camdetect/) | `swiftbackend` source: Roboflow detection + directional counts (deployed by Cloud Build on push to `main`) |
| [`Causeway/`](Causeway/) | data.gov.sg rainfall / 2-hour forecast history fetchers (CSV) and `load_bigquery.py` (manual append to the weather tables) |
| [`eval/`](eval/) | Layer A and B harnesses, significance, feature builder, camera backfill; report snapshot `eval/runs/report/` |
| [`sql/`](sql/) | BigQuery view and BQML definitions exported from project `swiftborder` |
| [`docs/`](docs/) | Report drafts (design, evaluation, findings), GCP inventory, roadmap, work plan |
| [`scripts/`](scripts/) | Test runners and repo doc checks |

GCP project `swiftborder` is the source of truth for what is deployed; [docs/inventory.md](docs/inventory.md) is the dated copy. The Maps fetcher, the `travel_times` loader and the camera table writers are not in this repo; the weather tables can be appended with `Causeway/load_bigquery.py` (manual).

The camera 2701 backfill for the joined experiment is run by the Roboflow key holder: [docs/handoff-camera-pilot.md](docs/handoff-camera-pilot.md).

## Testing

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r camdetect/requirements-dev.txt -r Causeway/requirements-dev.txt -r eval/requirements-dev.txt -r scripts/requirements-dev.txt
scripts\run_tests.ps1            # all suites (camdetect, Causeway, eval, repo doc checks)
scripts\run_tests.ps1 -Slow      # + TensorFlow tests (needs eval/requirements-notebook.txt)
scripts\run_tests.ps1 -Coverage
```

On Linux/macOS use `scripts/run_tests.sh [--slow] [--coverage]`. GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs the fast suites on Python 3.11 for every push and PR; it deploys nothing. Dependencies are pinned per suite; the full resolved environment is [requirements-lock-py311.txt](requirements-lock-py311.txt).

## Documentation

- [Docs index](docs/README.md) · [Design](docs/architecture.md) · [Evaluation](docs/evaluation.md) · [Findings and claims register](docs/findings.md)
- [Roadmap](docs/roadmap.md) · [Deep-learning assessment](docs/deep-learning-assessment.md) · [Active work plan](docs/plan-eval-integrity.md) · [Agent deploy instructions](docs/agent-deploy.md)
- [GCP inventory](docs/inventory.md) · [Release checklist for PR #2](docs/release-pr2.md) · [Changelog](CHANGELOG.md)

## Agents and grading

- [AGENTS.md](AGENTS.md) — harness-agnostic docs/review rules
- [Practice Module grading lens](docs/grading/nus-iss-practice-module.md)
- Skills: [documentation-generation](skills/documentation-generation/SKILL.md), [diagram-image-generation](skills/diagram-image-generation/SKILL.md), [architecture-code-review](skills/architecture-code-review/SKILL.md). Edit those, then mirror the body under `.cursor/skills/` and fix relative links (`../../` vs `../../../`).
