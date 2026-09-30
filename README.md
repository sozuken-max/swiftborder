# SwiftBorder

Woodlands-only causeway crossing-time forecasting (NUS-ISS Practice Module, Group 3). The **live** system serves a **30-minute** forecast of the Google Maps duration series; a 24-hour horizon and <= 15 min MAE are **product targets**, not results.

| Layer | Job |
| --- | --- |
| **A - Vision** | LTA camera frames -> Roboflow YOLO (label / train / serve) -> per-direction vehicle counts (camera 2701) via Cloud Run `swiftbackend` |
| **B - Forecasting** | Maps Distance Matrix durations every 5 minutes in BigQuery -> BQML `lin_h30` / `xgb_h30` -> `v_forecast_recent` (30 min; `lin_h30` or persistence per direction) |

**Principal risk:** queue counts are not crossing duration, and the label is Maps' own estimate. Every Layer B score is skill over persistence on the Maps series; there is no independent wait-time measurement.

## Results (13–30 Sep 2026, out-of-sample)

30-minute MAE on the Maps series, both directions: persistence 2.64 min, `lin_h30` 2.78, `xgb_h30` 2.49, ensemble 2.46. `xgb_h30` and the ensemble are significantly better than persistence (Diebold–Mariano with Holm correction, day-block bootstrap). Weather features gave no significant gain. Camera features and Layer A metrics are pending. Details, slices and caveats: [docs/evaluation.md](docs/evaluation.md); evidence bundle: [`eval/runs/report/`](eval/runs/report/).

## Techniques (Practice Module)

| Category | Where |
| --- | --- |
| Supervised learning | Roboflow-labelled YOLO; regression of future Maps duration (BQML, XGBoost, ridge) |
| Machine learning / deep learning | YOLO; `lin_h30`, `xgb_h30`; offline XGBoost; LSTM code (unscored) |
| Intelligent sensing | LTA frames -> directional occupancy with the camera 2701 dividing line |
| Hybrid / ensemble | Mean of `lin_h30` + `xgb_h30`, ridge + XGBoost, scored against the single models |

## Run the MVP

1. **Directional counts (Layer A, live):** call `swiftbackend` for camera 2701. It is public and each call uses Roboflow credits (see [docs/findings.md](docs/findings.md#known-risk-public-swiftbackend-documented-not-changed)).
   ```bash
   curl "https://swiftbackend-1095552466513.europe-west1.run.app/?camera_id=2701&format=json"
   curl -o frame.jpg "https://swiftbackend-1095552466513.europe-west1.run.app/?camera_id=2701&format=directional"
   ```
2. **30-minute forecast (Layer B, live, read-only):**
   ```bash
   bq query --nouse_legacy_sql --project_id swiftborder \
     'SELECT direction, bin_sgt, observed_now_min, serving_model, forecast_30min_min
      FROM `swiftborder.traffic_prediction.v_forecast_recent` ORDER BY bin_ts DESC LIMIT 4'
   ```
3. **Evaluation:** `cd eval && pip install -r requirements-dev.txt && python layer_b.py` (read-only), or the full report run in [eval/README.md](eval/README.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [`camdetect/`](camdetect/) | `swiftbackend` source: Roboflow detection + directional counts (deployed by Cloud Build on push to `main`) |
| [`Causeway/`](Causeway/) | data.gov.sg rainfall / 2-hour forecast history fetchers (CSV) |
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
- [Roadmap](docs/roadmap.md) · [Active work plan](docs/plan-eval-integrity.md) · [Agent deploy instructions](docs/agent-deploy.md)
- [GCP inventory](docs/inventory.md) · [Changelog](CHANGELOG.md)

## Agents and grading

- [AGENTS.md](AGENTS.md) — harness-agnostic docs/review rules
- [Practice Module grading lens](docs/grading/nus-iss-practice-module.md)
- Skills: [documentation-generation](skills/documentation-generation/SKILL.md), [diagram-image-generation](skills/diagram-image-generation/SKILL.md), [architecture-code-review](skills/architecture-code-review/SKILL.md). Edit those, then mirror the body under `.cursor/skills/` and fix relative links (`../../` vs `../../../`).
