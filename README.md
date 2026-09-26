# SwiftBorder

Woodlands-only **24-hour causeway crossing-time forecast** (NUS-ISS Practice Module, Group 3).

Two layers:

| Layer | Job |
| --- | --- |
| **A - Vision** | LTA camera frames -> Roboflow YOLO (label / train / serve) -> per-direction counts and congestion features |
| **B - Forecasting** | Maps Distance Matrix labels in BigQuery; live serve is **30 minutes** on Maps lags and time-of-day (`lin_h30` or persistence). Vision, weather, and holidays are planned joins, not in the current training view. |

**Principal risk:** queue counts are not the same as crossing duration. The label series is Maps' current duration estimate. The baseline is persistence of that series. The project has no independent wait-time measurement.

**Success target (final report):** <= 15 min MAE on an independent wait-time study remains a **product target**; scored Maps-series metrics and figures are in [docs/evaluation.md](docs/evaluation.md) and [`eval/runs/report/`](eval/runs/report/).

## Repository layout

| Path | Contents |
| --- | --- |
| [`camdetect/`](camdetect/) | Camera detection spike (2701 directional detect via Roboflow) |
| [`Causeway/`](Causeway/) | Weather / rainfall fetch and filter scripts |
| [`eval/`](eval/) | Layer B harness (`layer_b.py`), offline XGB/LSTM helpers, significance and comparison plots; committed report snapshot under `eval/runs/report/` |
| [`sql/`](sql/) | BigQuery view and BQML definitions exported from project `swiftborder` |
| [`docs/`](docs/) | Final-report drafts: design, evaluation and reasoning, findings; GCP inventory is the evidence appendix |

GCP project `swiftborder` is the source of truth for what is deployed. This repo is behind that project. Re-query the project before treating the docs as current.

## Documentation

Drafted against the Practice Module report sections:

- [Docs index](docs/README.md)
- [Design: tools, techniques, architecture](docs/architecture.md)
- [Performance: evaluation and reasoning](docs/evaluation.md)
- [Findings and claims](docs/findings.md)
- [Roadmap](docs/roadmap.md)
- [Agent instructions: local deploy to CI](docs/agent-deploy.md)
- [GCP evidence snapshot](docs/inventory.md) (dated copy of project `swiftborder`)
- [Changelog](CHANGELOG.md) (repo and deploy history)

## Current status

Live resources and row counts come from GCP project `swiftborder`. Refresh [docs/inventory.md](docs/inventory.md) after you query the project. The served forecast is 30 minutes on Maps lags (`lin_h30` or persistence). Layer B **methods, numbers, significance, and figures** are documented in [docs/evaluation.md](docs/evaluation.md) (citation bundle: [`eval/runs/report/`](eval/runs/report/)). A 24-hour horizon remains a product target. Claims register: [docs/findings.md](docs/findings.md).

## Contributing

Prefer small, reviewable PRs. Evaluation and Risk ownership (metrics harness, protocol docs, honest proposal framing) is a natural first footprint in this repo. Cite [`eval/runs/report/run.json`](eval/runs/report/run.json) for dataset and model provenance. Report skill on the Maps series vs persistence; do not claim "we beat Google" without an independent wait-time label.

## Agents and grading

- [AGENTS.md](AGENTS.md) — harness-agnostic docs/review rules
- [Practice Module grading lens](docs/grading/nus-iss-practice-module.md)
- Skills: `skills/documentation-generation`, `skills/architecture-code-review`. Edit those, then mirror the body under `.cursor/skills/` and fix relative links (`../../` vs `../../../`).
