# SwiftBorder

Woodlands-only **24-hour causeway crossing-time forecast** (NUS-ISS Practice Module, Group 3).

Two layers:

| Layer | Job |
| --- | --- |
| **A - Vision** | LTA camera frames -> Roboflow YOLO (label / train / serve) -> per-direction counts and congestion features |
| **B - Forecasting** | Maps Distance Matrix labels in BigQuery; live serve is **30 minutes** on Maps lags and time-of-day (`lin_h30` or persistence). Vision, weather, and holidays are planned joins, not in the current training view. |

**Principal risk:** queue counts are not the same as crossing duration. The label series is Maps' current duration estimate. The baseline is persistence of that series. The project has no independent wait-time measurement.

**Success target (final report):** <= 15 min MAE -- a **target**, not a measured result until evaluation lands.

## Repository layout

| Path | Contents |
| --- | --- |
| [`camdetect/`](camdetect/) | Camera detection spike (2701 directional detect via Roboflow) |
| [`Causeway/`](Causeway/) | Weather / rainfall fetch and filter scripts |
| [`eval/`](eval/) | Read-only Layer B harness (does not deploy `swiftbackend`) |
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

Live resources and row counts come from GCP project `swiftborder`. Refresh [docs/inventory.md](docs/inventory.md) after you query the project. The served forecast is 30 minutes on Maps lags (`lin_h30` or persistence). A 24-hour horizon and <= 15 min MAE are targets until the harness fills [docs/evaluation.md](docs/evaluation.md). Claims and reasoning: [docs/findings.md](docs/findings.md).

## Contributing

Prefer small, reviewable PRs. Evaluation and Risk ownership (metrics harness, protocol docs, honest proposal framing) is a natural first footprint in this repo. Do not claim measured MAE or "we beat Google" until the harness produces numbers for the final report.

## Agents and grading

- [AGENTS.md](AGENTS.md) — harness-agnostic docs/review rules
- [Practice Module grading lens](docs/grading/nus-iss-practice-module.md)
- Skills: `skills/documentation-generation`, `skills/architecture-code-review`. Edit those, then mirror the body under `.cursor/skills/` and fix relative links (`../../` vs `../../../`).
