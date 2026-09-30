# SwiftBorder docs

**Source of truth for deployed state:** GCP project `swiftborder`. Dated query results live in [inventory.md](inventory.md). Repo and deploy history live in [CHANGELOG.md](../CHANGELOG.md). When these pages disagree with the project, the project wins.

They are drafted as final-report sections (Practice Module: tools and design, performance, findings). The rubric is in [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md).

| Report section | Doc |
| --- | --- |
| Tools, techniques, system design | [architecture.md](architecture.md) |
| Performance: methods, reasoning, result tables | [evaluation.md](evaluation.md) |
| Findings and what may be claimed | [findings.md](findings.md) |
| What remains, and the order to do it | [roadmap.md](roadmap.md) |
| Should we use LSTM or Transformer forecasters? (deferred, with reasons) | [deep-learning-assessment.md](deep-learning-assessment.md) |
| Active work plan (evaluation integrity, hardening, joined experiment) | [plan-eval-integrity.md](plan-eval-integrity.md) |
| Handoff: camera 2701 backfill for the Roboflow key holder | [handoff-camera-pilot.md](handoff-camera-pilot.md) |
| How a teammate's agent checks a local deploy into CI | [agent-deploy.md](agent-deploy.md) |
| Evidence appendix (dated GCP copy) | [inventory.md](inventory.md) |
| Layer B harness and report figures | [../eval/README.md](../eval/README.md), committed snapshot [../eval/runs/report/](../eval/runs/report/) |
| BigQuery views and BQML | [../sql/README.md](../sql/README.md) |
| Repo and deploy history | [../CHANGELOG.md](../CHANGELOG.md) |
| Rubric | [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md) |

## Images and diagrams

**Mermaid source of truth:** [diagrams/](diagrams/) (`*.mmd`). [architecture.md](architecture.md) and [evaluation.md](evaluation.md) embed synced copies; run `python docs/diagrams/sync_mermaid.py` after editing a `.mmd` file. **Deck PNGs** under `images/` are exports for slides—refresh them when Mermaid topology changes ([diagrams/README.md](diagrams/README.md), [diagram-image-generation](../skills/diagram-image-generation/SKILL.md)).

Design figures are **high-level** and **detailed** views of one system, not an as-is / to-be pair. Live training (`v_training_set`) is Maps-only; the weather and camera-2701 congestion views exist and are not joined. Weather is joined only in the offline experiment (`eval/joined.py`), where it gave no gain. `traffic_images.metadata` is populated; `traffic_images.labels` is empty. See [architecture.md](architecture.md) and [inventory.md](inventory.md) for counts.

| Mermaid (edit first) | PNG export | Report use |
| --- | --- | --- |
| `diagrams/architecture-high-level.mmd` | `architecture-high-level.png` | High-level design |
| `diagrams/architecture-detailed.mmd` | `architecture-detailed.png` | Detailed design |
| `diagrams/eval-layer-a.mmd` | `eval-layer-a.png` | Performance: Layer A method |
| `diagrams/eval-layer-b.mmd` | `eval-layer-b.png` | Performance: Layer B method |
| — | `../eval/runs/report/{offline,bqml,joined}/*.png` | Performance: scored comparison plots (see [evaluation.md](evaluation.md)) |

The four deck PNGs predate the 2026-10-01 fact pass and are **stale** until regenerated ([diagrams/README.md](diagrams/README.md#png-exports-are-stale-regenerate-before-the-deck)). The Mermaid sources are current. There is no as-is / to-be pair.

Region layout is not part of the graded story.
