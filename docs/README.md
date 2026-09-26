# SwiftBorder docs

**Source of truth for deployed state:** GCP project `swiftborder`. Dated query results live in [inventory.md](inventory.md). Repo and deploy history live in [CHANGELOG.md](../CHANGELOG.md). When these pages disagree with the project, the project wins.

They are drafted as final-report sections (Practice Module: tools and design, performance, findings). The rubric is in [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md).

| Report section | Doc |
| --- | --- |
| Tools, techniques, system design | [architecture.md](architecture.md) |
| Performance: methods, reasoning, result tables | [evaluation.md](evaluation.md) |
| Findings and what may be claimed | [findings.md](findings.md) |
| What remains, and the order to do it | [roadmap.md](roadmap.md) |
| How a teammate's agent checks a local deploy into CI | [agent-deploy.md](agent-deploy.md) |
| Evidence appendix (dated GCP copy) | [inventory.md](inventory.md) |
| Layer B harness and report figures | [../eval/README.md](../eval/README.md), committed snapshot [../eval/runs/report/](../eval/runs/report/) |
| BigQuery views and BQML | [../sql/README.md](../sql/README.md) |
| Repo and deploy history | [../CHANGELOG.md](../CHANGELOG.md) |
| Rubric | [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md) |

## Images

Design figures are **high-level** and **detailed** views of one system, not an as-is / to-be pair. Training is Maps-only; weather and camera-2701 congestion are present and not joined. `traffic_images.metadata` is populated; `traffic_images.labels` is empty. See [architecture.md](architecture.md) and [inventory.md](inventory.md) for counts.

| File | Report use |
| --- | --- |
| `architecture-as-is.png` | High-level design (file name is legacy) |
| `dataflow-as-is.png` | Detailed design (file name is legacy) |
| `eval-layer-a.png` | Performance: Layer A method |
| `eval-layer-b.png` | Performance: Layer B method |
| `../eval/runs/report/offline/*.png`, `../eval/runs/report/bqml/*.png` | Performance: scored comparison plots (see [evaluation.md](evaluation.md)) |

Legacy as-is / to-be PNGs were removed from the tree. Use high-level and detailed figures only (see [architecture.md](architecture.md)).

Region layout is not part of the graded story.
