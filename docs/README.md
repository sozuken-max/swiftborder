# SwiftBorder docs

**Source of truth for deployed state:** GCP project `swiftborder`. These pages are a copy of a read-only `bq` / `gcloud` check around **13:40 SGT on 26 September 2026**. When they disagree with the project, the project wins.

They are drafted as final-report sections (Practice Module: tools and design, performance, findings). The rubric is in [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md).

| Report section | Doc |
| --- | --- |
| Tools, techniques, system design | [architecture.md](architecture.md) |
| Performance: methods, reasoning, result tables | [evaluation.md](evaluation.md) |
| Findings and what may be claimed | [findings.md](findings.md) |
| What remains, and the order to do it | [roadmap.md](roadmap.md) |
| How a teammate's agent checks a local deploy into CI | [agent-deploy.md](agent-deploy.md) |
| Evidence appendix (dated GCP copy) | [inventory.md](inventory.md) |
| Rubric | [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md) |

## Images

Design figures are **high-level** and **detailed** views of one system, not an as-is / to-be pair. PNGs were regenerated **26 Sep 2026 ~13:40 SGT**. Training is Maps-only; weather and camera-2701 congestion are present and not joined. The detailed dataflow PNG still labels `traffic_images.metadata` as 0 rows. That table has 368,905 rows. `traffic_images.labels` is empty. See [architecture.md](architecture.md).

| File | Report use |
| --- | --- |
| `architecture-as-is.png` | High-level design (file name is legacy) |
| `dataflow-as-is.png` | Detailed design (file name is legacy) |
| `eval-layer-a.png` | Performance: Layer A method |
| `eval-layer-b.png` | Performance: Layer B method |

`architecture-proposal.png` and `dataflow-target.png` are leftover as-is / to-be slides. Do not use them as a second system in the report.

Region layout is not part of the graded story.
