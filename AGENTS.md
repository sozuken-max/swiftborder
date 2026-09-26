# AGENTS.md — SwiftBorder

Harness-agnostic guidance for coding agents (Cursor, Claude Code, Copilot, Codex, etc.).
Read this before changing docs, diagrams, architecture notes, or reviewing code.

## Project in one paragraph

SwiftBorder is an NUS-ISS Practice Module (Pattern Recognition Systems) project: a Woodlands-only
causeway crossing-time forecast. **Layer A** turns LTA camera frames into directional counts
via Roboflow YOLO. **Layer B** forecasts duration from features plus Google Maps Distance
Matrix labels. Principal risk: **counts != crossing duration**. A 24-hour horizon and <=15 min
MAE are **targets** until a harness measures them.

## Source of truth

**GCP project `swiftborder` is the source of truth** for what exists and what is live: BigQuery datasets, views, and models; Cloud Run services and jobs; Cloud Scheduler; Cloud Storage; Cloud Build. Query it (`bq`, `gcloud`, or Console) before stating row counts, joins, serve horizon, or deploy path.

[docs/inventory.md](docs/inventory.md) is a **dated copy** of that query. When the copy and the project disagree, the project wins and the copy is updated. Do not paste volatile GCP facts into this file.

| Question | Authority |
| --- | --- |
| What is deployed, joined, served, or stored | GCP project `swiftborder` |
| What code and docs are checked in | This repo (`camdetect/`, `Causeway/`, `docs/`) |
| What the proposal may claim | Plan + status until a harness measures it |

Never collapse Git, the GCP project, and the proposal into one "done" story.

## Skills (read when relevant)

| Skill | When |
| --- | --- |
| [documentation-generation](skills/documentation-generation/SKILL.md) | Writing or revising README, report drafts under `docs/`, diagrams |
| [architecture-code-review](skills/architecture-code-review/SKILL.md) | Reviewing code, PRs, architecture, GCP vs git drift |
| [Local deploy to CI](docs/agent-deploy.md) | A teammate's agent is capturing a Cloud Run / Scheduler / BigQuery deploy into this repo |

Cursor discovers the mirror under `.cursor/skills/`. Edit `skills/` first, then copy the body
into `.cursor/skills/` and rewrite relative links: `../../` from `skills/<name>/`, `../../../`
from `.cursor/skills/<name>/`. The bodies should match. The files are not byte-identical.

## Grading lens (Practice Module)

See [docs/grading/nus-iss-practice-module.md](docs/grading/nus-iss-practice-module.md).
Graded: presentations, final report, runnable MVP, peer review, methods/metrics.
The system must show **at least three** of: supervised or unsupervised learning, ML/DL
techniques, a hybrid or ensemble approach, intelligent sensing. Name which three a change
supports.
**Not** graded: cloud region sprawl / multi-region hygiene narratives.

## Hard honesty rules

- Do not invent GCP buckets, datasets, services, or metrics.
- Do not claim measured MAE or "we beat Google/persistence" until harness output exists.
- `traffic-backfill` is a **Cloud Run Job**, not a BigQuery dataset.
- Layer A labelling lives in **Roboflow**; empty BQ `traffic_images.labels` != "cannot export".
- Roboflow Public: dataset export OK after version; **weights download = Core**.
- Prefer polished GCP-icon architecture diagrams over Mermaid screenshots for deck assets.
- Diagrams and prose must match a fresh query of project `swiftborder`. Draw that system twice: high-level, then detailed. Do not keep an as-is / to-be pair. Unused feature views are dashed callouts, not arrows into training. Unfinished scope goes in `docs/findings.md`. After the query, refresh `docs/inventory.md` so the snapshot matches the project.

## Ownership hints

Evaluation & Risk owns [docs/evaluation.md](docs/evaluation.md), [docs/findings.md](docs/findings.md), and the harness that fills the result tables.
Prefer small PRs that add docs/eval before claiming model wins.
