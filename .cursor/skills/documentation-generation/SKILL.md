---
name: documentation-generation
description: >-
  Use when writing or revising SwiftBorder README, docs/, architecture or evaluation
  diagrams, findings, or inventory snapshots — keep plan+status honesty and
  grading-aware framing.
---

# SwiftBorder documentation generation

## Goals

Produce accurate, proposal-ready docs that match verified GCP/git state. Stay
**plan + status** until evaluation numbers exist.

## Before writing

1. Read root [AGENTS.md](../../../AGENTS.md) and [docs/grading/nus-iss-practice-module.md](../../../docs/grading/nus-iss-practice-module.md).
2. **Source of truth is GCP project `swiftborder`.** Query it with `bq` / `gcloud` (or Console) before writing row counts, joins, serve horizon, or deploy path. `docs/inventory.md` is only the last dated copy.
3. When the query and the docs disagree, rewrite the docs to match the project.
4. Diff existing `docs/` — update rather than fork dated copies when content supersedes.
5. Record the query in `docs/inventory.md` and the README status snapshot, with the observation time. Do not copy volatile GCP facts into `AGENTS.md`.

## Document map

| Path | Purpose |
| --- | --- |
| `README.md` | Short pitch, Layer A/B, status snapshot, links into docs |
| `docs/architecture.md` | Report: tools, techniques, system design |
| `docs/evaluation.md` | Report: performance methods, reasoning, empty result tables |
| `docs/findings.md` | Report: findings, discussion, claims register |
| `docs/roadmap.md` | Ordered remaining work. Not a second architecture. |
| `docs/agent-deploy.md` | How to export a teammate's live deploy into git and Cloud Build. |
| `docs/inventory.md` | Evidence appendix. Dated copy of project `swiftborder`. Not the source of truth. |
| `docs/grading/nus-iss-practice-module.md` | Rubric. Report prose should be able to land in its four sections |
| `docs/images/` | Deck PNGs (polished GCP-icon style) |

## Diagram rules

- **Style:** polished GCP-icon architecture (Cloud Run, BigQuery, Scheduler, Storage icons; Layer A/B bands). Prefer this over Mermaid screenshots for slide assets.
- **One system, two depths.** High-level and detailed diagrams describe the same queried system. Do not add an as-is / to-be or proposal-target pair. Unfinished scope is prose in `docs/findings.md` and the order of work is `docs/roadmap.md`.
- Do not invent joins. If training is Maps-only, do not draw weather or camera congestion into `traffic_prediction`.
- Show present-but-unused sources as dashed callouts in deck PNGs and as `-.->` edges in the detailed Mermaid. Do not draw them as inputs to training.
- `traffic_images.labels` is the empty table. `traffic_images.metadata` is populated. Do not swap them.
- `traffic-backfill` is a Cloud Run Job that rebuilds metadata and `backfill_checkpoint`.
- Label serve horizon from a fresh query of project `swiftborder` (live horizon vs 24h intent). Do not hard-code that result into `AGENTS.md`.
- Include risks that matter for grading (missing harness, honesty on targets), not region-consolidation theatre.
- Layer A and Layer B each need an evaluation diagram under `docs/images/eval-layer-a.png` and `eval-layer-b.png`.

## Proposal language

- Soften: "deployed", Firebase live, <=15 min achieved, vision+weather feeding the model (if views unused).
- Keep: Woodlands-only, cameras 2701/2702, evaluation protocol, principal risk counts != duration.
- Move measured tables / rankings to the **final report**.

## Output checklist

- [ ] Facts dated; no invented resources
- [ ] Status box matches the GCP query it cites, and `docs/inventory.md` was refreshed from that same query
- [ ] Images referenced with relative paths and present on disk
- [ ] ASCII-safe punctuation in markdown if Windows checkouts mangle Unicode
- [ ] Grading weights / deliverables not contradicted
- [ ] At least three module technique categories are nameable for the system (see grading lens)
- [ ] This file's body copied to `.cursor/skills/documentation-generation/SKILL.md` with `../../` rewritten to `../../../`
