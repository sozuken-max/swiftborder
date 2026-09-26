---
name: architecture-code-review
description: >-
  Use when reviewing SwiftBorder PRs, code, architecture diagrams, or GCP-vs-git
  drift — harness-agnostic checklist for honesty, eval readiness, and Practice Module fit.
---

# SwiftBorder architecture and code review

## Goals

Catch inventiveness, proposal overclaim, and missing evaluation story. Reviews are
**harness-agnostic**: apply whether the change came from Cursor, Claude Code, humans, or CI.

## Read first

1. [AGENTS.md](../../AGENTS.md)
2. [docs/grading/nus-iss-practice-module.md](../../docs/grading/nus-iss-practice-module.md)
3. A fresh query of GCP project `swiftborder` when the change describes live resources. [docs/inventory.md](../../docs/inventory.md) is the last dated copy, not the authority. Report drafts: [docs/architecture.md](../../docs/architecture.md) (design), [docs/evaluation.md](../../docs/evaluation.md) (methods and reasoning), [docs/findings.md](../../docs/findings.md) (claims).

## Review dimensions

### A. Truthfulness

- Claims about live services match a query of project `swiftborder` (Scheduler, Run, BQ tables/views, models). A match to `docs/inventory.md` alone is not enough when that file is older than the claim.
- No fake buckets/datasets. `traffic-backfill` must be described as a **Cloud Run Job**.
- Training feature set accurate (Maps-only vs joined weather/vision).
- Serve horizon matches project `swiftborder` (live horizon versus the 24h intent).

### B. Evaluation readiness (Yingzhao / Evaluation & Risk)

- Layer A metrics path: mAP, precision/recall, count-error, day-night; labels export story.
- Layer B metrics path: MAE/RMSE vs **persistence** and vs **Maps**; direction/TOD slices.
- `model_registry` should be filled from harness scores, not preference.
- Flag PRs that claim wins without a checked-in harness or scored tables. Result cells in `docs/evaluation.md` stay `pending` until then. Claims stay inside `docs/findings.md`.

### C. Code and deploy

- What does Cloud Build / CI actually deploy? (Historically: `camdetect` -> `swiftbackend` only.)
- Are Maps fetcher and ingest loaders still outside git? View/BQML SQL should live under `sql/`; eval under `eval/`.
- Secrets: never commit Maps keys or tokens; flag leaked credentials.

### D. Practice Module fit

- Name **at least three** of: supervised or unsupervised learning; ML/DL techniques; hybrid or ensemble; intelligent sensing. A change that cannot name three is a grading gap.
- Runnable MVP path is clear.
- Final-report sections foreshadowed: tools, design/models, performance, findings.
- Peer / individual reflection not blocked by missing contribution footprint.

### E. Docs and diagrams

- Diagram style: GCP-icon preferred for decks; must match project `swiftborder`.
- High-level and detailed figures describe one system. Flag a second as-is / to-be architecture.
- Unused feature views are dashed, not drawn into training.
- `traffic_images.labels` is empty; `traffic_images.metadata` is not. Flag a diagram that swaps them.
- Both `eval-layer-a.png` and `eval-layer-b.png` present and referenced when evaluation docs change.
- If this skill's body changed, the `.cursor/skills/` mirror was updated and relative links adjusted (`../../` under `skills/`, `../../../` under `.cursor/skills/`).

## Review output format

```text
Verdict: approve | request changes | comment
Truthfulness: ...
Eval readiness: ...
Deploy/git drift: ...
Grading risk: ...
Must-fix before merge: ...
Nice-to-have: ...
```

## Non-goals

- Do not demand multi-region consolidation for its own sake.
- Do not block on tooling brand (Claude Code vs Cursor) — review the artifact.
