# Architecture and evaluation diagrams (Mermaid)

**Source of truth:** the `.mmd` files in this directory. They define topology, joins, and dashed unused edges.

**Deck exports:** PNGs under [`../images/`](../images/) are slide assets. After changing a `.mmd` file, regenerate the matching PNG with [diagram-image-generation](../../skills/diagram-image-generation/SKILL.md) so facts and layout stay aligned.

**Embedded copies:** [`architecture.md`](../architecture.md) and [`evaluation.md`](../evaluation.md) contain fenced Mermaid blocks between HTML comment markers. Regenerate them from here:

```bash
python docs/diagrams/sync_mermaid.py
```

Edit **only** the `.mmd` files, then run the sync script before commit. Volatile counts belong in [inventory.md](../inventory.md), not hard-coded in diagram nodes.

| Mermaid file | Markdown embed | PNG export |
| --- | --- | --- |
| `architecture-high-level.mmd` | `architecture.md` (high-level) | `architecture-high-level.png` |
| `architecture-detailed.mmd` | `architecture.md` (detailed) | `architecture-detailed.png` |
| `eval-layer-a.mmd` | `evaluation.md` (Layer A) | `eval-layer-a.png` |
| `eval-layer-b.mmd` | `evaluation.md` (Layer B) | `eval-layer-b.png` |

## PNG exports refreshed: 4 Oct 2026

All four deck PNGs were regenerated with built-in image generation and visually checked against the updated Mermaid topology and the fresh serving/storage observations in [inventory.md](../inventory.md). They retain the GCP icon language, blue/gold bands, rounded cards and explicit pending-results caveats.

- High-level architecture: local forecast API, shared Hosting UI, separate deploy pipelines, historical BQML and offline evaluation.
- Detailed architecture: camera backfill job and metadata, local serving, explicit deployment/offline references, and all seven buckets. References are repeated in the evaluation/deployment panel to avoid crossing connectors; they are not extra resources.
- Layer A: scorer present, held-out results pending, correct stage icons.
- Layer B: paired scoring, recorded run evidence, current local serving, and the open pooled-day inference caveat.

The detailed architecture required a full-canvas structured redraw after two reference-based drafts failed topology review. Rejected drafts were not copied into this repository. Final files retain the generator's native 1672 x 941 resolution (approximately 16:9), rather than resampling text to the requested 1280 x 720. No post-generation paint-over or image manipulation was used.

Prompts: [image-generation-prompts.json](image-generation-prompts.json). Visual QA checked node/edge meaning, bucket spelling, pending-versus-scored status, panel containment, clear captions and readable text. This refresh does not resolve the application or evaluation issues listed in the readiness review. Yingzhao remains final on presentation acceptance.
**Preview:** VS Code Mermaid preview, GitHub rendering on the markdown fences, or `npx @mermaid-js/mermaid-cli` against a `.mmd` file.
