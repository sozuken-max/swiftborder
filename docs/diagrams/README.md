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

## PNG exports are stale (regenerate before the deck)

The `.mmd` files were corrected on 2026-10-01 against a fresh query of project `swiftborder`. The PNGs predate that and **fail the diagram-skill fact review** on these points:

| PNG | Stale content |
| --- | --- |
| `architecture-detailed.png` | Bucket shown as `swiftborder-cloudbuild` (live name `swiftborder_cloudbuild`); `LIVE 11,830`; frame-cache drawn as a live data path (no writer in git); no `eval/` block; "Layer A scoring missing" |
| `architecture-high-level.png` | `LIVE 11,830`; frame-cache "cached frames for detect path"; "Layer A scoring missing"; no public-exposure note |
| `eval-layer-a.png` | "Layer A scoring script MISSING" (`eval/layer_a.py` now exists; results still pending) |
| `eval-layer-b.png` | `LIVE 11,830`; candidates without the ensemble; no fixed window, joined experiment or significance step |

Until they are regenerated, cite the Mermaid figures (rendered by GitHub and in `architecture.md` / `evaluation.md`) rather than these PNGs.

**Preview:** VS Code Mermaid preview, GitHub rendering on the markdown fences, or `npx @mermaid-js/mermaid-cli` against a `.mmd` file.
