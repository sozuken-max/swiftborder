---
name: diagram-image-generation
description: >-
  use this when creating, updating, or reviewing SwiftBorder docs/images deck
  PNGs (architecture, dataflow, eval-layer-a/b) so assets keep the locked
  GCP-icon visual language and inventory-backed facts.
---

# SwiftBorder diagram image generation

## Goals

Regenerate, edit, or **review** deck PNGs under `docs/images/` that stay
**visually consistent** with the approved GCP-icon designs and **factually
consistent** with a fresh query of GCP project `swiftborder` (or the dated
copy in [docs/inventory.md](../../docs/inventory.md)).

**Topology source of truth:** the matching `.mmd` file in [docs/diagrams/](../../docs/diagrams/) (`architecture-high-level`, `architecture-detailed`, `eval-layer-a`, `eval-layer-b`). Edit Mermaid first, run `python docs/diagrams/sync_mermaid.py`, then export or redraw the PNG so node set and edge meaning match the `.mmd` file. Mermaid is not a deck asset; PNGs are.

## When to use

- Creating or updating any PNG under `docs/images/`
- Especially: `architecture-high-level.png`, `architecture-detailed.png`, `eval-layer-a.png`, `eval-layer-b.png`
- Before any GenerateImage / image-edit / Pillow pass on those files
- **Review-only:** run the [Review pass](#review-pass-no-regen) section in-thread before commit or deck use (no new bot required)

For prose docs and Mermaid in markdown, use [documentation-generation](../documentation-generation/SKILL.md). For code/PR architecture review, use [architecture-code-review](../architecture-code-review/SKILL.md). For PNG assets and PNG fact/layout QA, **this skill wins**.

## Ownership (who runs what)

| Work | Who |
| --- | --- |
| Regen / edit PNGs that live in this repo | **Implementer** (or CloudAgent draft PR) following this skill |
| One-off deck ask outside a coding thread | Personal Assistant may run a light pass; still follow this skill |
| Review before commit / presentation | Anyone in-thread: run **Review pass** below; Yingzhao is final on deck quality |
| Code DRY/SOLID PR review | Code Reviewer — stretch to PNG layout only if the ask is "does this asset belong in the PR / regen needed" |

Do **not** create a Diagram Craft or Technical Diagram Reviewer bot unless Yingzhao overrides after this skill fails twice on standing volume (multiple decks / repeated regen).

## Visual language lock (non-negotiable)

**Canvas:** 1280 x 720 (16:9) unless the clean reference is a different size — then match the reference.

**Approved look:** polished **GCP product-icon** diagrams — Cloud Run, BigQuery, Cloud Scheduler, GCS buckets, Roboflow cubes, camera icons, soft shadows, rounded cards, Layer A / Layer B bands, optional right-hand Gaps/risks panel.

**Bands / palette (architecture + dataflow)**

| Region | Fill feel | Role |
| --- | --- | --- |
| Layer A — Vision | cool blue wash | LTA cameras, Roboflow, swiftbackend, BQ vision tables |
| Layer B — Forecasting | warm tan / gold wash | Scheduler, Maps fetcher, travel_times, traffic_prediction |
| GCS strip | neutral grey wash | detailed: the seven verified buckets (see below); high-level: the camera, cache and public-history buckets plus a pointer to the detailed figure |
| Gaps / risks | light red / rose card, red border | numbered honesty list |
| Serve TODAY (if present) | light green card | live serve horizon only |

**Stroke / type**

- Solid arrows = live data / deploy / trigger
- Dashed boxes and dashed arrows = present but unused / not joined / planned
- Body text dark charcoal; LIVE / PRESENCE callouts in green; risk emphasis in red
- Risk panel: clear padding from the red border (no text kissing the frame)
- Captions never sit on top of product icons
- **Inset rule (Mode C and panel edits):** every node, callout, badge, and dashed box must sit fully inside its band/panel with >=8 px clear margin from that band's inner edge. Nothing may cross Layer A / Layer B / GCS / sidebar dividers.
- **Callout rule:** status chips (Cloud Build, LIVE, harness PRESENT) must not cover flow arrows or sit on a node card's border. Prefer a dedicated row above or below the node row.
- **Badge rule:** LIVE / count badges sit in clear space beside or below the node label — not bisecting the icon card edge.

**Icon set (must appear when the topology includes that node)**

Camera (LTA), Roboflow cube, Cloud Run hex/glyph, BigQuery circle, Cloud Scheduler clock, GCS bucket, cylinder for `travel_times`, document/report for harness output. Mode C may use simplified GCP-colored glyphs, but must not drop the icon language for flat PowerPoint tiles.

**Canonical style references**

1. The **last clean committed** PNG for that file (or a verified clean base). Never use a mangled regen as the reference.
2. Committed peers under `docs/images/` of the same family (architecture vs eval).
3. Architecture: Layer A (vision, blue) + Layer B (forecasting, tan/gold) + bottom GCS strip + Gaps/risks — match the last approved architecture PNG.

**Forbidden**

- Flat card-only / PowerPoint-tile redesigns that drop GCP icons
- Mermaid or matplotlib screenshots as deck assets
- Inventing Vertex AI, Cloud Composer, fake services, or buckets not in inventory
- Region badges on services (grading brief)
- Opaque Pillow paint-overs that leave broken or mismatched patches
- OCR-localized chip patches (find text, white-out, rewrite) — cause overlapping / double text
- Stacking edits on a previously patched PNG (always restart from a clean base)
- Pasting misaligned rectangles or captions over icons
- Rearranging topology because an image model preferred a new layout
- Splitting `swiftborder-frame-cache` into two bucket labels

## Topology lock

Preserve node set and edge meaning from the clean reference. Allowed: refresh stale captions, counts, harness PRESENT/MISSING, risk list wording. Forbidden without an explicit human ask: adding/removing major nodes, inventing joins, collapsing Layer A and B, or replacing the GCS strip.

## Edit strategy (order) — pick ONE mode

**Mode A — Full-figure regenerate (preferred when many facts are stale)**

1. Pass the **clean** existing PNG as the sole layout reference.
2. Demand identical layout, icons, bands, and topology — change **only** the stale facts listed in the prompt.
3. Vision-check; if any acceptance item fails, **discard** and retry Mode A or escalate to Mode C. Do not "fix" with chips.

**Mode B — Whole-panel replace (one discrete panel only)**

1. Measure the exact bbox of one self-contained panel (e.g. the entire Gaps/risks card).
2. Replace **only** that panel; neighboring pixels outside the bbox must be byte-identical to the clean base.
3. Vision-check seams and padding. Reject if text collides with the border or adjacent bands.

**Mode C — Full-canvas structured redraw (last resort)**

Only when Modes A and B fail (including when Mode A redesigns topology twice). Redraw the **entire** canvas from a layout spec that mirrors the reference topology and keeps GCP product-icon language. Still forbidden: flat tiles without icons, invented nodes. Prefer Mode C over shipping overlapping chips. Mode C must obey the **inset / callout / badge rules** under Visual language lock — reserve empty gutters between node rows and band edges; place Cloud Build / LIVE / unused-join callouts in dedicated clear strips, never on arrows or straddling bands. A Mode C that clips fails the Review pass even if facts are correct.

**Never use**

- Partial OCR / Pillow text chips over live art
- Risk-card paste with wrong aspect or misaligned bbox
- Shipping any PNG that still shows ghost text under new labels

## Practical note (Mode A)

As of 2026-09-26, GenerateImage with a reference PNG often **redesigns** topology (invented Composer / Vertex / buckets). Treat those outputs as automatic rejects. Prefer **Mode C** when Mode A fails twice.

## Canonical GCS bucket strip

When the diagram includes the bucket strip, use the seven verified names below. Re-query inventory before accepting a fixed count; do not invent or rename buckets:

1. `sg-lta-traffic-cameras`
2. `swiftborder-frame-cache`
3. `run-sources-swiftborder-asia-southeast1`
4. `run-sources-swiftborder-europe-west1`
5. `swiftborder-public`
6. `swiftborder_cloudbuild` (underscore, as in the project)
7. `swiftborder_asia-southeast1_cloudbuild`

Use the exact bucket names from a fresh query / inventory. Never rewrite `swiftborder_cloudbuild` with a hyphen. `swiftborder-frame-cache` has no writer in git: draw edges into it dashed.

## Fact authority and inventory gates

Before changing or **accepting** counts, joins, serve horizon, Cloud Build wording, or harness status:

1. Query GCP project `swiftborder` (`bq` / `gcloud` / Console), **or** read a fresh [docs/inventory.md](../../docs/inventory.md).
2. If the query and inventory disagree, update inventory, then the diagram.
3. Do not invent row counts or services.

**Inventory-vs-diagram gates (must match inventory dated stamp)**

| Gate | Diagram must show |
| --- | --- |
| Maps live rows | `causeway.travel_times` LIVE count from the current inventory — once, not duplicated (or "LIVE, see inventory") |
| Labels vs metadata | `traffic_images.labels = 0` distinct from `traffic_images.metadata` row count — never swap |
| Backfill | `traffic-backfill` as **Cloud Run Job** rebuilding metadata + `backfill_checkpoint` |
| Roboflow | Public export OK; weights = Core |
| Training joins | Maps-only; weather and cam congestion = dashed unused, not training edges |
| Serve | Horizon from inventory (often 30 min); 24h = intent until measured |
| MAE / 24h | Targets until harness proves otherwise |
| Harnesses | Layer B: `eval/layer_b.py`, `eval/joined.py` -> `eval/runs/report/`. Layer A: `eval/layer_a.py` present; results pending until an export is scored |
| Cloud Build | separate tested pipelines for `camdetect` -> `swiftbackend` and `forecastapi` -> `forecast-api` |
| Cams | cam2701/2702 frozen date from inventory when still frozen |
| Firebase | Hosting UI and forecast API connection as verified in inventory |

## Per-file intents

| File | Intent |
| --- | --- |
| `docs/images/architecture-high-level.png` | High-level: Layer A + Layer B + GCS strip + optional Gaps/risks annotation panel. Topology matches `architecture-high-level.mmd`. |
| `docs/images/architecture-detailed.png` | Denser detailed flow; same icon language; topology matches `architecture-detailed.mmd`. |
| `docs/images/eval-layer-a.png` | Layer A evaluation protocol; scorer PRESENT, results pending until scored. |
| `docs/images/eval-layer-b.png` | Layer B evaluation protocol; harness path and MAE **target** honesty. |

One system, two depths (high-level + detailed). Do **not** add a separate as-is / to-be pair. Unfinished scope belongs in [docs/findings.md](../../docs/findings.md) and [docs/roadmap.md](../../docs/roadmap.md).

## Review pass (no regen)

Run this in-thread before commit or deck use. Open each PNG with vision (**Read the image** — zoom mentally on band edges and callouts). Open [docs/inventory.md](../../docs/inventory.md). **Any single fail = reject** (do not ship "almost fine").

### Hard fail — clipping / containment (most common miss)

Reject if vision finds **any** of these:

- [ ] A node, dashed callout, badge, or caption **crosses** a Layer A / Layer B / GCS / sidebar boundary
- [ ] A callout **covers or cuts through** a flow arrow
- [ ] A badge or chip **bisects** a node card edge (half inside / half outside)
- [ ] Text or bullets **kiss or clip** a panel border (< ~8 px inset), including the Gaps/risks red frame and Serve TODAY green frame
- [ ] Labels under icons cut off by the bottom of their card or by the next band
- [ ] Crowding that forces overlapping boxes (even if text is still readable)

These are **craft rejects**, not nits. The 2026-09-26 Mode C redraws failed this bar when Cloud Build sat on an arrow, traffic-backfill straddled A/B, and congestion callouts clipped into the GCS strip.

### Layout / craft (also reject on fail)

- [ ] Matches clean reference topology (bands, panels, icon set) — no reinvented graph
- [ ] No overlapping text, double labels, or ghost glyphs
- [ ] Bucket strip (detailed): all seven names spelled as in inventory (`swiftborder_cloudbuild`); `swiftborder-frame-cache` not split
- [ ] Mode C redraws still read as GCP-icon language (not bare PowerPoint tiles), or are explicitly flagged as interim **and** still pass clipping rules above

### Facts (also reject on fail)

- [ ] Every inventory gate in the table above that applies to this file is correct
- [ ] Unused joins are dashed, not training edges
- [ ] Gaps/risks match inventory (Cloud Build, harness, Firebase, holidays, 2701-only, labels, stale tables)
- [ ] Eval diagrams: harness PRESENT/MISSING and paths honest; MAE is TARGET unless measured

### Process

- [ ] Agent vision-checked band edges and callouts (not OCR alone; not "looks mostly fine")
- [ ] If any item fails: restore clean base or regen under Mode A/B/C with layout fixed — do not chip-patch
- [ ] Yingzhao final on deck quality for presentations

If the review pass fails, say **FAIL** and list the clipping/fact items — do not claim pass with caveats.
## Vision QA / acceptance checklist (before commit)

Same as Review pass. Reject and regenerate if **any** item fails. Also:

- [ ] Skill body mirrored under `.cursor/skills/diagram-image-generation/` when this file changed

## Lessons locked in (2026-09-26)

1. GenerateImage that redesigns topology / invents buckets — reject; do not patch afterward.
2. OCR + Pillow chip overlays — overlapping LIVE counts, double harness paths, mangled Cloud Build captions.
3. Partial risk-card paste with wrong size — restyled panel that collides with neighbors.
4. Seat gap vs craft gap — prefer expanding this skill over a Diagram Craft / Technical Diagram Reviewer bot until standing volume proves otherwise.
5. Facts-correct Mode C can still **FAIL** review when callouts clip band edges or cover arrows — never mark pass with "minor crowding" caveats.

Correct recovery: restore last clean committed PNG, then Mode A (or B for one panel only), else Mode C.

## Do not

- Commit binary experiments under `skills/`
- Change application code as part of a diagram-skill-only change
- Hard-code volatile GCP numbers into [AGENTS.md](../../AGENTS.md)
- Create a new bot for diagram craft or review without Yingzhao override after two skill failures on standing volume

## Mirror

After editing this file, copy the body to
`.cursor/skills/diagram-image-generation/SKILL.md` and rewrite relative links:
`../../` here becomes `../../../` in the Cursor mirror. Bodies should match; files are not byte-identical.