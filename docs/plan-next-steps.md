# Next steps after the model updates (planned 2026-10-10)

This plan covers what to do between now and the deadlines, given what has landed since 4 Oct:
- the horizon study and forecast curve (PR #7);
- TimesFM 2.5 and FCM + MLP ([evaluation.md §8](evaluation.md#8-timesfm-25-and-fcm--mlp-shared-13-30-sep-window-2026-10-05));
- Roboflow workflow v6 and count-band congestion labels;
- `hosting/` in git.

**Fixed dates:**
- **19 Oct 23:59 SGT:** evaluation data cutoff ([roadmap.md](roadmap.md#evaluation-freeze-decided-2026-10-03)).
- **30 Oct:** weather snapshot decision.
- **31 Oct:** deliverables.

## Where things stand

| Area | Evaluated | Exploratory only | Missing |
| --- | --- | --- | --- |
| Layer B, 30 min | Daily-refit `xgb[maps]` beats persistence (−0.37) and the served mix (−0.27) on 13–30 Sep | `fcm_mlp` + `xgb[maps]` mean (−0.14 vs `xgb[maps]`, formed after scoring); TimesFM ≈ `xgb[maps]` | October confirmation (Run B) |
| Layer B, 1–24 h | — | `xgb[maps+prof]` beats the profile to 3 h under Holm (5.5 h uncorrected); nothing consistent after | Frozen-run claims for any horizon |
| Layer A | — | v6 is live; count-band labels (from 14 frames) | Any detector score; v6 vs v4; observed counts for the Maps window |
| Demo | API and curve are live and public | Page shows a 5 h curve with no caveats | Hosting deployed from git; browser rehearsal |

The largest gap is still Layer A: nothing in it is measured. The largest risk is claiming exploratory Layer B results as findings.

## Before 19 Oct 23:59 SGT

### 1. Team decisions (this week; they change the signed-off plan)

| Decision | Options | Recommendation |
| --- | --- | --- |
| C5/C8 wording and the pooled-day rule (PR #6) | Accept, or revert to the 3 Oct plan | Accept; already in code and docs |
| **C9:** `fcm_mlp` + `xgb[maps]` mean vs `xgb[maps]` | Add to Run B, or leave it exploratory | Add it. It is the only 30-minute arm that beat `xgb[maps]` |
| **Horizon claims** | Add none; add H1 (`xgb[maps+prof]` vs profile at 2 h); add H1 + H2 (the same at 4 h) | Add H1 and H2. They are the parts of the study that survive Holm, plus the 4 h edge |
| Curve model cutoff | 330 min (uncorrected), or 180 min (Holm) | 180 min, unless the page is clearly labelled exploratory |
| Page copy | No caveats (current), or a one-word `status` cue in the tooltip | Add the cue. The page is graded as the MVP and should not overclaim |
| Detector to score first | v6 (Roboflow default), `model=local` (in-container YOLO26s), v4 | v6 and `model=local` on the same export; they are what the demo shows |

### 2. Layer A scoring (the critical path; needs the Roboflow key holder)

1. Export a Roboflow dataset version with a `test` split that none of the candidates trained on. The local YOLO26s was trained on dataset v6, so exclude v6's training images.
2. Score v6 and `model=local` (and v4 if possible) with [`eval/layer_a.py`](../eval/layer_a.py), recording mAP, precision/recall, count error by direction, and day vs night.
   - The local model can be scored offline from `camdetect/models/yolo26s_v6_boxfix.onnx`, with no Roboflow credits.
   - Score it at its served overlap setting (0.6), and with suppression off.
3. Record the model ids, the dataset version and the overlap setting, and fill the Layer A table in [evaluation.md](evaluation.md#layer-a--vision).
4. Run the 6–11 Sep backfill ([handoff-camera-pilot.md](handoff-camera-pilot.md), about 864 frames). It gives the first observed counts that overlap the Maps label (§7c).
   - With `model=local` the backfill costs no credits, so prefer it if its score is at least as good as v6's.
   - The backfill script calls Roboflow today, so it needs a `model=local` path first.
5. Check the count bands on the scored frames: do the four labels match a hand label? If not, report them as a display heuristic only.

If step 2 cannot happen by 19 Oct, the report describes Layer A as an unvalidated prototype ([final-report-readiness.md](final-report-readiness.md) §1).

### 3. Code for the frozen run (before 19 Oct, so Run A/B only run code that already exists)

- **`horizon_study.py`:** add `--test-start`, `--test-end` and `--data-cutoff`, mirroring `generate_comparison_plots.py`, so H1/H2 can be scored on Run A (13 Sep–19 Oct) and Run B (1–19 Oct). Promote them as a `horizon` component of the run manifest, not as a separate summary.
- **`fcm_mlp` and the C9 mean in the harness:**
  - Either call `Causeway/layer_b_fcm_mlp.py` from `eval/ensemble.py` (or a new component), on the same folds and rows, inside the run-wide multiplicity check.
  - Or run the §8 script on the Run B window from a clean commit and attach its output to the run.
  - The harness route is cleaner.
- **TimesFM:** keep it as an exploratory comparison. `AI.FORECAST` is a billed BigQuery call, outside the local-only evaluation, and it did not beat `xgb[maps]`. No Run B arm is needed.
- **Tests:** the new components' folds never train on labels from the test day, which mirrors `test_folds_never_train_on_labels_from_the_test_day`.

### 4. Demo readiness

- Deploy the page from `hosting/` and record which commit is live; today the live site differs from git ([inventory.md](inventory.md)).
- Rehearse the browser flow and record it, including what the page shows when the API returns 503 (stale data).
- Apply the page-copy decision from step 1.

## From 20 Oct

1. **Run A and Run B.** Use the [roadmap](roadmap.md#frozen-window-run-plan-proposed-2026-10-03-the-team-confirms-before-19-oct-2359-sgt) commands from a clean tree, plus the components added in step 3. Validate and promote both.
2. **Set `SERVED_SELECTION`** by the ADR 0004 rule.
   - If C9 is confirmed, serving the mix needs a small change to `forecast-api`: fit `fcm_mlp` there, or a precomputed artifact. Estimate half a day; decide only after Run B.
   - If H1/H2 are confirmed, the curve keeps the model to that horizon and the cutoff is set from Run A/B, not from the study.
3. **Refresh the evidence:**
   - fill `evaluation.md` and `findings.md` from the promoted runs;
   - mark the horizon study and §8 as superseded where Run A/B score the same question;
   - regenerate the deck PNGs (handled separately).
4. **Weather snapshots:** decide by 30 Oct whether to keep them; they expire 31 Oct 01:41 SGT ([release-pr2.md](release-pr2.md)).
5. **Report and submission:**
   - write the report;
   - prepare the submission data package or the access procedure ([final-report-readiness.md](final-report-readiness.md) §9);
   - record the video and the individual reflections.

## What not to start

- New model families. TimesFM and FCM + MLP already cover the foundation-model and fuzzy-hybrid angles for the rubric.
- A 24-hour claim. At 24 h nothing beats the calendar profile.
- Re-picking the "best single" model on October data.
