# Final-report readiness review

## Report assembly: 24 Oct 2026

This section is the structure for the Word report (`PRS-Practice-Module-Group-3-Project-Report.docx`). It follows the 24 Oct academic review of that draft. The review is a scoped audit of coherence and cogency. It is not a mark, and it did not re-check every figure in the docx. Repo evidence stays in [evaluation.md](evaluation.md) and [findings.md](findings.md). Do not add a model run to satisfy the review.

The graded surfaces are still tools and design, performance, findings, a runnable system, and the submission package ([grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md)). The six-file repo spine below stays the working set. This section says how those files become one report.

### One target, four questions

The observed forecast label is Google Maps `duration_in_traffic`. A low error on that series is not a measured traveller crossing time. Count error is not a duration error. Deployment of `mix[fcm_mlp+xgb]` does not promote that mix from exploratory to confirmed.

| | Question | Strongest committed answer | Status |
| --- | --- | --- | --- |
| Q1 | Does the fine-tuned detector improve vehicle **counts** over the hosted baseline on the 15-frame test split? | At the **served** point (`conf=0.20`, overlap 0.6) car MAE is **1.87**, versus Roboflow v6 **4.40**. The **1.73** figure is the threshold swept on those same frames (`conf=0.30`, overlap off). Directional count error is not scored. | Descriptive. Say so. |
| Q2 | Do short-horizon models beat persistence on future Maps duration? | 13–30 Sep, `y_30`, both directions: persistence **2.640 min**, daily-refit `xgb[maps]` **2.275 min** (about 22 seconds; Holm p 0.0117). The gain is inside the 0.5-minute bar. | Confirmatory on that window. Not a crossing-time result. |
| Q3 | Do camera counts add anything beyond a non-camera baseline? | Restricted 13–30 Sep tree: about **5 seconds**, and that tree still trails persistence (§7c). Counts **instead of** Maps inputs on 1–4 Oct: regression of about **3–4 min** (§7d). Current duration, counts plus calendar vs a Fourier calendar: XGBoost **6.570** vs **6.702**, interval crosses ±0.5 min (§7b). | Bound each sentence to its protocol. |
| Q4 | Do extra complexity and hybrids earn their place? | Original combiners do not beat `xgb[maps]` (§6). The equal mix point estimate is about **2.13 min** and was chosen after seeing September. It is the HTTP default. It is not a pre-registered win. | Exploratory for the mix. Confirmatory that the older combiners add nothing. |

Longer horizons, deep sequence models, fuzzy level, TimesFM, and the CNN-RNN notebooks are exploratory. They do not redefine Q1–Q4.

Three outcomes stay distinct: improvement detected, no improvement detected, and insufficient temporal evidence. "Not significant" is not equivalence.

### Report structure

Write the body in this order. Inside each results subsection use the same sequence: question, comparison, result, interpretation, limitation.

| # | Chapter | Take from | Leave out of the body |
| --- | --- | --- | --- |
| 1 | Introduction: problem, observed target, Q1–Q4, exclusions | [findings.md](findings.md) claims register; [evaluation.md](evaluation.md) "What would count as success" | 24-hour and ≤15 min crossing-time claims |
| 2 | Related work: why these comparisons | Draft §3, cut to the decision each subsection motivates | Model-family catalogues that do not choose a method |
| 3 | Data and measurements: sources, labels, splits, what a count is not | [inventory.md](inventory.md) latest block; Layer A vs Layer B split in evaluation | One date that mixes architecture, collection, and the freeze |
| 4 | Methods: detector operating points, baselines, feature sets, the mix | [architecture.md](architecture.md); [ADR 0004](adr/0004-serve-local-models.md) | The full callable-model list and HTTP catalogue |
| 5 | Protocol: time order, matched rows, threshold boundaries, uncertainty | Evaluation "Reasoning" and "Significance" | A single leaderboard across horizons |
| 6 | Results: Q1–Q4, then a labelled exploratory section | Frozen Evidence Index | Combining §7c, §7d, and §8 into one camera sentence |
| 7 | Discussion: answer each question, effect size, failure cases | [findings.md](findings.md) | Infrastructure as a finding |
| 8 | Conclusion: only the claims the experiments support | Claims register, "Say this" column | "We beat Google"; directional accuracy; untouched test-set MAE of 1.73 |
| 9 | Appendices: deploy, model catalogue, evidence register, who did what | [services/README.md](../services/README.md), [runbooks/forecast-api.md](runbooks/forecast-api.md), ADRs | — |

Dates to keep separate: architecture description, Maps and camera collection windows, each evaluation window, submission freeze at `199c2e5`, and later committed runs (counts substitute, camera-current, mix backtest). The forecast horizon in every headline is the bin that starts 30–40 minutes after the origin, about 9–19 minutes after a request when ingest is current. It is not "30 minutes from now".

Detector wording for Q1, from the review: the test frames were excluded from weight training and checkpoint selection. The comparison thresholds were selected using count error on those frames. Those rows are descriptive, threshold-selected comparisons. The served and backfill point is confidence 0.2 and overlap 0.6 (car MAE 1.87). Do not attach 1.73 to `local_counts`.

### Evidence register

Every number in the body needs one row. Fields: research question, table or figure id, task and target, dataset and dates, split and n, model and operating point, baseline and whether the rows match, artefact path and git revision, evidence status (`descriptive`, `exploratory`, `confirmatory`, `insufficient data`). The Frozen Evidence Index is that register for the committed runs. Cite the run or notebook, not this readiness file, as the source of a result.

### Canonical figures

Use these in the report body. A figure is canonical only for the question in its row. Deck PNGs are slide copies of the Mermaid sources; the `.mmd` files and the embeds in [architecture.md](architecture.md) and [evaluation.md](evaluation.md) are the system drawings. Regenerate a PNG when it disagrees with its `.mmd` ([diagrams/README.md](diagrams/README.md)).

| Use | Figure | Question | Do not use it to say |
| --- | --- | --- | --- |
| Body | [architecture-high-level.mmd](diagrams/architecture-high-level.mmd), [architecture-detailed.mmd](diagrams/architecture-detailed.mmd) | What is deployed | That weather or counts feed the served model |
| Body | [eval/runs/report/bqml/mae-diff-ci.png](../eval/runs/report/bqml/mae-diff-ci.png), [mae-by-direction.png](../eval/runs/report/bqml/mae-by-direction.png) | Q2, 13–30 Sep `y_30` | A crossing-time MAE, or that the served mix won this test |
| Body | [eval/runs/report/ensemble/ensemble-30min-mae-diff.png](../eval/runs/report/ensemble/ensemble-30min-mae-diff.png) | Q4, original combiners | That the later 50/50 mix was in this figure |
| Body | [counts-substitute paired difference](../eval/runs/20261017T083554Z_counts-substitute/paired_diff.png), [series](../eval/runs/20261017T083554Z_counts-substitute/series_xgb_mix.png) | Q3, counts in place of Maps inputs, 1–4 Oct, target `y_30` | That counts are useless as a sensor; §7c is a different test |
| Body | [camera-current paired difference](../eval/runs/20261017T084900Z_camera-current/paired_current.png), [series](../eval/runs/20261017T084900Z_camera-current/series_current.png) | Q3, current duration vs a calendar profile | A 30-minute forecast result |
| Exploratory | [horizon mae-by-horizon](images/horizon-study/mae-by-horizon.png), [skill-vs-profile](images/horizon-study/skill-vs-profile.png) | Horizon study | A confirmed 24-hour forecast |
| Exploratory | [deep-mae-diff.png](../eval/runs/report/deep/deep-mae-diff.png), [fuzzy-confusion.png](../eval/runs/report/fuzzy/fuzzy-confusion.png) | Extra model families | That deep or fuzzy models are served |
| Exploratory | [camfc-mae-diff.png](../eval/runs/report/joined/camfc-mae-diff.png), [camfc-profiles.png](../eval/runs/report/joined/camfc-profiles.png) | March–April count forecast as a calendar prior | Observed September counts, or `local_counts` |
| Appendix | [joined-mae-by-feature-set.png](../eval/runs/report/joined/joined-mae-by-feature-set.png) | Weather and extra Maps features on 13–30 Sep | That rain never matters |
| Appendix | [parity-mae-diff.png](../eval/runs/parity-bqml/parity/parity-mae-diff.png) | Local replica vs BigQuery ML | A serving-quality claim |
| Not canonical | [images/architecture-high-level.png](images/architecture-high-level.png) and the other three deck PNGs | Slides, after they match the `.mmd` | A result table |
| Not canonical | CNN-RNN notebook screenshots | A separate image-to-duration study | Chad's 15-frame detector comparison |
| Still to draw | One timeline: origin start, bin close, request, target interval `[t+30, t+40)` | Horizon definition in chapter 5 | A new measured error |

There is no canonical figure that shows directional count accuracy, an independent crossing time, or a single ranking of every model.

### Claim boundaries to apply while editing the docx

1. Disclose test-set threshold selection. Keep 1.73 and 1.87 in different rows.
2. Name the forecast target as the Maps route-duration estimate in the abstract, the introduction, the target definition, and the conclusion.
3. Directional assignment is implemented. Total-frame count error does not validate it. Call the output a visible vehicle count, not traffic density or flow.
4. Camera conclusions stay inside §7b, §7c, or §7d. Do not merge them with the §8 mix.
5. The deployed mix stays exploratory.
6. Report the 22-second `xgb[maps]` gain and its uncertainty. Do not call it a large traveller benefit. The mix's about 30-second point-estimate gap versus persistence is exploratory.
7. "Costs nothing per call" means no separate hosted-detector fee, not zero infrastructure cost.
8. In-service fitting was chosen to match the harness with less extra machinery. It is not the only design that can reproduce a model.
9. Literature percentages (small-object share, height versus area) get one population and one denominator before they stay in the report. This repo does not hold Table 3.2's source counts; do not invent a corrected percentage here.

Individual contribution (draft §2.8) still needs person-level artefacts if the rubric asks for them. That check is on Canvas, not in this file.

## Re-review: 17 Oct 2026, HEAD c98b4d5 (22:09 SGT)

**Verdict: acceptable to assemble the final report. Two new stale-prose items found; all previously listed open items remain open and unchanged.** This pass reads the full working tree at `c98b4d5` (the merge of `experiment/counts-for-maps` into `main`). It does not run experiments, query GCP resources beyond what is already in [inventory.md](inventory.md), or make any deployment. The team's no-new-runs constraint applies.

### New findings (not present in the earlier c98b4d5 pass)

| ID | File | Stale text | Correct text |
| --- | --- | --- | --- |
| **N1 — Medium** | [`README.md`](../README.md), Layer B table row | `the local lin_h30 replica for SG→MY, persistence for MY→SG` | `mix[fcm_mlp+xgb]` for both directions — what `SERVED_SELECTION` in [`forecastapi/main.py`](../services/forecastapi/main.py) says and what live revision `forecast-api-00021-qud` confirms |
| **N2 — Medium** | [`docs/diagrams/architecture-detailed.mmd`](diagrams/architecture-detailed.mmd) line 17, and the matching embed in [`docs/architecture.md`](architecture.md) | `SG_TO_MY: lin_bq frozen` / `MY_TO_SG: persistence` on the API node | Both directions now `mix[fcm_mlp+xgb]`. The high-level Mermaid in `architecture.md` is already correct; only the detailed `.mmd` source and its embed in `architecture.md` are stale |
| **N3 — Low** | [`docs/roadmap.md`](roadmap.md), "Already true" serving bullet | `lin_h30 replica for SG to MY and persistence for MY to SG` | Same as N1; the roadmap has not been updated to reflect the October mix promotion |

**Resolution without a new experiment run:** update the three prose locations listed above and re-sync the detailed `.mmd` into `architecture.md` (run `python docs/diagrams/sync_mermaid.py`). The deck PNGs must also be regenerated once the `.mmd` is corrected — this is already an open item carried forward.

### Previously open items — status unchanged

- **Deck PNGs** — still not regenerated. `architecture-high-level.png` and `architecture-detailed.png` (both regenerated 4 Oct) are stale on: the `forecast-api` node labels (pre-mix selection), the Hosting node still labelled as outside git in the PNG, and the `gmap-woodlands-fetcher` source path. `eval-layer-a.png` and `eval-layer-b.png` should be checked against the current Mermaid. **Action before submission:** regenerate all four PNGs after correcting N2 above.
- **Submission package** — recorded demo, slides, reflections, video, and Canvas submission archive. Out of repo scope; team responsibility.
- **§8 vs §7c camera overlap sentence** — `fcm_mlp_layer_a` / `timesfm_layer_a` detections end 22 Apr; `local_counts` is the 6 Sep–4 Oct window. These are different windows and must not be collapsed into one sentence in the report.
- **Optional hygiene** — root `README.md` quick-start still says `python layer_b.py`; the dispatcher is now `python run.py layer_b`. Correct alongside N1 in the same commit.

### Code and evidence verified correct at this checkout

The following were re-read and found consistent with [evaluation.md](evaluation.md), [findings.md](findings.md), and [inventory.md](inventory.md). No changes needed.

- `services/forecastapi/main.py` `SERVED_SELECTION` — confirmed `mix[fcm_mlp+xgb]` for both directions; `status_for` correctly labels the mix as `exploratory` and other 30-minute models as `evaluated`. Fallback chain `mix → xgb[maps] → persistence → profile` is implemented in `query_default_served`.
- `services/forecastapi/mix_bridge.py` — imports `layer_b_fcm_xgb_mix` from `shipped_eval/` or the repo root `eval/`; equal 0.5 / 0.5 weights match the backtest; `predict_value` returns `None` (→ fallback) when the mix is non-finite.
- `scripts/vendor_forecast_mix_eval.sh` — exists and is called in the Cloud Build `Test` and `Vendor-mix` steps; `services/forecastapi/tests/test_shipped_eval_import.py` asserts isolated import. Finding P1 is closed.
- Freeze boundary — `eval/runs/report/run.json` (`20261003T035807Z`, git `b3b6562`) is the primary evidence for §§1–7a. Later runs (`backtest-20261017T094617Z-d001a839`, `20261017T083554Z_counts-substitute`, `20261017T084900Z_camera-current`) are correctly separated in the Frozen Evidence Index. Finding P2 / P3 are closed.
- `architecture.md` high-level Mermaid embed — shows `hosting/ in git; manual deploy`, `mix[fcm_mlp+xgb]` in the techniques table, and `local models / target: bin 30–40 min after the newest closed bin` on the `forecast-api` node. All accurate.
- `findings.md` items 1–13 and claims register — consistent with committed run artifacts and inventory facts. No new false claim found.

### Finishing checklist (no new experiments required)

| Priority | Action | Status |
| --- | --- | --- |
| **High** | Fix N1: update `README.md` Layer B table row to say `mix[fcm_mlp+xgb]` for both directions | Done |
| **High** | Fix N2: update `docs/diagrams/architecture-detailed.mmd` API node labels; re-sync embed into `architecture.md` | Done |
| **High** | Fix N3: update `docs/roadmap.md` "Already true" serving bullet | Done |
| **High** | Regenerate all four deck PNGs after N2 correction | Blocked (upstream image-generator service 500 internal error; Mermaid source & prompts verified and saved) |
| **High** | Submission package: recorded demo, slides, video, reflections, Canvas archive | Open (team task) |
| **Low** | Fix `README.md` quick-start: `python layer_b.py` → `python run.py layer_b` | Done (already verified `python run.py layer_b` in README.md) |



---

## Latest re-review: 17 Oct 2026, c98b4d5

**Verdict: acceptable to assemble the final report from committed evidence, with packaging and figure work still open.** This review covers checkout `c98b4d5` (eval package refactor, October camera studies, refreshed §8 backtest). It does not run experiments, query cloud resources, or assess live availability. The team's no-new-runs constraint applies.

### Findings addressed in tree (after c98b4d5)

| ID | Finding | Resolution |
| --- | --- | --- |
| P1 | Forecast image vendored only mix shims (`import warehouse` failed in `shipped_eval/`) | [`scripts/vendor_forecast_mix_eval.sh`](../scripts/vendor_forecast_mix_eval.sh) copies shims plus `warehouse/` and required `harness/` modules; Cloud Build Test and Vendor-mix steps call it; [`services/forecastapi/tests/test_shipped_eval_import.py`](../services/forecastapi/tests/test_shipped_eval_import.py) asserts isolated import |
| P2 | Counts-substitute labels ignored the ten-day harness gate | [`eval/studies/counts_substitute.py`](../eval/studies/counts_substitute.py) forces `inconclusive` when `calendar_days < 10`; [evaluation.md §7d](evaluation.md#7d-counts-substituted-for-maps-inputs-1-4-oct-2026) qualifies committed `regression` manifest labels as **interval regression** on four days |
| P2 | Evidence freeze text did not separate `199c2e5` from later artifacts | [evaluation.md](evaluation.md) opening paragraphs and **Frozen Evidence Index** rows for camera-current and counts-substitute runs; dirty `run.json` flags documented as provenance |
| P2 | §8 prose contradicted mix as HTTP default | [evaluation.md §8](evaluation.md) **Served selection** aligned with ADR 0004 and `forecastapi/main.py` |

No rerun was required for these doc and packaging fixes.

### Still open (unchanged by this pass)

- **Deck PNGs** — factual freshness vs current architecture and Layer A/B claims ([diagrams/README.md](diagrams/README.md)).
- **Submission package** — recorded demo, slides, reflections (out of repo scope).
- **§8 vs §7c camera overlap** — `fcm_mlp_layer_a` / `timesfm_layer_a` detections end 22 Apr; `local_counts` is a different window; do not merge the sentences.
- **Optional hygiene** — diagram skill inventory paths under `eval/lb/` (agent docs only).

### Findings closed since the 3 Oct review (carried forward)

- CNN-RNN protocol, speed measurements, pooled parity, hybrid MAE citations, Layer A AP50 vs mAP50-95 split, fuzzy insufficient-data wording, detector CSVs in git, harness code fingerprint for nested `eval/`.

### Verification note

Re-run `scripts/run_tests.sh` and `bash scripts/vendor_forecast_mix_eval.sh` before merge; the isolated shipped-eval test was added with the vendor script. The historical **596-test** count is not automatically refreshed here.

The project has a strong basis for the report (supervised learning, ML/DL, hybrid/ensemble, intelligent sensing). Remaining work: regenerate figures, package the demonstration, finish submission materials. No new forecast experiments are required for the findings above.

## Still open after the 17 Oct honesty pass

The evidence-index cells (CNN-RNN protocol, speed timings, pooled parity, mix MAE 2.134 / 2.132), the Layer A notebook-versus-Ultralytics box tables, the fuzzy insufficient-data wording, and the `data/sql/` fingerprint path are updated in this tree. Detector CSVs under `eval/test-reports/runs/` are on `main`. The 30-minute HTTP default is `mix[fcm_mlp+xgb]`.

Still open, and not fixed here:

- The four deck PNGs were not regenerated. Their older captions can still disagree with the text.
- The submission package (recorded demo, slides, reflections) is out of scope for this change.
- Section 8's camera-overlap sentence is about `fcm_mlp_layer_a` and `timesfm_layer_a` (detections end 22 Apr). Section 7c's `local_counts` overlap is a different window. Do not collapse those two statements.

No experiment was rerun.

## Current review: 17 Oct 2026, frozen submission

**Status: superseded by [Latest re-review](#latest-re-review-17-oct-2026-c98b4d5) above** for open items and verdict. The narrative below is retained as review history.

**Verdict (historical): request changes to report claims and packaging; no further experiments required.** The project has enough substantive evidence to assemble a defensible final report. It is not yet ready to submit unchanged, because its documents mix historical, exploratory and confirmed results. This is a substantially stronger position than the 3 Oct review.

**Scope fixed by the team on 17 Oct:** use **all currently committed results**, with **no more experiment runs**. Checkout reviewed: `main` at merge `199c2e5` (same tree as content commit `910d715`, tree `51a722ab`). Live operation is intentionally curtailed to reduce costs. An unavailable live forecast is therefore **not a readiness blocker**. The earlier proposed 19 Oct cutoff and future Runs A/B are superseded as requirements; unperformed confirmation remains a limitation, not work that must be commissioned.

This review used repository code and result artifacts, the grading guide, all four deck PNGs, and fresh read-only GCP/HTTP observations recorded in [inventory.md](inventory.md). It follows the architecture review, documentation and diagram-review skills. No deployment, scheduler, access-policy or model change was made. No forecast experiments were refit for the review.

### What is now resolved or materially stronger

- Completed-bin selection, ingestion grace, stale-target rejection, gap handling and cache expiry are implemented. Serving builds time-based features from `v_bins_10min`; the earlier timing/parity findings should not be repeated as current defects.
- Pooled inference now counts shared calendar days and requires a joint day-bootstrap interval. Short fuzzy comparisons become **insufficient data**, rather than gaining independent evidence from two directions.
- Layer A is **partially measured**, not unmeasured: count errors, day/night slices, test-set AP50 and car precision/recall are committed. At the served local settings, car MAE is **1.87** on 15 frames; Roboflow v6 at its served settings scores **4.40**. The notebook reports local YOLO-S mAP50 **0.827** and car AP50 **0.899**. These are different scoring procedures and operating points, not one interchangeable result.
- Observed camera-count correlation and ablation are now available. The newer FCM/MLP/XGBoost work supplies a concrete hybrid experiment. CNN-RNN notebooks provide a separate image-to-duration research strand, explicitly outside the served stack.
- `gmapfetcher/`, `hosting/`, `firebase.json` and `.firebaserc` are all tracked. The previous missing-source/configuration findings are closed; remaining deployment instructions should stop claiming these files are absent.
- Local checks completed before the freeze clarification: **596 tests passed**, including slow tests (114 camera, 189 Causeway, 223 evaluation, 56 forecast API, 14 repository). Nine Keras/NumPy deprecation warnings did not fail the suite. Passing tests establish implementation checks, not independent scientific validity.

### Findings to resolve using the existing evidence

**1. High: qualify the detector test-set claim.** [The detector report](../eval/test-reports/README.md#box-metrics-test-split) says scoreboard thresholds minimize car MAE over a 0.10-0.50 sweep. The comparison notebook selects `car_sw["count_MAE/img"].idxmin()` for each split. Thus test images were excluded from weight training and checkpoint selection, but were used to choose reported confidence thresholds. The blanket sentence in [evaluation.md](evaluation.md#layer-a--vision) that no detector was selected on them is too broad. Describe the 1.73 car MAE as a **test-threshold-selected descriptive result**. Report the 1.87 served-setting rescore separately, without claiming that its operating point was independently preregistered. AP50 remains useful notebook evidence; do not equate it with mAP50-95 or the unrun `eval/la/layer_a.py` procedure. Fifteen 2025 frames, seven day/eight night, do not validate 2026 directional-count accuracy. With the freeze, disclose these limits rather than require new labels or a new run.

**2. High: reconcile significance and ensemble conclusions.** [Evaluation's significance section](evaluation.md#significance) correctly downgrades short fuzzy results, but subsequently lists "The XGB -> fuzzy hybrid beats the persistence level" among robust claims. That should be a lower point estimate with insufficient temporal evidence. Similarly, "weather adds nothing" should be "no improvement detected on this window." The older finding that ensembling adds nothing is scoped to the original combiners; it is no longer an adequate summary of all committed work. The newer FCM/XGB mix scored **2.132 min** versus XGBoost **2.275** and persistence **2.640** in its standalone September backtest. Retain those numbers, but explicitly retain the exploratory label: the mix was formed after inspecting September results, uses a separate comparison family, and has no completed October confirmation. Do not claim C9 confirmed, or that the different camera and hybrid gains have been demonstrated in a single combined model.

**3. Medium: narrow the camera-input conclusion.** The committed [ablation](../eval/runs/layer_a_eval/ablation.json) scores maps-only XGBoost **2.8445**, maps+camera **2.7618**, and persistence **2.6718** minutes. Camera features improve that restricted XGBoost by **0.0827 min**, but it still loses to persistence. [The implementation](../eval/studies/layer_a_eval.py) uses current duration, congestion ratio, speed and direction; it does not test the full lag/calendar/profile baseline. Consequently "signal beyond the Maps lags" is unsupported by this experiment's feature list. Report the fold t-test as exploratory and the Pearson/Spearman coefficients as descriptive association; ordinary row-level correlation p-values do not account for serial dependence. Do not infer causal information, directional ground-truth accuracy, or replacement of Maps from these correlations. No further run is needed if the conclusion is narrowed accordingly.

**4. Medium: identify every frozen evidence source.** The canonical `eval/runs/report/run.json` is the 3 Oct run from `b3b6562`, while detector metrics, observed-count ablation, horizon study and FCM experiments live elsewhere. This is legitimate historical provenance, not evidence that the old run is invalid. However, the opening statement in evaluation.md that sections 1-7 all come from that manifest now excludes section 7c's actual source. Add a final evidence index listing each included result, its artifact/notebook, evaluation dates, code/model version, operating point, comparator and confirmation status. Preserve the original manifests and dates. Record the freeze as an artifact selection at merge `199c2e5` (same tree as `910d715`), not an invented common data cutoff. Replace future-tense statements that final numbers will come from Runs A/B; do not imply those runs exist.

**5. Medium: refresh report prose and figures to the frozen system.** All four PNGs were visually inspected: **FAIL for current report use on factual freshness**, not a newly observed clipping defect. Architecture images still say five-minute collection, `v_training_set` serving input and Hosting source outside git; they omit the newer local detector/offline evidence. Layer A still says accuracy unmeasured. Layer B still says observed counts pending and pooled-day review open. Update topology and regenerate once the report claims are reconciled. Present the implemented architecture with a dated note that live collection was curtailed for cost; do not draw intentionally paused operation as a missing implementation. Longer-horizon studies are exploratory, and a calendar-profile extension is not a validated 24-hour crossing-time forecast.

**6. Medium: package a demonstrable frozen MVP.** Persistent live availability is not required by this review. Supply the recorded demonstration or a documented local/offline replay, the permitted datasets or tested access procedure, model files/checksums, dependency lock and exact commands associated with the cited artifacts. A working test suite and diagram do not by themselves prove that a grader can reproduce the demonstration. Hosting source is present. Rechecked 17 Oct 2026: public `https://swiftborder-92b45.web.app/app.js` and `hosting/app.js` are the same text (44,260 characters). The earlier 44,261-versus-45,165 comparison does not match either copy. If a later deploy changes the live file, name the version used in any recorded demo. CNN-RNN Colab checkpoints and human-rating outputs need explicit inclusion/provenance if their results are claimed; otherwise describe this strand as research methodology, with only committed outputs cited. Individual reflections, contribution mapping, final slides/video and the submission archive were not verified as complete.

### Grading and final-report disposition

| Surface | Current assessment | Completion without new experiments |
| --- | --- | --- |
| Tools, design and models | Strong basis | One consistent account of implemented, historical and offline components |
| Layer B performance | Substantial frozen evidence | Correct statistical wording; retain exploratory status and exact windows |
| Layer A performance | Useful but small evaluation | Disclose threshold selection, dataset size, operating points and missing directional/mAP50-95 evidence |
| Findings/discussion | Strong material, inconsistent summaries | Explain small gains, negative results, proxy labels and limitations |
| Runnable MVP | Implementation and tests present | Include a reproducible offline or recorded demonstration; no live restart required |
| Figures and submission package | Not ready unchanged | Refresh four PNGs, evidence index and deliverable checklist |

The work demonstrates **supervised learning**, **ML/deep learning**, **hybrid/ensemble methods**, and **intelligent sensing**: more than the three required categories. The defensible report is an evaluation of camera-derived vehicle counts and forecasting **future Google Maps duration estimates**. It does not establish independent physical crossing-time accuracy, superiority to Google as ground truth, or the original 24-hour/15-minute product target. Frozen limitations are acceptable findings when stated accurately.

**Recommended finishing order:** reconcile the claims; index the committed evidence; align architecture and figures; package and rehearse the frozen demonstration; finish report, presentation and individual submissions. No new model, data collection or cloud spending is necessary to address this review.

## Historical review: 3-4 Oct 2026

The material below is retained as review history. Its open/closed statuses and future-run requirements are superseded by the 17 Oct review above.

**Follow-up, 4 Oct 2026:** all four deck PNGs and their source topology were regenerated; see [diagram QA notes](diagrams/README.md#png-exports-refreshed-4-oct-2026). The figure finding below records the original review state. Application and statistical findings remain open.

Reviewed 2026-10-03 (SGT), checkout `be9281d`, against repository code, committed evaluation evidence, the module grading guide, four deck PNGs, and fresh read-only queries of GCP project `swiftborder`. Live observations and query scope are in [inventory.md](inventory.md#latest-verification-2026-10-03-approximately-2335-2340-sgt).

**Verdict: request changes before final submission.** The project is ready to start assembling the final report. It is not ready to freeze the report or claim the original crossing-time product goal is achieved. The strongest contribution is the evaluated forecasting methodology and its honest negative results. The biggest gaps are unmeasured vision quality, live/evaluation timing differences, statistical interpretation, and inconsistent documentation.

## Response (2026-10-04, branch `fix/final-report-readiness`)

| # | Finding | Status | Where |
| --- | --- | --- | --- |
| 1 | Layer A box metrics | **Partial (17 Oct).** Image-level count error is scored on 15 frames ([evaluation.md](evaluation.md#layer-a--vision)). mAP, precision, recall and directional count error still need a box export in `eval/la/layer_a.py`. `cam2701` / `cam2702` remain Mar–Apr model detections, not ground truth | [findings.md](findings.md) item 8 |
| 2 | Live API uses incomplete origin bins | **Fixed in code.** The origin is the newest bin closed for at least 60 s. The response carries `origin_closed_at`, `forecast_window_end` and `lead_min`. The target is described as the bin 30–40 min after the origin, 9–19 min ahead of the request | `forecastapi/main.py`, [runbook](runbooks/forecast-api.md), [ADR 0004](adr/0004-serve-local-models.md) amendment |
| 3 | Stale observations served as current | **Fixed in code.** A direction answers 503 once its target bin has started or within an hour of a >25 min gap, and a cached answer expires with its target. Tests cover each case, per direction | `forecastapi/tests/test_forecast.py` |
| 4 | Pooled inference ignores shared days | **Fixed in code; sensitivity re-run.** The gate counts shared calendar days, and the decision needs the joint calendar-day bootstrap CI; a day-clustered p is reported. 30-min pooled decisions are unchanged; the three fuzzy decisions become "insufficient data" | `eval/harness/significance.py`, [findings.md](findings.md#pooled-significance-sensitivity-2026-10-04), [evaluation.md](evaluation.md#significance) |
| 5 | Non-significance treated as confirmation | **Reworded.** C5/C8 have no "confirmed" outcome; "not significant" is reported as no improvement detected. No equivalence margin was adopted, which needs team confirmation before 19 Oct | [roadmap.md](roadmap.md) |
| 6 | Deployment and report architecture disagree | **Docs updated.** README, findings, architecture prose and Mermaid, runbook and ADR now describe `forecast-api` local serving; BQML results are labelled BQML-era | as listed; **PNGs need regeneration** (see 10) |
| 7 | Access policy contradicts the smoke test | **Reconciled to the live decision (public).** Deploy sets `--no-invoker-iam-check` and `--max-instances=3`. The smoke test calls anonymously with the Hosting origin and checks the annotation and CORS. Risk recorded | `forecastapi/cloudbuild.yaml`, [findings.md](findings.md#known-risk-public-forecast-api-by-decision) |
| 8 | Parity depends on gap-free data | **Fixed in code.** Serving and daily training build features from `v_bins_10min` with the harness's time-based rules. A test asserts they equal `eval/lb/features.maps_features` with skipped bins and a gap. `v_training_set` itself is unchanged | `forecastapi/local_models.py`, `eval/tests/test_forecastapi_models.py` |
| 9 | Reproducibility / submission package | **Open (team).** Static Hosting files are in `hosting/`. Still needs an immutable data package or access procedure, `firebase.json` / `.firebaserc` (not invented), the Maps fetcher source, and the frozen run from a clean revision | [roadmap.md](roadmap.md), [ADR 0002](adr/0002-firebase-hosting-source.md) |
| 10 | Deck PNGs | **Regenerated 4 Oct, and now stale on three labels** (forecast-api input, public access, and the Hosting node still drawn as outside git). Handled separately, as agreed | [architecture.md](architecture.md) |

## Readiness by graded surface

| Surface | Assessment | Remaining evidence |
| --- | --- | --- |
| Tools, techniques and model design | Substantial | Consolidate the model/data descriptions and current serving architecture |
| Layer B performance | Strong preliminary evidence | Frozen Runs A/B; resolve inference concerns below; publish exact windows per experiment |
| Layer A performance | Count error on 15 frames; box metrics open | mAP, precision, recall, directional count error, and whether those 15 images selected the checkpoint |
| Runnable MVP | Backend demonstrated; `hosting/` is in git | Full browser rehearsal, freshness handling, and a Hosting release taken from `hosting/` |
| Findings and discussion | Good foundation | Correct significance wording; distinguish historical BQML results from current local serving |
| Final figures | Fail | Refresh topology and regenerate all four deck PNGs |
| Reproducibility/submission package | Partial | Immutable input data package or access procedure, pinned code/environment, client source and run instructions |
| Individual reflection/peer review | Not established by this review | Contribution matrix and each member's reflection; confirm submission requirements on Canvas |

The project can name at least three required categories: **supervised learning** (future-duration regression and labelled YOLO), **ML/deep learning** (trees, YOLO, sequence models), and **hybrid/ensemble** (evaluated combinations and XGB-to-fuzzy classification). **Intelligent sensing** is also implemented, but its quality remains unmeasured. These are rubric categories, not claims of four independent scientific contributions.

## Priority findings

### 1. High: Layer A box metrics are still open

Image-level count error on 15 labelled frames is now in [evaluation.md](evaluation.md#layer-a--vision). The paragraph below is the 3 Oct review. Its "entirely pending" sentence described that date. mAP, precision, recall and directional count error are still pending, and the 15-frame set is not shown to be the checkpoint-selection split.

The results table in [evaluation.md](evaluation.md#layer-a--vision) was entirely pending on 3 Oct. The scorer exists and its fixtures pass. The live image demonstration cannot substitute for box measurements.

Before submission, score the served YOLO model on an untouched export and report mAP50, mAP50-95, precision/recall, count MAE by direction, day/night sample counts, and representative failures. Record dataset version, model/workflow version, class mapping, confidence threshold, and how near-duplicate frames were separated across splits. Compare a pretrained detector if feasible. If this cannot be completed, describe Layer A explicitly as an unvalidated prototype and narrow the report's claims. Optional ResNet work should not displace this.

### 2. High: the live API uses incomplete origin bins

[`forecastapi/main.py`](../services/forecastapi/main.py), `_LATEST_SQL`, picks the newest row without requiring `bin_ts + 10 minutes <= now`. The underlying view aggregates all readings currently present in each bin. The evaluation treats a bin's features as available only after the bin closes.

This occurred live: a response generated at **23:32:49 SGT** used the **23:30-23:40** origin bin. Later readings can change that bin's features. Therefore fixture agreement between local model definitions does not establish end-to-end serving/evaluation equivalence.

Use a completed-bin availability rule, including an explicit ingestion-delay allowance if required. Test requests before/after bin close and compare actual served inputs with the harness. Explain the target interval: the nominal 30-minute bin-start shift targets `[t+30,t+40)`, whose start is only 20 minutes after the completed origin becomes available. Do not present this as an unqualified 30-minute forecast from request time.

### 3. High: stale observations can be presented as a current forecast

The same query accepts origins up to **24 hours old**. `forecast_for` is always origin plus 30 minutes; neither assembly nor the HTTP handler rejects a target already in the past. During an ingestion outage the endpoint can therefore return HTTP 200 with a freshly generated response containing an expired forecast.

Define a maximum origin age appropriate to five-minute collection, reject expired targets with a clear unavailable response, and expose observation age. Verify both directions independently. This is a code-path finding, not a claim that an outage occurred during the review.

### 4. High: pooled inference ignores shared-day dependence

[`eval/harness/significance.py`](../eval/harness/significance.py) computes HAC covariance within each direction and draws bootstrap blocks independently for each direction. Its minimum-block gate sums blocks across directions. The fuzzy manifest records **10 blocks across two routes**, although both routes cover roughly the same five days; that produces a formal `challenger` decision while the single-route deep results remain `insufficient data`.

Duplicating the calendar coverage across directions is not evidence of ten independent traffic days. Common events can correlate direction errors. This does not invalidate the point estimates, and this review has not recomputed the decisions. It does mean the combined significance claims need a sensitivity check that preserves the joint direction vector when resampling calendar-day blocks, or another justified panel covariance method. Base the minimum temporal-coverage gate on shared calendar history, and report directional results separately. Revisit the fuzzy significance headline after that check.

### 5. High: non-significance is defined as confirmation of no gain

The [frozen-window plan](roadmap.md#frozen-window-run-plan-proposed-2026-10-03-the-team-confirms-before-19-oct-2359-sgt) says C5 and C8 are confirmed when their result is `not significant`. Failure to detect a difference does not establish equivalence or absence of a useful gain.

Keep the conclusion as “no improvement detected on this window,” or predefine a scientifically meaningful equivalence/no-useful-benefit margin and the corresponding interval decision before the confirmation run. Retain point estimates and uncertainty. The 0.5-minute product threshold can inform that discussion, but should not be silently retrofitted as an equivalence protocol. The October confirmation window is otherwise a valuable safeguard against choosing models on the September results.

### 6. High: current deployment and report architecture disagree

Fresh GCP and HTTP reads confirm a working **local-model forecast API**, with `lin_bq[frozen]` for SG-to-MY and persistence for MY-to-SG. README, architecture, findings, several ADR passages, and the inventory's earlier observations still describe `v_forecast_recent` as the current serving path. The architecture also calls LSTM unscored even though the evidence bundle includes deep results.

Separate historical BQML benchmark results, local-replica parity, and today's HTTP serving policy. Do not transfer the old served MAE of 2.542 minutes to the new replica without identifying the relevant evaluation. Update both architecture depths to one verified system. The current public frontend now calls the forecast API. At this review the client source was outside git. **Follow-up:** `hosting/` is in git (`index.html`, `app.js`, `style.css`); deploy stays manual. A successful API call and inspection of its client code do not replace a full browser rehearsal.

### 7. Medium: live access policy contradicts the deployment smoke test

The forecast service is anonymously accessible because its invoker check is disabled; a live request during the 3 Oct review returned HTTP 200 with CORS `*`. At that checkout [`forecastapi/cloudbuild.yaml`](../services/forecastapi/cloudbuild.yaml) expected an anonymous request to return 403.

**Follow-up (2026-10-04):** local `main` `7bd94e3` accepts anonymous 200 or 503 for `model=served`, and a fresh anonymous GET returned 200. See [inventory.md](inventory.md). Do not label the API private based only on an empty IAM policy.

The camera service's public billed-inference exposure is an already accepted, documented risk. This review does not reopen that team decision or change access settings.

### 8. Medium: model parity depends on gap-free data

The live SQL uses positional LAG/LEAD; [`eval/lb/features.py`](../eval/lb/features.py) uses timestamp lookups. A missing ten-minute bin can change SQL horizons/lags while the local builder leaves the missing time empty. The `gap_min > 25` guard does not catch a single skipped bin (20-minute gap).

The fresh query found **zero** such single-bin gaps and **zero** three-row horizon mismatches, so this is a latent reliability issue, not evidence that the published window is wrong. Align the definitions or enforce continuity and target-time checks before both scoring and serving; add a missing-bin regression case.

### 9. Medium: reproducibility and submission packaging remain incomplete

The committed report manifest validates and records a clean run at `b3b6562`, dataset fingerprints and environment metadata. Its code fingerprint differs from current HEAD, which has subsequent legitimate changes. That is a historical-run distinction, not proof of invalid results. Reproduce that snapshot from its recorded revision, and generate the final run from the frozen final revision.

Raw inputs are ignored in git. Hashes identify them but do not deliver them to a grader. Supply an immutable, permitted dataset package or a tested retrieval/access process, including Maps, weather, camera history, and held-out labels. Source for Maps ingestion is absent. **Follow-up:** the Hosting static files are in `hosting/`; `firebase.json` is still absent, so a clean checkout still cannot publish the live site. A bounded offline demo can be an acceptable fallback if documented and included. No leaked secret value was printed or used as report evidence; this was not an exhaustive history-wide secret audit.

### 10. Medium: all four deck PNGs fail factual review

All four images were opened visually under the diagram skill. Failures include stale live row counts, “Layer A scoring script MISSING,” an obsolete BQML-only serve path, and “Firebase Hosting unset.” The detailed image also uses a hyphenated Cloud Build bucket name instead of `swiftborder_cloudbuild` and crowds the unused-view annotation against the band edge. The high-level image labels the frame cache as an active detect cache without code evidence.

The project now has seven buckets, including `swiftborder_asia-southeast1_cloudbuild`; the skill's exact-six-bucket rule also needs correction. Update Mermaid topology and the mirrored diagram guidance from inventory first, then regenerate and visually check the PNGs. The current Mermaid is also behind the new HTTP serving architecture; rendering it alone is not enough.

## Evidence worth retaining

- The report correctly distinguishes Maps' estimate from independently measured crossing time. Neither the 24-hour horizon nor the <=15-minute real crossing-time target is achieved by these experiments.
- The September 30-minute results use paired rows and explicit baselines. Reported MAE is approximately 2.640 minutes for persistence and 2.275 for daily-refit Maps-only XGBoost. The gain is about 22 seconds, below the proposed 30-second practical threshold.
- Causal feature construction, rolling label cutoffs, seeds, run provenance, direction/time slices, family and run-wide Holm reporting, and explicit negative results provide a defensible methods section, subject to the pooled-inference finding above.
- Weather and ensemble complexity did not show a detected incremental benefit on the evaluated window. The Maps-profile control materially improves the interpretation of the camera-profile result.
- Deep models are scored, not merely listed. Their short single-route hold-out is appropriately marked insufficient for significance claims.
- CI, isolated suites, service/harness parity fixtures, deployment smoke checks and rollback-oriented deployment structure are useful engineering evidence. Passing fixtures do not establish model accuracy or field reliability.

## Verification performed and limits

Fast suites passed: **315 tests** (camdetect 41, Causeway 37, eval 204, forecastapi 18, repository 15). The separate TensorFlow suite passed **all five slow tests**, for **320 passing tests total**; it emitted nine Keras/NumPy deprecation warnings. Report manifest validation returned no issues. Live checks confirmed ingestion freshness, serving revision and model identity, anonymous access, scheduler state, model/view inventory and storage changes. No production training, deployment, registry change, backfill, or billed Roboflow inference was performed; tests trained small local fixture models.

The review is not a full rerun of the published experiments, a browser usability test, an exhaustive security audit, or validation of every external API/model source. No new MAE or significance result was generated. The pre-existing untracked `semantic-review/` directory was left untouched.

## Route to final submission

1. **Before the 19 October data freeze:** score Layer A; resolve bin completion/staleness; settle the statistical inference and no-gain wording; preserve the confirmation window; keep the Hosting static files in git (done in `hosting/`) and capture approved immutable data inputs. Avoid adding more model families.
2. **From 20 October:** run the agreed frozen evaluation from a clean revision, validate and promote Runs A/B, and make every table/figure traceable to its run, horizon, direction coverage, label and baseline. Keep insufficient-data conclusions where required.
3. **Assemble the report:** problem and scope; data and label validity; methods/model choices; current architecture; experimental protocol; results; error analysis and limitations; conclusion; references; reproducibility and contribution appendices. Lead with the few research questions, not the entire experiment catalog.
4. **Before submission:** regenerate the figures, rehearse a clean-start demo and browser flow, prepare an outage fallback, package code/data/instructions, complete the video/slides and individual reflections, and verify the 31 October deadline and submission rules on Canvas.

**Final acceptance bar:** every claimed result has frozen evidence; vision quality is measured or explicitly excluded from validated scope; served forecasts obey their availability/freshness contract; current diagrams match GCP; and a grader can run the submitted demonstration from the supplied materials.

## Proposed information architecture

**Status: repo spine (2026-10-04), superseded for the Word report by [Report assembly](#report-assembly-24-oct-2026).** The gap table at the end of this section is the 4 Oct snapshot. It is not the current figure list.

A lecturer should open six files and stop. Everything else is an ADR, a runbook, or a labelled archive.

| Spine | Why it exists |
| --- | --- |
| [../README.md](../README.md) | What the system is, and how to run the camera count plus the 30-minute forecast |
| [architecture.md](architecture.md) | Tools, techniques, and one system at two depths |
| [evaluation.md](evaluation.md) | What was measured, on which window, against which baseline, and which cells are still targets |
| [findings.md](findings.md) | Findings a report or a 10–15 min talk can cite, each tied to a run or a live query, plus the claims register |
| [inventory.md](inventory.md) | Dated copy of project `swiftborder`. Cite the latest verification block only |
| [README.md](README.md) | The map. After a later rewrite it should lead with this spine, then operational docs, then archive |

**Merge in a later pass (headings move; files stay until that pass):**

- [horizon-study.md](horizon-study.md) moves under a new `evaluation.md` heading, "Exploratory horizon study (not a report run)". The 3 h Holm cutoff, the 5.5 h uncorrected cutoff, and the 24 h product target stay in that one section. README, findings item 12, and the roadmap "Already true" list keep one sentence and a link.
- [deep-learning-assessment.md](deep-learning-assessment.md) "What it showed", "Why this is expected", and "When to revisit" fold into evaluation §4. The transformer design note can stay as a short paragraph there.
- Open readiness items (Layer A scores, submission package) move into findings as "Still open". Items already marked fixed stay in [CHANGELOG.md](../CHANGELOG.md), not in report prose.
- The callable-model catalog has one home: the parameter table in [runbooks/forecast-api.md](runbooks/forecast-api.md). ADR 0004 keeps the serving decision. [release-pr2.md](archive/release-pr2.md) curl comments and the 2026-10-03 changelog catalog stay historical.

**Label as archive. Do not delete:**

- [release-pr2.md](archive/release-pr2.md) — merged-PR checklist. Its "pending merge", anonymous 403, and `v_forecast_recent` curls are not the current contract.
- [plan-eval-integrity.md](archive/plan-eval-integrity.md) — execution log. Remaining work is the roadmap freeze and steps 4–7, not this task list.
- This file's original priority-finding body — a 3 Oct review memo. The Response table is the status. The finding prose is not current behaviour.
- Inventory blocks dated before the latest verification, and body rows that still quote older row counts, revisions, or "not on main".
- Roadmap steps already marked done (first presentation, published 13–30 Sep Layer B).

**Stay operational. Do not merge into report prose:** ADRs 0001–0004, the forecast-api runbook, [handoff-camera-pilot.md](archive/handoff-camera-pilot.md), [agent-deploy.md](agent-deploy.md), the roadmap freeze and remaining sequence, [../eval/README.md](../eval/README.md), [../data/sql/README.md](../data/sql/README.md), the grading lens, and `docs/diagrams/` plus `docs/images/`.

**Do not merge:** `eval/runs/report/` JSON and plots, replay CSVs, or `skills/`. `hosting/` is in this checkout. The per-point `model` fields in `forecastapi/` are a separate code change in the same branch; they are not a report-file merge.

**Report gaps this spine still leaves empty:**

| Graded section | Filled by | Still empty |
| --- | --- | --- |
| Tools | architecture.md | Deck PNGs stale on forecast input, public access, and the Hosting node. Mermaid in this checkout says `hosting/` is in git and deploy is manual |
| Design and models | architecture.md; decision in ADR 0004; feature matrix in ADR 0003 | ADR 0004's decision still shows example ids that did not ship. The shipped catalog is only in the runbook |
| Performance | evaluation.md, from `eval/runs/report/run.json` | Layer A mAP table is all `pending`. §7c observed counts are pending. Frozen Runs A and B are not run. The horizon study is exploratory and is not in `run.json` |
| Findings | findings.md | Item 5 (scored camera-profile prior) is easy to read as measured Layer A quality. 24 h and ≤15 min MAE stay targets |
