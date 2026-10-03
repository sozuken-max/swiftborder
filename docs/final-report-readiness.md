# Final-report readiness review

**Follow-up, 4 Oct 2026:** all four deck PNGs and their source topology were regenerated; see [diagram QA notes](diagrams/README.md#png-exports-refreshed-4-oct-2026). The figure finding below records the original review state. Application and statistical findings remain open.

Reviewed 2026-10-03 (SGT), checkout `be9281d`, against repository code, committed evaluation evidence, the module grading guide, four deck PNGs, and fresh read-only queries of GCP project `swiftborder`. Live observations and query scope are in [inventory.md](inventory.md#latest-verification-2026-10-03-approximately-2335-2340-sgt).

**Verdict: request changes before final submission.** The project is ready to start assembling the final report. It is not ready to freeze the report or claim the original crossing-time product goal is achieved. The strongest contribution is the evaluated forecasting methodology and its honest negative results. The biggest gaps are unmeasured vision quality, live/evaluation timing differences, statistical interpretation, and inconsistent documentation.

## Readiness by graded surface

| Surface | Assessment | Remaining evidence |
| --- | --- | --- |
| Tools, techniques and model design | Substantial | Consolidate the model/data descriptions and current serving architecture |
| Layer B performance | Strong preliminary evidence | Frozen Runs A/B; resolve inference concerns below; publish exact windows per experiment |
| Layer A performance | Not ready | Held-out detector and directional-count results, dataset/model version and split provenance |
| Runnable MVP | Backend demonstrated; browser implementation inspected | Full browser rehearsal, freshness handling and reproducible Hosting source |
| Findings and discussion | Good foundation | Correct significance wording; distinguish historical BQML results from current local serving |
| Final figures | Fail | Refresh topology and regenerate all four deck PNGs |
| Reproducibility/submission package | Partial | Immutable input data package or access procedure, pinned code/environment, client source and run instructions |
| Individual reflection/peer review | Not established by this review | Contribution matrix and each member's reflection; confirm submission requirements on Canvas |

The project can name at least three required categories: **supervised learning** (future-duration regression and labelled YOLO), **ML/deep learning** (trees, YOLO, sequence models), and **hybrid/ensemble** (evaluated combinations and XGB-to-fuzzy classification). **Intelligent sensing** is also implemented, but its quality remains unmeasured. These are rubric categories, not claims of four independent scientific contributions.

## Priority findings

### 1. High: no empirical Layer A quality evidence

The results table in [evaluation.md](evaluation.md#layer-a--vision) is entirely pending. The scorer exists and its fixtures pass, but neither held-out detector accuracy nor directional count accuracy has been established. The live image demonstration cannot substitute for those measurements. The observed-count Layer B experiment also remains unscored; a March-April camera-derived daily profile is not simultaneous camera/Maps integration.

Before submission, score the served YOLO model on an untouched export and report mAP50, mAP50-95, precision/recall, count MAE by direction, day/night sample counts, and representative failures. Record dataset version, model/workflow version, class mapping, confidence threshold, and how near-duplicate frames were separated across splits. Compare a pretrained detector if feasible. If this cannot be completed, describe Layer A explicitly as an unvalidated prototype and narrow the report's claims. Optional ResNet work should not displace this.

### 2. High: the live API uses incomplete origin bins

[`forecastapi/main.py`](../forecastapi/main.py), `_LATEST_SQL`, picks the newest row without requiring `bin_ts + 10 minutes <= now`. The underlying view aggregates all readings currently present in each bin. The evaluation treats a bin's features as available only after the bin closes.

This occurred live: a response generated at **23:32:49 SGT** used the **23:30-23:40** origin bin. Later readings can change that bin's features. Therefore fixture agreement between local model definitions does not establish end-to-end serving/evaluation equivalence.

Use a completed-bin availability rule, including an explicit ingestion-delay allowance if required. Test requests before/after bin close and compare actual served inputs with the harness. Explain the target interval: the nominal 30-minute bin-start shift targets `[t+30,t+40)`, whose start is only 20 minutes after the completed origin becomes available. Do not present this as an unqualified 30-minute forecast from request time.

### 3. High: stale observations can be presented as a current forecast

The same query accepts origins up to **24 hours old**. `forecast_for` is always origin plus 30 minutes; neither assembly nor the HTTP handler rejects a target already in the past. During an ingestion outage the endpoint can therefore return HTTP 200 with a freshly generated response containing an expired forecast.

Define a maximum origin age appropriate to five-minute collection, reject expired targets with a clear unavailable response, and expose observation age. Verify both directions independently. This is a code-path finding, not a claim that an outage occurred during the review.

### 4. High: pooled inference ignores shared-day dependence

[`eval/significance.py`](../eval/significance.py) computes HAC covariance within each direction and draws bootstrap blocks independently for each direction. Its minimum-block gate sums blocks across directions. The fuzzy manifest records **10 blocks across two routes**, although both routes cover roughly the same five days; that produces a formal `challenger` decision while the single-route deep results remain `insufficient data`.

Duplicating the calendar coverage across directions is not evidence of ten independent traffic days. Common events can correlate direction errors. This does not invalidate the point estimates, and this review has not recomputed the decisions. It does mean the combined significance claims need a sensitivity check that preserves the joint direction vector when resampling calendar-day blocks, or another justified panel covariance method. Base the minimum temporal-coverage gate on shared calendar history, and report directional results separately. Revisit the fuzzy significance headline after that check.

### 5. High: non-significance is defined as confirmation of no gain

The [frozen-window plan](roadmap.md#frozen-window-run-plan-proposed-2026-10-03-the-team-confirms-before-19-oct-2359-sgt) says C5 and C8 are confirmed when their result is `not significant`. Failure to detect a difference does not establish equivalence or absence of a useful gain.

Keep the conclusion as “no improvement detected on this window,” or predefine a scientifically meaningful equivalence/no-useful-benefit margin and the corresponding interval decision before the confirmation run. Retain point estimates and uncertainty. The 0.5-minute product threshold can inform that discussion, but should not be silently retrofitted as an equivalence protocol. The October confirmation window is otherwise a valuable safeguard against choosing models on the September results.

### 6. High: current deployment and report architecture disagree

Fresh GCP and HTTP reads confirm a working **local-model forecast API**, with `lin_bq[frozen]` for SG-to-MY and persistence for MY-to-SG. README, architecture, findings, several ADR passages, and the inventory's earlier observations still describe `v_forecast_recent` as the current serving path. The architecture also calls LSTM unscored even though the evidence bundle includes deep results.

Separate historical BQML benchmark results, local-replica parity, and today's HTTP serving policy. Do not transfer the old served MAE of 2.542 minutes to the new replica without identifying the relevant evaluation. Update both architecture depths to one verified system. The current public frontend now calls the forecast API; its source is still outside git. A successful API call and inspection of its client code do not replace a full browser rehearsal.

### 7. Medium: live access policy contradicts the deployment smoke test

The forecast service is anonymously accessible because its invoker check is disabled; a live request returned HTTP 200 with CORS `*`. [`forecastapi/cloudbuild.yaml`](../forecastapi/cloudbuild.yaml) expects an anonymous request to return 403. Its `--no-allow-unauthenticated` flag concerns IAM policy and does not explicitly resolve the separate disabled-invoker-check setting. Reconcile the intended access policy, deploy flags, smoke assertion and frontend before the next release. Do not label the API private based only on an empty IAM policy.

The camera service's public billed-inference exposure is an already accepted, documented risk. This review does not reopen that team decision or change access settings.

### 8. Medium: model parity depends on gap-free data

The live SQL uses positional LAG/LEAD; [`eval/features.py`](../eval/features.py) uses timestamp lookups. A missing ten-minute bin can change SQL horizons/lags while the local builder leaves the missing time empty. The `gap_min > 25` guard does not catch a single skipped bin (20-minute gap).

The fresh query found **zero** such single-bin gaps and **zero** three-row horizon mismatches, so this is a latent reliability issue, not evidence that the published window is wrong. Align the definitions or enforce continuity and target-time checks before both scoring and serving; add a missing-bin regression case.

### 9. Medium: reproducibility and submission packaging remain incomplete

The committed report manifest validates and records a clean run at `b3b6562`, dataset fingerprints and environment metadata. Its code fingerprint differs from current HEAD, which has subsequent legitimate changes. That is a historical-run distinction, not proof of invalid results. Reproduce that snapshot from its recorded revision, and generate the final run from the frozen final revision.

Raw inputs are ignored in git. Hashes identify them but do not deliver them to a grader. Supply an immutable, permitted dataset package or a tested retrieval/access process, including Maps, weather, camera history, and held-out labels. Source for Maps ingestion and Hosting is absent, so a clean checkout cannot reconstruct the whole deployed application. A bounded offline demo can be an acceptable fallback if documented and included. No leaked secret value was printed or used as report evidence; this was not an exhaustive history-wide secret audit.

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

1. **Before the 19 October data freeze:** score Layer A; resolve bin completion/staleness; settle the statistical inference and no-gain wording; preserve the confirmation window; capture the Hosting source and approved immutable data inputs. Avoid adding more model families.
2. **From 20 October:** run the agreed frozen evaluation from a clean revision, validate and promote Runs A/B, and make every table/figure traceable to its run, horizon, direction coverage, label and baseline. Keep insufficient-data conclusions where required.
3. **Assemble the report:** problem and scope; data and label validity; methods/model choices; current architecture; experimental protocol; results; error analysis and limitations; conclusion; references; reproducibility and contribution appendices. Lead with the few research questions, not the entire experiment catalog.
4. **Before submission:** regenerate the figures, rehearse a clean-start demo and browser flow, prepare an outage fallback, package code/data/instructions, complete the video/slides and individual reflections, and verify the 31 October deadline and submission rules on Canvas.

**Final acceptance bar:** every claimed result has frozen evidence; vision quality is measured or explicitly excluded from validated scope; served forecasts obey their availability/freshness contract; current diagrams match GCP; and a grader can run the submitted demonstration from the supplied materials.
