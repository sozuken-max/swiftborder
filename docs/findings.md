# Findings and discussion

**Report section:** findings and discussion
**Source of truth:** GCP project `swiftborder`
**GCP facts:** [inventory.md](inventory.md) (dated copy, 2026-10-01). **Methods and scored results:** [evaluation.md](evaluation.md), from [`eval/runs/report/run.json`](../eval/runs/report/run.json). **Design:** [architecture.md](architecture.md).

---

## Findings the query and the harness support

1. **A duration label is accumulating.** `causeway.travel_times` is written every five minutes for both directions (14,368 rows in the live table at the 1 Oct query, [inventory.md](inventory.md); the report run used a 14,366-row export taken minutes earlier). That supports an 18-day out-of-sample window at 30 minutes. It is not a seasonal record.
2. **The live forecast is 30 minutes of the Maps series, from Maps lags.** `v_training_set` joins neither weather nor camera congestion. `v_forecast_recent` publishes persistence or `lin_h30` per the registry. `xgb_h30` is trained and unused. `y_60` is computed and unused.
3. **On 13–30 Sep the served choice holds up, and a better one exists.** Against persistence: `lin_h30` is better on `SG_TO_MY` (−0.20 min) and worse on `MY_TO_SG` (+0.47); combined it is not significantly different. That matches the registry (lin for `SG_TO_MY`, persistence for `MY_TO_SG`). `xgb_h30` is better than persistence on `MY_TO_SG` (−0.17) and combined (−0.15), but only within the BQML family; under a run-wide Holm check it is not significant. The mean of `lin_h30` and `xgb_h30` is better than persistence combined (−0.18) and on `SG_TO_MY` (−0.31). All with Diebold–Mariano (DM) tests of whether two error series differ, Holm-adjusted for testing several models, and CIs from a bootstrap that resamples whole days ([evaluation.md §1](evaluation.md#1-production-models-30-minutes-evallayer_bpy)).
4. **Weather did not help.** Adding rainfall and the 2-hour forecast (joined by data.gov.sg acquisition time) to a daily-refit model changed MAE by +0.003 (XGBoost) and +0.011 (ridge) minutes, neither significant. A daily-refit Maps-only XGBoost beats persistence by 0.37 min ([evaluation.md §2](evaluation.md#2-joined-features-30-minutes-evaljoinedpy)).
5. **A Layer A forecast helps Layer B, but only as a daily-profile prior; observed counts are untested.** A camera-2701 queue forecast learned from the Mar–Apr detections cuts daily-refit XGBoost's 30-min MAE by 0.065 min (about 4 s; significant within its family, not run-wide) and ridge's by 0.053 (significant also run-wide), most at queue onsets, but a profile fitted on the Maps rows gives the same gain, and the camera forecast adds nothing on top of it ([evaluation.md §7a](evaluation.md#7a-layer-a-forecast-as-a-layer-b-input-evalcamera_forecastpy-scored-in-evaljoinedpy)). Observed counts cannot be tested yet: The only Layer A output in BigQuery (`Cam2701` / `Cam2702` detections) covers 13 Mar–22 Apr 2026; the Maps label starts on 5 Sep. Raw 2701 frames exist for 6–11 Sep (`traffic_images.metadata`) but have no detections. The joined experiment is ready for camera counts; the backfill is handed to the Roboflow key holder ([handoff-camera-pilot.md](handoff-camera-pilot.md)). The headroom is bounded: queue onsets and clearings (16.5% of rows) carry 39% of the best model's error ([evaluation.md §7](evaluation.md#7-is-layer-a-output-a-meaningful-layer-b-input)).
6. **Offline 60-minute XGBoost has lower MAE than persistence on one route after the evaluation fixes**, by 1.2 min (hold-out MAE 3.47 vs 4.68; `jb_to_woodlands` only), but the hold-out spans only 5 days, so the significance decision is "insufficient data". The earlier offline "win" of 3.7 min was an artefact of a mislabelled persistence baseline, a target-time feature and smoothed labels.
7. **Vision tables in BigQuery are historical side stores; weather is now current to 30 Sep.** Camera detection tables hold 13 Mar–22 Apr 2026 (last modified 18 Jul); image frames run to 11 Sep; image metadata 13 Sep (368,905 rows); `traffic_images.labels` is empty (labelling is in Roboflow). `rainfall.rainfall` and `weatherforecast.weatherforecast` held data to 31 Aug SGT; on 1 Oct the Woodlands rows for 1–30 Sep were appended from the Causeway CSVs by [`Causeway/load_bigquery.py`](../Causeway/load_bigquery.py) (snapshots kept). No scheduled loader keeps them current.
8. **Layer A has a scorer and no scores.** [`eval/layer_a.py`](../eval/layer_a.py) is tested on fixtures; no Roboflow export has been scored.
9. **Ensembling adds nothing over the best single model; changing what is served would.** On 13–30 Sep a rolling LAD stack, rolling selection and a fuzzy-gated stack over the BQML and daily-refit models are all within 0.02 min of the daily-refit `xgb[maps]` (2.275 min, not significant); an equal mean is worse. `xgb[maps]` or the stack beat the **served** registry forecast (2.542) by 0.27–0.29 min (significant); the served forecast is not significantly better than persistence ([evaluation.md §6](evaluation.md#6-ensembles-and-hybrids-of-the-layer-b-models-evalensemblepy)).
10. **Deep sequence models do not beat XGBoost.** At 60 min on one route, LSTM 3.97, GRU 4.03 and a patch Transformer 4.19 min against XGBoost 3.47 and persistence 4.68 (seed means; 5 day-blocks, so "insufficient data") ([evaluation.md §4](evaluation.md#4-deep-sequence-models-60-minutes-one-route-evaldeep_forecastpy)).
11. **Traffic level can be forecast, best by a hybrid.** Light / moderate / heavy 60 min ahead, both directions: XGBoost's forecast mapped through the fuzzy partition reaches 0.785 accuracy against 0.690 for the persistence level (significant); a learned fuzzy rule base reaches 0.718 (not significant vs persistence) with readable rules ([evaluation.md §5](evaluation.md#5-fuzzy-traffic-level-60-minutes-both-directions-evalfuzzy_trafficpy)).

## Known risk: public `swiftbackend` (documented, not changed)

`swiftbackend` has the IAM invoker check disabled with an empty policy, ingress `all`, CORS `*` (`ALLOWED_ORIGIN` unset) and `ROBOFLOW_API_KEY` as a plain env var ([inventory.md](inventory.md)). Anyone who finds the URL can trigger billed Roboflow inference and exhaust the free-tier credits the camera backfill also needs. The team chose to document this rather than change the live service. Minimum mitigation when approved: set `ALLOWED_ORIGIN`, move the key to Secret Manager, require an invoker identity or an API key.

---

## Claims register

Use this for the proposal, the presentations and the final report (31 Oct 2026). Weights are in [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md).

| Say this | Do not say this | Until |
| --- | --- | --- |
| Woodlands only; cameras 2701 and 2702 are in scope | The live divider covers 2702 | 2702 geometry exists and is demoed |
| Layer A measures detection; Layer B measures duration. Camera-derived queue counts follow the same daily cycle as Maps travel time, and a queue forecast learned from them lowers 30-min MAE significantly, as much as a Maps-derived daily profile does. Observed counts are untested (no overlap with the Maps window) | Layer A output is highly correlated with the Layer B target; camera counts can replace Distance Matrix data; live queue counts improve the forecast; the camera adds information beyond the daily cycle | An observed-count row beats `maps+mpfc` with sufficient coverage in [evaluation.md](evaluation.md) |
| Maps durations log live; serve horizon is 30 minutes | A 24-hour forecast is running | A scored horizon beyond 60 minutes is in the results tables |
| On 13–30 Sep, the ensemble has lower 30-min MAE than persistence (significant under family and run-wide Holm); `xgb_h30` does too within the BQML family | `xgb_h30` or the ensemble is the production model; a significant gain is a product-relevant gain | `model_registry` and `v_forecast_recent` are changed (an approved write) |
| A daily-refit XGBoost (or a stack) would cut the served 30-min MAE by ~0.27 min (~16 s) on 13–30 Sep | The served forecast is significantly better than persistence; ensembling beats the best single model | A longer window shows it |
| We tested LSTM, GRU and a patch Transformer; none beat XGBoost on 5 test days | Deep learning does not work for this problem | 3+ months of history are scored |
| A fuzzy traffic-level forecast (light / moderate / heavy) is scored; the XGB-to-fuzzy hybrid is best | The fuzzy rule base beats persistence | Its accuracy gain is significant |
| Skill over persistence on the Maps duration series | We beat Google | An independent wait-time label exists and is scored |
| <= 15 min MAE is the product target | <= 15 min MAE is achieved | An independent crossing-time label is scored against it (Maps-series MAE of 2–3 min does not test this target) |
| Weather features did not reduce 30-min MAE on 13–30 Sep | Weather does not matter for the causeway | A longer window with rain events is scored |
| Weather and congestion views exist in BigQuery | They feed the served model | `v_training_set` references them |
| Offline 60-min XGBoost is **JB → SG only**; its MAE is ~1.2 min lower than persistence on a <5-day hold-out (too short for a significance claim) | Offline XGB significantly beats persistence, covers both directions, or is the served model | A hold-out of 10+ days, and a second route, are scored |
| Layer A scorer exists and is tested on fixtures | We measured mAP / count error | A Roboflow export is scored in the Layer A table |
| Roboflow Public can export a dataset version; weight download is Core | Empty `traffic_images.labels` means export is impossible | — |
| `camdetect` runtime changes on `main` run pytest then deploy `swiftbackend`; tests also run in GitHub Actions | Forecast SQL/models deploy from git | A separate pipeline exists |
| `swiftbackend` is publicly callable (known risk) | The service is secured | The live service is changed and inventory re-checked |
| Firebase Hosting is planned | The UI is live | Hosting exists in the project |

Paste-ready status for a slide (refresh from [inventory.md](inventory.md) and `run.json` before the deck):

> Maps durations log every five minutes (both directions). **Serve:** 30 minutes (`v_forecast_recent`: persistence or `lin_h30`). **Out-of-sample 13–30 Sep:** persistence MAE 2.64 min; the `lin_h30`+`xgb_h30` ensemble 2.46, significantly better than persistence (`xgb_h30` 2.49 only within its test family). Weather features: no significant gain. Ensembles: no gain over a daily-refit XGBoost (2.28), which would beat the served forecast (2.54). Deep models and a fuzzy traffic-level forecast are scored offline. A Layer A queue forecast as input: significant gain, equal to a Maps daily profile. Observed camera counts and Layer A metrics: pending. **24-hour** horizon and <= 15 min MAE remain targets.

---

## Discussion points for the report

**Horizon.** Three bins ahead is implemented and scored (`y_30`). Six bins (`y_60`) is labelled and not served. The only 60-minute result is the offline one-route model. Twenty-four hours has no training target yet.

**Unused features.** Weather gave no measurable gain on 18 days with few rain events; that is a limitation of the window, not evidence that rain never matters. Camera queue depth is the untested hypothesis most tied to the proposal. It can only help where Maps lags miss a change (queue onsets and clearings), which is where the remaining error is concentrated.

**Ensembles and hybrids.** The combiners converge on the best single model, because the BQML models are older and weaker than a daily refit on the same features. The useful hybrid in this project is a regression forecast followed by fuzzy level assignment, not a blend of regressors.

**Registry.** The harness supports keeping persistence for `MY_TO_SG` over `lin_h30`, but `xgb_h30` would do better there, and a daily-refit XGBoost better still in both directions. A registry change, or a daily retrain job, is an approved write, not a harness output.

**Camera as a substitute for Distance Matrix.** It is an attractive idea (no API cost, an independent sensor), but it is untested and faces structural limits: counts saturate when the frame is full, do not show speed, cover one stretch of road, and still need a label to calibrate against. Fog, haze and glare make frames look empty. The cheapest test is scoring the 6–11 Sep frames that overlap the Maps label (about 864 Roboflow calls), then a camera-only estimate of current travel time against a calendar baseline ([evaluation.md §7b](evaluation.md#7b-could-layer-a-output-replace-the-distance-matrix-data)).

**Two clocks for vision.** A demo of `swiftbackend` shows a live frame. A chart of `Cam2701` shows detections from 13 Mar–22 Apr 2026. The report should say which one a figure is.

**Git is still behind ingest for Maps and camera.** The Maps fetcher, `travel_times` loader and camera table writers are not in the tree. Weather is now loadable from git ([`Causeway/load_bigquery.py`](../Causeway/load_bigquery.py), manual, no schedule). Views, BQML and serve SQL are in [sql/](../sql/) and match the live definitions (2026-09-30). The evaluation harnesses, weather fetchers and camera backfill are in the repo.

**Deep learning is scored and not served.** With about three weeks of 5-minute data, LSTM, GRU and a patch Transformer trail XGBoost by 0.5–0.7 min at 60 minutes. Reasons and the conditions for revisiting: [deep-learning-assessment.md](deep-learning-assessment.md).

**What to defer.** ResNet, Firebase and holiday calendars can wait. The order of work is in [roadmap.md](roadmap.md) and [plan-eval-integrity.md](plan-eval-integrity.md).
