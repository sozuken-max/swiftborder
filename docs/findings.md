# Findings and discussion

**Report section:** findings and discussion
**Source of truth:** GCP project `swiftborder`
**GCP facts:** [inventory.md](inventory.md) (dated copy, 2026-10-01). **Methods and scored results:** [evaluation.md](evaluation.md), from [`eval/runs/report/run.json`](../eval/runs/report/run.json). **Design:** [architecture.md](architecture.md).

---

## Findings the query and the harness support

1. **A duration label is accumulating.** `causeway.travel_times` is written every five minutes for both directions (14,368 rows in the live table at the 1 Oct query, [inventory.md](inventory.md); the report run used a 14,366-row export taken minutes earlier). That supports an 18-day out-of-sample window at 30 minutes. It is not a seasonal record.
2. **The live forecast is 30 minutes of the Maps series, from Maps lags.** `v_training_set` joins neither weather nor camera congestion. `v_forecast_recent` publishes persistence or `lin_h30` per the registry. `xgb_h30` is trained and unused. `y_60` is computed and unused.
3. **On 13–30 Sep the served choice holds up, and a better one exists.** Against persistence: `lin_h30` is better on `SG_TO_MY` (−0.20 min) and worse on `MY_TO_SG` (+0.47); combined it is not significantly different. That matches the registry (lin for `SG_TO_MY`, persistence for `MY_TO_SG`). `xgb_h30` is better than persistence on `MY_TO_SG` (−0.17) and combined (−0.15). The mean of `lin_h30` and `xgb_h30` is better than persistence combined (−0.18) and on `SG_TO_MY` (−0.31). All with Holm-adjusted DM tests and day-block CIs ([evaluation.md §1](evaluation.md#1-production-models-30-minutes-evallayer_bpy)).
4. **Weather did not help.** Adding rainfall and the 2-hour forecast (joined by data.gov.sg acquisition time) to a daily-refit model changed MAE by +0.003 (XGBoost) and +0.011 (ridge) minutes, neither significant. A daily-refit Maps-only XGBoost beats persistence by 0.37 min ([evaluation.md §2](evaluation.md#2-joined-features-30-minutes-evaljoinedpy)).
5. **Camera features are untested.** No camera-2701 frames have been scored for the Maps window; the BigQuery camera tables stop on 18 Jul. The joined experiment is ready to take them; the backfill is handed to the Roboflow key holder ([handoff-camera-pilot.md](handoff-camera-pilot.md)).
6. **Offline 60-minute XGBoost has lower MAE than persistence on one route after the evaluation fixes**, by 1.2 min (hold-out MAE 3.47 vs 4.68; `jb_to_woodlands` only), but the hold-out spans only 5 days, so the significance decision is "insufficient data". The earlier offline "win" of 3.7 min was an artefact of a mislabelled persistence baseline, a target-time feature and smoothed labels.
7. **Vision tables in BigQuery are historical side stores; weather is now current to 30 Sep.** Camera tables last changed 18 Jul; image metadata 13 Sep (368,905 rows); `traffic_images.labels` is empty (labelling is in Roboflow). `rainfall.rainfall` and `weatherforecast.weatherforecast` held data to 31 Aug SGT; on 1 Oct the Woodlands rows for 1–30 Sep were appended from the Causeway CSVs by [`Causeway/load_bigquery.py`](../Causeway/load_bigquery.py) (snapshots kept). No scheduled loader keeps them current.
8. **Layer A has a scorer and no scores.** [`eval/layer_a.py`](../eval/layer_a.py) is tested on fixtures; no Roboflow export has been scored.

## Known risk: public `swiftbackend` (documented, not changed)

`swiftbackend` has the IAM invoker check disabled with an empty policy, ingress `all`, CORS `*` (`ALLOWED_ORIGIN` unset) and `ROBOFLOW_API_KEY` as a plain env var ([inventory.md](inventory.md)). Anyone who finds the URL can trigger billed Roboflow inference and exhaust the free-tier credits the camera backfill also needs. The team chose to document this rather than change the live service. Minimum mitigation when approved: set `ALLOWED_ORIGIN`, move the key to Secret Manager, require an invoker identity or an API key.

---

## Claims register

Use this for the proposal, the presentations and the final report (31 Oct 2026). Weights are in [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md).

| Say this | Do not say this | Until |
| --- | --- | --- |
| Woodlands only; cameras 2701 and 2702 are in scope | The live divider covers 2702 | 2702 geometry exists and is demoed |
| Layer A measures detection; Layer B measures duration | Queue counts predict crossing time | A camera-feature row is scored with sufficient coverage in [evaluation.md](evaluation.md) |
| Maps durations log live; serve horizon is 30 minutes | A 24-hour forecast is running | A scored horizon beyond 60 minutes is in the results tables |
| On 13–30 Sep, `xgb_h30` and the ensemble have lower 30-min MAE than persistence (significant, Holm) | `xgb_h30` or the ensemble is the production model | `model_registry` and `v_forecast_recent` are changed (an approved write) |
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

> Maps durations log every five minutes (both directions). **Serve:** 30 minutes (`v_forecast_recent`: persistence or `lin_h30`). **Out-of-sample 13–30 Sep:** persistence MAE 2.64 min; `xgb_h30` 2.49 and the `lin_h30`+`xgb_h30` ensemble 2.46, both significantly better than persistence (Holm). Weather features: no significant gain. Camera features and Layer A metrics: pending. **24-hour** horizon and <= 15 min MAE remain targets.

---

## Discussion points for the report

**Horizon.** Three bins ahead is implemented and scored (`y_30`). Six bins (`y_60`) is labelled and not served. The only 60-minute result is the offline one-route model. Twenty-four hours has no training target yet.

**Unused features.** Weather gave no measurable gain on 18 days with few rain events; that is a limitation of the window, not evidence that rain never matters. Camera queue depth is the untested hypothesis most tied to the proposal.

**Registry.** The harness supports keeping persistence for `MY_TO_SG` over `lin_h30`, but `xgb_h30` would do better there. A registry change is an approved write, not a harness output.

**Two clocks for vision.** A demo of `swiftbackend` shows a live frame. A chart of `Cam2701` shows history that stopped on 18 Jul. The report should say which one a figure is.

**Git is still behind ingest.** The Maps fetcher, `travel_times` loader, and camera/weather table writers are not in the tree. Views, BQML and serve SQL are in [sql/](../sql/) and match the live definitions (2026-09-30). The evaluation harnesses, weather fetchers and camera backfill are in the repo.

**What to defer.** ResNet, Firebase and holiday calendars can wait. The order of work is in [roadmap.md](roadmap.md) and [plan-eval-integrity.md](plan-eval-integrity.md).
