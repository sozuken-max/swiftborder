# Findings and discussion

**Report section:** findings and discussion  
**Source of truth:** GCP project `swiftborder`  
**As of:** 26 September 2026, ~13:40 SGT  
**Evidence:** [inventory.md](inventory.md). Methods: [evaluation.md](evaluation.md). Design: [architecture.md](architecture.md).

These are findings the query supports. They are not performance results. Performance cells stay in the evaluation tables until a harness fills them.

---

## Findings the query supports

1. **A duration label is accumulating.** `causeway.travel_times` is written every five minutes for both directions (11,790 rows through 13:35 SGT). That is enough history to start a 30-minute hold-out. It is not yet a long seasonal record.
2. **The live forecast is 30 minutes of the Maps series, from Maps lags.** `v_training_set` does not join weather or camera congestion. `v_forecast_recent` publishes `forecast_30min_min` using persistence or `lin_h30`. `xgb_h30` is trained and unused by that view. `y_60` is computed and unused.
3. **Vision and weather are historical side stores.** Camera detection tables last changed on 18 Jul. Weather and rainfall last changed on 3 Sep. Image metadata last changed on 13 Sep (368,905 rows). `traffic_images.labels` is empty; labelling is in Roboflow.
4. **Detection in git and detection in BigQuery are different clocks.** `camdetect` on `main` is what Cloud Build deploys to `swiftbackend`. The BigQuery camera tables are not receiving those calls. Directional geometry in the repo covers camera 2701 only.
5. **Nothing in the repo scores a model.** `model_registry` has two rows from 12 Sep. No evaluation script is checked in.

---

## Claims register

Use this when drafting the proposal, the first presentation (30 Sep 2026), or the final report (31 Oct 2026). Weights are in [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md). Region layout is not a graded story.

| Say this | Do not say this | Until |
| --- | --- | --- |
| Woodlands only; cameras 2701 and 2702 are in scope | The live divider covers 2702 | 2702 geometry exists and is demoed |
| Layer A measures detection; Layer B measures duration | Queue counts predict crossing time | A joined model is scored in [evaluation.md](evaluation.md) |
| Maps durations log live; serve horizon is 30 minutes | A 24-hour forecast is running | A scored horizon beyond 30 minutes is in the results table |
| `lin_h30` or persistence is what the serve view can emit | `xgb_h30` is the production model, or we beat Google | The harness fills MAE/RMSE against persistence and a registry row cites that window |
| Weather and congestion views exist | They feed the model | `v_training_set` references them |
| Roboflow Public can export a dataset version; weight download is Core | Empty `traffic_images.labels` means export is impossible | — |
| Cloud Build deploys `camdetect` to `swiftbackend` on push to `main` | The project has test CI, or the forecast is deployed from git | A pipeline runs tests and deploys the forecast |
| Firebase Hosting is planned | The UI is live | Hosting exists in the project |
| <= 15 min MAE is the target | <= 15 min MAE was achieved | A results cell shows it |

Paste-ready status for a slide, tied to this query:

> **Progress (26 Sep 2026, ~13:40 SGT):** Maps durations log into BigQuery every 5 minutes (11,790 rows, both directions, through 13:35 SGT). The served forecast is 30 minutes ahead (`v_forecast_recent`: persistence or `lin_h30`). Training features are Maps lags and time-of-day. Weather and camera-2701 congestion views exist and are not joined. Camera tables last moved 18 Jul; weather tables 3 Sep. Layer A labels are in Roboflow. Push to `main` redeploys `camdetect` only. A 24-hour forecast and <= 15 min MAE remain targets.

---

## Discussion points for the report

**Horizon.** Three bins ahead is implemented (`y_30`). Six bins (`y_60`) is labelled and not served. Twenty-four hours is the product sentence and has no training target yet. The report should show the 30-minute result first, then say what extra history and features a longer horizon needs.

**Unused features.** Leaving rain and queue depth out is a limitation of the current model, not a finding that they do not matter. The next experiment is to join `v_weather_features_10min` and `cam2701.v_congestion_index_10min` and read the change in MAE. Camera 2702 still needs a congestion view before it can enter that join.

**Two clocks for vision.** A demo of `swiftbackend` shows a live frame. A chart of `Cam2701` shows history that stopped on 18 Jul. The report should say which one the figure is.

**Git is behind the project.** The fetcher, the views, and both BigQuery ML models are not in the tree. Reproducing the forecast from the repo alone is not possible today. Checking those definitions in is part of making the MVP runnable for grading, separate from the metric.

**What to defer.** ResNet, Firebase, and holiday calendars can wait. A region-consolidation story does not move the grade. The order of work is in [roadmap.md](roadmap.md).
