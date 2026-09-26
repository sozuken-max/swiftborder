# Findings and discussion

**Report section:** findings and discussion  
**Source of truth:** GCP project `swiftborder`  
**GCP facts:** [inventory.md](inventory.md) (dated snapshot). **Methods:** [evaluation.md](evaluation.md). **Design:** [architecture.md](architecture.md).

These are findings the query supports, plus **offline performance** on a full `travel_times` export (see [evaluation.md](evaluation.md)). Live GCP facts still come from [inventory.md](inventory.md).

---

## Findings the query supports

1. **A duration label is accumulating.** `causeway.travel_times` is written every five minutes for both directions (see [inventory.md](inventory.md) for row counts and latest `observed_at`). That is enough history to start a 30-minute hold-out. It is not yet a long seasonal record.
2. **The live forecast is 30 minutes of the Maps series, from Maps lags.** `v_training_set` does not join weather or camera congestion. `v_forecast_recent` publishes `forecast_30min_min` using persistence or `lin_h30`. `xgb_h30` is trained and unused by that view. `y_60` is computed and unused.
3. **Vision and weather are historical side stores.** Camera detection tables last changed on 18 Jul. Weather and rainfall last changed on 3 Sep. Image metadata last changed on 13 Sep (368,905 rows). `traffic_images.labels` is empty; labelling is in Roboflow.
4. **Detection in git and detection in BigQuery are different clocks.** `camdetect` on `main` is what Cloud Build deploys to `swiftbackend`. The BigQuery camera tables are not receiving those calls. Directional geometry in the repo covers camera 2701 only.
5. **Layer B has scored numbers, significance tests, and committed comparison plots.** Report sources: [`eval/runs/report/`](../eval/runs/report/) (`run.json` + PNGs; promote via [`promote_report_run.py`](../eval/promote_report_run.py)). **Offline (60 min):** hold-out paired MAE gain vs persistence is **large and significant** (`offline/holdout-mae-diff.png`). **Serve audit (30 min):** `xgb_h30` leads on combined MAE but **does not** clear block-bootstrap significance on `both` (`bqml/mae-diff-ci.png`). Narrative: [evaluation.md](evaluation.md#plot-analysis-2026-09-26-run). Layer A still has no scoring script.

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
| Changes to `camdetect` runtime files on `main` run pytest then deploy `swiftbackend` | The whole repo has CI, or the forecast SQL/models deploy from git | Forecast views and BQML are committed and a separate pipeline exists |
| Firebase Hosting is planned | The UI is live | Hosting exists in the project |
| <= 15 min MAE is the product target | <= 15 min MAE is proven on an independent wait-time study | Offline Maps-series MAE ~2–4 min on some slices (see evaluation) |
| Offline eval used a full `travel_times` CSV export | Layer B numbers came from live BigQuery at slide time | Methods section names data source (export vs `layer_b.py`) |
| Offline sklearn XGB is **JB → SG** (`jb_to_woodlands`) only | Offline XGB covers **SG → JB** or both causeway directions | A second `route_id` is scored in [evaluation.md](evaluation.md) |

Paste-ready status for a slide (refresh counts from [inventory.md](inventory.md) before the deck):

> Maps durations log every five minutes (both directions in BQ). **Serve:** 30 minutes (`v_forecast_recent`: persistence or `lin_h30`). **30 min harness (26 Sep):** both `SG_TO_MY` / `MY_TO_SG` via `layer_b.py`. **Offline 60 min** — **JB → SG only** (`jb_to_woodlands`); SG → JB **not** in the notebook; XGB MAE **~2.2 min** vs persistence **~3.6 min** (22–24 Sep slice). Weather/congestion **not** joined. Layer A in Roboflow; no Layer A metric table. **24-hour** forecast remains a target.

---

## Discussion points for the report

**Horizon.** Three bins ahead is implemented (`y_30`). Six bins (`y_60`) is labelled and not served. Twenty-four hours is the product sentence and has no training target yet. The report should show the 30-minute result first, then say what extra history and features a longer horizon needs.

**Unused features.** Leaving rain and queue depth out is a limitation of the current model, not a finding that they do not matter. The next experiment is to join `v_weather_features_10min` and `cam2701.v_congestion_index_10min` and read the change in MAE. Camera 2702 still needs a congestion view before it can enter that join.

**Two clocks for vision.** A demo of `swiftbackend` shows a live frame. A chart of `Cam2701` shows history that stopped on 18 Jul. The report should say which one the figure is.

**Git is still behind ingest.** Maps fetcher, `causeway.travel_times` loader, and camera/weather table writers are not in the tree. **Views, BQML, and serve SQL are in** [sql/](../sql/) (exported 2026-09-26). You can recreate Layer B logic in a project that already has `travel_times`, but you cannot replay ingest from this repo alone.

**What to defer.** ResNet, Firebase, and holiday calendars can wait. A region-consolidation story does not move the grade. The order of work is in [roadmap.md](roadmap.md).
