# Roadmap

**Source of truth:** GCP project `swiftborder`. Re-query it before starting a step that depends on row counts, joins, or what is served.  
**GCP snapshot:** [inventory.md](inventory.md)  
**Dates (briefing; confirm on Canvas):** first presentation **30 Sep 2026, 18:30–22:30**; final deliverables **31 Oct 2026**.  
**Rubric:** [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md). Methods, a runnable system, and an honest report are the graded work. Region layout is not.

**Active work plan:** [plan-eval-integrity.md](plan-eval-integrity.md) (evaluation fixes, significance upgrades, joined-feature experiment, Layer A scorer, code hardening). Its task table is the current execution order; the steps below are the report-level milestones.

The report describes **one system at two depths** (high-level, then detailed). This page is the sequence that gets the empty tables in [evaluation.md](evaluation.md) filled and the claims in [findings.md](findings.md) updated. It is not a second architecture.

---

## Already true

- Maps durations log every 5 minutes into `causeway.travel_times` (both directions; counts in [inventory.md](inventory.md)).
- `v_forecast_recent` serves a 30-minute forecast: persistence or `lin_h30`. `xgb_h30` is trained and not called. `y_60` is computed and not served.
- Weather and camera-2701 congestion views exist and are not joined.
- `camdetect` runtime changes on `main` run pytest then deploy `swiftbackend` via Cloud Build. Camera 2701 has a dividing line. BigQuery camera tables last moved 18 Jul.
- Labels live in Roboflow. `traffic_images.labels` is empty. `traffic_images.metadata` is not.
- Layer B is scored and promoted to [`eval/runs/report/`](../eval/runs/report/): production models at 30 min on 13–30 Sep (`layer_b.py`), the joined weather experiment (`joined.py`), and offline 60-min XGBoost (`timeseries_xgb.py`), all with significance tests (Diebold–Mariano on error differences, Holm correction across models). Numbers are in [evaluation.md](evaluation.md).
- Weather history for 5–30 Sep is fetched (`Causeway/`) and appended to BigQuery; weather gave no significant gain. The Layer A scorer exists (`eval/layer_a.py`); no export is scored.
- Also scored in `report/`: LSTM, GRU and a patch Transformer at 60 min (none beats XGBoost), a fuzzy light / moderate / heavy forecast (the XGB → fuzzy hybrid is best), and ensembles / hybrids of the 30-min models (no gain over a daily-refit XGBoost, which beats the served forecast).
- Report drafts exist: design, evaluation, findings, inventory (2026-10-01).

---

## Sequence

Do these in order. A later step that needs a number waits on the harness.

### 1. First presentation (by 30 Sep)

**Done (30 Sep).** Kept for the rules that still apply to later decks.

**Done when** the Zoom deck states goals, data, techniques, and progress. Any MAE on a slide must come from `eval/runs/report/run.json` with its window and baseline.

| Use | From |
| --- | --- |
| High-level picture | [architecture.md](architecture.md) |
| What the metrics will be, and why persistence is the baseline | [evaluation.md](evaluation.md) |
| What may be said aloud | Claims register in [findings.md](findings.md) |

Do not use removed legacy proposal/target PNGs in the deck. Say the horizon in production is 30 minutes and that 24 hours and <= 15 min MAE are targets.

### 2. Layer B evaluation — publish and align horizons

**Done (2026-10-01):** 30-min production models on the fixed 13–30 Sep window, offline 60-min XGBoost (after the baseline and leakage fixes), and the joined weather experiment. Numbers are in [evaluation.md](evaluation.md).

**Remaining**

1. Keep slide/report wording distinct: **60 min offline XGB (one route)** vs **30 min production serve (both directions)**.
2. Before the final report, extend the window (more days, more rain events) and re-run `generate_comparison_plots.py --bqml --joined` from a clean tree, then promote (see [eval/README.md](../eval/README.md#reproduce-the-report-run)).
3. Deep-learning forecasters are scored and not served; see [deep-learning-assessment.md](deep-learning-assessment.md) for when to revisit.

**Done when** the report cites data source, horizon, and persistence baseline for each table. Do not write "we beat Google." Maps is the label.

### 3. Make the forecast re-runnable from git

**In git (2026-09-26):** view DDL and BQML scripts under [sql/](../sql/). See [sql/README.md](../sql/README.md) for apply order.

**Remaining:** Maps ingest and any training refresh automation. BQML files reconstruct training from `bq show --model`; confirm with a dry run in a dev dataset before overwriting production models.

**Done when** a teammate can apply `sql/` against a project that already has `causeway.travel_times` and get the same views, models, and serve path as inventory describes.

### 4. Layer A harness

**How**

1. Export a Roboflow dataset version (Public allows this; weight download is Core) into `eval/data/layer_a/`. Score the `test` split only.
2. Run [`eval/layer_a.py`](../eval/layer_a.py) for the served (fine-tuned) model (`--predict-with-roboflow` uses credits) and, if available, a pretrained baseline. ResNet only if time remains.
3. Copy mAP, precision, recall, count error, and day versus night into the Layer A table.

**Done when** Layer A has numbers that are not crossing-time numbers. Keep the two tables separate.

### 5. Join the features the views already hold

**Done (weather, offline):** [`eval/joined.py`](../eval/joined.py) joins fresh data.gov.sg rainfall and forecasts; no significant MAE gain on 13–30 Sep. The BigQuery views were not used: at run time the weather tables ended on 31 Aug (1–30 Sep was appended on 1 Oct, see [inventory.md](inventory.md)), camera tables end on 18 Jul, and both sit in a different location from `traffic_prediction`.

**Remaining:** camera 2701 counts for 5–30 Sep (no Layer A output overlaps the Maps window; see [evaluation.md §7](evaluation.md#7-is-layer-a-output-a-meaningful-layer-b-input)) via [`eval/backfill_camera_counts.py`](../eval/backfill_camera_counts.py) (Roboflow credits: dry run, 50-call pilot, then budgeted run), then re-run `joined.py`. Ship a join into `v_training_set` only if MAE moves.

Camera 2702 has detections and no congestion view. Add that view before claiming both cameras feed the forecast. The dividing line for a live 2702 demo is a geometry change in `camdetect`, separate from the historical table.

### 6. Horizon, only after step 2

`y_60` is already in `v_training_set`. Score it with the same harness before serving it. A 24-hour target needs a longer lead and enough history to hold out; the series started on 6 Sep 2026, so a 24-hour model is a late experiment, not the first result.

**Done when** any horizon beyond 30 minutes has its own row, or the findings state that the served horizon stays 30 minutes.

### 7. Final report and MVP (by 31 Oct)

**How**

1. Refresh [inventory.md](inventory.md) from a new query of project `swiftborder`.
2. Write the performance section from the filled tables. Write findings from those cells, including negative results.
3. Demo two things that exist: a directional count from `swiftbackend` (camera 2701), and the 30-minute forecast (`v_forecast_recent` or the registry model).
4. Submit the team ZIP, the video, and each member's reflection. Peer review uses the course system.

**Done when** the runnable path and the report cite the same models and the same test window.

---

## From the codebase (in parallel with steps 2–3)

The eval sequence above does not by itself put the deployed fetcher or the forecast SQL into git.

| Gap | Why it matters | What to do |
| --- | --- | --- |
| No `cloudbuild.yaml` in git | The `swiftbackend` trigger is inline in Cloud Build. It runs **pytest** before deploy; `includedFiles` is runtime paths only (not test-only files). | Treat the live trigger as the source of truth (`gcloud builds triggers describe 76bbca35-c1b4-4836-9f34-d7adda53ea17 --project=swiftborder`). Behavior is documented in [camdetect/README.md](../camdetect/README.md). History: [CHANGELOG.md](../CHANGELOG.md). Do not add a second trigger. |
| Weather load is manual | `Causeway/load_bigquery.py` appends to `rainfall` / `weatherforecast` (snapshot first, refuses overlaps). Nothing schedules it, and the original pipeline that filled the tables to 31 Aug is still outside the repo. | Re-run fetchers plus the loader before a report refresh. Add a Cloud Run job or Scheduler only as an approved deploy. |
| Direction names differ | `camdetect` emits `SG-MY` / `MY-SG`. The congestion view expects `to_JB` / `to_Woodlands` and emits `SG_TO_MY` / `MY_TO_SG`. | Map them in the join (step 5). Do not treat the strings as already aligned. |
| Camera 2701 line only | 2702 detections become `Unknown`. | Add a line only after it is calibrated on a real frame. |

## Leave until the tables exist

| Item | Why it waits |
| --- | --- |
| Firebase Hosting | Not in the project. The grade does not require a hosted UI if the demo runs. |
| Holiday calendars | No calendar table yet. Add only if a residual error looks like a public holiday. |
| ResNet | Optional third detector. It does not unblock Layer B. |
| Serving a better 30-min model | A daily-refit XGBoost (or the rolling stack) beats the served registry forecast by ~0.27 min on 13–30 Sep ([evaluation.md §6](evaluation.md#6-ensembles-and-hybrids-of-the-layer-b-models-evalensemblepy)); blending on top adds nothing. Serving it needs a daily retrain job and a `v_forecast_recent` change: an approved deploy. |
| Region consolidation | Not graded. |

---

## After each step

Update the claims register in [findings.md](findings.md) so the slide sentence matches the query and the tables. If the project and this roadmap disagree, the project wins.
