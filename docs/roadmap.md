# Roadmap

**Source of truth:** GCP project `swiftborder`. Re-query it before starting a step that depends on row counts, joins, or what is served.  
**GCP snapshot:** [inventory.md](inventory.md)  
**Dates (briefing; confirm on Canvas):** first presentation **30 Sep 2026, 18:30–22:30**; final deliverables **31 Oct 2026**.  
**Rubric:** [grading/nus-iss-practice-module.md](grading/nus-iss-practice-module.md). Methods, a runnable system, and an honest report are the graded work. Region layout is not.

The report describes **one system at two depths** (high-level, then detailed). This page is the sequence that gets the empty tables in [evaluation.md](evaluation.md) filled and the claims in [findings.md](findings.md) updated. It is not a second architecture.

---

## Already true

- Maps durations log every 5 minutes into `causeway.travel_times` (both directions; counts in [inventory.md](inventory.md)).
- `v_forecast_recent` serves a 30-minute forecast: persistence or `lin_h30`. `xgb_h30` is trained and not called. `y_60` is computed and not served.
- Weather and camera-2701 congestion views exist and are not joined.
- `camdetect` runtime changes on `main` run pytest then deploy `swiftbackend` via Cloud Build. Camera 2701 has a dividing line. BigQuery camera tables last moved 18 Jul.
- Labels live in Roboflow. `traffic_images.labels` is empty. `traffic_images.metadata` is not.
- [`eval/layer_b.py`](../eval/layer_b.py) scores Layer B read-only; [`eval/README.md`](../eval/README.md).
- Report drafts exist: design, evaluation reasoning, findings, inventory. Layer B result cells are still `pending` until a harness run is recorded.

---

## Sequence

Do these in order. A later step that needs a number waits on the harness.

### 1. First presentation (by 30 Sep)

**Done when** the Zoom deck states goals, data, techniques, and progress without a measured MAE.

| Use | From |
| --- | --- |
| High-level picture | [architecture.md](architecture.md) |
| What the metrics will be, and why persistence is the baseline | [evaluation.md](evaluation.md) |
| What may be said aloud | Claims register in [findings.md](findings.md) |

Do not use removed legacy proposal/target PNGs in the deck. Say the horizon in production is 30 minutes and that 24 hours and <= 15 min MAE are targets.

### 2. Layer B harness — run and publish numbers

**In git:** [`eval/layer_b.py`](../eval/layer_b.py) (hold-out window, persistence / `lin_h30` / `xgb_h30` at 30 minutes, direction and peak slices). Offline tests: `python -m pytest` in `eval/`.

**Remaining**

1. Run `python layer_b.py` against project `swiftborder` (see [eval/README.md](../eval/README.md)).
2. Paste MAE/RMSE into the Layer B table in [evaluation.md](evaluation.md). Wording: skill against persistence on the Maps series.
3. Optionally update `model_registry` only with an explicit, reviewed write path (the script does not do this by default).
4. Update the claims register only where a cell now supports the sentence.

**Done when** the report can cite a filled row with a stated test window. Do not write "we beat Google" from that table. Maps is the label.

### 3. Make the forecast re-runnable from git

**In git (2026-09-26):** view DDL and BQML scripts under [sql/](../sql/). See [sql/README.md](../sql/README.md) for apply order.

**Remaining:** Maps ingest and any training refresh automation. BQML files reconstruct training from `bq show --model`; confirm with a dry run in a dev dataset before overwriting production models.

**Done when** a teammate can apply `sql/` against a project that already has `causeway.travel_times` and get the same views, models, and serve path as inventory describes.

### 4. Layer A harness

**How**

1. Export a Roboflow dataset version (Public allows this; weight download is Core). Hold out frames.
2. Score a pretrained detector and the fine-tuned YOLO on that hold-out. ResNet only if time remains after the tables below are filled.
3. Record mAP, precision, recall, count-error, and day versus night in the Layer A table.

**Done when** Layer A has numbers that are not crossing-time numbers. Keep the two tables separate.

### 5. Join the features the views already hold

**How:** add `v_weather_features_10min` and `cam2701.v_congestion_index_10min` to the training query. Re-score against the same hold-out and the same persistence baseline. Ship the join only if MAE moves.

**Done when** the "joined model" row in the Layer B table is filled, or the findings say the join did not help and the served model stays Maps-only.

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
| `Causeway/` writes CSV only | It does not load `rainfall` or `weatherforecast`. The BigQuery weather pipeline is still outside the repo. | Do not wire these scripts up as if they were that pipeline. Recover the loader with the same export procedure. |
| Direction names differ | `camdetect` emits `SG-MY` / `MY-SG`. The congestion view expects `to_JB` / `to_Woodlands` and emits `SG_TO_MY` / `MY_TO_SG`. | Map them in the join (step 5). Do not treat the strings as already aligned. |
| Camera 2701 line only | 2702 detections become `Unknown`. | Add a line only after it is calibrated on a real frame. |

## Leave until the tables exist

| Item | Why it waits |
| --- | --- |
| Firebase Hosting | Not in the project. The grade does not require a hosted UI if the demo runs. |
| Holiday calendars | No calendar table yet. Add only if a residual error looks like a public holiday. |
| ResNet | Optional third detector. It does not unblock Layer B. |
| Blended `lin_h30` + `xgb_h30` | Ensemble is a fourth technique. Score it only after the single models have rows. |
| Region consolidation | Not graded. |

---

## After each step

Update the claims register in [findings.md](findings.md) so the slide sentence matches the query and the tables. If the project and this roadmap disagree, the project wins.
