# Evaluation and reasoning

**Report section:** performance (methods now; numbers when a harness writes them)  
**Source of truth for live resources:** GCP project `swiftborder`  
**Live resources:** [inventory.md](inventory.md). **Rubric:** methods and metrics are graded. This file is the methods chapter. Result cells stay `pending` until a harness run is recorded here. Claims that depend on those cells belong in the final report, not in the proposal.

Diagrams: [images/eval-layer-a.png](images/eval-layer-a.png), [images/eval-layer-b.png](images/eval-layer-b.png).

---

## What would count as success

The product intent is a Woodlands-only forecast of causeway crossing time, up to 24 hours ahead, with mean absolute error at or below 15 minutes.

The live serve path is narrower than the product sentence. `v_forecast_recent` emits a **30-minute** forecast of Google Maps `duration_in_traffic`. Training features are lags and time-of-day from that same series. Layer A counts vehicles in a camera frame. Those counts are occupancy, not crossing time.

<= 15 min MAE and the 24-hour horizon stay **targets**.

---

## Reasoning

### The series is Maps' current estimate

`causeway.travel_times.duration_in_traffic_sec` is the Distance Matrix estimate at observation time. There is no second ground truth in the project (no probe-vehicle wait, no checkpoint timestamp). Every Layer B score is skill **on the Maps series**.

`v_training_set` defines:

- `y_persistence` as duration in the current 10-minute bin
- `y_30` and `y_60` as the duration 3 and 6 bins later (`LEAD`)

A model beats persistence when its error on a **future** Maps reading is lower than carrying the current reading forward. That is a real forecast test.

It is not, by itself, "we beat Google." Maps is the label generator. The project does not store a separate Google forecast at a 30-minute or 24-hour horizon. Wording for the report: **skill over persistence on the Maps duration series**. Reserve "beat Google" for a comparison against an independent wait measurement, which this dataset does not contain.

### Layer A cannot carry the product metric

A strong detector can still leave Layer B wrong, because a queue visible in one frame is not the time to cross. Camera refresh is about once a minute, so a frame cannot be turned into a flow count. Layer A is graded on detection quality (mAP, precision, recall, count-error, day versus night). Layer B is graded on duration error. The report should keep those tables separate.

### Weather and congestion are hypotheses, not current inputs

`v_weather_features_10min` and `cam2701.v_congestion_index_10min` exist. `v_training_set` does not reference them. The protocol includes them because the proposal's claim is a multi-source forecast. Until they are joined, a results table that credits rain or queue depth is unsupported. The honest current feature set is Maps lags, rolling means, and time-of-day.

### Significance (paired error differences)

Point forecasts are **paired** on the same `(direction, time)` rows. For each observation, compute
`d_i = |y_i - ŷ_i^A| - |y_i - ŷ_i^B|` in minutes (negative means A has lower absolute error than B).

| Method | Role |
| --- | --- |
| **Block bootstrap CI** on `mean(d)` | Primary test when bins are serially correlated (default block **6** × 10 min = 1 h for BQML; **12** × 5 min = 1 h for offline XGB). |
| **Paired t-test** on `d_i` | Supplementary; treat as approximate when autocorrelation is strong. |

**Interpretation (α = 0.05):** call challenger **better** than reference when the bootstrap CI for `mean(d)` lies entirely below 0 (one-sided improvement on MAE). A lower aggregate MAE with a CI that crosses 0 is **not** significant.

**Code:** [`eval/significance.py`](../eval/significance.py). BQML harness: `python layer_b.py --significance`. Offline XGB: `holdout_significance_vs_persistence()` in [`eval/timeseries_xgb.py`](../eval/timeseries_xgb.py).

### Why these baselines

| Comparison | Why it is in the protocol |
| --- | --- |
| Persistence (`y_t` predicts `y_{t+h}`) | Naive forecast. Required in every Layer B table. |
| `lin_h30` (linear regression) | The model `v_forecast_recent` actually calls. |
| `xgb_h30` (boosted tree) | Trained on 12 Sep and not called by the serve view. The harness decides whether it earns the registry row. |
| sklearn XGB (`eval/timeseries_xgb.py`) | Exploratory **60-minute** horizon on raw 5-minute series (`jb_to_woodlands` only). Not production; optional notebook backtests in [eval/notebooks/](../eval/notebooks/). |
| Day / night, direction, time-of-day | The border is not one regime. A single MAE can hide a peak-hour failure. |

`model_registry` has two rows (see [model_registry_reference.sql](../sql/bigquery/traffic_prediction/model_registry_reference.sql)). Training features and OPTIONS for `lin_h30` / `xgb_h30` are in [sql/bigquery/traffic_prediction/](../sql/bigquery/traffic_prediction/). The report should fill registry rows from harness scores (direction, serving model, reason, test window, date), not from preference alone.

---

## Techniques this evaluation is accountable for

The module asks for at least three of the categories below. Hybrid or ensemble is available as a fourth once the harness compares a blend. The serve view today selects `lin_h30` or persistence, so a blend is not yet demonstrated.

| Category | Where it shows up | What the harness must report |
| --- | --- | --- |
| Supervised learning | Roboflow labels; regression of future Maps duration | Hold-out detection metrics; time-based hold-out for `y_30` / `y_60` |
| Machine learning / deep learning | YOLO via Roboflow; BigQuery ML `lin_h30` and `xgb_h30` | Same hold-outs, one row per candidate |
| Intelligent sensing | LTA frames to directional occupancy (camera 2701 geometry in `camdetect`) | Count-error and day/night, not crossing time |
| Hybrid / ensemble | Not in the serve path | Only if a blend is scored against the single models |

---

## Layer A — vision

**Question:** On held-out frames, how well does a detector localize vehicles and recover directional counts?

**Candidates:** pretrained baseline, fine-tuned YOLO, optional ResNet. Same hold-out for all three.

**Procedure:**

1. Export a Roboflow dataset version and hold out frames. Do not train on that hold-out.
2. Score each candidate.
3. Report the metrics below, split by day and night.
4. Record scores before promoting a serving checkpoint. Roboflow remains the serve path. Public plan: dataset export after a version is allowed; manual weight download is Core.

**Status:** labels are in Roboflow. `traffic_images.labels` has 0 rows. `traffic_images.metadata` is populated and is not a label table. There is **no** checked-in Layer A scoring script; only the protocol and table below.

![Layer A evaluation](images/eval-layer-a.png)

### Results table (fill from the harness)

| Candidate | Split | mAP | Precision | Recall | Count-error | Day | Night |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | hold-out | pending | pending | pending | pending | pending | pending |
| Fine-tuned YOLO | hold-out | pending | pending | pending | pending | pending | pending |
| ResNet (optional) | hold-out | pending | pending | pending | pending | pending | pending |

---

## Layer B — crossing-time forecast

**Question:** On a later window of the Maps series, how far is each forecast from the observed duration, compared with persistence?

**Procedure:**

1. Build 10-minute bins from `causeway.travel_times` (`status = OK`). Add weather and congestion only after those views are joined. Add holiday flags only after a calendar exists.
2. Fit candidates on an earlier window. Always include persistence. Include `lin_h30` and `xgb_h30`.
3. Score MAE and RMSE on the held-out window, by direction (`SG_TO_MY`, `MY_TO_SG`) and by time-of-day (morning peak, evening peak, other).
4. Write the chosen serving model to `model_registry` with reason, test window, and date.
5. Leave <= 15 min MAE as a target until a cell in the table supports it.

**Status:** see [inventory.md](inventory.md) for live row counts and serve path. Production serve is **30 minutes** (`v_forecast_recent`: `lin_h30` or persistence). **Recorded runs (2026-09-26 SGT):** offline sklearn XGB on a cached full export of `causeway.travel_times`; BQML candidates via read-only `layer_b.py` on `v_training_set` (trailing 3-day hold-out).

![Layer B evaluation](images/eval-layer-b.png)

### Offline evaluation (full `travel_times` export)

**Data:** canonical table `swiftborder.causeway.travel_times` — BigQuery download with CSV cache ([`eval/data/README.md`](../eval/data/README.md)). **Export size (this run):** 11,842 rows canonical; 5,921 rows after `jb_to_woodlands` filter. **Route:** `jb_to_woodlands` (`SG_TO_MY`). **Code:** [`eval/timeseries_xgb.py`](../eval/timeseries_xgb.py), [`eval/timeseries_lstm.py`](../eval/timeseries_lstm.py) (LSTM optional; not scored here), [`eval/notebooks/causeway_xgb_timeseries.ipynb`](../eval/notebooks/causeway_xgb_timeseries.ipynb).

**Chronological 80/20 hold-out** on the 5-minute series (sklearn XGB, 60-minute horizon `H=12`):

| Metric | Value | Notes |
| --- | --- | --- |
| Test RMSE | **3.24 min** | Same-scale naive persistence not in this row; see day backtest |
| Dominant feature | `target_lag_1` (~70% importance) | Maps duration is highly persistent at 5-minute cadence |

**Fixed-day backtest** (60 min ahead; mean RMSE/MAE in minutes over 22–24 Sep 2026):

| Method | RMSE (min) mean | MAE (min) mean | vs persistence MAE (3.60) |
| --- | --- | --- | --- |
| Persistence T-60 | 5.35 | 3.60 | — |
| Naive D-1/D-7 blend | 4.30 | 2.72 | Better |
| **XGB (actual lag window)** | **3.57** | **2.18** | **Better** |
| XGB (blend window) | 4.46 | 2.81 | Better |
| XGB average of windows | 3.84 | 2.39 | Better |

This path is **not** the live serve model (`v_forecast_recent` is 30-minute BQML). It shows that a richer lag + calendar model on the **same Maps series** can beat persistence at **60 minutes** on this export.

### BQML serve-path harness (30 minutes; optional)

[`eval/layer_b.py`](../eval/layer_b.py) scores **persistence**, `lin_h30`, and `xgb_h30` on a trailing hold-out of `traffic_prediction.v_training_set` via **read-only BigQuery** (both directions, peak slices). Use this to audit **production** models; it does not re-read the offline CSV.

```bash
cd eval && pip install -r requirements.txt && python layer_b.py --project swiftborder --holdout-days 3
```

### Results table (production models at 30 min)

Headline rows use `layer_b.py --holdout-days 3` (n=866 per direction slice at `both` / `all`). Full peak and direction breakdown is in the harness stdout.

| Candidate | Horizon | Test window | MAE (min) | RMSE (min) | Persistence MAE | Direction | Time of day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 30 min | 2026-09-23 06:50 UTC .. 2026-09-26 06:50 UTC | 2.62 | 3.86 | — | both | all |
| `lin_h30` | 30 min | same | 2.89 | 3.83 | 2.62 | both | all |
| `xgb_h30` | 30 min | same | **2.51** | **3.76** | 2.62 | both | all |
| Joined model (weather + congestion) | 30 min | pending | pending | pending | pending | both | all |
| sklearn XGB (offline export) | **60 min** | full export; 80/20 + 22–24 Sep | **2.18** (XGB actual-window MAE mean) | **3.57** (RMSE mean) | 3.60 | `jb_to_woodlands` | all |
| Any 24 h model | 24 h | pending | pending | pending | pending | both | all |

On this 3-day window, **`xgb_h30` beats persistence on combined MAE**; **`lin_h30` does not** (worse than persistence overall, though better on `SG_TO_MY` alone). Serve view still selects `lin_h30` or persistence — registry promotion should cite this window if `xgb_h30` is claimed. Do not mix **60 min** offline XGB and **30 min** BQML in one headline MAE.

### Comparison plots (per run)

Regenerate after a new harness run:

```bash
cd eval
pip install -r requirements-dev.txt
python generate_comparison_plots.py --bqml
# or: python layer_b.py --holdout-days 3 --significance --plots
```

**Report citation target (committed):** [`eval/runs/report/`](../eval/runs/report/) — promote with `python eval/promote_report_run.py` when tables and prose match a harness run. Ephemeral runs live under **`eval/runs/<run_id>/`** (gitignored). Each run includes:

- **`run.json`** — datasets, models, horizons, significance summaries, artifact paths
- **`README.md`** — human-readable copy of the same metadata
- **`offline/`** — 60 min sklearn figures (when generated)
- **`bqml/`** — 30 min BQML figures (when generated)

The latest local run id is in [`eval/runs/LATEST.json`](../eval/runs/LATEST.json). Layout and gitignore rules: [`eval/runs/README.md`](../eval/runs/README.md).

| Path (under `eval/runs/report/`) | What it shows |
| --- | --- |
| `offline/backtest-mae.png` | Mean MAE by method across 22–24 Sep backtest days |
| `offline/holdout-sample.png` | Tail of chronological hold-out: actual vs XGB vs persistence T-60 |
| `offline/holdout-mae-diff.png` | Bootstrap CI for mean paired AE difference (offline hold-out) |
| `bqml/mae-by-direction.png` | BQML candidates vs persistence MAE by direction |
| `bqml/mae-diff-ci.png` | Paired MAE difference vs persistence for `lin_h30` / `xgb_h30` by direction |

Dataset and model details for the committed snapshot: [`eval/runs/report/run.json`](../eval/runs/report/run.json).

Protocol diagrams for methods (not scored runs) remain [`images/eval-layer-a.png`](images/eval-layer-a.png) and [`images/eval-layer-b.png`](images/eval-layer-b.png).

**How to read the forest plots:** each point is `mean(|err_ch| - |err_ref|)` in minutes; error bars are block-bootstrap 95% CIs. Intervals entirely left of zero mean the challenger has **significantly** lower MAE than persistence at alpha = 0.05 (see [Significance](#significance-paired-error-differences)).

### Plot analysis (2026-09-26 run)

**Offline 60 minutes (`jb_to_woodlands`).** The backtest bar chart ranks methods the same way as the table: **XGB (actual lag window)** has the lowest mean MAE (~2.2 min), ahead of naive D-1/D-7 blend and well ahead of persistence T-60 (~3.6 min). The hold-out time-series panel shows XGB tracking sharp moves in the Maps duration series more closely than persistence; gaps widen when the series turns after a plateau. On the full chronological hold-out (n = 1,138 supervised rows), the paired MAE-difference forest plot sits **far left of zero** (mean improvement ~3.7 min vs persistence T-60; bootstrap CI excludes zero). That is a much stronger separation than the 3-day BQML window — different horizon, route filter, and model — but it supports the same story: **rich lags beat naive carry-forward on this label**.

**BQML 30 minutes (both directions, trailing 3 days).** The direction bar chart shows **regime split**: persistence MAE is lower on `MY_TO_SG` than on `SG_TO_MY`, and both learned models struggle most on **morning peak** rows (see harness tables). `xgb_h30` has the lowest combined MAE, but the forest plot shows the **combined** (`both`) CI for `xgb_h30` **crosses zero** — the headline MAE gain (~0.1 min) is not significant under block bootstrap. `lin_h30` is **significantly worse than persistence combined** (CI entirely right of zero) while **significantly better on `SG_TO_MY` alone** — a pattern visible in the per-direction bars and worth stating explicitly in the report (do not quote a single "both" MAE without the direction plot). `xgb_h30` is **significantly better than persistence on `MY_TO_SG`** in this window; on `SG_TO_MY` the point estimate favors `xgb_h30` but the CI still overlaps zero.

**Reporting takeaway:** use the **bar charts** for magnitude and direction splits; use **forest plots** before claiming "model A beats persistence." The offline 60 min path can support a strong supervised-learning slide; the live 30 min BQML path supports **auditing serve candidates** with honest significance qualifiers.

---

## Principal risk

**Counts are not crossing duration.** Report Layer A and Layer B separately. A low count-error does not imply a low MAE.

**The label is not an independent clock.** Report skill against persistence on the Maps series. Do not write "we beat Google" from that comparison alone.
