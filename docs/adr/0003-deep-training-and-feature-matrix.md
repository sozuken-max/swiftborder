# Train the deep forecasters locally, and let the trees pick the cells

## Status

Accepted

## Decision

Train on the existing Windows machine. Do not create a GCP training job for the LSTM, GRU, or patch Transformer.

The scored deep component is about 16k to 21k weights on about 5.7k training windows and finishes in 10 to 15 minutes on CPU ([evaluation.md](../evaluation.md), Reproduce). A GPU VM or a Vertex / Agent Platform custom job would bill image pull, startup, and a management fee on top of the machine, for a fit that already runs here, and it would not add day-blocks. The hold-out stays 5 day-blocks either way, so every Diebold-Mariano decision on this split stays **insufficient data**.

**Cost ceiling if a later rerun leaves this machine:** 10 USD, one short-lived CPU VM in `asia-southeast1`, deleted the same day. That cap is a project limit. This pass did not read a Singapore GPU SKU price (the billing catalog returned HTTP 403).

**Trigger for that rerun:** at least three months of 5-minute history, which is the condition in [deep-learning-assessment.md](../deep-learning-assessment.md) for a hold-out of 10 or more day-blocks, or a model that does not fit in memory on this machine. A wish for a GPU is not a trigger. Do not enable `aiplatform.googleapis.com` for this matrix. Do not add a region.

Use the pinned TensorFlow in `eval/requirements-notebook.txt` so a new cell can sit next to the section 4 seed MAEs. `eval/requirements-tf-gpu-windows.txt` is a Python 3.10 DirectML plugin on TensorFlow 2.10. It conflicts with that pin. This machine has an NVIDIA GeForce RTX 3070 (8,192 MiB, `nvidia-smi` on 2026-10-03). That GPU is enough for these weights. The DirectML venv is a different build, so it stays off the scored comparison.

`use_dwt` stays off for the promoted section 3 model until a promoted `run.json` says otherwise.

## What was checked in GCP

Read-only, 2026-10-03. The GCP account used for the read-only check had a different active gcloud project; every call passed `--project=swiftborder`. Nothing was created, enabled, or billed on purpose. `gcloud ai custom-jobs list` stopped at the disabled-API prompt and was not confirmed.

| Check | Command | Result |
| --- | --- | --- |
| Enabled APIs | `gcloud services list --enabled --project=swiftborder` | `aiplatform`, `notebooks`, and `ml` are absent. `compute.googleapis.com` is enabled. |
| Agent Platform / notebooks / ML API state | `gcloud services list --available --project=swiftborder` filtered to those three names | All three `DISABLED`. |
| Custom jobs | `gcloud ai custom-jobs list --project=swiftborder --region=asia-southeast1` | `SERVICE_DISABLED` (`aiplatform.googleapis.com` has not been used on this project). |
| VMs | `gcloud compute instances list --project=swiftborder` | No instances. |
| GPU types in the zone catalog | `gcloud compute accelerator-types list --project=swiftborder` filtered to `asia-southeast1-a/b/c` | `nvidia-tesla-t4` is listed in each of those zones, along with larger GPUs. None is attached: there is no VM. |
| Cloud Run jobs | `gcloud run jobs list --project=swiftborder` | Only `traffic-backfill` in `asia-southeast1` (image metadata, not a trainer). |
| Billing catalog | `GET https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus` | HTTP 403. No Singapore hourly rate is quoted here. |

The only trained models already in the project are BigQuery ML `lin_h30` and `xgb_h30` ([inventory.md](../inventory.md)). They are not these sequence models.

Agent Platform custom training, as published on the [pricing page](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing) read the same day: no minimum training duration, usage in 30-second increments, and a management fee in addition to the VM and any accelerator. The job is billed for its whole lifetime, including startup.

## Context

Section 4 already scored LSTM(64), GRU(64), and the patch Transformer on the offline 60-minute split: same 1,428 test rows as section 3, one route (`jb_to_woodlands`), three seeds, seed-mean forecast. Point estimates trail XGBoost. Every decision is insufficient data (5 day-blocks, below 10). Details and the "do not serve" decision: [deep-learning-assessment.md](../deep-learning-assessment.md).

The next matrix adds three things the section 3 table does not have, on that same split:

- Window features already sketched in [`eval/timeseries_xgb.py`](../../eval/timeseries_xgb.py): lags; lags plus a causal first difference and a 60-minute rolling mean; lags plus a per-window z-score Daubechies **db2** wavelet at level 2 (level 3 optional; **db4 is not the default**). Window mean and standard deviation stay as columns. `use_dwt` is false for the promoted model.
- Feature-importance agreement between XGBoost and a new Random Forest, on the training rows.
- A Layer A block crossed with those feature blocks: none, `camfc`, `mpfc`, `camfc+mpfc`.

Unpromoted A/B/C point estimates from a local `python timeseries_xgb.py` are in [eval/README.md](../../eval/README.md). They are not in `eval/runs/report/run.json`. This plan does not copy them into [evaluation.md](../evaluation.md).

## Protocol the harness must keep

Do not invent a new split.

| Item | Value |
| --- | --- |
| Cache | `eval/data/causeway_gdata.csv`. Do not pass `--refresh-bq`. A newer export moves the boundary. |
| Fingerprint the run must match | sha256 `13a119980c640061a2b7d9cc8cae7909787771314bf6a1391c7cfb492b9bbf07`, 3,484,144 bytes ([eval/README.md](../../eval/README.md)) |
| Route | `jb_to_woodlands` only (`MY_TO_SG`) |
| Config | `TimeSeriesConfig` defaults: `window_size` 36, `horizon_steps` 12 (60 min), `keep_lags` 12, `train_fraction` 0.8 |
| Boundary | Target time **2026-09-26 01:35** SGT. 5,710 train / 1,428 test. 5 day-blocks. |
| Tree seed | `random_state` 42, existing `XGBTrainConfig` |
| Deep seeds | `(0, 1, 2)` (`deep_forecast.DEFAULT_SEEDS`). Scored forecast is the seed mean. |
| Deep protocol | Unchanged: standardize on the fit rows, last 15% of training rows for early stopping, persistence-anchored target, Huber, AdamW. |
| Significance | Diebold-Mariano, day-block bootstrap, Holm, in a **new** family. Below 10 day-blocks the decision is insufficient data and the command prints point estimates only, which is what `compare_window_feature_sets` already does. |
| Where numbers go | A future `run.json` after promotion. Not this note, and not a hand edit of the report file. |

Recompute the section 3 column set (block S3, camera arm none) in the same run. It is the promoted model. If that row disagrees with section 3, the cache or the code drifted: stop, and do not publish the matrix.

Block C (lags plus wavelet) is a window comparison against A and B. S3 is a separate block so a lags-plus-wavelet MAE is not read as a challenge to the promoted section 3 model.

## Feature blocks

One supervised frame, built with `include_trends=True` and `use_dwt=True` (db2, level 2, periodization, per-window z-score). Extra columns do not change which origins exist, so the split stays the section 3 split. Subset columns per cell.

| Block | Columns | Role |
| --- | --- | --- |
| A | `target_lag_*` | Lags only |
| B | A + `trend_diff_1` + `trend_roll_mean` | Causal difference (one 5-minute bin) and the mean of the last 12 bins |
| C | A + `dwt_mean` + `dwt_std` + `dwt_a_*` / `dwt_d*_*` | Lags plus normalized db2 level 2 |
| S3 | lags + `duration_sec` + calendar (`hour`, `minute`, `dayofweek`, `day`, `month`, `nonworkday`) + `d1` + `d7` | Section 3 set. No trend columns. No wavelet columns. |
| S3T | S3 + the two trend columns | Trends on the promoted set |
| S3W | S3 + the wavelet columns | Wavelet on the promoted set |

Level 3 is not in this cross. db4 is not run.

## Layer A block, crossed with every tree block

Same rows. Joining the profile must not drop rows and must not move the boundary.

| Arm | Columns | Fit |
| --- | --- | --- |
| none | no camera columns | |
| camfc | `camfc_now`, `camfc_60`, `camfc_delta` | Fourier profile from Mar-Apr camera 2701 counts (`camera_forecast.py`). Calendar only, so it is known for every September origin. |
| mpfc | `mp_now`, `mp_60`, `mp_delta` | The same estimator, fit on **this split's training rows** of the Maps series, then applied to train and test. |
| camfc+mpfc | both triples | |

`camfc_30` in section 7a is the **30-minute** label (`t + 35 min`) on the rolling daily folds in `joined.py`. This matrix is 60 minutes on one chronological split. Evaluate the profile at the origin and at the 60-minute label, and name the columns `camfc_60` / `mp_60`. Do not paste the section 7a MAE into this table.

Direction is `MY_TO_SG` only. The congestion history cache `eval/data/cam2701_congestion_10min.csv` is already on this machine, so the profile does not need a new query. A read-only refresh of `cam2701.v_congestion_index_10min` is allowed if that file is missing. It is still March-April counts, not September detections.

**Observed counts are not an arm.** Detections in BigQuery end on 22 Apr 2026. The Maps label starts on 5 Sep. Raw 2701 frames for 6-11 Sep have no detections until the Roboflow backfill in [handoff-camera-pilot.md](../handoff-camera-pilot.md) (`backfill_camera_counts.py --mode full --start 2026-09-06 --end 2026-09-11`, about 864 calls). That overlap is the first place a live count could be tested. It is blocked until the key holder runs it.

## What the trees run

XGBoost on all 6 blocks x 4 camera arms = **24 cells**, one seed (42). It is the forecast baseline and one importance judge.

Random Forest on the same 24 cells, same rows, same seed. `RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=50, random_state=42, n_jobs=1)`. No hyperparameter search. The forest forecasts so the two models can agree on which cell has lower MAE. It is not a serving candidate. Its job is the rank agreement below.

That is the full factorial. It is seconds to a few minutes on this CPU.

**Level 3, after importance, only if a level-2 `dwt_*` column beats the noise column** (rule below). Then refit XGBoost and the forest on blocks C and S3W only, at `dwt_level=3`, for camera arm none and for the single camera arm whose level-2 point MAE is lowest. Four extra fits per model. If nothing beats noise, skip level 3.

## Importance (training rows only)

Fit one XGBoost and one forest on the training rows of **S3W + camfc + mpfc** (the widest column set). Add one column, `noise_gauss`, standard normal, seed 42, same length as those rows.

| Judge | Statistic |
| --- | --- |
| XGBoost | total gain, and total cover (`get_score`) |
| Random Forest | impurity (`feature_importances_`), and permutation importance on those same training rows (`n_repeats=5`, `random_state=42`, negative MAE) |

Report Spearman rank correlation of XGBoost gain against forest impurity, and of XGBoost cover against forest permutation importance.

A wavelet or camfc column **ranks above noise** only when both XGBoost gain and forest impurity place it strictly above `noise_gauss`. Permutation importance is printed in the same table. It does not add a deep cell by itself.

This is a judgment about columns. It is not a significance test and it does not change the MAE decision rule.

## What the deep models run

`build_lstm_split` already forces the same train and test rows as `split_supervised`. The sequence tensor is not the tree's column list. Default channels are the 36-step window of 9 series: the duration input, `duration_sec`, `hour`, `minute`, `dayofweek`, `month`, `nonworkday`, `d1`, `d7`. Trees see 12 lag columns plus whatever block is selected. Same rows, same boundary, different representation. Say that in the run notes. Do not describe it as one shared design matrix.

Wavelet coefficients are one vector per window (`wavedec` of the z-scored 36 bins), which is what the tree stores as `dwt_*` columns. They are **a tabular head, not extra channels**. After the recurrent stack, or after the Transformer's flatten, concatenate those scalars and finish with one linear unit. Broadcasting a coefficient across 36 steps would be a different object from the tree's columns, and it would repeat a window the sequence already contains. Camera scalars, if a deep cell uses them, use that same head.

The sequence path stays the 9 channels. Trends are not a deep cell: the difference and the rolling mean are functions of a window the sequence already sees.

**Already scored, do not retrain for this matrix:** LSTM, GRU, anchored Transformer, and the raw-target Transformer, 9 channels, no tabular head, seeds 0, 1, 2. That is section 4.

**New deep cells, LSTM only, seeds 0, 1, 2, and only after the tree importance:**

| Cell | Run it when |
| --- | --- |
| Tabular head = the level-2 `dwt_*` columns, camera arm none | At least one `dwt_*` column ranks above noise |
| Tabular head = `camfc_now`, `camfc_60`, `camfc_delta` | At least one `camfc_*` column ranks above noise, and the matching tree point MAE is lower with that arm than without it |
| Tabular head = those wavelet columns and those three camfc columns together | Both rows above fired |

At most three new LSTM trainings, nine seed fits. GRU, the Transformer, and the raw-target ablation stay on the section 4 cell. `mpfc` and `camfc+mpfc` stay on the trees. Blocks A, B, and S3T stay on the trees.

If nothing ranks above noise, this matrix trains **no** new deep model.

## What is too expensive to cross, and is skipped

3 architectures x 6 blocks x 4 camera arms x 3 seeds is 216 deep fits, plus a raw-target ablation. Skip that. The trees already cover the factorial. The deep models run the two or three cells the importance rule names, or none.

Also skipped: db4, a second split, more than three deep seeds, both directions, the 30-minute `joined.py` folds, serving any of these models, and any GCP job.

## What can run now, and what is blocked

| Piece | Data | Status |
| --- | --- | --- |
| Blocks A, B, C, S3, S3T, S3W, camera arm none, both trees | `eval/data/causeway_gdata.csv` (present) | Can run |
| mpfc arms | The same Maps cache, profile fit inside the training rows | Can run |
| camfc and camfc+mpfc arms | Maps cache plus `eval/data/cam2701_congestion_10min.csv` (present; Mar-Apr profile) | Can run |
| Importance, including the noise column | Training rows of that frame | Can run |
| New LSTM cells | Only after that importance | Can run locally, maybe zero cells |
| Observed September counts as a feature | Roboflow backfill, especially 6-11 Sep | Blocked ([handoff-camera-pilot.md](../handoff-camera-pilot.md)) |
| A significance claim on this hold-out | 10 or more day-blocks | Blocked by the length of the series, not by compute |

## Consequences

- The team runs the tree factorial locally, then at most three LSTM heads. They do not enable Agent Platform and they do not start a VM for this matrix.
- A GCP rerun waits on the trigger above and stays inside the 10 USD ceiling, in `asia-southeast1`, on CPU.
- Point estimates from this matrix enter the report only through a promoted `run.json`. With 5 day-blocks the decision text is insufficient data.
- The promoted section 3 model keeps `use_dwt` off until that promotion.
- An observed-count arm waits on the camera backfill. The honest camera arms until then are none, camfc, mpfc, and camfc+mpfc.

## What this pass did not verify

- A Singapore GPU or CPU SKU price (billing catalog HTTP 403).
- That `cam2701_congestion_10min.csv` still matches the live view. The file is on disk; it was not re-queried.
- A DirectML training run. `nvidia-smi` shows the RTX 3070; `check_tf_gpu.py` was not run.
- Any new MAE. No model was trained.
