# CNN-RNN: camera 2701 frames to Google duration bands

Design note only. No training, preprocessing, or model code in this change. Contemporaneous classification of one Maps duration, not the 30-minute-ahead `y_30` forecast in `forecast-api` / `v_training_set`.

## Goal

Map LTA camera **2701** frames to a probability distribution over Google Maps `duration_in_traffic` bands (softmax classifier). Score with a confusion matrix and with MAE after converting that distribution back to minutes.

## Scope

| Item | Choice |
| --- | --- |
| Camera | `2701` only |
| Window start | **2026-09-06 02:00:00 Asia/Singapore** (2026-09-05 18:00:00 UTC), inclusive |
| Window end | **Assumption:** through **2026-10-04 23:59:59.999 Asia/Singapore**. The user said 4 Oct 2026 inclusive and did not give an end clock time. Query as half-open `[2026-09-05 18:00:00Z, 2026-10-04 16:00:00Z)`. |
| Images | GCS `gs://sg-lta-traffic-cameras/` (`asia-southeast1`). Console listing uses hive prefixes `camera_id=2701/month=2026-09`. The window includes 1–4 Oct, so also list `camera_id=2701/month=2026-10` if that prefix exists. |
| Labels | BigQuery `swiftborder.causeway.travel_times` (location **US**). |

Images and `travel_times` are in different BigQuery/GCS locations. Join them offline. Do not expect one cross-location SQL join.

`traffic_images.metadata.capture_timestamp` is documented only through **2026-09-11 23:55** (inventory; table last modified 2026-09-13). It does not cover this window. List GCS. `traffic_images.labels` is empty and is not a duration label.

Inventory’s latest Maps snapshot in-repo ends at `observed_at` **2026-10-04 06:40:04 UTC** (14:40:04 SGT) with the scheduler still on `*/5 * * * *` `Asia/Singapore`. Rows through 23:59 SGT on 4 Oct are not guaranteed by that snapshot. Drop images whose 10-minute bin has no OK duration.

### Route (one scalar label)

Camera 2701 shows both carriageways. One softmax needs one duration series.

**Proposed label:** `route_id = jb_to_woodlands` (`direction = MY_TO_SG`, JB → Woodlands / Malaysia → Singapore). That is the offline series in `eval/timeseries_xgb.py` (`DEFAULT_ROUTE_ID`). The other route, `mandai_to_shell_jb` (`SG_TO_MY`), is a separate run, not a tenth class.

`duration_in_traffic_sec` is the label. `duration_sec` is Maps’ duration without traffic and is not the class target. `camdetect` congestion strings (`Free Flow`, `Quarter Way`, `Half Way`, `Back to Back`) are vertical spread of detections, not these minute bands.

### What 2701 frames

`camdetect` treats 2701 as the Causeway camera with a dividing line on a **1920×1080** reference frame. Foot point above the line is `SG-MY` (Singapore → Malaysia); on or below the line is `MY-SG` (Malaysia → Singapore). Stored detection names differ: `to_JB` / `to_Woodlands` map to `SG_TO_MY` / `MY_TO_SG` in `cam2701.v_congestion_index_10min`. Those detection rows end **2026-04-22** and do not overlap this window. Use the line only as a crop guide, not as a join key.

## Alignment rule

Google samples are about every 5 minutes (`Gmap-Woodlands`, `*/5`). Stored 2701 frames are about every 10 minutes (inventory: 132–144 frames/day on 5–11 Sep). `observed_at` is the actual sample time, not a perfect `:00`/`:05` grid.

Match the existing 10-minute bin, which **averages** the 5-minute samples inside the bin:

1. Take the image capture time and convert it to UTC. `observed_at` is UTC. Eval code derives naive `observed_at_sgt` with `Asia/Singapore` (`eval/timeseries_xgb.py` `ensure_observed_at_sgt`). Split dates and the window bounds are SGT. Do not floor a naive SGT clock as if it were UTC.
2. `bin_ts = floor(capture_utc, 10 minutes)`, same as `eval/features.py` `maps_bins` (`observed_at` floored to `10min`) and `sql/bigquery/traffic_prediction/v_bins_10min.sql`: `TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(observed_at), 600) * 600)`. The bin is half-open `[bin_ts, bin_ts + 10 min)`.
3. Label minutes = mean of `duration_in_traffic_sec / 60` over rows with `route_id = jb_to_woodlands`, `status = 'OK'`, and non-null `duration_in_traffic_sec` in that interval. That is `AVG` in `v_bins_10min` (`n_obs` is how many 5-minute samples landed in the bin). One sample is still a valid mean. Zero samples: drop the frame.
4. Class = the band of that mean (table below). Keep the continuous mean for MAE.

Do not use nearest-5-minute matching. The repo’s 10-minute feature path averages the bin.

`observed_date_sgt` is the partition column (inventory; table DDL is not in `sql/`). Use it to limit the scan to **2026-09-06 .. 2026-10-04**, and still filter on `observed_at` for the 02:00 SGT start. Columns confirmed in repo code: `observed_at`, `route_id`, `status`, `duration_sec`, `duration_in_traffic_sec`, `error_message` (`eval/timeseries_xgb.py`); `direction`, `congestion_ratio`, `speed_kmh` (`v_bins_10min.sql`).

**Unverified (check GCS / BigQuery, do not guess):** object names under `month=YYYY-MM`; whether `month=2026-10` exists; which timestamp inside the object or `traffic_images.metadata.capture_timestamp` is the capture time and which zone a naive stamp uses. If a stamp is naive, confirm one object against `observed_at` before treating it as SGT.

## Label bands

Nine labels, as listed: `20`, `30`, `40`, `50`, `60`, `70`, `80`, `90`, `100+`. Left-closed, right-open. The label is the lower edge of the nominal 10-minute band, except `100+`.

**Assumption:** durations under 20 minutes are class `20`. They are not dropped and they are not a tenth class.

| class_index | label | duration `d` (minutes) | MAE representative (minutes) |
| --- | --- | --- | --- |
| 0 | 20 | `[0, 30)` | 25 |
| 1 | 30 | `[30, 40)` | 35 |
| 2 | 40 | `[40, 50)` | 45 |
| 3 | 50 | `[50, 60)` | 55 |
| 4 | 60 | `[60, 70)` | 65 |
| 5 | 70 | `[70, 80)` | 75 |
| 6 | 80 | `[80, 90)` | 85 |
| 7 | 90 | `[90, 100)` | 95 |
| 8 | 100+ | `[100, +∞)` | 110 |

Edges: `d = 20` → class 0; `d = 30` → class 1; `d = 100` → class 8. `d = 0` stays in class 0.

Representatives are midpoints of the nominal bands `[20,30)`, `[30,40)`, …, `[90,100)`. Class 0 uses **25**, the midpoint of `[20, 30)`, including when `d < 20` (the lower tail does not pull the representative down to 10 or 15). Class 8 uses **110** (one 10-minute bin width above 100). A later train-only mean of `d | d ≥ 100` may replace 110. Do not fit that mean on the validation dates.

## Split

Chronological **block by calendar date** in `Asia/Singapore`. Not a shuffle of frames. Not the row-count cut in `eval/timeseries_xgb.py` `split_supervised` (that function cuts on `target_ts` order at `train_fraction = 0.8`).

The window touches **29** dates: 2026-09-06 through 2026-10-04. Twenty percent of 29 is 5.8 days, so hold out the last **6** dates (6/29 ≈ 21%).

| Set | Dates (SGT) | Days |
| --- | --- | --- |
| Train | 2026-09-06 02:00 through 2026-09-28 23:59 | 23 (first day starts at 02:00) |
| Validation | 2026-09-29 00:00 through 2026-10-04 23:59 | 6 |

A congestion event that crosses 28/29 Sep can put similar frames on both sides. That leakage is accepted for this date cut; do not move individual frames across it. A CNN-RNN sequence is train-only or validation-only. No sequence may include both 28 Sep and 29 Sep.

## Model intent

Softmax over the 9 classes. The frame (or sequence) yields `p_k`, not a regression head.

**Baseline — single-frame CNN.** One 2701 image → conv stack → dense → softmax. This is the control for every filter and conv variant.

**Named model — short-sequence CNN-RNN.** Same conv stack, weights shared across time, on **K** consecutive 2701 frames (about `10*(K-1)` minutes). A small recurrent layer (LSTM or GRU) reads the K embeddings in time order. Softmax on the last step predicts the band of the **last** frame’s aligned duration, not a future horizon. Start the experiment at `K = 1` (the baseline) and `K = 6` (about one hour). Pick K only among values whose whole sequence sits inside one split.

`eval/timeseries_lstm.py` is an LSTM on the Maps **numeric** series (`jb_to_woodlands`, 60-minute horizon). It is not an image model. Do not reuse its input window as this K.

## Metrics

On the validation dates only:

1. **Confusion matrix** on `argmax_k p_k` versus the true class index (9×9).
2. **Argmax minutes:** representative of the argmax class. Report alongside the matrix; it is not the MAE target.
3. **Expected minutes:** `d_hat = Σ_k p_k * r_k` with `r_k` from the table above (class 8 contributes 110).
4. **MAE** = mean of `|d_hat - d|` where `d` is the continuous 10-minute mean from the alignment rule, in minutes, not the representative of the true class.

## Pre-DNN filter experiments

Checklist. Compare each variant to the full-frame single-frame CNN on the same split, bands, and route.

- [ ] Full frame, no crop (baseline pixels).
- [ ] Resize only (record the pixel size; 1920×1080 is the `camdetect` reference, not a proven stored size for every object).
- [ ] ROI on the **MY-SG / `jb_to_woodlands`** side of the 2701 dividing line (on or below the polyline). Rescale the line if the stored frame is not 1920×1080.
- [ ] ROI on the other carriageway only if that run’s label is `mandai_to_shell_jb`.
- [ ] Conv-stack variants (depth, width, kernel) on the chosen crop. Keep the softmax and the band table fixed so the stack is the only change.

## Non-goals

- No training code, preprocessing code, or layer code in this change.
- Existing docs (`README.md`, `docs/adr/*`, `docs/architecture.md`, `docs/inventory.md`, and the other files already in `docs/`) are not updated by this note.
- Not a forecast of `y_30` / `y_60`. Not Roboflow detection counts. Not both routes in one 9-way softmax.

## Open assumptions

- End of window is 2026-10-04 23:59:59.999 SGT (exclusive end 2026-10-05 00:00:00 SGT).
- Label route is `jb_to_woodlands`.
- Durations in `[0, 20)` use class `20` and representative 25.
- `100+` representative is 110 until a train-only conditional mean replaces it.
- October objects live under `camera_id=2701/month=2026-10` with the same layout as September. Unverified in git.
- Capture-time field and its timezone on GCS objects are unverified in git. Align only after that stamp is confirmed, then convert to UTC before the 10-minute floor.
