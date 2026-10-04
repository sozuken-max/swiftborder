# Layer B: FCM + MLP results (real data, 6–30 Sep 2026)

Produced on 4 Oct 2026 by the `train` logic of [`layer_b_fcm_mlp.py`](layer_b_fcm_mlp.py) on the BigQuery view `traffic_prediction.v_bins_10min`, 3,589 ten-minute bins per direction. No October labels were used (the protected window 1–19 Oct was respected).

## Protocol
- **Model:** Fuzzy C-Means (c = 3, m = 2) on travel time now, its 30-minute change and yesterday's change over the coming hour. The membership degrees, plus lag, D-1/D-7 and calendar features, feed an MLP (32-16 ReLU, Adam). Each model is an average of 5 networks with different seeds, predicting the change at 10–60 minutes.
- **Splits:** time-ordered and day-disjoint.
  - Fit: 6–16 Sep.
  - Early stopping and L2 choice: 17–18 Sep.
  - Validation, used for the decision rules: 19–25 Sep.
  - Test, scored once: 26–30 Sep.
- **Statistics:** paired day-block bootstrap, with both directions resampled together by calendar day.
- **Layer A coverage:** zero (`layer_a_counts` does not exist yet), so the vision gate could not run.

## Test MAE in minutes (26–30 Sep, both directions, identical rows; n ≈ 1,430 per step)

| Horizon | Persistence | TimesFM 2.5 (zero-shot) | MLP without FCM | FCM + MLP |
|---|---|---|---|---|
| 10 min | 1.465 | 1.442 | 1.369 | 1.371 |
| 20 min | 2.138 | 1.996 | 1.880 | 1.882 |
| 30 min | 2.804 | 2.449 | 2.330 | 2.314 |
| 40 min | 3.394 | 2.825 | 2.676 | 2.639 |
| 50 min | 3.986 | 3.202 | 3.009 | 3.001 |
| 60 min | 4.559 | 3.525 | 3.392 | 3.369 |

## FCM + MLP against each comparator (95% CI of the paired difference)

**Against persistence:** better at every step.
- 30 min: −0.49 [−0.59, −0.41]
- 60 min: −1.19 [−1.45, −0.94]

**Against TimesFM 2.5:** better at 10–50 min; at 60 min the interval just includes zero.
- 30 min: −0.135 [−0.234, −0.066]
- 60 min: −0.156 [−0.408, +0.013]

**Against the same network without FCM (ablation):** no significant difference at any step on the test days. On the validation days, MY→SG at 60 min was significantly worse with FCM (+0.097 [+0.037, +0.170]).

**±15 minutes:** 98.9–100% of test forecasts fall within ±15 min of the actual travel time.

## Fuzzy regimes (centres, fit and inner days)

| Direction | Light | Moderate | Heavy |
|---|---|---|---|
| SG→MY | 16.1 min, steady | 30.7 min, clearing (−3.1 min per 30 min) | 34.7 min, building (+4.0) |
| MY→SG | 19.7 min | 32.7 min, clearing | 40.2 min, building |

Partition coefficient is 0.56–0.58 and Xie–Beni 0.34–0.42 (textbook, unsaturated degrees). The upper two regimes differ mainly by trend, not by level.

## Caveats for the report
- There are only 5 test days and 7 validation days. The per-step gates have no multiplicity correction.
- The fuzzy front end adds interpretability (regime labels with degrees) but no measurable accuracy over the identical network without it.
- The TimesFM figures are zero-shot, without the residual calibration in `layer_b_timesfm.py`.
- Horizons count from the start of the origin bin, so a "60 min" target's readings arrive 50–55 minutes after the forecast is issued.
