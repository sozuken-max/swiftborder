> **Committed report snapshot.** Ephemeral runs stay gitignored; this folder is the citation target for the final report. Promoted from `20260930T191544Z_offline-bqml-joined-deep-fuzzy-ensemble`.

# Eval run `20260930T191544Z_offline-bqml-joined-deep-fuzzy-ensemble`

Created (UTC): 2026-09-30T19:27:56Z. Schema v2.
Git `9597e2099657` (dirty: False); code SHA-256 `012b8dc3bfc4`.

## offline

- **Horizon:** 60 min
- **Window:** 2026-09-26 01:35:00 .. 2026-10-01 00:30:00
- **Models:** sklearn.XGBRegressor (timeseries_xgb.train_xgb), Persistence T-60, Maps typical duration at origin

| Candidate | Slice | n | MAE (min) | RMSE (min) |
| --- | --- | --- | --- | --- |
| XGB (sklearn) | holdout | 1428 | 3.469 | 4.958 |
| Persistence T-60 | holdout | 1428 | 4.677 | 6.386 |
| Maps typical duration at origin | holdout | 1428 | 14.715 | 16.880 |

| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGB (sklearn) | Persistence T-60 | 1428 | -1.208 | [-1.594, -0.660] | 6.5e-07 | 6.5e-07 | insufficient data |
| XGB (sklearn) | Maps typical duration at origin | 1428 | -11.245 | [-12.319, -10.266] | 7.2e-77 | 1.4e-76 | insufficient data |

Figures:
- `offline/backtest-mae.png`
- `offline/holdout-sample.png`
- `offline/holdout-mae-diff.png`

## bqml

- **Horizon:** 30 min
- **Window:** 2026-09-12T16:00:00+00:00 .. 2026-09-30T15:50:00+00:00
- **Models:** Persistence (y_persistence), lin_h30, xgb_h30, ensemble_mean = (lin_h30 + xgb_h30) / 2

| Candidate | Slice | n | MAE (min) | RMSE (min) |
| --- | --- | --- | --- | --- |
| Persistence | both/all | 5184 | 2.640 | 3.894 |
| lin_h30 | both/all | 5184 | 2.777 | 3.777 |
| xgb_h30 | both/all | 5184 | 2.493 | 3.678 |
| ensemble_mean | both/all | 5184 | 2.459 | 3.526 |

| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lin_h30 (both/all) | Persistence | 5184 | 0.136 | [0.026, 0.259] | 0.1022 | 0.3297 | not significant |
| xgb_h30 (both/all) | Persistence | 5184 | -0.147 | [-0.233, -0.061] | 0.00086 | 0.0060 | challenger |
| ensemble_mean (both/all) | Persistence | 5184 | -0.182 | [-0.253, -0.095] | 0.00013 | 0.0011 | challenger |
| lin_h30 (SG_TO_MY/all) | Persistence | 2592 | -0.198 | [-0.337, -0.076] | 0.0058 | 0.0288 | challenger |
| xgb_h30 (SG_TO_MY/all) | Persistence | 2592 | -0.124 | [-0.268, 0.005] | 0.0824 | 0.3297 | not significant |
| ensemble_mean (SG_TO_MY/all) | Persistence | 2592 | -0.309 | [-0.414, -0.225] | 9.6e-10 | 9.6e-09 | challenger |
| lin_h30 (MY_TO_SG/all) | Persistence | 2592 | 0.471 | [0.310, 0.706] | 2.4e-06 | 2.1e-05 | reference |
| xgb_h30 (MY_TO_SG/all) | Persistence | 2592 | -0.170 | [-0.258, -0.052] | 0.00096 | 0.0060 | challenger |
| ensemble_mean (MY_TO_SG/all) | Persistence | 2592 | -0.054 | [-0.155, 0.111] | 0.4235 | 0.8469 | not significant |
| ensemble_mean (both/all) | lin_h30 | 5184 | -0.318 | [-0.378, -0.255] | 6.1e-12 | 6.7e-11 | challenger |
| ensemble_mean (both/all) | xgb_h30 | 5184 | -0.034 | [-0.109, 0.049] | 0.4731 | 0.8469 | not significant |
| lin_h30 (both/time_of_day=morning peak) | Persistence | 1080 | 0.274 | [0.081, 0.434] | 0.0528 | 1.0000 | not significant |
| xgb_h30 (both/time_of_day=morning peak) | Persistence | 1080 | -0.073 | [-0.259, 0.143] | 0.5015 | 1.0000 | not significant |
| ensemble_mean (both/time_of_day=morning peak) | Persistence | 1080 | -0.070 | [-0.223, 0.077] | 0.3987 | 1.0000 | not significant |
| lin_h30 (both/time_of_day=evening peak) | Persistence | 1296 | 0.276 | [0.035, 0.501] | 0.0634 | 1.0000 | not significant |
| xgb_h30 (both/time_of_day=evening peak) | Persistence | 1296 | -0.135 | [-0.321, 0.011] | 0.1312 | 1.0000 | not significant |
| ensemble_mean (both/time_of_day=evening peak) | Persistence | 1296 | -0.198 | [-0.359, -0.050] | 0.0468 | 1.0000 | not significant |
| lin_h30 (both/time_of_day=other) | Persistence | 2808 | 0.019 | [-0.082, 0.141] | 0.7607 | 1.0000 | not significant |
| xgb_h30 (both/time_of_day=other) | Persistence | 2808 | -0.181 | [-0.250, -0.095] | 1.4e-05 | 0.00069 | challenger |
| ensemble_mean (both/time_of_day=other) | Persistence | 2808 | -0.217 | [-0.287, -0.131] | 2e-07 | 1.1e-05 | challenger |
| lin_h30 (both/day_type=weekday) | Persistence | 3744 | 0.098 | [-0.007, 0.238] | 0.3562 | 1.0000 | not significant |
| xgb_h30 (both/day_type=weekday) | Persistence | 3744 | -0.237 | [-0.320, -0.164] | 1.1e-08 | 6.4e-07 | challenger |
| ensemble_mean (both/day_type=weekday) | Persistence | 3744 | -0.227 | [-0.308, -0.125] | 0.00011 | 0.0050 | challenger |
| lin_h30 (both/day_type=weekend) | Persistence | 1440 | 0.237 | [0.077, 0.427] | 0.0203 | 0.6696 | not significant |
| xgb_h30 (both/day_type=weekend) | Persistence | 1440 | 0.086 | [-0.010, 0.267] | 0.2169 | 1.0000 | not significant |
| ensemble_mean (both/day_type=weekend) | Persistence | 1440 | -0.063 | [-0.161, 0.080] | 0.3328 | 1.0000 | not significant |
| lin_h30 (both/light=day) | Persistence | 2592 | 0.384 | [0.227, 0.576] | 0.0032 | 0.1271 | not significant |
| xgb_h30 (both/light=day) | Persistence | 2592 | -0.129 | [-0.246, 0.005] | 0.0484 | 1.0000 | not significant |
| ensemble_mean (both/light=day) | Persistence | 2592 | -0.017 | [-0.128, 0.116] | 0.8257 | 1.0000 | not significant |
| lin_h30 (both/light=night) | Persistence | 2592 | -0.112 | [-0.206, -0.012] | 0.0454 | 1.0000 | not significant |
| xgb_h30 (both/light=night) | Persistence | 2592 | -0.166 | [-0.263, -0.066] | 0.00095 | 0.0407 | challenger |
| ensemble_mean (both/light=night) | Persistence | 2592 | -0.346 | [-0.416, -0.267] | 1.2e-18 | 7.3e-17 | challenger |
| lin_h30 (SG_TO_MY/time_of_day=morning peak) | Persistence | 540 | -0.381 | [-0.557, -0.233] | 2.6e-05 | 0.0012 | challenger |
| xgb_h30 (SG_TO_MY/time_of_day=morning peak) | Persistence | 540 | 0.144 | [-0.189, 0.544] | 0.4424 | 1.0000 | not significant |
| ensemble_mean (SG_TO_MY/time_of_day=morning peak) | Persistence | 540 | -0.257 | [-0.444, -0.074] | 0.0085 | 0.3060 | not significant |
| lin_h30 (SG_TO_MY/time_of_day=evening peak) | Persistence | 648 | -0.238 | [-0.531, 0.009] | 0.1112 | 1.0000 | not significant |
| xgb_h30 (SG_TO_MY/time_of_day=evening peak) | Persistence | 648 | -0.299 | [-0.581, -0.071] | 0.0229 | 0.7335 | not significant |
| ensemble_mean (SG_TO_MY/time_of_day=evening peak) | Persistence | 648 | -0.527 | [-0.759, -0.354] | 2.8e-06 | 0.00014 | challenger |
| lin_h30 (SG_TO_MY/time_of_day=other) | Persistence | 1404 | -0.108 | [-0.237, -0.006] | 0.0750 | 1.0000 | not significant |
| xgb_h30 (SG_TO_MY/time_of_day=other) | Persistence | 1404 | -0.147 | [-0.267, -0.048] | 0.0096 | 0.3345 | not significant |
| ensemble_mean (SG_TO_MY/time_of_day=other) | Persistence | 1404 | -0.229 | [-0.335, -0.157] | 5.6e-07 | 2.9e-05 | challenger |
| lin_h30 (SG_TO_MY/day_type=weekday) | Persistence | 1872 | -0.333 | [-0.425, -0.245] | 1.8e-13 | 1.1e-11 | challenger |
| xgb_h30 (SG_TO_MY/day_type=weekday) | Persistence | 1872 | -0.212 | [-0.352, -0.096] | 0.0016 | 0.0676 | not significant |
| ensemble_mean (SG_TO_MY/day_type=weekday) | Persistence | 1872 | -0.401 | [-0.487, -0.333] | 6.5e-25 | 4.1e-23 | challenger |
| lin_h30 (SG_TO_MY/day_type=weekend) | Persistence | 720 | 0.154 | [-0.160, 0.353] | 0.2926 | 1.0000 | insufficient data |
| xgb_h30 (SG_TO_MY/day_type=weekend) | Persistence | 720 | 0.104 | [-0.059, 0.347] | 0.3111 | 1.0000 | insufficient data |
| ensemble_mean (SG_TO_MY/day_type=weekend) | Persistence | 720 | -0.070 | [-0.271, 0.106] | 0.4747 | 1.0000 | insufficient data |
| lin_h30 (SG_TO_MY/light=day) | Persistence | 1296 | -0.129 | [-0.351, 0.047] | 0.2450 | 1.0000 | not significant |
| xgb_h30 (SG_TO_MY/light=day) | Persistence | 1296 | -0.058 | [-0.282, 0.159] | 0.6061 | 1.0000 | not significant |
| ensemble_mean (SG_TO_MY/light=day) | Persistence | 1296 | -0.227 | [-0.397, -0.101] | 0.0042 | 0.1594 | not significant |
| lin_h30 (SG_TO_MY/light=night) | Persistence | 1296 | -0.267 | [-0.358, -0.174] | 1.4e-08 | 8.1e-07 | challenger |
| xgb_h30 (SG_TO_MY/light=night) | Persistence | 1296 | -0.191 | [-0.335, -0.065] | 0.0057 | 0.2105 | not significant |
| ensemble_mean (SG_TO_MY/light=night) | Persistence | 1296 | -0.392 | [-0.489, -0.298] | 6.3e-16 | 3.8e-14 | challenger |
| lin_h30 (MY_TO_SG/time_of_day=morning peak) | Persistence | 540 | 0.930 | [0.603, 1.226] | 3.6e-09 | 2.1e-07 | reference |
| xgb_h30 (MY_TO_SG/time_of_day=morning peak) | Persistence | 540 | -0.290 | [-0.456, -0.135] | 0.00046 | 0.0203 | challenger |
| ensemble_mean (MY_TO_SG/time_of_day=morning peak) | Persistence | 540 | 0.117 | [-0.121, 0.356] | 0.3222 | 1.0000 | not significant |
| lin_h30 (MY_TO_SG/time_of_day=evening peak) | Persistence | 648 | 0.790 | [0.441, 1.204] | 2.8e-05 | 0.0013 | reference |
| xgb_h30 (MY_TO_SG/time_of_day=evening peak) | Persistence | 648 | 0.028 | [-0.199, 0.236] | 0.7991 | 1.0000 | not significant |
| ensemble_mean (MY_TO_SG/time_of_day=evening peak) | Persistence | 648 | 0.131 | [-0.081, 0.392] | 0.2748 | 1.0000 | not significant |
| lin_h30 (MY_TO_SG/time_of_day=other) | Persistence | 1404 | 0.146 | [-0.010, 0.372] | 0.1391 | 1.0000 | not significant |
| xgb_h30 (MY_TO_SG/time_of_day=other) | Persistence | 1404 | -0.216 | [-0.294, -0.068] | 0.0004 | 0.0181 | challenger |
| ensemble_mean (MY_TO_SG/time_of_day=other) | Persistence | 1404 | -0.205 | [-0.304, -0.044] | 0.0032 | 0.1271 | not significant |
| lin_h30 (MY_TO_SG/day_type=weekday) | Persistence | 1872 | 0.528 | [0.336, 0.798] | 7.4e-06 | 0.00037 | reference |
| xgb_h30 (MY_TO_SG/day_type=weekday) | Persistence | 1872 | -0.262 | [-0.349, -0.161] | 5.4e-08 | 3e-06 | challenger |
| ensemble_mean (MY_TO_SG/day_type=weekday) | Persistence | 1872 | -0.053 | [-0.187, 0.148] | 0.5380 | 1.0000 | not significant |
| lin_h30 (MY_TO_SG/day_type=weekend) | Persistence | 720 | 0.320 | [0.196, 0.661] | 0.0121 | 0.4114 | insufficient data |
| xgb_h30 (MY_TO_SG/day_type=weekend) | Persistence | 720 | 0.069 | [-0.060, 0.300] | 0.4694 | 1.0000 | insufficient data |
| ensemble_mean (MY_TO_SG/day_type=weekend) | Persistence | 720 | -0.057 | [-0.152, 0.161] | 0.5126 | 1.0000 | insufficient data |
| lin_h30 (MY_TO_SG/light=day) | Persistence | 1296 | 0.898 | [0.680, 1.264] | 1.2e-08 | 6.6e-07 | reference |
| xgb_h30 (MY_TO_SG/light=day) | Persistence | 1296 | -0.200 | [-0.303, -0.059] | 0.0018 | 0.0725 | not significant |
| ensemble_mean (MY_TO_SG/light=day) | Persistence | 1296 | 0.194 | [0.050, 0.445] | 0.0666 | 1.0000 | not significant |
| lin_h30 (MY_TO_SG/light=night) | Persistence | 1296 | 0.044 | [-0.127, 0.219] | 0.6155 | 1.0000 | not significant |
| xgb_h30 (MY_TO_SG/light=night) | Persistence | 1296 | -0.140 | [-0.262, 0.032] | 0.0513 | 1.0000 | not significant |
| ensemble_mean (MY_TO_SG/light=night) | Persistence | 1296 | -0.301 | [-0.408, -0.175] | 4.7e-07 | 2.5e-05 | challenger |

Figures:
- `bqml/mae-by-direction.png`
- `bqml/mae-diff-ci.png`

## joined

- **Horizon:** 30 min
- **Window:** 2026-09-13 .. 2026-09-30
- **Models:** persistence (y_persistence), maps_typical (Maps duration without traffic, same bin), ridge (median impute + indicators, standardised, alpha=1), xgb (XGBRegressor, seed 42), ensemble = mean(ridge, xgb)

| Candidate | Slice | n | MAE (min) | RMSE (min) |
| --- | --- | --- | --- | --- |
| persistence | both/all | 5184 | 2.640 | 3.894 |
| maps_typical | both/all | 5184 | 12.816 | 15.650 |
| ridge[maps] | both/all | 5184 | 2.500 | 3.553 |
| xgb[maps] | both/all | 5184 | 2.275 | 3.311 |
| ensemble[maps] | both/all | 5184 | 2.306 | 3.324 |
| ridge[maps+weather] | both/all | 5184 | 2.511 | 3.573 |
| xgb[maps+weather] | both/all | 5184 | 2.277 | 3.314 |
| ensemble[maps+weather] | both/all | 5184 | 2.313 | 3.337 |

| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgb[maps] (both/all) | persistence | 5184 | -0.366 | [-0.486, -0.246] | 2.2e-08 | 2.7e-07 | challenger |
| xgb[maps] (both/all) | maps_typical | 5184 | -10.541 | [-11.524, -9.615] | 3e-89 | 4.2e-88 | challenger |
| xgb[maps+weather] (both/all) | xgb[maps] | 5184 | 0.003 | [-0.015, 0.022] | 0.7792 | 1.0000 | not significant |
| ridge[maps+weather] (both/all) | ridge[maps] | 5184 | 0.011 | [-0.002, 0.025] | 0.1289 | 0.9022 | not significant |
| ensemble[maps+weather] (both/all) | xgb[maps+weather] | 5184 | 0.036 | [-0.003, 0.075] | 0.0816 | 0.6524 | not significant |
| xgb[maps] (SG_TO_MY/all) | persistence | 2592 | -0.378 | [-0.557, -0.237] | 2.4e-05 | 0.00026 | challenger |
| xgb[maps] (SG_TO_MY/all) | maps_typical | 2592 | -9.726 | [-11.435, -8.232] | 1.1e-29 | 1.5e-28 | challenger |
| xgb[maps+weather] (SG_TO_MY/all) | xgb[maps] | 2592 | -0.008 | [-0.032, 0.023] | 0.5532 | 1.0000 | not significant |
| ridge[maps+weather] (SG_TO_MY/all) | ridge[maps] | 2592 | 0.014 | [-0.003, 0.033] | 0.1449 | 0.9022 | not significant |
| ensemble[maps+weather] (SG_TO_MY/all) | xgb[maps+weather] | 2592 | -0.004 | [-0.063, 0.045] | 0.8841 | 1.0000 | not significant |
| xgb[maps] (MY_TO_SG/all) | persistence | 2592 | -0.354 | [-0.498, -0.132] | 0.00021 | 0.0021 | challenger |
| xgb[maps] (MY_TO_SG/all) | maps_typical | 2592 | -11.356 | [-12.345, -10.259] | 5.5e-96 | 8.2e-95 | challenger |
| xgb[maps+weather] (MY_TO_SG/all) | xgb[maps] | 2592 | 0.014 | [-0.013, 0.039] | 0.3034 | 1.0000 | not significant |
| ridge[maps+weather] (MY_TO_SG/all) | ridge[maps] | 2592 | 0.007 | [-0.011, 0.030] | 0.4764 | 1.0000 | not significant |
| ensemble[maps+weather] (MY_TO_SG/all) | xgb[maps+weather] | 2592 | 0.076 | [0.023, 0.134] | 0.0071 | 0.0638 | not significant |

Figures:
- `joined/joined-mae-diff.png`
- `joined/joined-mae-by-feature-set.png`

## deep

- **Horizon:** 60 min
- **Window:** 2026-09-26 01:35:00 .. 2026-10-01 00:30:00
- **Models:** LSTM(64), anchored, GRU(64), anchored, Patch Transformer, anchored, Patch Transformer, raw target (ablation), XGB (sklearn), Persistence T-60

| Candidate | Slice | n | MAE (min) | RMSE (min) |
| --- | --- | --- | --- | --- |
| Persistence T-60 | holdout | 1428 | 4.677 | 6.386 |
| XGB (sklearn) | holdout | 1428 | 3.469 | 4.958 |
| LSTM(64), anchored, seed mean | holdout | 1428 | 3.970 | 5.423 |
| GRU(64), anchored, seed mean | holdout | 1428 | 4.030 | 5.433 |
| Patch Transformer, anchored, seed mean | holdout | 1428 | 4.192 | 5.772 |
| Patch Transformer, raw target (ablation), seed mean | holdout | 1428 | 4.491 | 6.222 |

| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM(64), anchored, seed mean | Persistence T-60 | 1428 | -0.707 | [-1.313, -0.211] | 0.0058 | 0.0521 | insufficient data |
| LSTM(64), anchored, seed mean | XGB (sklearn) | 1428 | 0.501 | [-0.160, 0.987] | 0.0770 | 0.2309 | insufficient data |
| GRU(64), anchored, seed mean | Persistence T-60 | 1428 | -0.647 | [-0.899, -0.346] | 0.00011 | 0.0011 | insufficient data |
| GRU(64), anchored, seed mean | XGB (sklearn) | 1428 | 0.561 | [0.073, 0.996] | 0.0129 | 0.1033 | insufficient data |
| Patch Transformer, anchored, seed mean | Persistence T-60 | 1428 | -0.485 | [-1.000, 0.003] | 0.0409 | 0.2045 | insufficient data |
| Patch Transformer, anchored, seed mean | XGB (sklearn) | 1428 | 0.723 | [-0.013, 1.384] | 0.0327 | 0.1961 | insufficient data |
| Patch Transformer, raw target (ablation), seed mean | Persistence T-60 | 1428 | -0.186 | [-0.753, 0.502] | 0.5314 | 0.5314 | insufficient data |
| Patch Transformer, raw target (ablation), seed mean | XGB (sklearn) | 1428 | 1.021 | [0.174, 1.920] | 0.0181 | 0.1264 | insufficient data |
| Patch Transformer, anchored | Patch Transformer, raw target | 1428 | -0.298 | [-0.604, -0.003] | 0.0540 | 0.2162 | insufficient data |
| Patch Transformer, anchored | LSTM(64), anchored | 1428 | 0.222 | [-0.144, 0.510] | 0.1620 | 0.3239 | insufficient data |

Figures:
- `deep/deep-mae-by-seed.png`
- `deep/deep-mae-diff.png`

## fuzzy

- **Horizon:** 60 min
- **Window:** 2026-09-26 01:35:00 .. 2026-10-01 00:30:00
- **Models:** Persistence level, Majority level (train), Fuzzy rule base, XGB forecast -> fuzzy level

| Candidate | Slice | n | Accuracy | Macro-F1 | Severe errors | RPS |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence level | holdout | 2856 | 0.690 | 0.678 | 0.006 | 0.120 |
| Majority level (train) | holdout | 2856 | 0.470 | 0.213 | 0.000 | - |
| Fuzzy rule base | holdout | 2856 | 0.718 | 0.698 | 0.002 | 0.116 |
| XGB forecast -> fuzzy level | holdout | 2856 | 0.785 | 0.770 | 0.001 | 0.082 |

Significance loss: 0/1 misclassification; mean_ae_diff_min is an error-rate difference.

| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fuzzy rule base | Persistence level | 2856 | -0.028 | [-0.075, 0.017] | 0.2105 | 0.2105 | not significant |
| XGB forecast -> fuzzy level | Persistence level | 2856 | -0.096 | [-0.113, -0.067] | 5.9e-12 | 1.8e-11 | challenger |
| Fuzzy rule base | XGB forecast -> fuzzy level | 2856 | 0.067 | [0.021, 0.105] | 0.0018 | 0.0035 | reference |

Figures:
- `fuzzy/fuzzy-memberships.png`
- `fuzzy/fuzzy-confusion.png`

## ensemble

- **Horizon:** 30 min
- **Window:** 2026-09-13 .. 2026-09-30
- **Models:** Persistence, Served (registry), lin_h30, xgb_h30, ensemble_mean (lin_h30 + xgb_h30), ridge[maps] (daily refit), xgb[maps] (daily refit), Equal mean of 4 models, Rolling LAD stack, Rolling best-model selection, Fuzzy-gated stack (hybrid)

| Challenger | Reference | n | Mean AE diff | CI | DM p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Served (registry) | Persistence | 5184 | -0.099 | [-0.169, -0.040] | 0.0123 | 0.0738 | not significant |
| Equal mean of 4 models | xgb[maps] (daily refit) | 5184 | 0.083 | [0.030, 0.152] | 0.0148 | 0.0742 | not significant |
| Rolling LAD stack | xgb[maps] (daily refit) | 5184 | -0.021 | [-0.074, 0.042] | 0.4961 | 1.0000 | not significant |
| Rolling best-model selection | xgb[maps] (daily refit) | 5184 | 0.016 | [-0.042, 0.090] | 0.6453 | 1.0000 | not significant |
| Fuzzy-gated stack (hybrid) | xgb[maps] (daily refit) | 5184 | -0.007 | [-0.063, 0.060] | 0.8374 | 1.0000 | not significant |
| Fuzzy-gated stack (hybrid) | Rolling LAD stack | 5184 | 0.015 | [-0.003, 0.038] | 0.1564 | 0.6256 | not significant |
| Rolling LAD stack | Served (registry) | 5184 | -0.288 | [-0.349, -0.199] | 6.5e-13 | 5.8e-12 | challenger |
| Fuzzy-gated stack (hybrid) | Served (registry) | 5184 | -0.274 | [-0.341, -0.177] | 3.6e-10 | 2.9e-09 | challenger |
| xgb[maps] (daily refit) | Served (registry) | 5184 | -0.267 | [-0.371, -0.146] | 7.6e-06 | 5.3e-05 | challenger |
| mean[XGB+deep] | XGB (sklearn) | 1428 | 0.186 | [-0.236, 0.592] | 0.3519 | 0.7038 | insufficient data |
| stack[XGB+deep] | XGB (sklearn) | 1428 | 0.092 | [-0.112, 0.287] | 0.3836 | 0.7038 | insufficient data |
| mean[deep] | XGB (sklearn) | 1428 | 0.460 | [-0.113, 0.971] | 0.0813 | 0.2438 | insufficient data |

Figures:
- `ensemble/ensemble-30min-mae-diff.png`
- `ensemble/ensemble-60min-mae-diff.png`

Machine-readable metadata: [`run.json`](run.json).
