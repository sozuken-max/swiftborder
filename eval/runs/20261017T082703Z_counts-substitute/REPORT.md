# Counts substituted for Maps inputs

The target on every row is Google Maps duration in traffic at the 30-minute horizon: the mean `duration_in_traffic` of the bin that starts 30 minutes after the origin (`y_30`), in minutes. The Maps version is that served estimator refit on September 2026 only, reading Maps duration, lags, rolling means, slope, congestion ratio and speed where that model uses them. The counts version puts the directional camera-2701 count in those duration slots (`sg_my` for SG_TO_MY, `my_sg` for MY_TO_SG) and does not read a Maps duration, lag, speed or distance feature at predict time. Calendar terms stay. Persistence-of-Maps repeats the current Maps duration. Persistence-of-counts repeats the current directional count and does not see Maps at predict time; that output is a vehicle count, so it is not scored as a minute forecast. The counts profile is the same Fourier rule fit on September counts and read at the target clock; that output is a typical count, not a duration. Learned counts models are fit to September rows whose label is Maps `y_30`, so they can emit minutes.

## Protocol

Callable forecast-api ids confirmed in `services/forecastapi/main.py`: `persistence`, `profile`, `ridge[maps]`, `xgb[maps]`, `xgb[maps+prof]`, `lin_bq[daily]`, `xgb_bq[daily]`, `lin_bq[frozen]`, `xgb_bq[frozen]`, and `mix[fcm_mlp+xgb]`. `served` at 30 minutes is the mix in both directions, so it is not a second fit. `fcm_mlp` alone and `timesfm` are not served. `lin_bq[frozen]` and `xgb_bq[frozen]` are refit on all September rows and reported as `lin_bq[sept-refit]` and `xgb_bq[sept-refit]`, not as the 12 Sep production cutoff. Those September fits use the same rows and estimators as `lin_bq[daily]` and `xgb_bq[daily]` (BQ feature list, `lag_60` required), so each pair is one fit listed under both names. No October row is used to fit. The mix's L2 penalty still uses the module's last two September days as inner validation (`fit_day` for serving day 2026-10-01).

**Practical bar.** `docs/evaluation.md` states two different bars. "What would count as success" is the product target of mean absolute error at or below 15 minutes on crossing time; that target is not measured here, and this run does not claim it. The practical bar used for this decision is the one in the Significance section: a serving change needs a drop of at least 0.5 min (30 s), about the rounding of a forecast shown in whole minutes. The same +/-0.5 min band is the practical threshold elsewhere on that page. A difference inside that band is not a meaningful improvement. Substitution also requires the counts version to be able to run when Maps bins are absent (it does not read `v_bins_10min` at predict time) and its October MAE to sit within 0.5 min of both the Maps version and persistence-of-Maps. Label `regression` when the paired interval lies entirely above +0.5 min, `improvement` when it lies entirely below -0.5 min, `substitute` when the interval lies inside +/-0.5 min and the counts MAE is also within 0.5 min of persistence, and `inconclusive` when a direction has fewer than 50 rows or the interval crosses +/-0.5 min.

## Window

Clocks: bin timestamps are UTC. The month cut is the Asia/Singapore calendar date. Train: origin date 2026-09-06 through 2026-09-30 SGT (Maps logging and `local_counts` both start 6 Sep; 1-5 Sep have neither series) and `bin_ts + 40 min` at or before 2026-10-01 00:00 SGT. Eval: October 2026 SGT origins that are scorable on both series (`y_30` present, `after_gap = 0`). Maps `v_bins_10min` rows in the download: 8618 (2026-09-05T17:50:00+00:00 to 2026-10-05T15:50:00+00:00). October SGT bin-days in that download: 2026-10-01: 288 bins, mean n_obs 2.00; 2026-10-02: 288 bins, mean n_obs 2.00; 2026-10-03: 288 bins, mean n_obs 2.00; 2026-10-04: 288 bins, mean n_obs 2.00; 2026-10-05: 288 bins, mean n_obs 2.00. `cam2701.local_counts` frames in the download: 4052 (2026-09-05T16:05:46+00:00 to 2026-10-04T15:55:48+00:00). October SGT frame counts by day: {'2026-10-01': 143, '2026-10-02': 138, '2026-10-03': 139, '2026-10-04': 135}. Camera 2702 is not used: `features.py` builds counts from one camera (2701).

## Decision table (pooled October rows)

Paired difference is counts MAE minus Maps-version MAE, in minutes. Positive means the counts version is worse. The interval is the harness day-block bootstrap (`eval/significance.py`, 4,999 resamples, seed 0, Asia/Singapore day boundary). With both directions the interval is the joint calendar-day bootstrap. Holm is not applied across this family.

| Model | n | MAE Maps | MAE counts | MAE persistence (Maps) | Counts − Maps (95% CI) | Label |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `persistence` | 1068 | 3.204 | — | 3.204 | — | inconclusive |
| `profile` | 1068 | 6.789 | — | 3.204 | — | inconclusive |
| `ridge[maps]` | 1068 | 2.936 | 7.225 | 3.204 | 4.289 [3.117, 5.345] | regression |
| `xgb[maps]` | 1068 | 2.786 | 5.912 | 3.204 | 3.126 [1.814, 4.307] | regression |
| `xgb[maps+prof]` | 1068 | 2.790 | 5.517 | 3.204 | 2.727 [1.470, 3.858] | regression |
| `lin_bq[daily]` | 1068 | 2.935 | 7.240 | 3.204 | 4.305 [3.091, 5.397] | regression |
| `lin_bq[sept-refit]` | 1068 | 2.935 | 7.240 | 3.204 | 4.305 [3.091, 5.397] | regression |
| `xgb_bq[daily]` | 1068 | 3.273 | 7.080 | 3.204 | 3.807 [2.413, 5.061] | regression |
| `xgb_bq[sept-refit]` | 1068 | 3.273 | 7.080 | 3.204 | 3.807 [2.413, 5.061] | regression |
| `mix[fcm_mlp+xgb]` | 1068 | 2.737 | 5.689 | 3.204 | 2.952 [1.452, 4.302] | regression |

## Per direction

| Model | Direction | n | MAE Maps | MAE counts | MAE persistence (Maps) | Counts − Maps (95% CI) | Label |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `persistence` | SG_TO_MY | 534 | 3.236 | — | 3.236 | — | inconclusive |
| `persistence` | MY_TO_SG | 534 | 3.172 | — | 3.172 | — | inconclusive |
| `profile` | SG_TO_MY | 534 | 7.858 | — | 3.236 | — | inconclusive |
| `profile` | MY_TO_SG | 534 | 5.721 | — | 3.172 | — | inconclusive |
| `ridge[maps]` | SG_TO_MY | 534 | 2.892 | 8.838 | 3.236 | 5.946 [4.167, 8.313] | regression |
| `ridge[maps]` | MY_TO_SG | 534 | 2.979 | 5.612 | 3.172 | 2.633 [2.118, 3.201] | regression |
| `xgb[maps]` | SG_TO_MY | 534 | 2.798 | 6.596 | 3.236 | 3.799 [2.227, 5.824] | regression |
| `xgb[maps]` | MY_TO_SG | 534 | 2.775 | 5.228 | 3.172 | 2.453 [1.900, 3.051] | regression |
| `xgb[maps+prof]` | SG_TO_MY | 534 | 2.916 | 6.137 | 3.236 | 3.222 [1.500, 5.211] | regression |
| `xgb[maps+prof]` | MY_TO_SG | 534 | 2.664 | 4.896 | 3.172 | 2.232 [1.726, 2.828] | regression |
| `lin_bq[daily]` | SG_TO_MY | 534 | 2.891 | 8.908 | 3.236 | 6.017 [4.129, 8.493] | regression |
| `lin_bq[daily]` | MY_TO_SG | 534 | 2.979 | 5.572 | 3.172 | 2.592 [2.077, 3.174] | regression |
| `lin_bq[sept-refit]` | SG_TO_MY | 534 | 2.891 | 8.908 | 3.236 | 6.017 [4.129, 8.493] | regression |
| `lin_bq[sept-refit]` | MY_TO_SG | 534 | 2.979 | 5.572 | 3.172 | 2.592 [2.077, 3.174] | regression |
| `xgb_bq[daily]` | SG_TO_MY | 534 | 3.439 | 8.010 | 3.236 | 4.571 [2.485, 7.129] | regression |
| `xgb_bq[daily]` | MY_TO_SG | 534 | 3.107 | 6.150 | 3.172 | 3.043 [2.705, 3.603] | regression |
| `xgb_bq[sept-refit]` | SG_TO_MY | 534 | 3.439 | 8.010 | 3.236 | 4.571 [2.485, 7.129] | regression |
| `xgb_bq[sept-refit]` | MY_TO_SG | 534 | 3.107 | 6.150 | 3.172 | 3.043 [2.705, 3.603] | regression |
| `mix[fcm_mlp+xgb]` | SG_TO_MY | 534 | 2.751 | 6.197 | 3.236 | 3.446 [1.843, 5.678] | regression |
| `mix[fcm_mlp+xgb]` | MY_TO_SG | 534 | 2.723 | 5.181 | 3.172 | 2.458 [1.686, 3.459] | regression |

## Notes

- `persistence`: Univariate. Counts version repeats the current vehicle count and is not a minute forecast.
- `profile`: Univariate calendar rule. Counts version is fit on September counts and emits a typical count, not minutes. It does not read the live Maps bin.
- `ridge[maps]`: Harness ridge (`joined._ridge`). Counts features omit congestion ratio and speed.
- `xgb[maps]`: Harness XGBoost (`joined._xgb`). Counts features omit congestion ratio and speed.
- `xgb[maps+prof]`: Harness XGBoost plus the Fourier profile. The counts version uses the profile of the count series, not the Maps profile.
- `lin_bq[daily]`: BQ linear replica (`bq_replica.fit_lin` on the Maps side). September only, `lag_60` required.
- `lin_bq[sept-refit]`: Same September fit as the daily replica. Not the 12 Sep production frozen model. Callable id this stands in for: `lin_bq[frozen]`.
- `xgb_bq[daily]`: BQ XGBoost replica (`bq_replica.fit_xgb`, five seeds). September only, `lag_60` required.
- `xgb_bq[sept-refit]`: Same September fit as the daily replica. Not the 12 Sep production frozen model. Callable id this stands in for: `xgb_bq[frozen]`.
- `mix[fcm_mlp+xgb]`: Same `fit_day` / `predict_direction` as the service. Counts version: duration slots are directional counts, congestion and speed are missing, and the FCM residual is Maps minutes minus the count.

Persistence-of-counts and the counts profile are not minute forecasts. On these same October rows the mean absolute difference between Maps `y_30` (minutes) and the vehicle-count output is 28.523 for persistence-of-counts and 16.502 for the counts profile. Those figures are not duration errors and are not labelled improvement or regression.

This run does not fill the live forecast-api 503, does not change the service, and does not show a crossing-time error. A counts model that still needed `v_bins_10min` at predict time would not be a substitute; the counts versions above do not read it then.
