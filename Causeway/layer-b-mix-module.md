# Layer B mix module: `Causeway/layer_b_fcm_xgb_mix.py`

Added on 10 Oct 2026 on top of `main` f90e62b. Not deployed: a read of project `swiftborder` that afternoon showed no `layer_b_mix_*` or `layer_a_counts` table in `traffic_prediction`, and no Cloud Run Job or Scheduler entry runs this module. Code: [`layer_b_fcm_xgb_mix.py`](layer_b_fcm_xgb_mix.py). Slide numbers follow the 17-slide proposal deck.

## What it does
- **Forecast:** the Google Maps travel time per direction, 30 minutes ahead (target bin [t+30, t+40) = `y_30`), as `mix = 0.5 × fcm_mlp + 0.5 × xgb[maps]`.
- **`fcm_mlp`:** imported from [`layer_b_fcm_mlp.py`](layer_b_fcm_mlp.py). Byte-identical to its `train_direction` on the same days, and covered by a test.
- **`xgb[maps]`:** the harness XGBoost (`eval/joined.py`, `eval/features.py`). Row-for-row identical to `forecastapi/local_models.py`, also covered by a test.
- **Refit:** both members are refitted daily on every label observed before 00:00 SGT since `MIX_HISTORY_START` (default 2026-09-06). This is the evaluated daily refit; `MIX_LOOKBACK_DAYS` is only an optional cap.
- **Layer A:** the camera 2701 counts and congestion level (shared `layer_a_counts` table) are optional, gated inputs (slide 10).
  - The Layer A mix is served only if the camera covers at least 50% of validation rows and, on those rows, it beats both the identical no-vision mix trained on the same rows and the otherwise-served model (slides 13 and 15).
- **Persistence gate:** anything served must beat persistence on 7 rolling out-of-sample validation days (paired day-block bootstrap, slides 11 and 15).
- **Protected window:** 1–19 Oct is refused, and its labels are never scored without `--allow-protected-window`.
- **Commands:** `train` (daily), `run-cycle` (every 10 min), `backtest`, `monitor`, `ensure-tables` and `show-sql`.
- **New tables** in `traffic_prediction`, created on first run: `layer_b_mix_forecasts`, `layer_b_mix_evaluation` and `layer_b_mix_registry`.

## Other files in the change
- `Causeway/tests/test_layer_b_fcm_xgb_mix.py`: 42 offline tests.
- `Causeway/requirements.txt`: adds `xgboost==3.2.0`.
- `Causeway/requirements-dev.txt`: adds `pandas==3.0.6`, so the parity tests run in CI.
- `Causeway/Procfile` and `Causeway/.python-version` (3.11).
- A `Causeway/README.md` row and a `CHANGELOG.md` entry.

## Verification (10 Oct 2026)
- **Local backtest:** run on bins exported from BigQuery for 13–30 Sep with `--allow-protected-window`, 5,184 rows.

  | Model | MAE (min) | [evaluation.md §8](../docs/evaluation.md#8-timesfm-25-and-fcm--mlp-shared-13-30-sep-window-2026-10-05) (min) |
  |---|---|---|
  | persistence | 2.640 | 2.640 |
  | `fcm_mlp` | 2.230 | 2.230 |
  | `xgb[maps]` | 2.273 | 2.276 |
  | mix | 2.130 | 2.134 |

  The small differences come from the data export: `xgb[maps]` is identical to the forecast-api code on the same bins. Without the flag, 5,178 rows are scored, because six labels fall on 1 Oct.
- **`train` on 24–30 Sep:** took 88 s and served the mix in both directions (validation mix 2.09 against persistence 2.74). The registry round trip is exact, and the registry entry is 0.74 MB.
- **Tests and checks:** on `main` f90e62b the Causeway suite passes (189 tests, including a fresh CI-like venv), and so do the repo doc checks. ruff and mypy are clean.
- **Independent review:** two passes. All findings were fixed, including the lookback default (now all history) and the registry age.

## Open items for the team
- The mix is not among frozen-run claims C1–C8. [roadmap.md](../docs/roadmap.md) proposes it as C9 for Run B; until the team confirms that before 19 Oct 23:59 SGT, any October score is exploratory.
- The daily train job exits 2 until `PROTECTED_WINDOW` is cleared after the frozen run. Register a September model first: `train --end 2026-09-30 --register`.
- The Procfile buildpack build has not been run on GCP.
- No Layer A frames overlap the labels yet, so the vision variants will not train until frames are logged.
- `xgboost` pulls about 0.5 GB of NVIDIA NCCL. `xgboost-cpu==3.2.0` gave identical predictions and could serve a deploy-only image, but never install both packages in one environment.
