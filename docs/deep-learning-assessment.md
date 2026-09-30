# Deep-learning forecasters (LSTM, Transformer): assessment

**Decision (2026-10-01):** do not build or serve an LSTM or Transformer forecaster for the final report. Keep the existing LSTM code in `eval/` as an unscored option, and revisit when the conditions below are met.

**Scope:** Layer B, forecasting the Google Maps duration series at 30–60 minutes. Layer A already uses deep learning (Roboflow YOLO), so the Practice Module "ML/DL techniques" criterion does not depend on this decision ([grading lens](grading/nus-iss-practice-module.md)).

**Evidence status:** no deep-learning row is in [`eval/runs/report/run.json`](../eval/runs/report/run.json), so this page cites no deep-learning MAE. A scratch run on the offline split (LSTM, GRU and a small Transformer encoder, same rows as offline XGBoost) did not beat XGBoost. It is not recorded or reproducible from the repo, so it must not be quoted as a result.

## Why not now

1. **Too little data for the parameter count.** The Maps series starts on 6 Sep 2026. The offline 60-minute split has about 5.7k training windows for one route, and the chronological validation split used for early stopping takes a share of those. A small LSTM or Transformer has 10k–20k weights, which is more than the training examples and far more than the number of independent days (about 25). XGBoost on engineered lags regularises better at this size.
2. **The signal at 30–60 minutes is mostly recent level plus time of day.** Persistence is already a strong baseline ([evaluation.md](evaluation.md)). The gains that exist come from lags, D-1/D-7 values and calendar features, which XGBoost uses directly. A sequence model has to learn the same structure from the raw window, with less data.
3. **Attention adds little on short windows.** The input window is 36 five-minute steps (3 hours). Attention pays off on long contexts and many series; on a short, single, smooth series a Transformer is the least data-efficient choice and was the most seed-sensitive in the scratch run.
4. **Cost and reproducibility.** TensorFlow on native Windows is CPU-only (see [eval/README.md](../eval/README.md#lstm-on-windows-gpu)). Results vary with the seed, so a fair comparison needs several seeds per model, each scored with the same significance rules. With about 25 days, a day-block test has little power to separate two models that differ by tenths of a minute.
5. **The report's main risk is elsewhere.** The open questions are whether camera counts add anything ([handoff-camera-pilot.md](handoff-camera-pilot.md)), whether any horizon beyond 30 minutes holds up, and the fact that the label is Maps' own estimate ([findings.md](findings.md)). A more complex forecaster answers none of them.

## When to revisit

Reconsider when at least one of these holds:

- **At least 3 months of history**, so that there are enough independent days for training and a day-block test with at least 10 blocks.
- **Multi-horizon or 24-hour targets.** Sequence-to-sequence models can predict the whole horizon in one pass, where XGBoost needs one model per horizon.
- **Several aligned input series** (both directions, camera counts, weather, holidays) that a shared sequence encoder could learn from jointly.
- **A camera count series with enough coverage** to test whether queue dynamics help beyond Maps lags.

## How to test it properly

If it is revisited, a deep-learning row counts only if it follows the same protocol as the rows already in the report:

1. Train and score on the same rows as offline XGBoost (`split_supervised` in `timeseries_xgb.py`; `build_lstm_split` in `timeseries_lstm.py` already aligns to them).
2. Run several seeds and report the mean and spread. `timeseries_lstm._require_keras()` currently resets the seed to 42 on every model build, so it has to take a seed argument first.
3. Compare against persistence and XGBoost with `significance.compare_absolute_errors` (Diebold–Mariano, Holm, day-block bootstrap). If the run has fewer than 10 day blocks, the decision is "insufficient data".
4. Write the result into a run folder via `generate_comparison_plots.py` and promote it with `promote_report_run.py` before any number appears in the docs.

## Cheaper alternatives first

- **LightGBM or tuned XGBoost** on the joined feature table (`features.py`). This is fast, deterministic and uses the same significance path.
- **Multi-horizon XGBoost** (one model per horizon, or the horizon as a feature) to test 60 minutes and beyond on both directions before any 24-hour claim.
- **Quantile regression** (XGBoost or LightGBM) for prediction intervals, which a traveller-facing forecast needs more than a small MAE gain.
