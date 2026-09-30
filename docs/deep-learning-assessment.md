# Deep-learning forecasters (LSTM, GRU, Transformer): assessment

**Decision (2026-10-01):** keep the deep models as a scored, documented comparison; do not serve them. At the current data size they do not beat XGBoost, and combining them with XGBoost does not help either.

**Scope:** Layer B, forecasting the Google Maps duration series 60 minutes ahead on `jb_to_woodlands`. Layer A already uses deep learning (Roboflow YOLO), so the Practice Module "ML/DL techniques" criterion does not depend on this result ([grading lens](grading/nus-iss-practice-module.md)).

**Evidence:** [evaluation.md §4](evaluation.md#4-deep-sequence-models-60-minutes-one-route-evaldeep_forecastpy) and [§6](evaluation.md#6-ensembles-and-hybrids-of-the-layer-b-models-evalensemblepy), from `eval/runs/report/run.json`.

## What was built

All models use the same rows as offline XGBoost and one training protocol ([`eval/deep_forecast.py`](../eval/deep_forecast.py)), so differences come from the architecture:

- **LSTM(64)** and **GRU(64)** ([`timeseries_lstm.py`](../eval/timeseries_lstm.py)).
- **Patch Transformer** ([`timeseries_transformer.py`](../eval/timeseries_transformer.py)), designed for a short, single, smooth series with about 5.7k training windows:
  - **Patching.** Six 30-minute patches replace 36 five-minute tokens, which gives each token local shape and keeps attention small (PatchTST, Nie et al. 2023).
  - **Pre-LayerNorm blocks.** These train stably without warm-up.
  - **Flatten head.** It keeps position information that average pooling would discard.
  - **Persistence-anchored target.** The model predicts the change from the current value, so an under-trained model degrades toward persistence, not toward the mean.
  - **Size.** About 19k weights, matched to the LSTM's 21k.
  - **Ablation.** The raw-target version isolates the anchoring.
- **Training.** Huber loss, AdamW, early stopping on the last 15% of training rows, three seeds, seed-mean forecast. Seeds are now respected: `LSTMTrainConfig.seed` and `TransformerConfig.seed`. Before this, `_require_keras()` reset every model to seed 42.

## What it showed

- XGBoost 3.469 min MAE. LSTM 3.970, GRU 4.030, patch Transformer 4.192, Transformer without anchoring 4.491. Persistence 4.677.
- Seed spread is about 0.1–0.2 min per architecture.
- All three deep models point to an improvement over persistence; all are about 0.5–0.7 min behind XGBoost.
- Anchoring gained the Transformer about 0.3 min.
- Every decision is "insufficient data": the hold-out has 5 day-blocks.
- An equal mean of XGBoost and the deep models (3.655) and a rolling stack (3.561) did not beat XGBoost alone.

## Why this is expected

1. **Too little data for the parameter count.** The series starts on 6 Sep 2026: about 5.7k training windows on one route and about 20 independent days. XGBoost on engineered lags regularises better at this size.
2. **The signal at 60 minutes is mostly recent level plus time of day.** The lags, D-1/D-7 values and calendar features that XGBoost uses directly have to be learned from the raw window by a sequence model.
3. **Attention has little to attend to.** Six tokens from one series give attention no long context and no cross-series structure to exploit. The Transformer was the weakest of the three, as expected.
4. **The error that remains is at queue onsets and clearings** ([evaluation.md §6](evaluation.md#6-ensembles-and-hybrids-of-the-layer-b-models-evalensemblepy)), which no Maps-only model anticipates well. A new input (camera counts, [§7](evaluation.md#7-is-layer-a-output-a-meaningful-layer-b-input)) is a better bet than a bigger model.

## When to revisit

- **At least 3 months of history.** This gives enough independent days to train and at least 10 day-blocks in the hold-out.
- **Multi-horizon or 24-hour targets.** A sequence model predicts all horizons in one pass, where XGBoost needs one model per horizon.
- **Several aligned inputs.** Both directions, camera counts, weather and holidays, which a shared encoder could learn from jointly.

Re-run with `generate_comparison_plots.py --deep` (more seeds via `--deep-seeds`) and promote as usual. Numbers go into the docs only from `run.json`.

## Cheaper options first

- LightGBM or tuned XGBoost on the joined feature table (`features.py`).
- Multi-horizon XGBoost on both directions before any 24-hour claim.
- Quantile regression for prediction intervals, which a traveller-facing forecast needs more than a small MAE gain.
