# eval

Evaluation harnesses for Layer A and Layer B. Nothing here deploys `swiftbackend` or writes to GCP: BigQuery access is read-only (`SELECT`, `ML.PREDICT`, model metadata). Results that the report cites live in [`runs/report/`](runs/report/) ([`runs/README.md`](runs/README.md)); the narrative is [docs/evaluation.md](../docs/evaluation.md).

**What the numbers mean:** error on the Google Maps `duration_in_traffic` series (skill vs persistence on that series), not independent wait time, and not "we beat Google".

## Modules

| Module | Role |
| --- | --- |
| [`layer_b.py`](layer_b.py) | Production models at 30 min on a **fixed** window after training (default 13 Sep 00:00 SGT to the latest labelled bin): persistence, `lin_h30`, `xgb_h30`, ensemble; slices by direction, time of day, day type, day/night; refuses models trained inside the window |
| [`joined.py`](joined.py) | Offline joined-feature experiment at 30 min: rolling daily folds; Maps vs + weather vs + camera; ridge, XGBoost, ensemble; persistence and Maps-typical baselines |
| [`features.py`](features.py) | Causal 10-min feature table: `v_training_set` logic in pandas (parity-checked against the live view), rainfall, 2-h forecast, camera counts |
| [`timeseries_xgb.py`](timeseries_xgb.py) | Offline 60-min sklearn XGBoost, `jb_to_woodlands` only (JB → SG); causal inputs, raw labels, split on target time |
| [`timeseries_lstm.py`](timeseries_lstm.py), [`train_lstm.py`](train_lstm.py) | Recurrent architectures and a tuning CLI on the same rows as offline XGB (seeded by `LSTMTrainConfig.seed`) |
| [`timeseries_transformer.py`](timeseries_transformer.py) | Patch Transformer encoder (6 × 30-min patches, pre-LN, flatten head, about 19k weights) |
| [`deep_forecast.py`](deep_forecast.py) | Scored deep component: LSTM, GRU, patch Transformer (+ raw-target ablation), one training protocol, 3 seeds, DM/Holm vs persistence and XGB ([assessment](../docs/deep-learning-assessment.md)) |
| [`fuzzy_traffic.py`](fuzzy_traffic.py) | Light / moderate / heavy at 60 min, both directions: fuzzy partition, learned fuzzy rule base, XGB → fuzzy level hybrid; accuracy, macro-F1, severe errors, RPS; DM on 0/1 loss |
| [`camera_forecast.py`](camera_forecast.py) | Layer A queue forecast for Layer B: camera-2701 profile learned from the Mar-Apr detections (cached read-only pull), CV-chosen Fourier ridge; `camfc_*` features and the Maps-profile control used by `joined.py` |
| [`ensemble.py`](ensemble.py) | Ensembles and hybrids of Layer B models: served (registry), equal mean, rolling LAD stack, rolling selection, fuzzy-gated stack; error by regime; optional 60-min XGB + deep pool |
| [`significance.py`](significance.py) | Diebold–Mariano test on loss differences (HAC variance, i.e. corrected for autocorrelation, with at least one day of lags; HLN small-sample correction), moving day-block bootstrap CIs within direction, Holm correction across comparisons |
| [`layer_a.py`](layer_a.py) | Layer A scorer: mAP, precision/recall, count error (overall and per direction), day/night |
| [`backfill_camera_counts.py`](backfill_camera_counts.py) | Camera 2701 counts per 10-min bin via `camdetect.detect_frame` (**billed Roboflow**) |
| [`generate_comparison_plots.py`](generate_comparison_plots.py) | One run folder with offline + BQML + joined components and figures |
| [`run_artifacts.py`](run_artifacts.py), [`promote_report_run.py`](promote_report_run.py) | Schema-v2 `run.json` (provenance, windows, metrics, significance), validation, promotion to `runs/report/` |
| [`plots.py`](plots.py), [`metrics.py`](metrics.py) | Figures and MAE/RMSE helpers |

## Setup and tests

```bash
cd eval
pip install -r requirements-dev.txt        # pinned; Python 3.11
python -m pytest                           # fast suite (slow tests excluded)
python -m pytest -m "slow or not slow"     # + TensorFlow tests (pip install -r requirements-notebook.txt)
```

Or from the repo root: `scripts/run_tests.ps1 [-Slow] [-Coverage]` / `scripts/run_tests.sh [--slow] [--coverage]`. BigQuery calls need Application Default Credentials (`gcloud auth application-default login`).

## Reproduce the report run

```bash
cd eval
# weather CSVs for the joined experiment (see ../Causeway/README.md)
python ../Causeway/fetch_rainfall_history.py --start-date 2026-09-05 --end-date 2026-09-30
python ../Causeway/fetch_forecast_history.py --start-date 2026-09-05 --end-date 2026-09-30
python ../Causeway/filter_station_history.py
python ../Causeway/filter_area_forecast.py
# all components into one run folder, then validate and promote
pip install -r requirements-notebook.txt   # TensorFlow for --deep
python generate_comparison_plots.py --bqml --window-end "2026-09-30 23:50" --joined --deep --fuzzy --ensemble
python promote_report_run.py --check
python promote_report_run.py <run_id>
```

`--ensemble` needs `--bqml` and `--joined` (it reuses their out-of-sample rows) and adds the 60-minute pool when `--deep` is given. The offline component reads the cached `data/causeway_gdata.csv`. Add `--refresh-bq` only when that cache is missing: it re-downloads `travel_times`, which now has rows past the report cut, so the offline split and its numbers will differ from the promoted run. The BQML window is pinned by `--window-end` either way. Promotion refuses a run made from a dirty tree unless `--allow-dirty` is given; do not use that for the report.

Individual harnesses:

```bash
python layer_b.py                                   # print tables (read-only BigQuery)
python layer_b.py --plots --window-end "2026-09-30 23:50"
python joined.py --start 2026-09-13 --end 2026-09-30
python features.py --build --parity                 # feature table + parity with the live view
python layer_a.py --coco <export>/_annotations.coco.json --predictions preds.json --frame-times times.csv
```

## Camera backfill (Roboflow free tier)

`backfill_camera_counts.py` spends Roboflow credits (one call per sampled bin). The free tier has no overage billing: when credits run out, calls fail until the monthly reset.

```bash
python backfill_camera_counts.py --dry-run                                     # count calls; no requests
export ROBOFLOW_API_KEY="<key>"                                                 # never commit it
python backfill_camera_counts.py --max-calls 50 --start 2026-09-20 --end 2026-09-20   # pilot
python backfill_camera_counts.py --coverage
```

In PowerShell set the key with `$env:ROBOFLOW_API_KEY = "<key>"` instead of `export`. The full procedure is in [docs/handoff-camera-pilot.md](../docs/handoff-camera-pilot.md).

Check the credit change on the Roboflow Credit Usage page after the pilot before a full run (2,730 calls for 5–30 Sep peak-first; 3,743 with `--mode full`). A quota or payment error stops the run with everything so far saved; re-run to resume.

## LSTM on Windows GPU

Native Windows TensorFlow 2.11+ is CPU-only. For DirectML use a separate Python 3.10 venv with [`requirements-tf-gpu-windows.txt`](requirements-tf-gpu-windows.txt) and `python check_tf_gpu.py`; it conflicts with the pinned TensorFlow in `requirements-notebook.txt`.

```bash
python train_lstm.py --mode train --architecture bilstm_attention
python train_lstm.py --mode compare
```

Notebook: [notebooks/causeway_xgb_timeseries.ipynb](notebooks/causeway_xgb_timeseries.ipynb). Local data caches: [data/README.md](data/README.md).
