# eval

Read-only evaluation helpers outside `camdetect/` so changes here do not redeploy `swiftbackend`.

## Layer B — two evaluation paths

| Path | Data | Horizon | Models | Status in report |
| --- | --- | --- | --- | --- |
| **Offline (canonical BQ table)** | Live `sync_canonical_travel_times` + CSV cache [`data/causeway_gdata.csv`](data/README.md) | **60 min** (`H=12` × 5 min) | sklearn XGB on **`jb_to_woodlands` (JB → SG only)**; LSTM hook in [`timeseries_lstm.py`](timeseries_lstm.py) | **Scored** — see [evaluation.md](../docs/evaluation.md) |
| **BQML harness** | Live read-only `traffic_prediction.v_training_set` | **30 min** (`y_30`) | persistence, `lin_h30`, `xgb_h30` | Optional cross-check of **serve** models; not the same split as the CSV run |

**What numbers mean:** error on the Google Maps `duration_in_traffic` series (skill vs persistence on that series), not independent wait time and not "we beat Google."

### Offline (full `travel_times` export)

```bash
cd eval
pip install -r requirements-dev.txt
jupyter notebook notebooks/causeway_xgb_timeseries.ipynb
```

Open the notebook; set `REFRESH_FROM_BQ = True` to pull canonical `travel_times`, or use the cache ([data/README.md](data/README.md)).

**LSTM (JB→SG, 60 min):** TensorFlow required. On **Windows** with an NVIDIA GPU, native `pip install tensorflow` is usually **CPU-only** (TF 2.11+); use DirectML instead:

```bash
pip install -r requirements-tf-gpu-windows.txt
python check_tf_gpu.py
```

(Linux/WSL or CPU-only: `pip install -r requirements-notebook.txt`.) Then:

```bash
python train_lstm.py --mode train --architecture bilstm_attention
python train_lstm.py --mode compare
python train_lstm.py --mode tune --max-trials 20 --out runs/lstm_tune.json
```

Architectures and grid: [`timeseries_lstm.py`](timeseries_lstm.py).

### BQML serve-path harness (optional)

```bash
cd eval
pip install -r requirements.txt
python layer_b.py --project swiftborder --holdout-days 3
python layer_b.py --significance   # paired MAE vs persistence (block bootstrap + t-test)
```

`model_registry` is **not** updated unless `--write-registry` is implemented.

Paired significance helpers live in [`significance.py`](significance.py); protocol in [evaluation.md](../docs/evaluation.md#significance-paired-error-differences).

### Comparison plots (per-run folders)

Each harness run writes **`eval/runs/<run_id>/`** (gitignored). For the **final report**, promote one run to **`eval/runs/report/`** (committed): `python promote_report_run.py`. See [`runs/README.md`](runs/README.md).

```bash
pip install -r requirements-dev.txt
python generate_comparison_plots.py              # offline only
python generate_comparison_plots.py --bqml       # offline + BQML
python generate_comparison_plots.py --bqml-only  # BQML only
python layer_b.py --holdout-days 3 --plots     # BQML plots into a new run folder
```

Narrative in [evaluation.md](../docs/evaluation.md#comparison-plots-per-run). Implementation: [`plots.py`](plots.py), [`run_artifacts.py`](run_artifacts.py).

### Offline tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
