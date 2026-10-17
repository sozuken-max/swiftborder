# eval

Evaluation harnesses for Layer A and Layer B. Nothing here deploys `swiftbackend` or writes to GCP: BigQuery access is read-only (`SELECT`, `ML.PREDICT`, model metadata). Results that the report cites live in [`runs/report/`](runs/report/) ([`runs/README.md`](runs/README.md)); the narrative is [docs/evaluation.md](../docs/evaluation.md).

**What the numbers mean:** error on the Google Maps `duration_in_traffic` series (skill vs persistence on that series), not independent wait time, and not "we beat Google".

## Layout

Code lives in topic packages under `eval/`. The root keeps [`run.py`](run.py) (CLI dispatcher), [`_flat_imports.py`](_flat_imports.py) (legacy `import joined` names for tests), and the two warehouse entrypoints Cloud Build copies into `forecast-api` ([`layer_b_fcm_mlp.py`](layer_b_fcm_mlp.py), [`layer_b_fcm_xgb_mix.py`](layer_b_fcm_xgb_mix.py)).

```text
eval/
  run.py                 # python run.py <name> [args...]
  harness/               # run.json, promotion, significance, report driver
  lb/                    # Layer B scoring (joined, features, ensemble, …)
  timeseries/            # 60-min offline XGB, LSTM, Transformer, deep_forecast
  la/                    # Layer A scorer, rescoring, camera backfill
  studies/               # horizon study, layer_a_eval, counts_substitute, …
  warehouse/             # FCM+MLP, TimesFM, mix (implementations; two shims at root for deploy)
  tests/  data/  runs/  notebooks/  test-reports/
```

Run any harness from `eval/`:

```bash
python run.py joined --start 2026-09-13 --end 2026-09-30
python -m lb.joined --start 2026-09-13 --end 2026-09-30   # same
python run.py --help                                      # lists module names
```

### Packages (selected modules)

| Package | Module | Role |
| --- | --- | --- |
| `lb` | `layer_b` | Production models at 30 min on a **fixed** window: persistence, `lin_h30`, `xgb_h30`, ensemble; slices by direction, time of day, day type, day/night |
| `lb` | `joined` | Offline joined-feature experiment at 30 min: rolling daily folds; Maps vs + weather vs + camera |
| `lb` | `features` | Causal 10-min feature table (`v_training_set` parity), rainfall, forecast, camera counts |
| `lb` | `fuzzy_traffic`, `ensemble`, `camera_forecast`, `bq_replica`, `bqml_parity` | Fuzzy classifier, ensembles/hybrids, camera profile features, local BQML replicas |
| `timeseries` | `timeseries_xgb`, `deep_forecast`, … | 60-min offline models and deep sequence scoring |
| `harness` | `generate_comparison_plots`, `promote_report_run`, `replay_series` | Full report run, promotion to `runs/report/`, CSV replay from `run.json` |
| `harness` | `significance`, `metrics`, `plots`, `run_artifacts` | DM/Holm, MAE/RMSE, figures + CSV sidecars, schema-v2 manifests |
| `la` | `layer_a`, `score_layer_a_*`, `backfill_camera_counts`, … | Box/count scorer, rescoring, Roboflow backfill |
| `studies` | `horizon_study`, `layer_a_eval`, `counts_substitute`, `camera_current` | Exploratory studies (not the promoted report run) |
| `warehouse` | `layer_b_fcm_mlp`, `layer_b_timesfm`, `layer_b_fcm_xgb_mix` | Exploratory Layer B warehouse jobs ([mix docs](../docs/layer-b-mix-module.md)) |

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
# weather CSVs for the joined experiment (see ../data/weather/README.md)
python ../data/weather/fetch_rainfall_history.py --start-date 2026-09-05 --end-date 2026-09-30
python ../data/weather/fetch_forecast_history.py --start-date 2026-09-05 --end-date 2026-09-30
python ../data/weather/filter_station_history.py
python ../data/weather/filter_area_forecast.py
# all components into one run folder, then validate and promote
pip install -r requirements-notebook.txt   # TensorFlow for --deep
python run.py generate_comparison_plots --bqml --window-end "2026-09-30 23:50" --joined --deep --fuzzy --ensemble
python run.py promote_report_run --check
python run.py promote_report_run <run_id>
```

`--ensemble` needs `--bqml` and `--joined` (it reuses their out-of-sample rows) and adds the 60-minute pool when `--deep` is given. The offline component reads the cached `data/causeway_gdata.csv`. Add `--refresh-bq` only when that cache is missing: it re-downloads `travel_times`, which now has rows past the report cut, so the offline split and its numbers will differ from the promoted run. The BQML window is pinned by `--window-end` either way. Promotion refuses a run made from a dirty tree unless `--allow-dirty` is given; do not use that for the report.

Each figure PNG has a CSV of the same stem beside it (`offline/holdout-sample.png` and `offline/holdout-sample.csv`, and the same pattern for `bqml/`, `joined/` including `camfc-profiles`, `deep/`, `fuzzy/`, and `ensemble/`). The CSV header is the series that figure plots. A figure that exists only with `--bqml`, `--joined`, `--deep`, `--fuzzy`, or `--ensemble` gets its CSV in that same branch. `run.json` lists both paths under the component `artifacts`, relative to the run folder.

The committed snapshot in [`runs/report/`](runs/report/) stays as cited. Its numbers are unchanged. Figure CSVs for that snapshot were replayed from its `run.json` (no model fit, no BigQuery) into [`runs/replay-20261003T035807Z/`](runs/replay-20261003T035807Z/) (any `runs/replay-<run_id>/*/*.csv` is tracked by git). Do not refit the snapshot to add them. Re-running `generate_comparison_plots.py` with the same flags and the cached `data/causeway_gdata.csv` reproduces these numbers on this machine: the runs from `b89bc4e`, `eb435dd` and `b3b6562` (the last two span the wavelet code, which is off by default) gave identical metrics, decisions and per-seed deep MAEs on the same TensorFlow build. A re-run would still be a **new** run: new run id and `code_sha256` (the plot writers changed the code), so it would need re-promoting and every doc reference updating. The team chose not to. The numbers can move only if the inputs move: `--refresh-bq` pulls `travel_times` rows past the cached export, a different TensorFlow build or CPU can change deep rows in the last digits, and the 2026-10-19 cutoff applies to the next, longer window, not to this pinned 13–30 Sep one. Series for every figure except `offline/holdout-sample` are already in `run.json` (MAE, bootstrap CIs, seeds, profile curves, confusion counts, fuzzy cuts). Replay them with no model fit and no BigQuery:

```bash
python run.py replay_series path/to/run_dir --output-dir path/to/csv_dir
```

That command writes the CSVs only and refuses `runs/report/`. Omit `--output-dir` to write next to the source `run.json` when that directory is not the snapshot. Older schema-v2 promotions (`20260930T202959Z`, `20260930T194805Z`, `20260930T191544Z`, `20260930T175104Z`) replay the same way from the `run.json` in the commit that promoted them. `20260926T073406Z` is schema v1 and is not replayable. The hold-out tail (timestamps, actual minutes, and the XGB and persistence traces) was never stored, so `offline/holdout-sample` is absent from the replay and that one CSV needs a new run.

**Data cutoff.** `--data-cutoff "2026-10-19 23:59"` (SGT) drops every `travel_times` observation, rainfall reading and forecast acquired after that time in all components, so no scored label is later than the cutoff. It also sets the BQML `--window-end` to the last origin whose 30-minute label is observed by then, refuses an explicit `--window-end` or `--joined-end` past it, and records `data_cutoff` in `run.json`. `promote_report_run.py --target report-confirm` promotes the October-only confirmation run beside `runs/report/`. The frozen-run commands are in [docs/roadmap.md](../docs/roadmap.md#frozen-window-run-plan-proposed-2026-10-03-the-team-confirms-before-19-oct-2359-sgt).

Individual harnesses:

```bash
python run.py layer_b                                   # print tables (read-only BigQuery)
python run.py layer_b --plots --window-end "2026-09-30 23:50"
python run.py joined --start 2026-09-13 --end 2026-09-30
python run.py features --build --parity                 # feature table + parity with the live view
python run.py layer_a --coco <export>/_annotations.coco.json --predictions preds.json --frame-times times.csv
```

## Window-feature ablation (offline XGB)

Chad's day-2 note: a discrete wavelet gives trees a trend and a bend that raw lags do not. This is an extra comparison in [`timeseries_xgb.py`](timeseries_xgb.py), not a change to the model in [`runs/report/`](runs/report/). `use_dwt` stays off for that snapshot.

Same chronological split, same `XGBTrainConfig`, same raw labels, same persistence baseline as the offline 60-minute path. The three sets are only the window columns (calendar, D-1/D-7, and origin `duration_sec` stay on the default model and out of this comparison):

| Set | Columns |
| --- | --- |
| A | lags (`target_lag_*`, default 12 bins) |
| B | A, plus the first difference (one 5-minute bin) and a causal rolling mean |
| C | A, plus the window mean, the window standard deviation, and wavelet coefficients |

The rolling mean uses 12 bins (60 minutes), the same span as `keep_lags` and the 60-minute horizon, ending at the forecast origin. The wavelet is Daubechies **db2 at level 2** (level 3 is `--dwt-level 3`). Each window is z-scored before the transform so the coefficients describe shape; mean and std are kept as columns so the level is not discarded. A zero-std window becomes a zero shape and still emits mean and std. **db4 is not the default** because it smooths the bends. Nothing in the window is after the origin.

```bash
cd eval
python run.py timeseries_xgb
python run.py timeseries_xgb --dwt-level 3 --json-out runs/wavelet-ablation.json
```

The cache is `data/causeway_gdata.csv` (gitignored). The command prints MAE and RMSE in minutes against persistence. Significance follows [`significance.py`](significance.py): Diebold-Mariano, day-block bootstrap, Holm. Below 10 day-blocks the decision is **insufficient data** and the command reports point estimates only (no p-values). Leave those estimates out of `docs/evaluation.md` and `runs/report/` unless a promoted run produced them.

Local run of `python timeseries_xgb.py --json-out runs/wavelet-ablation.json` on `data/causeway_gdata.csv` (sha256 `13a119980c640061a2b7d9cc8cae7909787771314bf6a1391c7cfb492b9bbf07`, 3,484,144 bytes). Route `jb_to_woodlands`, 7,183 rows, observed 2026-09-06 01:53:30 to 2026-10-01 00:30:03 SGT. Split boundary 2026-09-26 01:35 (5,710 train / 1,428 test), the same boundary as section 3 of [docs/evaluation.md](../docs/evaluation.md). A/B/C use only the window columns above. Persistence T-60 on that hold-out: MAE 4.6770 min, RMSE 6.3858 min.

| Set | MAE (min) | RMSE (min) |
| --- | --- | --- |
| A | 4.0830 | 5.4636 |
| B | 4.0722 | 5.4507 |
| C (db2, level 2) | 3.9919 | 5.4846 |

Printed to 4 decimal places (unrounded values are in the gitignored JSON). The test labels cover 6 calendar dates and **5 day-blocks** under the harness rule, so the decision is **insufficient data**. These are point estimates of error on the Maps duration series for this window comparison.

## Camera backfill (Roboflow free tier)

`backfill_camera_counts.py` spends Roboflow credits (one call per sampled bin). The free tier has no overage billing: when credits run out, calls fail until the monthly reset.

```bash
python backfill_camera_counts.py --dry-run                                     # count calls; no requests
export ROBOFLOW_API_KEY="<key>"                                                 # never commit it
python backfill_camera_counts.py --max-calls 50 --start 2026-09-20 --end 2026-09-20   # pilot
python backfill_camera_counts.py --coverage
```

In PowerShell set the key with `$env:ROBOFLOW_API_KEY = "<key>"` instead of `export`. The full procedure is in [docs/archive/handoff-camera-pilot.md](../docs/archive/handoff-camera-pilot.md).

Check the credit change on the Roboflow Credit Usage page after the pilot before a full run (2,730 calls for 5–30 Sep peak-first; 3,743 with `--mode full`). A quota or payment error stops the run with everything so far saved; re-run to resume.

## LSTM on Windows GPU

Native Windows TensorFlow 2.11+ is CPU-only. For DirectML use a separate Python 3.10 venv with [`requirements-tf-gpu-windows.txt`](requirements-tf-gpu-windows.txt) and `python check_tf_gpu.py`; it conflicts with the pinned TensorFlow in `requirements-notebook.txt`.

```bash
python train_lstm.py --mode train --architecture bilstm_attention
python train_lstm.py --mode compare
```

Notebook: [notebooks/causeway_xgb_timeseries.ipynb](notebooks/causeway_xgb_timeseries.ipynb). Local data caches: [data/README.md](data/README.md).
