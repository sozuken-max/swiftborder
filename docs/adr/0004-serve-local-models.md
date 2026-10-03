# Serve the 30-minute forecast from local models, not BigQuery ML

## Status

Proposed (2026-10-03). The team accepts or changes it before the frozen run. The model selection itself waits for that run (on or after 20 Oct).

## Context

- The evaluation is local only. BigQuery ML stays out of it ([roadmap.md](../roadmap.md#frozen-window-run-plan-proposed-2026-10-03-the-team-confirms-before-19-oct-2359-sgt)). The local replicas of `lin_h30` and `xgb_h30` forecast as well as the BQML models: within ±0.5 min, slightly better in fact ([evaluation.md §1a](../evaluation.md#1a-local-replicas-of-the-bqml-models-evalbqml_paritypy)). The team does not want to keep maintaining the BQML models.
- `forecast-api` should serve a **selection of the local models**. Today its callable ids (`served`, `lin_h30`, `xgb_h30`) all go through BigQuery ML.
- The policy being evaluated is the **daily refit**: one model per day, trained on every label observed before 00:00 SGT that day (`joined.fold_split`).
- `v_training_set` already holds the serving features, and the harness rebuilds the same columns locally with zero difference (parity check in `run.json`). A model trained in `eval/` can therefore score the latest `v_training_set` row directly.
- Hard constraint: `forecast-api` is built from `forecastapi/` only (`pack --path=forecastapi`), so it cannot import `eval/`.
- This is a student project. Nothing depends on the served values staying the same, so the main risk is a mismatch between what was evaluated and what is served.

## Options

| | A. Fit in the service, once per day | B. Scheduled training job, artifacts in GCS | C. Batch predictions into a table |
| --- | --- | --- | --- |
| How | On the first request of an SGT day, `forecast-api` reads `v_training_set` rows whose label was observed before 00:00 SGT, fits the selected models (seeded), and caches them in memory until the next day | A Cloud Run Job, triggered daily by Cloud Scheduler, fits the models, writes `model.json` / `.joblib` plus metadata to a bucket; the service loads them by version | A job writes forecasts every 5 minutes into BigQuery; the service reads the table |
| Served = evaluated | **Exactly**: the same rows and seeds as the harness fold for that day | Same, if the job uses the same code | Same, if the job uses the same code |
| New GCP pieces | None (one more query per instance per day) | Bucket, job, scheduler, IAM for both | Job, scheduler, table, IAM |
| Versioning / rollback | The version is the training cutoff (`labels_before=YYYY-MM-DDT00:00+08:00`) plus the image commit; roll back by redeploying the previous image | Real artifact versions; `version=` selects one | Per-row model tag in the table |
| Cold start | One BigQuery read (about 15k rows) plus a fit: ridge is instant; XGBoost (300 trees, ~15k rows) takes about a second | Load a small file | None |
| Image size | Adds scikit-learn, XGBoost and pandas (a few hundred MB) | Same in the service, plus the job image | Smallest service |
| Fits the timeline (deploy before 31 Oct) | Yes: one service change | Tight | Tight, over-built |

## Decision (proposed)

**Option A.** It serves the model the evaluation scored, with no new GCP resources, and it matches the daily-refit policy by construction. Option B is the upgrade path if the service ever needs explicit artifact versions or faster cold starts.

**Shape of the change**

1. `forecastapi/local_models.py`: the model definitions the service fits.
   - `xgb[maps]` and `ridge[maps]` use the harness settings in `eval/joined.py`.
   - `lin_bq` and `xgb_bq` use `eval/bq_replica.py`.
   - The features are the `v_training_set` columns plus a 0/1 direction flag.
   - A repo test asserts that the settings and feature lists equal the `eval/` definitions, so the two copies cannot drift.
2. Catalog. Each local model is a callable id with `deploy_state: local`, for example `xgb_maps_daily`, `ridge_maps_daily` and `persistence`. `served` becomes a per-direction selection among those ids, kept in `main.py` like the old registry and set from the frozen run. The BQML ids (`lin_h30`, `xgb_h30`) and the `v_forecast_recent` path are removed once the local path is live.
3. Response. `version` is the training cutoff, and `model_meta` records the rows used, the seeds and the image commit.
4. IAM: none new. `forecast-api@` already reads `traffic_prediction` and `causeway`.
5. Tests: a fixture-based fit and predict test; the cache refreshes at the SGT day boundary; two requests on the same day reuse one fit.

**Selection rule for `served`** (fixed before the frozen run): for each direction, serve the local model that beats persistence in Run A and is confirmed in Run B, both significant under the run-wide Holm check. Otherwise serve persistence.

**Open question for the team:** the 0.5-minute practical threshold in the frozen-run plan was written for "change what is served". On 13–30 Sep no 30-minute model beat persistence by 0.5 min (`xgb[maps]` −0.37). Applied literally, the threshold would keep persistence in both directions. Decide before the run whether it applies to this selection, or only to the report's "product-relevant" wording.

## Consequences

- BigQuery ML models and `v_forecast_recent` are no longer maintained. They stay in the project, unchanged, until someone deletes them, and the report describes them as history.
- `forecast-api` gets heavier dependencies and one BigQuery read per instance per day.
- The served forecast changes once a day, at 00:00 SGT, and is reproducible from the harness for any day.
