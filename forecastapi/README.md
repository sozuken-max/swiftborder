# forecastapi

Public HTTP read of the Woodlands 30-minute Maps-duration forecast, served from local models that the service fits itself, once per SGT day, on the same rows and settings as the evaluation harness ([ADR 0004](../docs/adr/0004-serve-local-models.md)). The response fields below are this tree. They are not on live revision `forecast-api-00009-hax` until this merges and the `forecast-api` trigger runs.

- **Callable ids:** `served` (per-direction selection), `persistence`, `xgb[maps]`, `ridge[maps]`, `xgb_bq[daily]`, `lin_bq[daily]`, `xgb_bq[frozen]` and `lin_bq[frozen]`. Other ids in `GET /?list=models` are harness results and answer 400.
- **CI/CD:** [cloudbuild.yaml](cloudbuild.yaml) runs tests, builds the image, deploys a no-traffic candidate, smoke-tests it, then promotes it (trigger `949ff029`, `main` only).
- **Scope:** it is not `camdetect`, and it does not deploy `swiftbackend`.

Runbook: [docs/runbooks/forecast-api.md](../docs/runbooks/forecast-api.md).

## Response

Top-level `model` is the id the caller asked for (`served` by default). Each direction, and each point on `?curve=forecast`, has its own `model`: the id that produced that value. A single-model value omits `components`. A blend adds `components`, a list of `{model, weight}` whose weights sum to 1. The service does not fill in weights.

`served` at 30 minutes is not a weighted ensemble. It is the registry choice `lin_bq[frozen]` for `SG_TO_MY` and `persistence` for `MY_TO_SG`. On the forecast curve the model also changes with lead time: that 30-minute choice, then `xgb[maps]` at 60 minutes, `xgb[maps+prof]` from 90 minutes through 5.5 hours, then `profile`.

```json
{
  "model": "served",
  "version": "registry-2026-09-12-local-replica",
  "horizon_min": 30,
  "directions": {
    "SG_TO_MY": {"forecast_min": 19.9, "model": "lin_bq[frozen]"},
    "MY_TO_SG": {"forecast_min": 35.5, "model": "persistence"}
  }
}
```

A blend point (not what `served` returns) looks like `"model": "mean[models]"` plus `"components": [{"model": "lin_bq[frozen]", "weight": 0.5}, {"model": "persistence", "weight": 0.5}]`.
