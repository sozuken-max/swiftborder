# forecastapi

Private HTTP read of the Woodlands 30-minute Maps-duration forecast, served from local models that the service fits itself, once per SGT day, on the same rows and settings as the evaluation harness ([ADR 0004](../docs/adr/0004-serve-local-models.md)).

- **Callable ids:** `served` (per-direction selection), `persistence`, `xgb[maps]`, `ridge[maps]`, `xgb_bq[daily]`, `lin_bq[daily]`, `xgb_bq[frozen]` and `lin_bq[frozen]`. Other ids in `GET /?list=models` are harness results and answer 400.
- **CI/CD:** [cloudbuild.yaml](cloudbuild.yaml) runs tests, builds the image, deploys a no-traffic candidate, smoke-tests it, then promotes it (trigger `949ff029`, `main` only).
- **Scope:** it is not `camdetect`, and it does not deploy `swiftbackend`.

Runbook: [docs/runbooks/forecast-api.md](../docs/runbooks/forecast-api.md).
