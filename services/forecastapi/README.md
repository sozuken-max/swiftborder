# forecastapi

Public HTTP read of the Woodlands 30-minute Maps-duration forecast, served from local models that the service fits itself, once per SGT day, on the same rows and settings as the evaluation harness ([ADR 0004](../../docs/adr/0004-serve-local-models.md)).

- **Callable ids:** `served` (per-direction selection), `mix[fcm_mlp+xgb]`, `persistence`, `xgb[maps]`, `ridge[maps]`, `xgb_bq[daily]`, `lin_bq[daily]`, `xgb_bq[frozen]` and `lin_bq[frozen]`. Other ids in `GET /?list=models` are harness results and answer 400.
- **CI/CD:** [cloudbuild.yaml](cloudbuild.yaml) runs tests (`services/forecastapi` + `eval/tests/test_forecastapi_models.py`), builds the image (`--path=services/forecastapi`), deploys a no-traffic candidate, smoke-tests it, then promotes it (trigger `949ff029`, `main` only; `services/forecastapi/README.md` and `services/forecastapi/tests/**` are ignored).
- **Scope:** it is not `camdetect`, and it does not deploy `swiftbackend`.

Runbook: [docs/runbooks/forecast-api.md](../../docs/runbooks/forecast-api.md).

## Response

Top-level `model` is the id the caller asked for (`served` by default). Each direction, and each point on `?curve=forecast`, has its own `model`: the id that produced that value. A single-model value omits `components`. A blend adds `components`, a list of `{model, weight}` whose weights sum to 1. The service does not fill in weights.

`served` at 30 minutes is `mix[fcm_mlp+xgb]` in both directions: half Fuzzy C-Means MLP and half daily-refit `xgb[maps]`. That is the lowest measured September mean absolute error on the Maps series, and the weights were chosen after that window was scored. It is not a pre-registered Holm result, not a 24-hour forecast, and not a crossing-time error. On the forecast curve the model changes with lead time: the mix at 30 minutes, `xgb[maps]` at 60 minutes, `xgb[maps+prof]` from 90 minutes through 5.5 hours, then `profile`.

```json
{
  "model": "served",
  "version": "mix-equal-20261010T080013Z",
  "horizon_min": 30,
  "directions": {
    "SG_TO_MY": {"forecast_min": 19.9, "model": "mix[fcm_mlp+xgb]"},
    "MY_TO_SG": {"forecast_min": 35.5, "model": "mix[fcm_mlp+xgb]"}
  }
}
```

A blend point (not what `served` returns) looks like `"model": "mean[models]"` plus `"components": [{"model": "lin_bq[frozen]", "weight": 0.5}, {"model": "persistence", "weight": 0.5}]`.

## Model cards

`GET /?list=cards` (same body as `GET /?view=cards`) returns a JSON array the frontend can render as cards. `GET /?list=models` is unchanged. A card's `callable` flag matches the forecast allow-list. `timesfm` is a card only. `fcm_mlp` is not its own query id; the mix calls that network as one half.

```json
[
  {
    "id": "served",
    "name": "Served default",
    "summary": "Predicts Maps travel time. At 30 minutes both directions use mix[fcm_mlp+xgb].",
    "kind": "mix",
    "callable": true,
    "default_for": "request with no model parameter; public curve (?curve=forecast)",
    "members": ["mix[fcm_mlp+xgb]", "xgb[maps]", "xgb[maps+prof]", "profile"]
  },
  {
    "id": "profile",
    "name": "Typical day",
    "summary": "What the Hosting chart labels Typical. It is the calendar profile, a single baseline.",
    "kind": "single",
    "callable": true,
    "default_for": "Hosting typical-day line (?baseline=profile); public curve after 5.5 h"
  }
]
```
