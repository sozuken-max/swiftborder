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

## Commute Buddy (`?advice=commute`)

Feeds the Commute Buddy card on the Hosting forecast page. Code: [`commute_advisor.py`](commute_advisor.py); calendar: [`holidays_sg_my.csv`](holidays_sg_my.csv); tests: [`tests/test_commute_advisor.py`](tests/test_commute_advisor.py). Generated text, not a measured result; not part of the evaluated forecast.

- **Input:** none from the caller. It uses the same cached 5-hour curve as `?curve=forecast&hours=5` (`cached_curve`), plus the latest reading in the public `traffic-24h.json`.
- **Facts, in code:** for each direction, the best window to leave from now to 5 h ahead. Leaving now counts as a slot (the latest Maps reading, if under 20 min old). The window is the run of slots within 2 min of the best one. Also the worst time, `vs_typical_avg_min`, `biggest_gap_vs_typical`, and every slot's time and forecast for the page's clock. Calendar: SG/MY public holidays, eves, SG long weekends, school holidays, and public holidays in the next 21 days.
- **LLM:** OpenAI Chat Completions (`OPENAI_MODEL`, default `gpt-4o-mini`) returns JSON with one `insight` line (at most 22 words) and at most 2 `tips`. The insight names the direction furthest from usual, by how much and when, and the likely cause from the calendar and a few well-known Causeway patterns in the prompt (Friday evenings and holiday eves heavy towards Johor, Sunday evenings and holiday ends heavy towards Singapore, weekday commuter peaks). It must hedge ("likely") and may not invent events, accidents, weather or checkpoint conditions. No events or incident APIs are read.
- **Cost:** one cached answer per forecast origin (10 min) per instance. A failed OpenAI call is retried after 60 s, not per visit.
- **Key:** `OPENAI_API_KEY` if set (local runs), else Secret Manager secret `openai-api-key` in `BQ_PROJECT`, read at run time with the instance's metadata token and cached for an hour. The key is **not** mounted at deploy, so a missing secret or permission cannot fail a deploy. The response then has `facts`, `advice: null` and `advice_error`, and the card shows the clocks without AI text.
- **Errors:** `400` for another `advice` value, `503` when there is no servable curve, `502` on an unexpected failure. The other endpoints are unchanged.
- **School holidays** are filled in for 2026 only. For other years the service treats them as unknown and says nothing, rather than "none". Add 2027 terms before January 2027.

`200` body (abridged):

```json
{"kind": "commute advice", "llm_model": "gpt-4o-mini",
 "facts": {"directions": {"SG_TO_MY": {"best": {"window": "15:10-15:40", "window_start_iso": "...", "window_end_iso": "...",
                                                "crossing_min": 32.1, "leave_now": false},
                                       "vs_typical_avg_min": 8.2, "biggest_gap_vs_typical": {...}, "forecast": [...], "now": {...}},
                          "MY_TO_SG": {...}},
           "calendar": {"weekday": "Friday", "sg_school_holidays": "start 2026-11-21 (in 8 days)", ...}},
 "advice": {"insight": "To Johor runs about 8 min slower than usual, peaking at 5:40 PM, likely the Friday evening rush.",
            "tips": ["...", "..."]}}
```

**One-time setup (project owner).** Nothing here changes the deploy; until it is done the card shows the clocks without AI text.

```bash
gcloud services enable secretmanager.googleapis.com --project=swiftborder
```

```bash
read -rs OPENAI_KEY && printf '%s' "$OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=- --replication-policy=automatic --project=swiftborder; unset OPENAI_KEY
```

```bash
gcloud secrets add-iam-policy-binding openai-api-key --member=serviceAccount:forecast-api@swiftborder.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor --project=swiftborder
```

A new key version is picked up within an hour, or on the next instance start.

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
