# commute-advisor

Cloud Run service (`commute_advice`, `asia-southeast1`) that turns the 5-hour forecast into commuter advice for the **Commute Buddy** card on the Hosting forecast page. It is **not deployed** until the one-time setup below has been run; check project `swiftborder` for what is live.

It is separate from `forecast-api` and never changes it. It reads the same public URLs the page reads:

- `forecast-api ?curve=forecast&hours=5`: the 5-hour curve and the typical-day `baseline_min` at each point;
- `traffic-24h.json`: the latest Maps travel time ("now").

## What it does

1. **Facts, in code** (`build_facts`): for each direction, the best time to leave in the next 5 hours (and the run of 30-minute slots within 2 min of it), the worst time, the average difference from the typical day, and the saving against leaving now.
2. **Calendar, from [`data/holidays_sg_my.csv`](data/holidays_sg_my.csv)** (`calendar_facts`): today's Singapore and Malaysia public holidays, eves and Singapore long weekends; whether school holidays are on or start within 21 days; public holidays in the next 21 days.
3. **Advice, from OpenAI** (`ask_llm`): the facts go to the Chat Completions API (`OPENAI_MODEL`, default `gpt-4o-mini`) with a "seasoned Causeway commuter" system prompt. It may only use the facts, and replies as JSON: `headline`, `to_johor`, `to_singapore`, up to 3 `tips`.

The card shows the times and minutes from `facts`; only the sentences come from the LLM. If OpenAI fails, the response still has `facts` and `advice: null`, and the card shows the times alone. If the service is down, the card stays hidden; the forecast is unaffected.

The advice is generated text, not a measured result. It is not part of the evaluated forecast.

## Holiday calendar

`data/holidays_sg_my.csv`: one row per day, 2025-01-01 to 2027-12-31. Columns: `sg_public`, `my_public`, `sg_school`, `my_school`, `sg_public_eve`, `my_public_eve`, `sg_long_weekend`, `sg_long_weekend_last_day` (0/1), `sg_names`, `my_names`.

**School holidays are filled in for 2026 only.** The service treats a year with no school-holiday rows as unknown and says nothing about school holidays then, rather than "none". Add 2027 school terms (MOE and the Malaysian school calendar) before January 2027. Public holidays cover 2025 to 2027.

## API

`GET /` (no parameters). `200`:

```json
{"kind": "commute advice", "llm_model": "gpt-4o-mini",
 "facts": {"generated_at_sgt": "...", "forecast_origin": {...},
           "directions": {"SG_TO_MY": {"best": {"leave_at": "15:40", "crossing_min": 32.1, "window": "15:10-15:40"},
                                       "worst": {...}, "vs_typical_avg_min": 8.2, "now": {...}, "forecast": [...]}, "MY_TO_SG": {...}},
           "calendar": {"weekday": "Friday", "sg_school_holidays": "start 2026-11-21 (in 8 days)", ...}},
 "advice": {"headline": "...", "to_johor": "...", "to_singapore": "...", "tips": ["..."]},
 "note": "..."}
```

`503` when `forecast-api` has no forecast, `405` for anything but `GET`/`OPTIONS`. CORS `*` (`ALLOWED_ORIGIN`): it takes no input, so preview channels may call it.

**Cost control.** No input reaches the prompt. One answer per forecast origin (a new origin every 10 minutes) is cached per instance, an answer is reused for 60 s without re-reading `forecast-api`, a failed OpenAI call is retried after a minute rather than per visitor, and `--max-instances=2`.

## One-time setup (project owner)

Run once, in this order. Nothing here touches `forecast-api`.

```bash
gcloud services enable secretmanager.googleapis.com --project=swiftborder
```

Store the OpenAI key without it landing in shell history:

```bash
read -rs OPENAI_KEY && printf '%s' "$OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=- --replication-policy=automatic --project=swiftborder; unset OPENAI_KEY
```

Runtime service account, allowed to read only that secret:

```bash
gcloud iam service-accounts create commute-advisor --display-name="commute-advisor runtime" --project=swiftborder
```

```bash
gcloud secrets add-iam-policy-binding openai-api-key --member=serviceAccount:commute-advisor@swiftborder.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor --project=swiftborder
```

Let the Cloud Build service account deploy as it (the same account the other triggers build as):

```bash
gcloud iam service-accounts add-iam-policy-binding commute-advisor@swiftborder.iam.gserviceaccount.com --member=serviceAccount:1095552466513-compute@developer.gserviceaccount.com --role=roles/iam.serviceAccountUser --project=swiftborder
```

Trigger on push to `main` under this folder (same GitHub connection as the `forecast-api` trigger):

```bash
gcloud builds triggers create github --name=commute-advisor --repo-owner=sozuken-max --repo-name=swiftborder --branch-pattern='^main$' --build-config=services/commuteadvisor/cloudbuild.yaml --included-files='services/commuteadvisor/**' --service-account=projects/swiftborder/serviceAccounts/1095552466513-compute@developer.gserviceaccount.com --project=swiftborder
```

First deploy, from `main`:

```bash
gcloud builds triggers run commute-advisor --branch=main --project=swiftborder
```

The build tests, deploys, and smoke-tests the service. Its URL is `https://commute-advisor-1095552466513.asia-southeast1.run.app/`, which is `ADVISOR_API` in `hosting/app.js`. To rotate the key, add a new secret version; the next deploy or instance start picks up `latest`.

## Local test

```bash
pip install -r requirements-dev.txt && python -m pytest -q
```
