# forecastapi

HTTP function, not deployed until a `main` build of [cloudbuild.yaml](cloudbuild.yaml) runs. `GET /?list=models` returns the curated catalog in `main.py` (eval ids this service knows, plus the BigQuery models it can query). It is not every harness-scored baseline: the 60-minute majority-level and Maps-typical baselines are not entries. `GET /?model=` returns a forecast only for a callable id: `served`, `lin_h30`, `xgb_h30`, and `persistence`. Other ids answer 400. It is not `camdetect` and it does not deploy `swiftbackend`.

Which id is which, and how each kind should be deployed: [docs/runbooks/forecast-api.md](../docs/runbooks/forecast-api.md).
