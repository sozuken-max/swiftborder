# forecastapi

HTTP function, not deployed until a `main` build of [cloudbuild.yaml](cloudbuild.yaml) runs. `GET /?list=models` lists every forecast model the eval harness scores, plus the BigQuery models a request can run today. `GET /?model=` returns a forecast only for a served model (`lin_h30`, `xgb_h30`, `persistence`). Other ids answer 400. It is not `camdetect` and it does not deploy `swiftbackend`.

Which id is which, and how each kind should be deployed: [docs/runbooks/forecast-api.md](../docs/runbooks/forecast-api.md).
