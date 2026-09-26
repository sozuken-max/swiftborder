# Causeway weather scripts (CSV research)

One-off **CSV** downloads from data.gov.sg for rainfall and the 2-hour forecast.
Output goes under `Causeway/data/`, which is gitignored.

These scripts are **not** the BigQuery loaders for `rainfall.rainfall` or
`weatherforecast.weatherforecast` in project `swiftborder`. That pipeline lives
outside this repo.

## Scripts

| Script | Role |
| --- | --- |
| `fetch_rainfall_history.py` | Per-day rainfall CSVs under `data/rainfall/` |
| `fetch_forecast_history.py` | Per-day forecast CSVs under `data/forecast/` |
| `filter_station_history.py` | One station from rainfall day files |
| `filter_area_forecast.py` | One area from forecast day files |

## Setup and tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Fetch scripts need network access. Tests are offline.

## Example

```bash
python fetch_rainfall_history.py --days 7
python filter_station_history.py --station-id S210
```
