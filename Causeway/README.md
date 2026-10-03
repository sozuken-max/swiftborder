# Causeway weather scripts (CSV research)

Day-by-day **CSV** downloads from data.gov.sg for rainfall and the 2-hour forecast.
Output goes under `Causeway/data/`, which is gitignored.

They feed the offline joined-feature experiment in [`../eval/`](../eval/README.md), and
`load_bigquery.py` appends the Woodlands rows to `rainfall.rainfall` and
`weatherforecast.weatherforecast` in project `swiftborder`. The original loader for those tables
(which filled them to 31 Aug 2026) is not in this repo, and nothing runs these scripts on a schedule.

## Loading into BigQuery

```bash
python load_bigquery.py            # dry run: rows and UTC range per table
python load_bigquery.py --check    # dry run + refuse if BigQuery already has rows in that range
python load_bigquery.py --execute  # snapshot each table (30-day expiry), then append
```

The loader keeps the tables' contract: UTC timestamps, S210 stored as `Woodlands Centre Road`,
one forecast row per issue (latest revision), plus a nullable `update_timestamp`. It only appends,
and refuses when the target already has rows in the load's time range. Needs Application Default
Credentials with write access to the two datasets.

## Scripts

| Script | Role |
| --- | --- |
| `datagov.py` | Shared client: retries, pagination, completeness rules, atomic writes |
| `fetch_rainfall_history.py` | Per-day rainfall CSVs `data/rainfall/rainfall_YYYY-MM-DD.csv` + `stations.csv` |
| `fetch_forecast_history.py` | Per-day forecast CSVs `data/forecast/forecast_YYYY-MM-DD.csv` + `areas.csv` |
| `filter_station_history.py` | One station (default `S210` Woodlands Centre), sorted and de-duplicated |
| `filter_area_forecast.py` | One area (default `Woodlands`); exact duplicates collapsed, revisions kept |
| `load_bigquery.py` | Append the Woodlands rows to the BigQuery weather tables (dry run by default) |

## Completeness rules

- A day's CSV is written only when **every page** was fetched. A failed day (network, 5xx or 429
  after retries, `code != 0`, repeated pagination token) writes nothing and is retried on the next
  run; the script exits 1 and prints a per-day completeness report.
- A 404 on the first page (or zero rows) means no data yet: a `*.nodata` marker is written instead of a CSV, and the day is retried on the next run.
- Forecast rows keep `update_timestamp` (when data.gov.sg acquired the forecast); causal joins use it rather than NEA's issue time.
- Today (SGT) is incomplete, so it is skipped unless `--allow-partial`, which writes
  `*_YYYY-MM-DD.partial.csv`. Filters ignore partial files unless `--include-partial`.
- Future dates are rejected. Writes are atomic (`.tmp` then rename).
- Retries: exponential backoff on network errors, 429 (honours `Retry-After` seconds or HTTP-date)
  and 5xx. Other 4xx (e.g. invalid pagination token) fail the day at once.
- Optional `DATAGOV_API_KEY` in the environment is sent as `x-api-key` for higher rate limits.
  Never commit it.

## Setup and tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests use a mocked HTTP session and are offline. The fetch scripts need network access.

## Example (joined-experiment window)

```bash
python fetch_rainfall_history.py --start-date 2026-09-05 --end-date 2026-09-30
python fetch_forecast_history.py --start-date 2026-09-05 --end-date 2026-09-30
python filter_station_history.py --station-id S210
python filter_area_forecast.py --area Woodlands
```

The 2-hour forecast schema is in `2hourWeatherForecast.json` (OpenAPI).
