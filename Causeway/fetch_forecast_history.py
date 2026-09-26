"""Fetch historical 2-hour weather forecasts across Singapore from data.gov.sg.

Like the rainfall API, the v2 real-time two-hour-forecast API only returns
readings for a single day at a time, paginated via `paginationToken`. To
build a historical dataset we loop day-by-day over the requested window and
page through each day's results.

API: GET https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast?date=YYYY-MM-DD

Usage:
    python fetch_forecast_history.py                     # past 365 days, ending today (SGT)
    python fetch_forecast_history.py --days 30
    python fetch_forecast_history.py --start-date 2025-08-30 --end-date 2026-08-30
    python fetch_forecast_history.py --output-dir ./data/forecast --sleep 0.5
"""

import argparse
import csv
import datetime
import os
import sys
import time
try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python 3.8
    from backports.zoneinfo import ZoneInfo

import requests

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
SGT = ZoneInfo("Asia/Singapore")
MAX_RETRIES = 5


def fetch_day(date_str, session):
    """Return (areas, records) for one day.

    areas: dict area_name -> (latitude, longitude)
    records: list of (issue_timestamp, valid_start, valid_end, area, forecast)
    """
    areas = {}
    records = []
    pagination_token = None

    while True:
        params = {"date": date_str}
        if pagination_token:
            params["paginationToken"] = pagination_token

        resp = _get_with_retries(session, params)
        if resp is None:
            break

        if resp.status_code == 404:
            break
        resp.raise_for_status()

        payload = resp.json()
        data = payload.get("data", {})

        for area in data.get("area_metadata", []):
            loc = area.get("label_location", {}) or {}
            areas[area["name"]] = (loc.get("latitude", ""), loc.get("longitude", ""))

        for item in data.get("items", []):
            issue_timestamp = item.get("timestamp")
            valid_period = item.get("valid_period", {}) or {}
            valid_start = valid_period.get("start")
            valid_end = valid_period.get("end")
            for forecast in item.get("forecasts", []):
                records.append(
                    (issue_timestamp, valid_start, valid_end, forecast.get("area"), forecast.get("forecast"))
                )

        pagination_token = data.get("paginationToken")
        if not pagination_token:
            break

    return areas, records


def _get_with_retries(session, params):
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"  request failed after {MAX_RETRIES} attempts: {exc}", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", delay))
            print(f"  rate limited, backing off {retry_after:.1f}s", file=sys.stderr)
            time.sleep(retry_after)
            delay *= 2
            continue

        return resp

    return None


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += datetime.timedelta(days=1)


def write_day_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["issue_timestamp", "valid_start", "valid_end", "area", "forecast"])
        writer.writerows(records)


def update_areas_csv(path, areas):
    existing = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["area"]] = (row["latitude"], row["longitude"])

    existing.update(areas)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["area", "latitude", "longitude"])
        for area, (lat, lon) in sorted(existing.items()):
            writer.writerow([area, lat, lon])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365, help="Number of days back from --end-date (default: 365)")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, overrides --days")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, default: today (SGT)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "forecast"))
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API pages")
    parser.add_argument("--overwrite", action="store_true", help="Refetch days that already have a CSV")
    args = parser.parse_args()

    end_date = (
        datetime.date.fromisoformat(args.end_date)
        if args.end_date
        else datetime.datetime.now(SGT).date()
    )
    start_date = (
        datetime.date.fromisoformat(args.start_date)
        if args.start_date
        else end_date - datetime.timedelta(days=args.days - 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    areas_path = os.path.join(args.output_dir, "areas.csv")

    dates = list(daterange(start_date, end_date))
    print(f"Fetching 2-hour forecast data for {len(dates)} days: {start_date} to {end_date}")

    session = requests.Session()
    total_records = 0

    for i, day in enumerate(dates, 1):
        date_str = day.isoformat()
        day_path = os.path.join(args.output_dir, f"forecast_{date_str}.csv")

        if os.path.exists(day_path) and not args.overwrite:
            print(f"[{i}/{len(dates)}] {date_str}: already have data, skipping")
            continue

        print(f"[{i}/{len(dates)}] {date_str}: fetching...")
        areas, records = fetch_day(date_str, session)

        write_day_csv(day_path, records)
        if areas:
            update_areas_csv(areas_path, areas)

        total_records += len(records)
        print(f"[{i}/{len(dates)}] {date_str}: {len(records)} records, {len(areas)} areas")

        time.sleep(args.sleep)

    print(f"Done. {total_records} records written to {args.output_dir}")


if __name__ == "__main__":
    main()
