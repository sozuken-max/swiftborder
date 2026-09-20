"""Fetch historical rainfall readings across Singapore from data.gov.sg.

The v2 real-time rainfall API (see RainfallacrossSingapore.json) only returns
readings for a single day at a time, paginated via `paginationToken`. To build
a historical dataset we loop day-by-day over the requested window and page
through each day's results.

API: GET https://api-open.data.gov.sg/v2/real-time/api/rainfall?date=YYYY-MM-DD

Usage:
    python fetch_rainfall_history.py                     # past 365 days, ending today (SGT)
    python fetch_rainfall_history.py --days 30
    python fetch_rainfall_history.py --start-date 2025-08-30 --end-date 2026-08-30
    python fetch_rainfall_history.py --output-dir ./data/rainfall --sleep 0.5
"""

import argparse
import csv
import datetime
import os
import sys
import time
from zoneinfo import ZoneInfo

import requests

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
SGT = ZoneInfo("Asia/Singapore")
MAX_RETRIES = 5


def fetch_day(date_str, session):
    """Return (stations, readings) for one day.

    stations: dict station_id -> (name, latitude, longitude)
    readings: list of (timestamp, station_id, value)
    """
    stations = {}
    readings = []
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

        for station in data.get("stations", []):
            loc = station.get("location", {}) or {}
            stations[station["id"]] = (
                station.get("name", ""),
                loc.get("latitude", ""),
                loc.get("longitude", ""),
            )

        for reading in data.get("readings", []):
            timestamp = reading.get("timestamp")
            for point in reading.get("data", []):
                readings.append((timestamp, point.get("stationId"), point.get("value")))

        pagination_token = data.get("paginationToken")
        if not pagination_token:
            break

    return stations, readings


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


def write_day_csv(path, readings):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "station_id", "value_mm"])
        writer.writerows(readings)


def update_stations_csv(path, stations):
    existing = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["station_id"]] = (row["name"], row["latitude"], row["longitude"])

    existing.update({sid: info for sid, info in stations.items()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "name", "latitude", "longitude"])
        for sid, (name, lat, lon) in sorted(existing.items()):
            writer.writerow([sid, name, lat, lon])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365, help="Number of days back from --end-date (default: 365)")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, overrides --days")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, default: today (SGT)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "rainfall"))
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
    stations_path = os.path.join(args.output_dir, "stations.csv")

    dates = list(daterange(start_date, end_date))
    print(f"Fetching rainfall data for {len(dates)} days: {start_date} to {end_date}")

    session = requests.Session()
    total_readings = 0

    for i, day in enumerate(dates, 1):
        date_str = day.isoformat()
        day_path = os.path.join(args.output_dir, f"rainfall_{date_str}.csv")

        if os.path.exists(day_path) and not args.overwrite:
            print(f"[{i}/{len(dates)}] {date_str}: already have data, skipping")
            continue

        print(f"[{i}/{len(dates)}] {date_str}: fetching...")
        stations, readings = fetch_day(date_str, session)

        write_day_csv(day_path, readings)
        if stations:
            update_stations_csv(stations_path, stations)

        total_readings += len(readings)
        print(f"[{i}/{len(dates)}] {date_str}: {len(readings)} readings, {len(stations)} stations")

        time.sleep(args.sleep)

    print(f"Done. {total_readings} readings written to {args.output_dir}")


if __name__ == "__main__":
    main()
