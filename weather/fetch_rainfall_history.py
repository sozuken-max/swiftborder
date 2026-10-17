"""Fetch historical rainfall readings across Singapore from data.gov.sg (one CSV per day).

The v2 real-time rainfall API returns one day per request, paginated via ``paginationToken``.
This script loops day by day and pages through each day. Completeness rules (complete days
only, atomic writes, today skipped unless ``--allow-partial``) are in ``datagov.py``.

API: GET https://api-open.data.gov.sg/v2/real-time/api/rainfall?date=YYYY-MM-DD

Usage:
    python fetch_rainfall_history.py                     # past 365 days, ending today (SGT)
    python fetch_rainfall_history.py --days 30
    python fetch_rainfall_history.py --start-date 2026-09-05 --end-date 2026-09-30
    python fetch_rainfall_history.py --output-dir ./data/rainfall --sleep 0.5

Exit code 1 if any day failed (re-run to retry those days; complete days are skipped).
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import requests

import datagov

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
HEADER = ["timestamp", "station_id", "value_mm"]
STATION_HEADER = ["station_id", "name", "latitude", "longitude"]


def parse_pages(pages: Sequence[dict]) -> Tuple[List[tuple], Dict[str, tuple]]:
    """(readings, stations) from the ``data`` objects of one day's pages."""
    stations: Dict[str, tuple] = {}
    readings: List[tuple] = []
    for data in pages:
        for station in data.get("stations", []) or []:
            sid = station.get("id")
            if not sid:
                continue
            loc = station.get("location", {}) or {}
            stations[sid] = (station.get("name", ""), loc.get("latitude", ""), loc.get("longitude", ""))
        for reading in data.get("readings", []) or []:
            ts = reading.get("timestamp")
            for point in reading.get("data", []) or []:
                readings.append((ts, point.get("stationId"), point.get("value")))
    readings.sort(key=lambda r: (r[0] or "", r[1] or ""))
    return readings, stations


def fetch_day(date_str: str, session) -> Tuple[List[tuple], Dict[str, tuple]]:
    return parse_pages(datagov.fetch_day_pages(session, API_URL, date_str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=365, help="Number of days back from --end-date (default: 365)")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, overrides --days")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, default: today (SGT)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "rainfall"))
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between days")
    parser.add_argument("--overwrite", action="store_true", help="Refetch days that already have a complete CSV")
    parser.add_argument("--allow-partial", action="store_true", help="Also fetch today (SGT) into rainfall_<date>.partial.csv")
    args = parser.parse_args(argv)

    try:
        start, end = datagov.resolve_dates(args.days, args.start_date, args.end_date)
    except ValueError as exc:
        parser.error(str(exc))
    dates = list(datagov.daterange(start, end))
    print(f"Fetching rainfall for {len(dates)} days: {start} to {end}")

    session = requests.Session()
    stations_path = os.path.join(args.output_dir, "stations.csv")
    report = datagov.run_backfill(
        dates,
        output_dir=args.output_dir,
        prefix="rainfall",
        header=HEADER,
        fetch_rows=lambda ds: fetch_day(ds, session),
        on_metadata=lambda meta: datagov.merge_metadata_csv(stations_path, STATION_HEADER, "station_id", meta),
        overwrite=args.overwrite,
        allow_partial=args.allow_partial,
        sleep_between_days=args.sleep,
    )
    print("\nCompleteness report:")
    print("\n".join(report.lines()))
    print(f"Summary: {report.summary()}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
