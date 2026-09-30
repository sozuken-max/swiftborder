"""Fetch historical 2-hour weather forecasts across Singapore from data.gov.sg (one CSV per day).

The v2 two-hour-forecast API returns one day per request, paginated via ``paginationToken``
(schema: ``2hourWeatherForecast.json``). This script loops day by day and pages through each
day. Completeness rules are in ``datagov.py``.

API: GET https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast?date=YYYY-MM-DD

Usage:
    python fetch_forecast_history.py                     # past 365 days, ending today (SGT)
    python fetch_forecast_history.py --days 30
    python fetch_forecast_history.py --start-date 2026-09-05 --end-date 2026-09-30

Exit code 1 if any day failed (re-run to retry those days; complete days are skipped).
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import requests

import datagov

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
# update_timestamp is when data.gov.sg acquired the forecast from NEA (available from then on);
# issue_timestamp is NEA's issue time. Causal joins must use update_timestamp.
HEADER = ["issue_timestamp", "valid_start", "valid_end", "area", "forecast", "update_timestamp"]
AREA_HEADER = ["area", "latitude", "longitude"]


def parse_pages(pages: Sequence[dict]) -> Tuple[List[tuple], Dict[str, tuple]]:
    """(records, areas) from the ``data`` objects of one day's pages."""
    areas: Dict[str, tuple] = {}
    records: List[tuple] = []
    for data in pages:
        for area in data.get("area_metadata", []) or []:
            name = area.get("name")
            if not name:
                continue
            loc = area.get("label_location", {}) or {}
            areas[name] = (loc.get("latitude", ""), loc.get("longitude", ""))
        for item in data.get("items", []) or []:
            period = item.get("valid_period", {}) or {}
            for fc in item.get("forecasts", []) or []:
                text = fc.get("forecast")
                if isinstance(text, dict):  # tolerate {"code", "text"} variants
                    text = text.get("text")
                records.append(
                    (item.get("timestamp"), period.get("start"), period.get("end"), fc.get("area"), text, item.get("update_timestamp"))
                )
    records = sorted(set(records), key=lambda r: (r[0] or "", r[3] or "", r[5] or ""))
    return records, areas


def fetch_day(date_str: str, session) -> Tuple[List[tuple], Dict[str, tuple]]:
    return parse_pages(datagov.fetch_day_pages(session, API_URL, date_str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=365, help="Number of days back from --end-date (default: 365)")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, overrides --days")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, default: today (SGT)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "forecast"))
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between days")
    parser.add_argument("--overwrite", action="store_true", help="Refetch days that already have a complete CSV")
    parser.add_argument("--allow-partial", action="store_true", help="Also fetch today (SGT) into forecast_<date>.partial.csv")
    args = parser.parse_args(argv)

    try:
        start, end = datagov.resolve_dates(args.days, args.start_date, args.end_date)
    except ValueError as exc:
        parser.error(str(exc))
    dates = list(datagov.daterange(start, end))
    print(f"Fetching 2-hour forecasts for {len(dates)} days: {start} to {end}")

    session = requests.Session()
    areas_path = os.path.join(args.output_dir, "areas.csv")
    report = datagov.run_backfill(
        dates,
        output_dir=args.output_dir,
        prefix="forecast",
        header=HEADER,
        fetch_rows=lambda ds: fetch_day(ds, session),
        on_metadata=lambda meta: datagov.merge_metadata_csv(areas_path, AREA_HEADER, "area", meta),
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
