"""Consolidate one area's 2-hour forecasts from the per-day forecast CSVs into one chronological CSV.

Reads ``forecast_YYYY-MM-DD.csv`` (complete days); ``*.partial.csv`` are ignored unless
``--include-partial``. A forecast issued just before midnight can appear in two day files; exact
duplicates collapse to one row, revisions (different ``update_timestamp``) are kept. Area matching
is case-insensitive.

Usage:
    python filter_area_forecast.py
    python filter_area_forecast.py --area Woodlands --output ./data/forecast/Woodlands.csv
"""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

from filter_station_history import day_files

HEADER = ["issue_timestamp", "valid_start", "valid_end", "area", "forecast", "update_timestamp"]


def rows_for_area(files: Iterable[str], area: str) -> List[Tuple[str, ...]]:
    """Forecast rows for one area, sorted by issue then acquisition time.

    Exact duplicates (a forecast repeated across two day files) collapse to one row. Revisions of
    the same issue with a different ``update_timestamp`` are all kept, so a causal join can pick
    the version that was available at a given time. Old files without ``update_timestamp`` give ''.
    """
    wanted = area.strip().lower()
    rows = set()
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("area") or "").strip().lower() != wanted:
                    continue
                rows.add(
                    (row["issue_timestamp"], row["valid_start"], row["valid_end"], row["area"], row["forecast"], row.get("update_timestamp") or "")
                )
    return sorted(rows, key=lambda r: (r[0], r[5]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=str, default="Woodlands")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "forecast"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args(argv)

    output_path = args.output or os.path.join(args.input_dir, f"{args.area}.csv")
    files = day_files(args.input_dir, "forecast", args.include_partial)
    if not files:
        print(f"No forecast_YYYY-MM-DD.csv files in {args.input_dir}")
        return 1

    matched = rows_for_area(files, args.area)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(matched)

    date_range = f"{matched[0][0]} to {matched[-1][0]}" if matched else "n/a"
    print(f"Scanned {len(files)} day files")
    print(f"Wrote {len(matched)} rows for area {args.area} to {output_path}")
    print(f"Date range: {date_range}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
