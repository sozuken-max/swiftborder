"""Consolidate one station's readings from the per-day rainfall CSVs into one chronological CSV.

Reads ``rainfall_YYYY-MM-DD.csv`` (complete days). ``*.partial.csv`` files are ignored unless
``--include-partial``. Output rows are sorted by timestamp and de-duplicated.

Usage:
    python filter_station_history.py
    python filter_station_history.py --station-id S210 --output ./data/rainfall/S210_Woodlands_Centre.csv
"""

import argparse
import csv
import glob
import os
from typing import Iterable, List, Tuple

HEADER = ["timestamp", "station_id", "value_mm"]


def day_files(input_dir: str, prefix: str, include_partial: bool = False) -> List[str]:
    """Per-day CSVs for ``prefix`` (``rainfall`` / ``forecast``), sorted by date."""
    files = []
    for path in glob.glob(os.path.join(input_dir, f"{prefix}_*.csv")):
        name = os.path.basename(path)
        if name.endswith(".partial.csv") and not include_partial:
            continue
        date_part = name[len(prefix) + 1 :].split(".")[0]
        if len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-":
            files.append(path)
    return sorted(files)


def rows_for_station(day_files: Iterable[str], station_id: str) -> List[Tuple[str, str, str]]:
    """(timestamp, station_id, value_mm) rows for one station, sorted and de-duplicated."""
    matched = set()
    for path in day_files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["station_id"] == station_id:
                    matched.add((row["timestamp"], row["station_id"], row["value_mm"]))
    return sorted(matched)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-id", type=str, default="S210")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "rainfall"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args(argv)

    output_path = args.output or os.path.join(args.input_dir, f"{args.station_id}_Woodlands_Centre.csv")
    files = day_files(args.input_dir, "rainfall", args.include_partial)
    if not files:
        print(f"No rainfall_YYYY-MM-DD.csv files in {args.input_dir}")
        return 1

    matched = rows_for_station(files, args.station_id)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(matched)

    date_range = f"{matched[0][0]} to {matched[-1][0]}" if matched else "n/a"
    print(f"Scanned {len(files)} day files")
    print(f"Wrote {len(matched)} rows for station {args.station_id} to {output_path}")
    print(f"Date range: {date_range}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
