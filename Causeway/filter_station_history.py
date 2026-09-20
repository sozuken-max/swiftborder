"""Consolidate one station's readings out of the per-day rainfall CSVs
produced by fetch_rainfall_history.py into a single chronological CSV.

Usage:
    python filter_station_history.py
    python filter_station_history.py --station-id S210 --output ./data/rainfall/S210_Woodlands_Centre.csv
"""

import argparse
import csv
import glob
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-id", type=str, default="S210")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "rainfall"),
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        args.input_dir, f"{args.station_id}_Woodlands_Centre.csv"
    )

    pattern = os.path.join(args.input_dir, "rainfall_*.csv")
    day_files = sorted(glob.glob(pattern))
    if not day_files:
        print(f"No day files found matching {pattern}")
        return

    matched_rows = []
    for path in day_files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["station_id"] == args.station_id:
                    matched_rows.append((row["timestamp"], row["station_id"], row["value_mm"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "station_id", "value_mm"])
        writer.writerows(matched_rows)

    date_range = (
        f"{matched_rows[0][0]} to {matched_rows[-1][0]}" if matched_rows else "n/a"
    )
    print(f"Scanned {len(day_files)} day files")
    print(f"Wrote {len(matched_rows)} rows for station {args.station_id} to {output_path}")
    print(f"Date range: {date_range}")


if __name__ == "__main__":
    main()
