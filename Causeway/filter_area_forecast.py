"""Consolidate one area's 2-hour forecasts out of the per-day forecast CSVs
produced by fetch_forecast_history.py into a single chronological CSV.

Usage:
    python filter_area_forecast.py
    python filter_area_forecast.py --area Woodlands --output ./data/forecast/Woodlands.csv
"""

import argparse
import csv
import glob
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=str, default="Woodlands")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "forecast"),
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.input_dir, f"{args.area}.csv")

    pattern = os.path.join(args.input_dir, "forecast_*.csv")
    day_files = sorted(glob.glob(pattern))
    if not day_files:
        print(f"No day files found matching {pattern}")
        return

    matched_rows = []
    for path in day_files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["area"] == args.area:
                    matched_rows.append(
                        (row["issue_timestamp"], row["valid_start"], row["valid_end"], row["area"], row["forecast"])
                    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["issue_timestamp", "valid_start", "valid_end", "area", "forecast"])
        writer.writerows(matched_rows)

    date_range = (
        f"{matched_rows[0][0]} to {matched_rows[-1][0]}" if matched_rows else "n/a"
    )
    print(f"Scanned {len(day_files)} day files")
    print(f"Wrote {len(matched_rows)} rows for area {args.area} to {output_path}")
    print(f"Date range: {date_range}")


if __name__ == "__main__":
    main()
