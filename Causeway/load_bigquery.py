#!/usr/bin/env python3
"""Append the Woodlands weather history from the Causeway CSVs to the BigQuery weather tables.

Targets (project ``swiftborder``, location ``asia-southeast1``), keeping their existing contract:

- ``rainfall.rainfall`` (``timestamp`` UTC, ``station_id``, ``value_mm``): one station, stored by
  **name** (``S210`` -> ``Woodlands Centre Road``, from ``data/rainfall/stations.csv``).
- ``weatherforecast.weatherforecast`` (``issue_timestamp``, ``valid_start``, ``valid_end`` UTC,
  ``area``, ``forecast``): Woodlands, **one row per issue**. When data.gov.sg revised an issue, the
  latest acquired text is kept. A nullable ``update_timestamp`` (data.gov.sg acquisition time) is
  added to the schema; older rows have NULL.

Safety rules:

- Dry run by default: prints what would be loaded. ``--execute`` performs the load.
- Refuses to load if the target already has any row inside the load's time range (no duplicates,
  no silent overwrite). Complete days only (``filter_*`` output from complete day files).
- Before any snapshot or append, checks every selected table for an overlapping range.
  Then, for each table, creates a snapshot ``<table>_snapshot_<YYYYMMDD>`` (expires in
  30 days) so the load can be reverted with ``CREATE OR REPLACE TABLE ... CLONE <snapshot>``.
- Append only (``WRITE_APPEND``); nothing is deleted or updated.

    python load_bigquery.py                      # dry run for both tables
    python load_bigquery.py --execute            # snapshot + append
    python load_bigquery.py --table rainfall --execute
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

PROJECT = "swiftborder"
LOCATION = "asia-southeast1"
HERE = os.path.dirname(os.path.abspath(__file__))
RAIN_CSV = os.path.join(HERE, "data", "rainfall", "S210_Woodlands_Centre.csv")
STATIONS_CSV = os.path.join(HERE, "data", "rainfall", "stations.csv")
FORECAST_CSV = os.path.join(HERE, "data", "forecast", "Woodlands.csv")
RAIN_TABLE = f"{PROJECT}.rainfall.rainfall"
FORECAST_TABLE = f"{PROJECT}.weatherforecast.weatherforecast"
SNAPSHOT_DAYS = 30


class LoadRefused(RuntimeError):
    pass


def to_utc_iso(value: str) -> str:
    """ISO-8601 with offset -> 'YYYY-MM-DD HH:MM:SS' UTC (the tables store true UTC)."""
    ts = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"timestamp without offset: {value!r}")
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# The table stores stations by name. data.gov.sg has returned S210 as both "Woodlands Centre Road"
# and "Woodlands Centre" (stations.csv keeps whichever day was merged last), so the table's name is
# pinned here; otherwise one station splits into two ids and drops out of v_weather_features_10min.
TABLE_STATION_NAMES = {"S210": "Woodlands Centre Road"}


def station_name(station_id: str, stations_csv: str = STATIONS_CSV) -> str:
    if station_id in TABLE_STATION_NAMES:
        return TABLE_STATION_NAMES[station_id]
    with open(stations_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["station_id"] == station_id:
                return row["name"]
    raise KeyError(f"station {station_id} not in {stations_csv}")


def rainfall_rows(path: str = RAIN_CSV, stations_csv: str = STATIONS_CSV) -> List[dict]:
    rows = []
    names: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r["station_id"]
            if sid not in names:
                names[sid] = station_name(sid, stations_csv)
            rows.append({"timestamp": to_utc_iso(r["timestamp"]), "station_id": names[sid], "value_mm": float(r["value_mm"])})
    keys = {(r["timestamp"], r["station_id"]) for r in rows}
    if len(keys) != len(rows):
        raise LoadRefused("duplicate (timestamp, station_id) rows in the rainfall CSV")
    return sorted(rows, key=lambda r: r["timestamp"])


def forecast_rows(path: str = FORECAST_CSV) -> List[dict]:
    """One row per (issue_timestamp, area): the latest acquired revision."""
    latest: Dict[tuple, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            upd = r.get("update_timestamp") or ""
            row = {
                "issue_timestamp": to_utc_iso(r["issue_timestamp"]),
                "valid_start": to_utc_iso(r["valid_start"]),
                "valid_end": to_utc_iso(r["valid_end"]),
                "area": r["area"],
                "forecast": r["forecast"],
                "update_timestamp": to_utc_iso(upd) if upd else None,
            }
            key = (row["issue_timestamp"], row["area"])
            prev = latest.get(key)
            if prev is None or (row["update_timestamp"] or "") >= (prev["update_timestamp"] or ""):
                latest[key] = row
    return sorted(latest.values(), key=lambda r: r["issue_timestamp"])


@dataclass
class Plan:
    table: str
    time_column: str
    rows: List[dict]
    add_update_timestamp: bool = False

    @property
    def start(self) -> str:
        return self.rows[0][self.time_column]

    @property
    def end(self) -> str:
        return self.rows[-1][self.time_column]


def plans(which: Sequence[str]) -> List[Plan]:
    out = []
    if "rainfall" in which:
        out.append(Plan(RAIN_TABLE, "timestamp", rainfall_rows()))
    if "forecast" in which:
        out.append(Plan(FORECAST_TABLE, "issue_timestamp", forecast_rows(), add_update_timestamp=True))
    return out


def _count_in_range(client, plan: Plan) -> int:
    from google.cloud import bigquery

    job = client.query(
        f"SELECT COUNT(*) AS n FROM `{plan.table}` WHERE {plan.time_column} BETWEEN @s AND @e",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("s", "TIMESTAMP", plan.start),
                bigquery.ScalarQueryParameter("e", "TIMESTAMP", plan.end),
            ]
        ),
        location=LOCATION,
    )
    return int(list(job.result())[0]["n"])


def _snapshot(client, table: str, today: dt.date) -> str:
    snap = f"{table}_snapshot_{today:%Y%m%d}"
    expires = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=SNAPSHOT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    client.query(
        f"CREATE SNAPSHOT TABLE IF NOT EXISTS `{snap}` CLONE `{table}` "
        f"OPTIONS (expiration_timestamp = TIMESTAMP '{expires} UTC')",
        location=LOCATION,
    ).result()
    return snap


def _load(client, plan: Plan) -> int:
    from google.cloud import bigquery

    table = client.get_table(plan.table)
    schema = list(table.schema)
    if plan.add_update_timestamp and "update_timestamp" not in {f.name for f in schema}:
        schema.append(bigquery.SchemaField("update_timestamp", "TIMESTAMP", mode="NULLABLE"))
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION] if plan.add_update_timestamp else None,
    )
    rows = plan.rows if plan.add_update_timestamp else [{k: v for k, v in r.items() if k != "update_timestamp"} for r in plan.rows]
    job = client.load_table_from_json(rows, plan.table, job_config=job_config, location=LOCATION)
    job.result()
    return int(job.output_rows or len(rows))


def run(
    which: Sequence[str],
    *,
    execute: bool,
    client=None,
    today: Optional[dt.date] = None,
    make_plans: Callable[[Sequence[str]], List[Plan]] = plans,
    log=sys.stdout,
) -> List[dict]:
    results = []
    selected = make_plans(which)
    if execute and client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=PROJECT)
    # Overlap checks for every selected table finish before the first snapshot,
    # so a refusal on a later table cannot leave an earlier table already loaded.
    existing_by_table: Dict[str, int] = {}
    if client is not None:
        for plan in selected:
            existing = _count_in_range(client, plan)
            existing_by_table[plan.table] = existing
            if existing:
                raise LoadRefused(f"{plan.table} already has {existing} rows in {plan.start} .. {plan.end}; refusing to append")
    for plan in selected:
        info = {"table": plan.table, "rows": len(plan.rows), "start_utc": plan.start, "end_utc": plan.end}
        print(f"{plan.table}: {len(plan.rows)} rows, {plan.start} .. {plan.end} UTC", file=log)
        if client is not None:
            info["existing_rows_in_range"] = existing_by_table[plan.table]
        if execute:
            info["snapshot"] = _snapshot(client, plan.table, today or dt.date.today())
            info["loaded"] = _load(client, plan)
            print(f"  snapshot {info['snapshot']}; appended {info['loaded']} rows", file=log)
        else:
            print("  dry run (use --execute to load)", file=log)
        results.append(info)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", choices=("rainfall", "forecast", "both"), default="both")
    parser.add_argument("--execute", action="store_true", help="snapshot and append (default: dry run)")
    parser.add_argument("--check", action="store_true", help="dry run that also queries BigQuery for overlaps")
    args = parser.parse_args(argv)
    which = ("rainfall", "forecast") if args.table == "both" else (args.table,)
    client = None
    if args.check and not args.execute:
        from google.cloud import bigquery

        client = bigquery.Client(project=PROJECT)
    try:
        run(which, execute=args.execute, client=client)
    except LoadRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
