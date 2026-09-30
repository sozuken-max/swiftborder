"""Shared data.gov.sg v2 real-time API client for the day-by-day history fetchers.

Rules that keep the CSV history honest:

- A day's CSV is written only after **every page** of that day was fetched. Any failure (network,
  5xx after retries, 429 after retries, bad JSON, ``code != 0``, a repeated pagination token)
  raises ``FetchError``; the day is reported as failed and nothing is written, so the next run
  retries it.
- A 404 on the **first** page (or a day with zero rows) is reported as ``no_data``: a
  ``*.nodata`` marker is written instead of a CSV, so the day is retried on the next run. A 404 on
  a later page is a failure.
- Files are written atomically (``.tmp`` then rename).
- The current SGT day is incomplete by definition. It is skipped unless ``allow_partial`` is set,
  and then written as ``*.partial.csv`` so it never blocks the complete file later.
- Future dates are rejected.

Optional ``DATAGOV_API_KEY`` (environment) is sent as ``x-api-key`` for higher rate limits.
"""

from __future__ import annotations

import csv
import datetime
import email.utils
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python 3.8
    from backports.zoneinfo import ZoneInfo

import requests

SGT = ZoneInfo("Asia/Singapore")
MAX_RETRIES = 5
TIMEOUT_SEC = 30
API_KEY_ENV = "DATAGOV_API_KEY"


class FetchError(RuntimeError):
    """A day could not be fetched completely. Nothing should be written for it."""


def parse_retry_after(value: Optional[str], default: float, now: Optional[datetime.datetime] = None) -> float:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP-date)."""
    if not value:
        return default
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if when is None:
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _headers() -> Dict[str, str]:
    key = os.environ.get(API_KEY_ENV)
    return {"x-api-key": key} if key else {}


def get_with_retries(
    session,
    url: str,
    params: Dict[str, str],
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    log=sys.stderr,
):
    """GET with exponential backoff on network errors, 429 and 5xx. Returns 2xx or 404 responses.

    Raises FetchError when retries are exhausted or on any other 4xx.
    """
    delay = base_delay
    last = "no attempt"
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, headers=_headers(), timeout=TIMEOUT_SEC)
        except requests.RequestException as exc:
            last = f"network error: {exc.__class__.__name__}"
        else:
            if resp.status_code < 300 or resp.status_code == 404:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
                if attempt < max_retries:
                    wait = parse_retry_after(resp.headers.get("Retry-After"), delay) if resp.status_code == 429 else delay
                    print(f"  {last}, retry {attempt}/{max_retries - 1} in {wait:.1f}s", file=log)
                    sleep(wait)
                    delay *= 2
                continue
            raise FetchError(f"HTTP {resp.status_code} for {params}: {_error_msg(resp)}")
        if attempt < max_retries:
            print(f"  {last}, retry {attempt}/{max_retries - 1} in {delay:.1f}s", file=log)
            sleep(delay)
            delay *= 2
    raise FetchError(f"gave up after {max_retries} attempts ({last}) for {params}")


def _error_msg(resp) -> str:
    try:
        return str(resp.json().get("errorMsg") or "")[:200]
    except ValueError:
        return ""


def fetch_day_pages(session, url: str, date_str: str, **kwargs) -> List[dict]:
    """Every page of ``data`` for one day, or [] if the API has no data for that day."""
    pages: List[dict] = []
    token: Optional[str] = None
    seen = set()
    while True:
        params = {"date": date_str}
        if token:
            params["paginationToken"] = token
        resp = get_with_retries(session, url, params, **kwargs)
        if resp.status_code == 404:
            if not pages:
                return []
            raise FetchError(f"404 on page {len(pages) + 1} of {date_str}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise FetchError(f"non-JSON response for {date_str}") from exc
        if payload.get("code", 0) != 0:
            raise FetchError(f"API code {payload.get('code')} for {date_str}: {payload.get('errorMsg')}")
        data = payload.get("data") or {}
        pages.append(data)
        token = data.get("paginationToken")
        if not token:
            return pages
        if token in seen:
            raise FetchError(f"pagination token repeated for {date_str}")
        seen.add(token)


def atomic_write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def daterange(start_date: datetime.date, end_date: datetime.date) -> Iterator[datetime.date]:
    current = start_date
    while current <= end_date:
        yield current
        current += datetime.timedelta(days=1)


def today_sgt() -> datetime.date:
    return datetime.datetime.now(SGT).date()


def resolve_dates(
    days: int,
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    today: Optional[datetime.date] = None,
) -> Tuple[datetime.date, datetime.date]:
    """Validate CLI dates. Default window ends today (SGT); future dates are rejected."""
    today = today or today_sgt()
    end = datetime.date.fromisoformat(end_date) if end_date else today
    if start_date:
        start = datetime.date.fromisoformat(start_date)
    else:
        if days < 1:
            raise ValueError("--days must be at least 1")
        start = end - datetime.timedelta(days=days - 1)
    if start > end:
        raise ValueError(f"start date {start} is after end date {end}")
    if end > today:
        raise ValueError(f"end date {end} is in the future (today SGT is {today})")
    return start, end


@dataclass
class DayOutcome:
    date: str
    status: str  # fetched | no_data | skipped | partial | skipped_today | failed
    rows: int = 0
    path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BackfillReport:
    outcomes: List[DayOutcome] = field(default_factory=list)

    @property
    def failed(self) -> List[DayOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for o in self.outcomes:
            counts[o.status] = counts.get(o.status, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    def lines(self) -> List[str]:
        out = []
        for o in self.outcomes:
            extra = f" ({o.error})" if o.error else ""
            out.append(f"{o.date}  {o.status:<13} rows={o.rows}{extra}")
        return out


def run_backfill(
    dates: Sequence[datetime.date],
    *,
    output_dir: str,
    prefix: str,
    header: Sequence[str],
    fetch_rows: Callable[[str], Tuple[List[Sequence], Dict]],
    on_metadata: Optional[Callable[[Dict], None]] = None,
    today: Optional[datetime.date] = None,
    overwrite: bool = False,
    allow_partial: bool = False,
    sleep_between_days: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    log=sys.stdout,
) -> BackfillReport:
    """Fetch each day with ``fetch_rows(date) -> (rows, metadata)`` and write complete days only."""
    today = today or today_sgt()
    os.makedirs(output_dir, exist_ok=True)
    report = BackfillReport()
    for i, day in enumerate(dates, 1):
        ds = day.isoformat()
        final_path = os.path.join(output_dir, f"{prefix}_{ds}.csv")
        partial_path = os.path.join(output_dir, f"{prefix}_{ds}.partial.csv")
        tag = f"[{i}/{len(dates)}] {ds}"
        is_today = day >= today
        if is_today and not allow_partial:
            print(f"{tag}: today (SGT) is incomplete, skipping (use --allow-partial)", file=log)
            report.outcomes.append(DayOutcome(ds, "skipped_today"))
            continue
        if not is_today and os.path.exists(final_path) and not overwrite:
            print(f"{tag}: already complete, skipping", file=log)
            report.outcomes.append(DayOutcome(ds, "skipped", path=final_path))
            continue
        try:
            rows, meta = fetch_rows(ds)
        except FetchError as exc:
            print(f"{tag}: FAILED, nothing written ({exc})", file=log)
            report.outcomes.append(DayOutcome(ds, "failed", error=str(exc)))
            continue
        if not rows and not is_today:
            # No data is not proof the day is complete (upstream can be late). Leave a marker
            # instead of a final CSV so the next run tries again.
            marker = os.path.join(output_dir, f"{prefix}_{ds}.nodata")
            open(marker, "w", encoding="utf-8").close()
            print(f"{tag}: no_data (marker written; retried next run)", file=log)
            report.outcomes.append(DayOutcome(ds, "no_data", path=marker))
            if sleep_between_days:
                sleep(sleep_between_days)
            continue
        path = partial_path if is_today else final_path
        atomic_write_csv(path, header, rows)
        if not is_today:
            for stale in (partial_path, os.path.join(output_dir, f"{prefix}_{ds}.nodata")):
                if os.path.exists(stale):
                    os.remove(stale)
        if meta and on_metadata:
            on_metadata(meta)
        status = "partial" if is_today else "fetched"
        print(f"{tag}: {status}, {len(rows)} rows", file=log)
        report.outcomes.append(DayOutcome(ds, status, rows=len(rows), path=path))
        if sleep_between_days:
            sleep(sleep_between_days)
    return report


def merge_metadata_csv(path: str, header: Sequence[str], key: str, new: Dict[str, Sequence]) -> None:
    """Merge {key: values} into a small metadata CSV (stations, areas), sorted by key."""
    existing: Dict[str, Tuple] = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row[key]] = tuple(row[h] for h in header if h != key)
    existing.update({k: tuple(v) for k, v in new.items()})
    atomic_write_csv(path, header, ([k, *v] for k, v in sorted(existing.items())))
