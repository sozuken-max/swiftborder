"""Shared evaluation constants (SGT, directions, label lag, report window)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple

import pandas as pd

SGT = timezone(timedelta(hours=8))
DIRECTIONS: Tuple[str, ...] = ("SG_TO_MY", "MY_TO_SG")

# y_30 is the mean duration over [origin+30m, origin+40m); observed at origin + 40 min.
LABEL_LAG = pd.Timedelta(minutes=40)
LABEL_LAG_30 = LABEL_LAG

DEFAULT_REPORT_WINDOW_START_SGT = "2026-09-13 00:00:00"
DEFAULT_REPORT_WINDOW_END_SGT = "2026-09-30 23:50:00"

DAY_START_HOUR = 7
DAY_END_HOUR = 19  # day slice [07:00, 19:00) SGT


def parse_sgt(text: str) -> datetime:
    """Parse ``YYYY-MM-DD[ HH:MM[:SS]]``; naive values are SGT; result is UTC-aware."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SGT)
    return dt.astimezone(timezone.utc)
