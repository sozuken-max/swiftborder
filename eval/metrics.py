"""Offline helpers for Layer B scoring (no BigQuery)."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

DIRECTIONS: Tuple[str, ...] = ("SG_TO_MY", "MY_TO_SG")
TIME_OF_DAY_SLICES: Tuple[str, ...] = ("morning peak", "evening peak", "other", "all")


def _as_bool(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip() in ("1", "true", "True", "yes")


def peak_time_of_day(is_morning_peak: object, is_evening_peak: object) -> str:
    """Map view flags to evaluation slice labels (06–10 and 16–21 SGT in the view)."""
    if _as_bool(is_morning_peak):
        return "morning peak"
    if _as_bool(is_evening_peak):
        return "evening peak"
    return "other"


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        return float("nan")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        return float("nan")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def score_slices(
    rows: Iterable[dict],
    actual_key: str,
    predicted_key: str,
) -> List[dict]:
    """Aggregate MAE/RMSE by direction and time-of-day slice."""
    buckets: dict[tuple[str, str], tuple[list[float], list[float]]] = {}

    for row in rows:
        direction = str(row["direction"])
        tod = peak_time_of_day(row.get("is_morning_peak"), row.get("is_evening_peak"))
        actual = float(row[actual_key])
        predicted = float(row[predicted_key])
        for dir_slice in (direction, "both"):
            for tod_slice in (tod, "all"):
                key = (dir_slice, tod_slice)
                if key not in buckets:
                    buckets[key] = ([], [])
                buckets[key][0].append(actual)
                buckets[key][1].append(predicted)

    out: List[dict] = []
    for (direction, tod), (actuals, preds) in sorted(buckets.items()):
        out.append(
            {
                "direction": direction,
                "time_of_day": tod,
                "mae": mae(actuals, preds),
                "rmse": rmse(actuals, preds),
                "n": len(actuals),
            }
        )
    return out
