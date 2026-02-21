from __future__ import annotations

import calendar
import math
from datetime import datetime
from typing import Any, Literal

import numpy as np

TaskType = Literal["classification", "regression"]


def infer_task(y_raw: list[Any]) -> TaskType:
    numeric = True
    numeric_values: list[float] = []
    for value in y_raw:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            numeric = False
            break

    if not numeric:
        return "classification"

    unique_count = len(set(numeric_values))
    if unique_count <= max(20, int(len(numeric_values) * 0.05)):
        return "classification"
    return "regression"


def batch_indices(n: int, batch_size: int) -> list[np.ndarray]:
    indices = np.arange(n)
    np.random.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, n, batch_size)]


def parse_date_like(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    formats = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def expand_date_columns(
    rows: list[dict[str, Any]],
    date_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows or not date_columns:
        return rows

    wanted = {column for column in date_columns if column}
    if not wanted:
        return rows

    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        for column in wanted:
            if column not in row:
                continue
            dt = parse_date_like(row.get(column))
            if dt is None:
                continue

            day_of_week = dt.weekday()
            day_of_year = dt.timetuple().tm_yday
            month = dt.month
            prefix = f"{column}__"

            enriched[f"{prefix}year"] = dt.year
            enriched[f"{prefix}month"] = month
            enriched[f"{prefix}day"] = dt.day
            enriched[f"{prefix}day_of_week"] = day_of_week
            enriched[f"{prefix}day_of_year"] = day_of_year
            enriched[f"{prefix}week_of_year"] = int(dt.strftime("%V"))
            enriched[f"{prefix}is_month_start"] = 1 if dt.day == 1 else 0
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            enriched[f"{prefix}is_month_end"] = 1 if dt.day == last_day else 0

            enriched[f"{prefix}sin_month"] = math.sin(2 * math.pi * month / 12.0)
            enriched[f"{prefix}cos_month"] = math.cos(2 * math.pi * month / 12.0)
            enriched[f"{prefix}sin_day_of_week"] = math.sin(2 * math.pi * day_of_week / 7.0)
            enriched[f"{prefix}cos_day_of_week"] = math.cos(2 * math.pi * day_of_week / 7.0)
            enriched[f"{prefix}sin_day_of_year"] = math.sin(2 * math.pi * day_of_year / 365.0)
            enriched[f"{prefix}cos_day_of_year"] = math.cos(2 * math.pi * day_of_year / 365.0)
        out.append(enriched)
    return out
