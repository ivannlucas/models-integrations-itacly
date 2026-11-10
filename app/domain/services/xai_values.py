"""``xai_feature_values`` of a prediction: the numeric inputs the XAI service explains."""
from __future__ import annotations

import math
from typing import Any, Iterable


def xai_feature_values(row: Any, columns: Iterable[str]) -> dict[str, float]:
    """Numeric value of each column present in ``row`` (a dict or a pandas Series).

    Missing columns, ``None``, values that are not numbers and NaN are left out, so the response
    never carries a NaN (invalid JSON). Plugins that answer ``None`` when nothing is left use
    ``xai_feature_values(...) or None``. Without these values in each prediction the platform's
    XAI-context builder has no input features and silently skips the explain service.
    """
    values: dict[str, float] = {}
    for col in columns:
        val = row.get(col)
        if val is None:
            continue
        try:
            number = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            values[col] = number
    return values
