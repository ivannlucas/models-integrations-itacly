"""Feature-row resolution for inline/batch prediction.

Design note (see inbox/a15/manifest.yaml known_issues): the AI team's original
src/predict/predictor.py reconstructs the 16 lagged/exogenous feature_columns internally from
two bundled historical CSVs (global_phytosanitary_prices_b2020.csv, financial_proxies_monthly.csv)
that would need periodic refreshing to stay valid, and whose exogenous columns require a
base-2020 reindexing anchor. Rather than bundling and refreshing that reference data inside the
plugin, this module follows the same convention already used by ml17_meat_market_price_analysis
(the closest sibling model — Ridge, sklearn, already-lagged exogenous panel): the caller supplies
the already-computed feature vector (the exact feature_columns contract of the artifact). Only
the four calendar features (month_sin, month_cos, quarter, is_spring_risk) — pure functions of
the calendar, not of any external series — are derived automatically when omitted, from a
supplied 'date'/'origin_date'.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

_EMPTY_TOKENS = {"", "nan", "none", "null", "nat"}


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in _EMPTY_TOKENS


def _parse_date(date_val: Any) -> datetime:
    s = str(date_val).strip()
    if not s:
        raise ValueError("empty date value")
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _calendar_features(date_val: Any) -> dict[str, float]:
    """Port of src/predict/predictor.py::build_inference_panel_from_history's calendar block."""
    try:
        month = _parse_date(date_val).month
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Fecha inválida '{date_val}': {exc}") from exc
    quarter = (month - 1) // 3 + 1
    return {
        "month_sin": math.sin(2.0 * math.pi * month / 12.0),
        "month_cos": math.cos(2.0 * math.pi * month / 12.0),
        "quarter": float(quarter),
        "is_spring_risk": 1.0 if month in (4, 5, 6) else 0.0,
    }


def build_feature_row(
    row: dict[str, Any], feature_columns: list[str], calendar_derived_columns: tuple[str, ...],
) -> dict[str, float]:
    """Resolve every column in *feature_columns* from *row*, deriving calendar columns from
    'date'/'origin_date' when they're missing and every other feature is not.

    Raises ValueError naming exactly which feature(s) are missing/blank — this is what maps to
    HTTP 422 via MissingRequiredFeatureError in plugin.py, not a generic 500.
    """
    resolved = dict(row)

    missing_calendar = [
        col for col in calendar_derived_columns
        if col in feature_columns and _is_blank(resolved.get(col))
    ]
    if missing_calendar:
        date_val = resolved.get("date") if not _is_blank(resolved.get("date")) else resolved.get("origin_date")
        if date_val is None:
            raise ValueError(
                f"Faltan las variables de calendario {missing_calendar} y no se aportó "
                "'date' ni 'origin_date' (YYYY-MM-DD) para derivarlas automáticamente."
            )
        resolved.update(_calendar_features(date_val))

    missing = [col for col in feature_columns if _is_blank(resolved.get(col))]
    if missing:
        raise ValueError(f"Faltan variables obligatorias del modelo: {missing}")

    return {col: float(resolved[col]) for col in feature_columns}


def build_feature_frame(
    row: dict[str, Any], feature_columns: list[str], calendar_derived_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build the single-row, column-ordered DataFrame model.predict() expects."""
    resolved = build_feature_row(row, feature_columns, calendar_derived_columns)
    return pd.DataFrame([resolved])[feature_columns]


def resolve_origin_target_dates(row: dict[str, Any], horizon: int) -> tuple[str | None, str | None]:
    """Best-effort (origin_date, target_date) ISO strings from a row's 'date'/'origin_date'."""
    date_val = row.get("date") if not _is_blank(row.get("date")) else row.get("origin_date")
    if date_val is None:
        return None, None
    try:
        origin = _parse_date(date_val)
    except (ValueError, TypeError):
        return None, None
    origin_month = pd.Timestamp(origin).to_period("M").to_timestamp()
    target_month = origin_month + pd.DateOffset(months=horizon)
    return origin_month.strftime("%Y-%m-%d"), target_month.strftime("%Y-%m-%d")
