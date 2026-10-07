"""Feature-row resolution for inline/batch prediction.

The caller may supply the already-computed 16-feature vector directly (the exact
feature_columns contract of the artifact — matches the convention used by
ml17_meat_market_price_analysis, the closest sibling model), or omit some of them:

- The four calendar features (month_sin, month_cos, quarter, is_spring_risk) are always
  derivable from a supplied 'date'/'origin_date' — pure functions of the calendar, no
  external data needed.
- The other 11 features (ipi_national_current + 6 autoregressive lags + 5 exogenous lags)
  are derivable too, from the bundled reference history (see history.py) — a caller who
  only knows the current national IPI value (and optionally its date) does not have to
  compute lags/exogenous variables by hand, exactly like the AI team's original
  src/predict/predictor.py did for its --input ipi_history.csv mode. If the caller also
  supplies 'ipi_national_current' for that date, it overrides the bundled reference value
  there (useful for dates the bundled snapshot doesn't cover yet).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

from app.plugins.ml15_wine_ipi_price_forecast import history

_EMPTY_TOKENS = {"", "nan", "none", "null", "nat"}


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in _EMPTY_TOKENS


def _parse_date(date_val: Any) -> datetime:
    s = str(date_val).strip()
    if not s:
        raise ValueError("empty date value")
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _resolve_date_value(row: dict[str, Any]) -> Any | None:
    date_val = row.get("date")
    return date_val if not _is_blank(date_val) else row.get("origin_date")


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


def _derive_from_history(resolved: dict[str, Any], missing: list[str], date_val: Any) -> dict[str, float]:
    """Best-effort derivation of *missing* feature(s) from the bundled reference history.

    Raises ValueError (chaining the underlying cause) if the bundled reference data is
    unavailable, or doesn't cover the dates needed for one of the requested lags.
    """
    try:
        origin_date = pd.Timestamp(_parse_date(date_val))
        prices_df, financial_df = history.load_reference_data()

        current_override = resolved.get("ipi_national_current")
        if not _is_blank(current_override):
            prices_df = history.merge_user_points_into_prices(
                prices_df, [{"date": origin_date, "value": float(current_override)}],
            )

        derived_row = history.derive_feature_rows([origin_date], prices_df, financial_df).iloc[0]
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise ValueError(
            f"No se pudieron derivar automáticamente {missing} desde el histórico de "
            f"referencia bundled: {exc}"
        ) from exc

    return {col: float(derived_row[col]) for col in missing if col in derived_row}


def build_feature_row(
    row: dict[str, Any], feature_columns: list[str], calendar_derived_columns: tuple[str, ...],
) -> dict[str, float]:
    """Resolve every column in *feature_columns* from *row*.

    Fills in calendar columns and/or (current value +) lag/exogenous columns from
    'date'/'origin_date' + the bundled reference history when missing. Raises ValueError
    naming exactly which feature(s) are still missing/blank once every derivation path has
    been tried — this is what maps to HTTP 422 via MissingRequiredFeatureError in plugin.py,
    not a generic 500.
    """
    resolved = dict(row)
    date_val = _resolve_date_value(resolved)

    missing_calendar = [
        col for col in calendar_derived_columns
        if col in feature_columns and _is_blank(resolved.get(col))
    ]
    if missing_calendar:
        if date_val is None:
            raise ValueError(
                f"Faltan las variables de calendario {missing_calendar} y no se aportó "
                "'date' ni 'origin_date' (YYYY-MM-DD) para derivarlas automáticamente."
            )
        resolved.update(_calendar_features(date_val))

    missing = [col for col in feature_columns if _is_blank(resolved.get(col))]
    if missing and date_val is not None:
        resolved.update(_derive_from_history(resolved, missing, date_val))

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
    date_val = _resolve_date_value(row)
    if date_val is None:
        return None, None
    try:
        origin = _parse_date(date_val)
    except (ValueError, TypeError):
        return None, None
    origin_month = pd.Timestamp(origin).to_period("M").to_timestamp()
    target_month = origin_month + pd.DateOffset(months=horizon)
    return origin_month.strftime("%Y-%m-%d"), target_month.strftime("%Y-%m-%d")
