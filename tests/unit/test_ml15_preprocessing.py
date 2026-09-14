"""Unit tests for ml15 preprocessing pure logic (no artifacts needed)."""
import math

import pandas as pd
import pytest

from app.plugins.ml15_wine_ipi_price_forecast.constants import CALENDAR_DERIVED_COLUMNS, FEATURE_COLUMNS
from app.plugins.ml15_wine_ipi_price_forecast.preprocessing import (
    build_feature_frame,
    build_feature_row,
    resolve_origin_target_dates,
)

_BASE_ROW = {
    "ipi_national_current": 123.85, "ipi_national_lag_1": 123.72, "ipi_national_lag_2": 123.87,
    "ipi_national_lag_3": 124.06, "ipi_national_lag_4": 124.06, "ipi_national_lag_5": 124.45,
    "ipi_national_lag_6": 124.19, "chem_sector_lag_11": 153.72, "copper_lag_14": 132.92,
    "eur_usd_lag_17": 95.62, "oil_brent_lag_12": 183.46, "usa_lag_1": 110.27,
}


def test_build_feature_row_derives_calendar_features_from_date():
    """month_sin/month_cos/quarter/is_spring_risk must be derived from 'date' when omitted."""
    row = {**_BASE_ROW, "date": "2025-04-01"}  # April -> spring risk
    resolved = build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)
    assert resolved["month_sin"] == pytest.approx(math.sin(2 * math.pi * 4 / 12))
    assert resolved["month_cos"] == pytest.approx(math.cos(2 * math.pi * 4 / 12))
    assert resolved["quarter"] == 2
    assert resolved["is_spring_risk"] == 1


def test_build_feature_row_non_spring_month_has_zero_risk_flag():
    row = {**_BASE_ROW, "date": "2025-01-01"}
    resolved = build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)
    assert resolved["quarter"] == 1
    assert resolved["is_spring_risk"] == 0


def test_build_feature_row_accepts_explicit_calendar_features_without_date():
    """If the caller already supplies month_sin/cos/quarter/is_spring_risk, no date is needed."""
    row = {**_BASE_ROW, "month_sin": 0.5, "month_cos": 0.87, "quarter": 1, "is_spring_risk": 0}
    resolved = build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)
    assert resolved["month_sin"] == pytest.approx(0.5)


def test_build_feature_row_missing_feature_raises_value_error_naming_it():
    row = {k: v for k, v in _BASE_ROW.items() if k != "copper_lag_14"}
    row["date"] = "2025-01-01"
    with pytest.raises(ValueError, match="copper_lag_14"):
        build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)


def test_build_feature_row_missing_date_and_calendar_raises_value_error():
    with pytest.raises(ValueError, match="date"):
        build_feature_row(dict(_BASE_ROW), FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)


def test_build_feature_frame_is_column_ordered():
    row = {**_BASE_ROW, "date": "2025-01-01"}
    frame = build_feature_frame(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 1


def test_resolve_origin_target_dates_shifts_by_horizon():
    origin, target = resolve_origin_target_dates({"date": "2025-01-15"}, horizon=6)
    assert origin == "2025-01-01"  # normalised to first-of-month
    assert target == "2025-07-01"


def test_resolve_origin_target_dates_none_when_no_date():
    origin, target = resolve_origin_target_dates({}, horizon=6)
    assert origin is None
    assert target is None
