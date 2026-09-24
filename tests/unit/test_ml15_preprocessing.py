"""Unit tests for ml15 preprocessing pure logic (no real bundled reference CSVs needed —
tests that exercise the history-derivation path monkeypatch history.load_reference_data
with tiny synthetic DataFrames instead, since artifacts/ is gitignored and not guaranteed
present in CI)."""
import math

import pandas as pd
import pytest

from app.plugins.ml15_wine_ipi_price_forecast import history
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


def test_build_feature_row_missing_feature_without_date_raises_value_error_naming_it():
    """Without a date, a missing feature can't be derived from history either — must fail."""
    row = {k: v for k, v in _BASE_ROW.items() if k != "copper_lag_14"}
    row.update({"month_sin": 0.5, "month_cos": 0.87, "quarter": 1, "is_spring_risk": 0})
    with pytest.raises(ValueError, match="copper_lag_14"):
        build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)


def test_build_feature_row_missing_date_and_calendar_raises_value_error():
    with pytest.raises(ValueError, match="date"):
        build_feature_row(dict(_BASE_ROW), FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)


def _synthetic_reference_data():
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    prices_df = pd.DataFrame({
        "date": dates, "spain": [100.0 + i for i in range(24)], "usa": [200.0] * 24,
    })
    fin_dates = pd.date_range("2020-01-01", periods=78, freq="MS")
    n = len(fin_dates)
    financial_df = history.reindex_financial_to_base_2020(pd.DataFrame({
        "date": fin_dates, "chem_sector_avg": [50.0] * n, "copper_avg": [10.0] * n,
        "oil_brent_avg": [70.0] * n, "eur_usd_avg": [1.1] * n,
    }))
    return prices_df, financial_df


def test_build_feature_row_derives_missing_feature_from_history_when_date_given(monkeypatch):
    """A feature missing from the row, but resolvable via the bundled reference history
    (mocked here), must be filled in automatically instead of raising — this is the fix for
    the reported bug where a simple date+ipi_national_current input failed."""
    monkeypatch.setattr(history, "load_reference_data", lambda: _synthetic_reference_data())

    row = {"date": "2025-01-01"}  # nothing else -- not even ipi_national_current
    resolved = build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)

    assert resolved["ipi_national_current"] == pytest.approx(112.0)  # 2025-01 = idx12
    assert resolved["ipi_national_lag_1"] == pytest.approx(111.0)  # 2024-12


def test_build_feature_row_explicit_ipi_national_current_overrides_bundled_history(monkeypatch):
    monkeypatch.setattr(history, "load_reference_data", lambda: _synthetic_reference_data())

    row = {"date": "2025-01-01", "ipi_national_current": 999.0}
    resolved = build_feature_row(row, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)

    assert resolved["ipi_national_current"] == pytest.approx(999.0)
    # lag_1 (2024-12) is untouched by the single-point override -- still from bundled data
    assert resolved["ipi_national_lag_1"] == pytest.approx(111.0)


def test_build_feature_row_history_derivation_failure_raises_clear_value_error(monkeypatch):
    def _raise():
        raise FileNotFoundError("reference CSV not found")

    monkeypatch.setattr(history, "load_reference_data", _raise)

    with pytest.raises(ValueError, match="histórico de referencia"):
        build_feature_row({"date": "2025-01-01"}, FEATURE_COLUMNS, CALENDAR_DERIVED_COLUMNS)


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
