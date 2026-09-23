"""Unit tests for ml15's history.py (bundled-reference-based feature derivation).

Pure logic — uses small synthetic DataFrames, never touches the real bundled reference
CSVs (those live under artifacts/, which is gitignored and not guaranteed present in CI).
"""
import pandas as pd
import pytest

from app.plugins.ml15_wine_ipi_price_forecast import history
from app.plugins.ml15_wine_ipi_price_forecast.constants import FEATURE_COLUMNS


def _make_prices_df() -> pd.DataFrame:
    """18 months of a synthetic national IPI series (Jan-2024 .. Jun-2025) + usa column."""
    dates = pd.date_range("2024-01-01", periods=18, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "spain": [100.0 + i for i in range(18)],
        "usa": [200.0 + i for i in range(18)],
    })


def _make_financial_df() -> pd.DataFrame:
    """Enough months (incl. all of 2020, needed for the base-2020 reindex) + the raw cols."""
    dates = pd.date_range("2020-01-01", periods=78, freq="MS")  # 2020-01 .. 2026-06
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "chem_sector_avg": [50.0] * n,
        "copper_avg": [10.0] * n,
        "oil_brent_avg": [70.0] * n,
        "eur_usd_avg": [1.1] * n,
    })


def test_reindex_financial_to_base_2020_sets_2020_mean_to_100():
    df = _make_financial_df()
    out = history.reindex_financial_to_base_2020(df)
    year_2020 = out["date"].dt.year == 2020
    for col in ("chem_sector_avg", "copper_avg", "oil_brent_avg", "eur_usd_avg"):
        assert out.loc[year_2020, f"{col}_idx"].mean() == pytest.approx(100.0)


def test_reindex_financial_to_base_2020_missing_column_raises():
    df = _make_financial_df().drop(columns=["copper_avg"])
    with pytest.raises(ValueError, match="copper_avg"):
        history.reindex_financial_to_base_2020(df)


def test_reindex_financial_to_base_2020_no_2020_data_raises():
    df = _make_financial_df()
    df = df[df["date"].dt.year != 2020]
    with pytest.raises(ValueError, match="2020"):
        history.reindex_financial_to_base_2020(df)


def test_derive_feature_rows_produces_all_feature_columns_in_input_order():
    prices_df = _make_prices_df()
    financial_df = history.reindex_financial_to_base_2020(_make_financial_df())

    # Origin dates deliberately NOT in chronological order, to assert order preservation.
    origin_dates = [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-01-01")]
    out = history.derive_feature_rows(origin_dates, prices_df, financial_df)

    assert list(out["date"]) == origin_dates  # input order preserved, not re-sorted
    assert set(FEATURE_COLUMNS).issubset(set(out.columns))


def test_derive_feature_rows_autoregressive_lags_match_source_series():
    prices_df = _make_prices_df()
    financial_df = history.reindex_financial_to_base_2020(_make_financial_df())

    origin = pd.Timestamp("2025-06-01")  # spain=117.0 (index 17: Jan-2024=100 + 17)
    out = history.derive_feature_rows([origin], prices_df, financial_df).iloc[0]

    assert out["ipi_national_current"] == pytest.approx(117.0)
    assert out["ipi_national_lag_1"] == pytest.approx(116.0)  # 2025-05
    assert out["ipi_national_lag_6"] == pytest.approx(111.0)  # 2024-12


def test_derive_feature_rows_missing_lag_history_raises_value_error():
    prices_df = _make_prices_df()  # only starts 2024-01
    financial_df = history.reindex_financial_to_base_2020(_make_financial_df())

    # origin=2024-06: lag_1..lag_5 resolve fine (2024-05..2024-01), but lag_6 needs
    # 2023-12, one month before the synthetic series starts.
    with pytest.raises(ValueError, match="ipi_national_lag_6"):
        history.derive_feature_rows([pd.Timestamp("2024-06-01")], prices_df, financial_df)


def test_merge_user_points_into_prices_overrides_existing_date():
    prices_df = _make_prices_df()
    merged = history.merge_user_points_into_prices(
        prices_df, [{"date": pd.Timestamp("2025-06-01"), "value": 999.0}],
    )
    row = merged[merged["date"] == pd.Timestamp("2025-06-01")].iloc[0]
    assert row["spain"] == pytest.approx(999.0)


def test_merge_user_points_into_prices_extends_beyond_bundled_range():
    """A date past the bundled series' last month must be inserted, not dropped."""
    prices_df = _make_prices_df()  # ends 2025-06
    merged = history.merge_user_points_into_prices(
        prices_df, [{"date": pd.Timestamp("2025-07-01"), "value": 123.4}],
    )
    row = merged[merged["date"] == pd.Timestamp("2025-07-01")]
    assert len(row) == 1
    assert row.iloc[0]["spain"] == pytest.approx(123.4)


def test_find_simple_ipi_column_priority_order():
    assert history.find_simple_ipi_column(["date", "spain", "ipi_national_current"]) == "ipi_national_current"
    assert history.find_simple_ipi_column(["date", "spain"]) == "spain"
    assert history.find_simple_ipi_column(["date", "foo"]) is None


def test_is_simple_history_frame_true_for_ipi_history_shape():
    columns = ["date", "year", "month", "ipi_national_current"]
    assert history.is_simple_history_frame(columns, FEATURE_COLUMNS) is True


def test_is_simple_history_frame_false_for_full_feature_panel():
    columns = ["date"] + list(FEATURE_COLUMNS)
    assert history.is_simple_history_frame(columns, FEATURE_COLUMNS) is False


def test_is_simple_history_frame_false_without_a_value_column():
    assert history.is_simple_history_frame(["date", "year", "month"], FEATURE_COLUMNS) is False


def test_resolve_row_date_prefers_origin_date_then_date_then_year_month():
    assert history.resolve_row_date({"origin_date": "2025-03-15", "date": "2025-01-01"}) == pd.Timestamp("2025-03-01")
    assert history.resolve_row_date({"date": "2025-01-15"}) == pd.Timestamp("2025-01-01")
    assert history.resolve_row_date({"year": 2025, "month": 4}) == pd.Timestamp("2025-04-01")


def test_resolve_row_date_raises_when_unresolvable():
    with pytest.raises(ValueError):
        history.resolve_row_date({"foo": "bar"})
