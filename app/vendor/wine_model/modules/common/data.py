"""
Data loading and basic ETL utilities for wine price forecasting.

This module is responsible for:
- Reading the raw CSV file with campaign, week and price information.
- Parsing viticulture campaigns (e.g. '2022/2023') and week numbers into
  real calendar dates using the August–July season convention.
- Cleaning the dataset by dropping invalid rows and sorting by date.
- Returning a time-indexed DataFrame with a standardized 'price' column
  ready for feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_CONFIG, RAW_DATA_DIR


@dataclass
class RawDataSchema:
    campaign: str
    week: str
    price: str


def _detect_price_column(df: pd.DataFrame) -> str:
    """
    Detect the red wine price column using configured candidates.

    Raises:
        ValueError: if no candidate column is found.
    """
    for col in DATA_CONFIG.price_column_candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of the expected price columns {DATA_CONFIG.price_column_candidates} "
        f"were found in {list(df.columns)}"
    )


def parse_campaign_date(row: pd.Series) -> Optional[datetime]:
    """
    Convert viticulture campaign + week to a real calendar date.

    Assumptions:
    - Campaign is a string like '2022/2023'.
    - Season runs from August of the first year to July of the next year.
    - Weeks >= 31 belong to the first year, weeks < 31 to the second year.
    - Return Monday of the ISO week.
    """
    try:
        campaign_str = str(row[DATA_CONFIG.campaign_column])  # e.g. "2022/2023"
        # Split by '/'
        parts = campaign_str.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid campaign format: {campaign_str}")

        year_start = int(parts[0])
        year_end = int(parts[1])
        week = int(row[DATA_CONFIG.week_column])

        # Decide real year based on week number (season Aug–Jul)
        real_year = year_start if week >= 31 else year_end

        # ISO week date: year-week-1 (Monday)
        date_str = f"{real_year}-W{week:02d}-1"
        return datetime.strptime(date_str, "%G-W%V-%u")
    except Exception:
        # In ETL we will drop rows where this fails
        return None



def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw CSV with wine prices.

    Args:
        path: Optional explicit path to the raw CSV. If None, use default from config.

    Returns:
        Raw DataFrame as loaded from CSV.
    """
    if path is None:
        path = RAW_DATA_DIR / DATA_CONFIG.raw_filename

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    return df


def clean_and_index_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns, parse dates and return a clean time-indexed DataFrame.

    Steps:
    - Detect price column (red wine).
    - Compute 'fecha' using campaign + week.
    - Drop rows with invalid dates or missing price.
    - Sort by date and set date as index.
    - Return DataFrame with at least the price column.
    """
    df = df_raw.copy()

    # Detect price column
    price_col = _detect_price_column(df)

    # Parse datetime from campaign and week
    df["fecha"] = df.apply(parse_campaign_date, axis=1)

    # Drop invalid rows
    df = df.dropna(subset=["fecha", price_col])

    # Sort and index
    df = df.sort_values("fecha")
    df = df.set_index("fecha")

    # Keep only the price column (and any others you might need later)
    df = df[[price_col]].rename(columns={price_col: "price"})

    return df
