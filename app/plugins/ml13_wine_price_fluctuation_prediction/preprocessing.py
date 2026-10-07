"""ETL + feature engineering for ml13 — faithful port of the AI team's modules/common/data.py and
modules/common/features.py (inbox/a13/codigo/).

Any change here changes model inputs: the indicators must stay byte-for-byte equivalent to the
original pipeline the delivered scaler/model were fitted on.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientDataError
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    BOLLINGER_WINDOW,
    BULLETIN_COLUMN,
    CAMPAIGN_COLUMN,
    FEATURE_COLUMNS,
    MIN_INFERENCE_WEEKS,
    PRICE_COLUMN_CANDIDATES,
    RSI_PERIOD,
    SMA_LONG_WINDOW,
    WEEK_COLUMN,
)

_BULLETIN_RE = re.compile(r"(?i)^\s*semana\s+(\d{1,2})/(\d{4})\s*$")


# ── data.py ──────────────────────────────────────────────────────────────────

def parse_bulletin_to_campaign_week(value: str) -> tuple[str, int]:
    """'SEMANA 50/2023' -> ('2022/2023', 50). year_end is always the campaign's second year."""
    m = _BULLETIN_RE.match(str(value))
    if not m:
        raise ValueError(
            f"Malformed bulletin value: {value!r}. "
            "Expected format: 'SEMANA <week>/<year_end>' (e.g. 'SEMANA 50/2023')."
        )
    week = int(m.group(1))
    year_end = int(m.group(2))
    if not 1 <= week <= 53:
        raise ValueError(f"Invalid ISO week number {week} in bulletin: {value!r}. Must be 1..53.")
    return f"{year_end - 1}/{year_end}", week


def detect_price_column(df: pd.DataFrame) -> str:
    """First matching column of PRICE_COLUMN_CANDIDATES (data.py::_detect_price_column)."""
    for col in PRICE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of the expected price columns {PRICE_COLUMN_CANDIDATES} were found in "
        f"{list(df.columns)}"
    )


def parse_campaign_date(row: pd.Series) -> datetime | None:
    """Campaign (Aug-Jul) + ISO week -> Monday of that ISO week; None if unparseable.

    week >= 31 -> first campaign year; week < 31 -> second campaign year.
    """
    try:
        parts = str(row[CAMPAIGN_COLUMN]).split("/")
        if len(parts) != 2:
            raise ValueError
        year_start, year_end = int(parts[0]), int(parts[1])
        week = int(row[WEEK_COLUMN])
        real_year = year_start if week >= 31 else year_end
        return datetime.strptime(f"{real_year}-W{week:02d}-1", "%G-W%V-%u")
    except Exception:  # pylint: disable=broad-exception-caught
        # Same as the original ETL: unparseable rows are dropped downstream.
        return None


def normalize_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    """data.py::load_raw_data schema handling (without the file read).

    campaign+week take precedence; otherwise they are derived from 'bulletin'.
    """
    df = df.copy()
    if CAMPAIGN_COLUMN in df.columns and WEEK_COLUMN in df.columns:
        return df
    if BULLETIN_COLUMN in df.columns:
        parsed = df[BULLETIN_COLUMN].map(parse_bulletin_to_campaign_week)
        df[CAMPAIGN_COLUMN] = parsed.map(lambda x: x[0])
        df[WEEK_COLUMN] = parsed.map(lambda x: x[1])
        return df
    raise ValueError(
        f"Raw CSV must contain either ({CAMPAIGN_COLUMN!r} and {WEEK_COLUMN!r}) or a "
        f"{BULLETIN_COLUMN!r} column. Found columns: {list(df.columns)}"
    )


def clean_and_index_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """data.py::clean_and_index_data — date-indexed frame with a single 'price' column."""
    df = df_raw.copy()
    price_col = detect_price_column(df)
    df["fecha"] = df.apply(parse_campaign_date, axis=1)
    df = df.dropna(subset=["fecha", price_col])
    df = df.sort_values("fecha").set_index("fecha")
    return df[[price_col]].rename(columns={price_col: "price"})


# ── features.py ──────────────────────────────────────────────────────────────

def _compute_technical_indicators(df_price: pd.DataFrame) -> pd.DataFrame:
    """Shared indicator block of generate_technical_features[_inference]."""
    if "price" not in df_price.columns:
        raise ValueError("Expected a 'price' column in df_price.")
    df = df_price.copy()
    price = df["price"]

    df["logret"] = np.log(price / price.shift(1))

    sma_long = price.rolling(window=SMA_LONG_WINDOW, min_periods=SMA_LONG_WINDOW).mean()
    df["distsma12"] = (price - sma_long) / price

    delta = price.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    loss = loss.replace(0, 1e-9)
    df["rsi14"] = 100 - (100 / (1 + gain / loss))

    roll_mean = price.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    roll_std = price.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std()
    df["bollingerpos"] = (price - roll_mean) / (2 * roll_std.replace(0, np.nan))

    weeks = df.index.isocalendar().week.astype(int)
    df["weeksin"] = np.sin(2 * np.pi * weeks / 52.0)
    df["weekcos"] = np.cos(2 * np.pi * weeks / 52.0)
    return df


def generate_technical_features_inference(df_price: pd.DataFrame) -> pd.DataFrame:
    """Features WITHOUT target; drops only rows whose features are incomplete (warm-up)."""
    df = _compute_technical_indicators(df_price)
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["price"]).copy()
    if len(df_clean) == 0:
        raise InsufficientDataError(
            "No valid rows remain after feature generation (e.g. constant price over the "
            f"{BOLLINGER_WINDOW}-week Bollinger window)."
        )
    return df_clean


def generate_technical_features(
    df_price: pd.DataFrame, target_window: int, return_threshold: float,
) -> pd.DataFrame:
    """Features + binary target: mean(price[t+1..t+window]) > price_t * (1 + threshold)."""
    df = _compute_technical_indicators(df_price)
    future_mean = (
        df["price"]
        .shift(-target_window)
        .rolling(window=target_window, min_periods=target_window)
        .mean()
    )
    df["target"] = np.nan
    valid_future = future_mean.notna()
    df.loc[valid_future, "target"] = (
        future_mean[valid_future] > df.loc[valid_future, "price"] * (1 + return_threshold)
    ).astype(int)
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["price", "target"]).copy()
    if len(df_clean) == 0:
        raise ValueError("No valid rows remain after feature and target generation.")
    return df_clean


# ── inference.py::predict_from_csv input stage ──────────────────────────────

def prepare_inference_frame(df_raw_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Normalize schema, clean, enforce the 20-week minimum and build inference features.

    Returns (df_raw_normalized, df_features, original_price_column).
    """
    df_raw = normalize_time_keys(df_raw_input)
    df_price = clean_and_index_data(df_raw)
    if len(df_price) < MIN_INFERENCE_WEEKS:
        raise InsufficientDataError(
            f"Insufficient historical data: {len(df_price)} weeks provided, but at least "
            f"{MIN_INFERENCE_WEEKS} are required to compute all features "
            "(RSI-14, SMA-12, Bollinger-20 dominant window)."
        )
    return df_raw, generate_technical_features_inference(df_price), detect_price_column(df_raw)


def merge_predictions(
    df_raw: pd.DataFrame, df_features: pd.DataFrame, price_col: str,
) -> pd.DataFrame:
    """Left-merge features + pred_proba_up back onto every input row (predict_from_csv step 7)."""
    id_cols = [CAMPAIGN_COLUMN, WEEK_COLUMN]
    if BULLETIN_COLUMN in df_raw.columns:
        id_cols = [BULLETIN_COLUMN] + id_cols
    df_raw_with_date = df_raw.copy()
    df_raw_with_date["fecha"] = df_raw_with_date.apply(parse_campaign_date, axis=1)
    merge_cols = ["fecha"] + FEATURE_COLUMNS + ["pred_proba_up"]
    df_output = df_raw_with_date.merge(
        df_features.reset_index()[merge_cols], on="fecha", how="left",
    )
    if price_col != "price":
        df_output = df_output.rename(columns={price_col: "price"})
    df_output = df_output.set_index("fecha").sort_index()
    output_cols = id_cols + ["price"] + FEATURE_COLUMNS + ["pred_proba_up"]
    return df_output[[c for c in output_cols if c in df_output.columns]]
