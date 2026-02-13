"""
Feature engineering and preprocessing utilities for wine price forecasting.

This module builds all model inputs from the cleaned price series:
- Technical indicators such as log returns, distance to long-term average,
  RSI and Bollinger band position.
- Seasonal features based on calendar week (sine and cosine encoding).
- Binary target indicating whether the average price over the next N weeks
  exceeds the current price by a given threshold.

It also provides functions to fit a scaler and transform features to numeric arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rsi_period: int = 14
    sma_short_window: int = 4
    sma_long_window: int = 12
    bollinger_window: int = 20


FEATURE_CONFIG = FeatureConfig()


def _compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute log returns of the price series."""
    return np.log(df["price"] / df["price"].shift(1))


def _compute_moving_averages(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute short and long moving averages."""
    sma_short = df["price"].rolling(
        window=FEATURE_CONFIG.sma_short_window,
        min_periods=FEATURE_CONFIG.sma_short_window,
    ).mean()
    sma_long = df["price"].rolling(
        window=FEATURE_CONFIG.sma_long_window,
        min_periods=FEATURE_CONFIG.sma_long_window,
    ).mean()
    return sma_short, sma_long


def _compute_rsi(df: pd.DataFrame) -> pd.Series:
    """Compute Relative Strength Index (RSI) over the price series."""
    period = FEATURE_CONFIG.rsi_period
    delta = df["price"].diff()

    gain = delta.where(delta > 0, 0.0).rolling(
        window=period, min_periods=period
    ).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(
        window=period, min_periods=period
    ).mean()

    # Avoid division by zero
    loss = loss.replace(0, 1e-9)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_bollinger_position(df: pd.DataFrame) -> pd.Series:
    """
    Compute a simple Bollinger band position indicator.

    Position is defined as:
        (price - rolling_mean) / (2 * rolling_std)
    """
    window = FEATURE_CONFIG.bollinger_window
    roll_mean = df["price"].rolling(window=window, min_periods=window).mean()
    roll_std = df["price"].rolling(window=window, min_periods=window).std()

    # Avoid division by zero
    denom = 2 * roll_std.replace(0, np.nan)
    boll_pos = (df["price"] - roll_mean) / denom
    return boll_pos


def _compute_seasonal_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Encode week-of-year seasonality using sine and cosine transforms."""
    weeks = df.index.isocalendar().week.astype(int)
    weeksin = np.sin(2 * np.pi * weeks / 52.0)
    weekcos = np.cos(2 * np.pi * weeks / 52.0)
    return weeksin, weekcos


def _compute_future_mean_price(df: pd.DataFrame, target_window: int) -> pd.Series:
    """
    Compute the mean price over the next 'target_window' weeks.

    Uses a forward-looking rolling window.
    """
    # Use a forward-looking window indexer if available, or shift and rolling
    future_mean = (
        df["price"]
        .shift(-target_window + 1)
        .rolling(window=target_window, min_periods=target_window)
        .mean()
    )
    return future_mean


def generate_technical_features(
    df_price: pd.DataFrame,
    target_window: int,
    return_threshold: float,
) -> pd.DataFrame:
    """
    Generate technical features and binary target from a price time series.

    Args:
        df_price: DataFrame indexed by date, with at least a 'price' column.
        target_window: Number of weeks to look ahead for the target.
        return_threshold: Minimum relative increase (e.g. 0.005 = 0.5%)
                          to label the target as 1.

    Returns:
        DataFrame with columns:
            - price
            - logret
            - distsma12
            - rsi14
            - bollingerpos
            - weeksin
            - weekcos
            - target  (0 or 1)
        Rows with incomplete features or missing target are dropped.
    """
    if "price" not in df_price.columns:
        raise ValueError("Expected a 'price' column in df_price.")

    df = df_price.copy()

    # 1. Technical indicators
    df["logret"] = _compute_log_returns(df)

    sma_short, sma_long = _compute_moving_averages(df)
    df["distsma12"] = (df["price"] - sma_long) / df["price"]

    df["rsi14"] = _compute_rsi(df)
    df["bollingerpos"] = _compute_bollinger_position(df)

    weeksin, weekcos = _compute_seasonal_features(df)
    df["weeksin"] = weeksin
    df["weekcos"] = weekcos

    # 2. Target: average price over next target_window weeks vs current price
    future_mean = _compute_future_mean_price(df, target_window)
    df["target"] = np.nan

    valid_future = future_mean.notna()
    df.loc[valid_future, "target"] = (
        (future_mean[valid_future] >= df.loc[valid_future, "price"] * (1 + return_threshold))
        .astype(int)
    )

    # 3. Drop rows with incomplete features or missing target
    feature_cols = ["logret", "distsma12", "rsi14", "bollingerpos", "weeksin", "weekcos"]
    required_cols = feature_cols + ["price", "target"]

    df_clean = df.dropna(subset=required_cols).copy()

    # Optionally warn if many rows were dropped (can be logged elsewhere)
    if len(df_clean) == 0:
        raise ValueError("No valid rows remain after feature and target generation.")

    return df_clean


# ----------------------------------------------------------------------
# Preprocessing utilities: scaler
# ----------------------------------------------------------------------

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the given feature DataFrame.

    Args:
        X: DataFrame of features (no target).

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler


def transform_features(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """
    Transform a feature DataFrame into a scaled numpy array.

    Args:
        scaler: Fitted StandardScaler.
        X: DataFrame of features.

    Returns:
        Scaled features as a 2D numpy array (n_samples, n_features).
    """
    return scaler.transform(X.values)
