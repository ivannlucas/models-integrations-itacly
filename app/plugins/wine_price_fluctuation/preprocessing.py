from datetime import datetime

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientDataError
from app.plugins.wine_price_fluctuation.predict_dto import WeeklyPriceRecord


def _parse_campaign_date(campaign: str, week: int) -> datetime:
    """Convert viticulture campaign + ISO week to a calendar date (Monday)."""
    parts = campaign.split("/")
    year_start, year_end = int(parts[0]), int(parts[1])
    real_year = year_start if week >= 31 else year_end
    date_str = f"{real_year}-W{week:02d}-1"
    return datetime.strptime(date_str, "%G-W%V-%u")


def build_features(records: list[WeeklyPriceRecord], feature_columns: list[str]) -> pd.DataFrame:
    """Convert raw weekly price records into a feature DataFrame."""
    rows = [r.model_dump() for r in records]
    df_raw = pd.DataFrame(rows)
    df_raw["fecha"] = df_raw.apply(
        lambda row: _parse_campaign_date(row["campaign"], row["week"]), axis=1
    )
    df_raw = df_raw.dropna(subset=["fecha", "price_red"])
    df_raw = df_raw.sort_values("fecha").set_index("fecha")
    df = df_raw[["price_red"]].rename(columns={"price_red": "price"})

    df["logret"] = np.log(df["price"] / df["price"].shift(1))

    sma12 = df["price"].rolling(window=12, min_periods=12).mean()
    df["distsma12"] = (df["price"] - sma12) / df["price"]

    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
    loss = loss.replace(0, 1e-9)
    df["rsi14"] = 100 - (100 / (1 + gain / loss))

    roll_mean = df["price"].rolling(window=20, min_periods=20).mean()
    roll_std = df["price"].rolling(window=20, min_periods=20).std()
    roll_std_safe = roll_std.where(roll_std > 1e-9, 1e-9)
    df["bollingerpos"] = (df["price"] - roll_mean) / (2 * roll_std_safe)

    weeks = df.index.isocalendar().week.astype(int)
    df["weeksin"] = np.sin(2 * np.pi * weeks / 52.0)
    df["weekcos"] = np.cos(2 * np.pi * weeks / 52.0)

    df_clean = df.dropna(subset=feature_columns)
    if len(df_clean) == 0:
        raise InsufficientDataError(
            "Not enough records to compute features. "
            "Provide at least ~22 weekly price records "
            "(Bollinger band requires a 20-week lookback)."
        )

    return df_clean[feature_columns]
