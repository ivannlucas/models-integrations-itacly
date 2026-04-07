import re

import pandas as pd

from app.domain.services.exceptions import InsufficientRowsError
from app.plugins.meat_price_forecast.predict_dto import MeatPriceRow


def build_lagged_features(
    rows: list[MeatPriceRow],
    required_features: list[str],
) -> pd.DataFrame:
    """Convert raw weekly price rows into a lag-feature DataFrame."""
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    lag_specs: dict[str, set[int]] = {}
    for feat in required_features:
        match = re.match(r"(.+)_lag(\d+)$", feat)
        if match:
            base, lag = match.group(1), int(match.group(2))
            lag_specs.setdefault(base, set()).add(lag)

    for base, lags in lag_specs.items():
        if base not in df.columns:
            continue
        for lag in sorted(lags):
            df[f"{base}_lag{lag}"] = df[base].shift(lag)

    available = [f for f in required_features if f in df.columns]
    df_clean = df.dropna(subset=available).reset_index(drop=True)

    if len(df_clean) == 0:
        raise InsufficientRowsError(
            "No rows remain after constructing lag features and dropping NaN. "
            "Provide at least 4 weekly rows."
        )

    return df_clean
