"""Input preprocessing for the meat-traceability scoring plugin."""
from __future__ import annotations

import pandas as pd

from app.plugins.ml30_meat_traceability_detection.constants import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce feature columns to numeric/str with safe defaults for missing values."""
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown") if col in df.columns else "unknown"
    return df


def build_dataframe_from_features(features: dict) -> pd.DataFrame:
    """Build a single-row DataFrame matching the feature contract."""
    row: dict = {}
    for col in NUMERIC_FEATURES:
        val = features.get(col)
        row[col] = float(val) if val is not None else 0.0
    for col in CATEGORICAL_FEATURES:
        val = features.get(col)
        row[col] = str(val) if val is not None else "unknown"
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def build_dataframe_from_csv(data_path: str) -> pd.DataFrame:
    """Read a CSV and coerce feature columns; non-feature columns (ids) are kept."""
    return _coerce(pd.read_csv(data_path))
