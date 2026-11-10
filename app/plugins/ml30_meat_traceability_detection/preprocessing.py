"""Input preprocessing for the meat-traceability scoring plugin."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.plugins.ml30_meat_traceability_detection.constants import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce feature columns to numeric/str.

    Missing numeric values stay NaN: the fitted TabularPreprocessor imputes each one with its
    training median, as the AI team's pipeline does. They used to be filled with 0.0 here, before
    the preprocessor could see them — pH 0 or a 0 kg carcass instead of the median (5.74,
    246 kg) — which changed the score of every event with an empty sensor (test AUC 0.7117
    instead of the audited 0.7185).
    """
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown") if col in df.columns else "unknown"
    return df


def build_dataframe_from_features(features: dict) -> pd.DataFrame:
    """Build a single-row DataFrame matching the feature contract."""
    row: dict = {}
    for col in NUMERIC_FEATURES:
        val = features.get(col)
        row[col] = float(val) if val is not None else np.nan  # imputed by the preprocessor
    for col in CATEGORICAL_FEATURES:
        val = features.get(col)
        row[col] = str(val) if val is not None else "unknown"
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def build_dataframe_from_csv(data_path: str) -> pd.DataFrame:
    """Read a CSV and coerce feature columns; non-feature columns (ids) are kept."""
    return _coerce(pd.read_csv(data_path))
