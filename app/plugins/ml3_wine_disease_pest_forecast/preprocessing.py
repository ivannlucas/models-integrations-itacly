"""Window building for ml3 — replicates run_inference from inbox/a03 predictor.py.

The functional input is a 168-hour window per series (never a single row). The plugin
reproduces the delivered inference behavior: per series, keep the last 168 rows (tail),
pad with the first row repeated if shorter, THEN apply feature engineering and scale.
"""
from __future__ import annotations

import pandas as pd

# X / X_tensor son los nombres del código entregado (predictor.py).
# pylint: disable=invalid-name

from app.plugins.ml3_wine_disease_pest_forecast.constants import (
    DATE_COLUMN,
    MODEL_FEATURES,
    RAW_FIXED_COLUMNS,
    SERIES_COLUMN,
    WINDOW_SIZE,
)
from app.plugins.ml3_wine_disease_pest_forecast.feature_engineering import apply_feature_engineering


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a raw DataFrame from an inline list of row dicts."""
    return pd.DataFrame(rows)


def validate_raw_columns(raw_df: pd.DataFrame) -> None:
    """Reject a batch CSV / inline row-set missing required raw sensor columns."""
    missing = [c for c in RAW_FIXED_COLUMNS + [DATE_COLUMN] if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas del contrato de entrada: {missing}")


def _tail_or_pad(series_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Return the last *window_size* rows, or pad with the first row if the series is shorter."""
    n_filas = len(series_df)
    if n_filas > window_size:
        out = series_df.tail(window_size).copy()
    elif n_filas < window_size:
        pad = window_size - n_filas
        df_pad = pd.concat([series_df.iloc[[0]]] * pad, ignore_index=True)
        out = pd.concat([df_pad, series_df], ignore_index=True)
    else:
        out = series_df.copy()
    out.reset_index(drop=True, inplace=True)
    return out


def build_window_tensor(
    series_df: pd.DataFrame,
    scaler,
    window_size: int = WINDOW_SIZE,
    model_features: list[str] | None = None,
    date_column: str = DATE_COLUMN,
) -> tuple:
    """Build the (1, window, n_features) scaled tensor for a single series' last window.

    Returns (X_tensor, last_fecha, window_df). ``window_df`` is the post-feature-engineering
    window, used to snapshot the last raw row for the XAI service.
    """
    feats = model_features or MODEL_FEATURES
    series_df = _tail_or_pad(series_df, window_size)
    if date_column in series_df.columns:
        series_df = series_df.sort_values(date_column).reset_index(drop=True)
        series_df = _tail_or_pad(series_df, window_size)

    window_df = apply_feature_engineering(series_df, date_column=date_column)

    X = scaler.transform(window_df[feats])
    X_tensor = X.reshape(1, window_size, len(feats))
    last_fecha = window_df[date_column].iloc[-1] if date_column in window_df.columns else None
    return X_tensor, last_fecha, window_df


def prepare_series_groups(
    raw_df: pd.DataFrame,
    series_column: str = SERIES_COLUMN,
) -> list[tuple]:
    """Split a raw DataFrame into per-series groups, preserving the input series id.

    Returns a list of (series_name, series_df); when no series column is present the whole
    input is treated as one series named "Única" (same fallback as the delivered code).
    """
    if series_column in raw_df.columns and raw_df[series_column].nunique() > 1:
        groups = []
        for name, group in raw_df.groupby(series_column, sort=False):
            groups.append((name, group.copy()))
        return groups
    if series_column in raw_df.columns and len(raw_df) > 0:
        return [(raw_df[series_column].iloc[0], raw_df.copy())]
    return [("Única", raw_df.copy())]
