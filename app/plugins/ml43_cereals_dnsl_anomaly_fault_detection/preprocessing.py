"""Input validation, null imputation and sequence building for ml43."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientSensorWindowError
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.preprocess import create_sequences
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import (
    ID_COLUMN,
    NORMAL_TOKENS,
    PARTIAL_NULL_MAX_RATIO,
    SENSOR_COLUMNS,
    SEQ_LENGTH,
    SOLAPAMIENTO_BETA,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)

logger = logging.getLogger(__name__)


def validate_sensor_data(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Validate required sensor columns and detect partial nulls.

    Adapted from inbox/a43/codigo/src/predict/predictor.py (validate_input_data),
    simplified to this plugin's fixed sensor columns.

    Returns:
        {sensor: {"null_count", "null_ratio"}} for columns with partial nulls within
        the allowed threshold (PARTIAL_NULL_MAX_RATIO).

    Raises:
        InsufficientSensorWindowError: if sensor columns are missing, a column is
            empty/non-numeric, or nulls exceed the allowed ratio.
    """
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise InsufficientSensorWindowError(
            f"CSV falta columnas de sensor requeridas: {missing}. "
            f"Se requieren las {len(SENSOR_COLUMNS)} columnas: {SENSOR_COLUMNS}."
        )

    partial_null_stats: dict[str, dict[str, float]] = {}
    exceeded: list[str] = []

    for col in SENSOR_COLUMNS:
        series = pd.to_numeric(df[col].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        if series.isna().all():
            raise InsufficientSensorWindowError(
                f"Columna de sensor completamente vacía o no numérica: '{col}'."
            )

        null_count = int(series.isna().sum())
        if null_count == 0:
            continue

        null_ratio = float(series.isna().mean())
        if null_ratio > PARTIAL_NULL_MAX_RATIO:
            exceeded.append(f"{col} ({null_count} nulos, {null_ratio:.2%})")
        else:
            partial_null_stats[col] = {"null_count": float(null_count), "null_ratio": null_ratio}

    if exceeded:
        raise InsufficientSensorWindowError(
            "Columnas con nulos parciales por encima del umbral permitido "
            f"({PARTIAL_NULL_MAX_RATIO:.0%}): {exceeded}."
        )

    return partial_null_stats


def temporal_impute_partial_nulls(
    df: pd.DataFrame,
    partial_null_stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Impute allowed partial nulls with temporal criteria (interpolation + carry).

    Adapted from inbox/a43/codigo/src/predict/predictor.py (_temporal_impute_partial_nulls).
    """
    columns_to_impute = [c for c in partial_null_stats if c in df.columns]
    if not columns_to_impute:
        return df

    df = df.copy()
    has_id = ID_COLUMN in df.columns
    has_ts = TIMESTAMP_COLUMN in df.columns

    def _fill_series(series: pd.Series) -> pd.Series:
        filled = series.interpolate(method="linear", limit_direction="both")
        return filled.ffill().bfill()

    if has_id and has_ts:
        df = df.sort_values([ID_COLUMN, TIMESTAMP_COLUMN])
        for col in columns_to_impute:
            df[col] = df.groupby(ID_COLUMN, sort=False)[col].transform(_fill_series)
    elif has_ts:
        df = df.sort_values([TIMESTAMP_COLUMN])
        for col in columns_to_impute:
            df[col] = _fill_series(df[col])
    elif has_id:
        for col in columns_to_impute:
            df[col] = df.groupby(ID_COLUMN, sort=False)[col].transform(_fill_series)
    else:
        for col in columns_to_impute:
            df[col] = _fill_series(df[col])

    for col in columns_to_impute:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(median_val) else median_val)

    logger.info("Temporal imputation applied for columns with partial nulls: %s", columns_to_impute)
    return df


def prepare_batch_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, list | None]:
    """Validate, impute and window a raw sensor CSV into model-ready sequences.

    Returns (X_arr [N,180,13], y_arr [N] or None, cycle_ids or None).
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    partial_null_stats = validate_sensor_data(df)
    df = temporal_impute_partial_nulls(df, partial_null_stats)

    has_target = TARGET_COLUMN in df.columns
    return create_sequences(
        df,
        feature_cols=SENSOR_COLUMNS,
        seq_length=SEQ_LENGTH,
        solapamiento_beta=SOLAPAMIENTO_BETA,
        id_column=ID_COLUMN if ID_COLUMN in df.columns else None,
        timestamp_column=TIMESTAMP_COLUMN if TIMESTAMP_COLUMN in df.columns else None,
        target_column=TARGET_COLUMN if has_target else None,
        normal_tokens=NORMAL_TOKENS,
    )


def build_inline_window(features: dict) -> np.ndarray:
    """Build a steady-state synthetic window [1, SEQ_LENGTH, 13] from a sensor snapshot.

    Each of the 13 sensor values is repeated SEQ_LENGTH times to simulate a cycle where
    the sensors hold the provided readings — approximates the model's response to a
    constant operating point. Same pattern as the DNSL family precedents in this repo
    (m47_dnsl_fallas_maquinaria_pasteurizado, ml45) — see manifest known_issues.
    """
    sensor_values = []
    for col in SENSOR_COLUMNS:
        v = features.get(col)
        if v is None:
            raise InsufficientSensorWindowError(f"Campo de sensor requerido no encontrado: '{col}'")
        sensor_values.append(float(v))
    return np.array([[sensor_values] * SEQ_LENGTH], dtype=np.float32)
