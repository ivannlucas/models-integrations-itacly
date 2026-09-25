"""Raw weekly history -> 39-feature LSTM input, faithful port of
src/data_processing/feature_engineering.py::build_modeling_features_from_raw from the
delivered a14 code. The client always supplies its own raw weekly history (date + 6 base
columns) — there is no bundled reference series to fall back on (unlike ml15), see
inbox/a14/manifest.yaml::artifacts.extra_bundled_data_required_at_inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientDataError
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    DATE_COL,
    DIFF_MONTH_PERIOD,
    EXPECTED_FREQUENCY_DAYS,
    EXPECTED_WEEKDAY,
    FEATURES_TO_PROCESS,
    LAGS,
    MIN_HISTORY_ROWS,
    MOVING_AVERAGE_WINDOWS,
    RAW_VALUE_COLS,
    SEQ_LEN,
)

_DATE_ALIASES = ("date", "FECHA")


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts, normalizing the date column name."""
    if not rows:
        raise InsufficientDataError(
            f"Historial vacío: se requieren al menos {MIN_HISTORY_ROWS} filas semanales."
        )
    df = pd.DataFrame(rows)
    if DATE_COL not in df.columns:
        alias = next((c for c in df.columns if str(c).strip().lower() == "fecha"), None)
        if alias is not None:
            df = df.rename(columns={alias: DATE_COL})
    return df


def _missing_raw_columns(df: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    if DATE_COL not in df.columns:
        missing.append("date")
    missing.extend(col for col in RAW_VALUE_COLS if col not in df.columns)
    return missing


def _validate_weekly_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Reject duplicate/non-weekly dates and return rows sorted chronologically."""
    result = df.copy()
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], format="%Y-%m-%d", errors="coerce")
    invalid_dates = int(result[DATE_COL].isna().sum())
    if invalid_dates:
        raise InsufficientDataError(
            f"La columna de fecha contiene {invalid_dates} valores no parseables (formato esperado YYYY-MM-DD)."
        )

    duplicated = result[DATE_COL].duplicated(keep=False)
    if duplicated.any():
        dupes = result.loc[duplicated, DATE_COL].dt.strftime("%Y-%m-%d").drop_duplicates().tolist()
        raise InsufficientDataError(f"El histórico contiene fechas duplicadas: {dupes[:5]}.")

    result = result.sort_values(DATE_COL).reset_index(drop=True)
    invalid_weekdays = result[DATE_COL].dt.weekday != EXPECTED_WEEKDAY
    if invalid_weekdays.any():
        examples = result.loc[invalid_weekdays, DATE_COL].dt.strftime("%Y-%m-%d").head(5).tolist()
        raise InsufficientDataError(
            f"La frecuencia temporal requerida es semanal W-SUN (domingo); fechas no alineadas: {examples}."
        )

    if len(result) > 1:
        deltas = result[DATE_COL].diff().dt.days.iloc[1:]
        invalid = deltas != EXPECTED_FREQUENCY_DAYS
        if invalid.any():
            first_index = int(invalid[invalid].index[0])
            previous_date = result.loc[first_index - 1, DATE_COL].date()
            current_date = result.loc[first_index, DATE_COL].date()
            delta_days = int(deltas.loc[first_index])
            raise InsufficientDataError(
                "La frecuencia temporal debe ser semanal y consecutiva "
                f"({EXPECTED_FREQUENCY_DAYS} días); entre {previous_date} y {current_date} "
                f"hay {delta_days} días."
            )
    return result


def build_modeling_features(rows: list[dict]) -> pd.DataFrame:
    """Validate a raw weekly history and derive the 39 modeling features.

    Raises InsufficientDataError for any structural/content problem (missing columns,
    invalid/duplicate/non-weekly dates, nulls, non-finite values, not enough rows before or
    after feature derivation) — never a silent/degraded prediction.
    """
    df = build_raw_dataframe(rows)
    missing = _missing_raw_columns(df)
    if missing:
        raise InsufficientDataError(
            f"Faltan columnas mínimas en el histórico: {missing}. "
            f"Columnas esperadas: {list(_DATE_ALIASES)} y {list(RAW_VALUE_COLS)}."
        )

    result = _validate_weekly_dates(df)

    for col in RAW_VALUE_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    null_counts = result[list(RAW_VALUE_COLS)].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        detail = {col: int(count) for col, count in null_counts.items()}
        raise InsufficientDataError(
            f"El histórico contiene valores vacíos o no numéricos en columnas requeridas: {detail}."
        )

    non_finite = {
        col: int((~np.isfinite(result[col].to_numpy(dtype=float))).sum())
        for col in RAW_VALUE_COLS
    }
    non_finite = {col: count for col, count in non_finite.items() if count > 0}
    if non_finite:
        raise InsufficientDataError(
            f"El histórico contiene valores no finitos en columnas requeridas: {non_finite}."
        )

    if len(result) < MIN_HISTORY_ROWS:
        raise InsufficientDataError(
            "Historial insuficiente para inferencia: "
            f"registros_recibidos={len(result)}, registros_minimos_requeridos={MIN_HISTORY_ROWS} "
            f"(feature_lookback + seq_len = {MIN_HISTORY_ROWS - SEQ_LEN} + {SEQ_LEN})."
        )

    result["MONTH"] = result[DATE_COL].dt.month
    result["MONTH_SIN"] = np.sin(2 * np.pi * result["MONTH"] / 12)
    result["MONTH_COS"] = np.cos(2 * np.pi * result["MONTH"] / 12)

    for col in FEATURES_TO_PROCESS:
        for lag in LAGS:
            result[f"{col}_LAG_{lag}"] = result[col].shift(lag)

    for col in FEATURES_TO_PROCESS:
        for window in MOVING_AVERAGE_WINDOWS:
            result[f"{col}_MA_{window}"] = result[col].rolling(window=window).mean()

    for col in FEATURES_TO_PROCESS:
        result[f"{col}_DIFF_MONTH"] = result[col].diff(DIFF_MONTH_PERIOD)

    result = result.dropna().reset_index(drop=True)
    if len(result) < SEQ_LEN:
        raise InsufficientDataError(
            "Historial insuficiente tras generar variables derivadas: se necesitan "
            f"{SEQ_LEN} filas modelables y hay {len(result)}."
        )
    return result
