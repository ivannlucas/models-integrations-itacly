"""Checks that input data follows the fitted preprocessor's contract (vocabulary and scale).

The preprocessor is never refit (train() reuses it), so an unknown category silently maps to
__MISSING__ and a value on another scale (a ratio sent as a percentage) is standardised to
hundreds of standard deviations, saturating the MLP at 0/1 without any error.
"""
from __future__ import annotations

import pandas as pd

from app.domain.services.exceptions import DataContractError
from app.plugins.ml30_meat_traceability_detection.constants import (
    CATEGORICAL_FEATURES,
    CONTRACT_SCALE_ERROR_Z,
    CONTRACT_SCALE_WARNING_Z,
    NUMERIC_FEATURES,
)

# Values _coerce()/pandas produce for a missing categorical; they map to __MISSING__ on purpose.
_MISSING_TOKENS = {"", "nan", "none", "<na>", "unknown", "__missing__"}
# The platform only surfaces an error detail up to 300 characters (extractPlainDetailString).
_MAX_DETAIL_LEN = 300


def _names(columns: list[str], limit: int = 4) -> str:
    shown = ", ".join(columns[:limit])
    return shown + (f" y {len(columns) - limit} más" if len(columns) > limit else "")


def check_data_contract(
    df: pd.DataFrame, preprocessor, present_columns
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for ``df`` (already coerced) against ``preprocessor``.

    ``present_columns`` are the columns of the raw CSV/payload, so a column that was absent
    (and got a default value from _coerce) is reported as missing, not as a scale problem.
    """
    present = set(present_columns)
    unknown_cols: list[str] = []
    scale_cols: list[str] = []
    warnings: list[str] = []

    for col in CATEGORICAL_FEATURES:
        if col not in present:
            warnings.append(f"Falta la columna '{col}'; se tratará como valor ausente.")
            continue
        values = df[col].astype(str)
        given = values[~values.str.strip().str.lower().isin(_MISSING_TOKENS)]
        if given.empty:
            continue
        unknown = given[~given.isin(set(preprocessor.categorical_vocab[col]))]
        if len(unknown) == len(given):
            unknown_cols.append(col)
        elif len(unknown):
            examples = ", ".join(sorted(unknown.unique())[:3])
            warnings.append(
                f"La columna '{col}' tiene un {100 * len(unknown) / len(given):.0f}% de valores "
                f"desconocidos para el modelo (p. ej. {examples}); se tratarán como valor ausente."
            )

    for col in NUMERIC_FEATURES:
        if col not in present:
            warnings.append(f"Falta la columna '{col}'; se usará un valor por defecto.")
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        z = ((series - preprocessor.numeric_means[col]) / preprocessor.numeric_stds[col]).abs().median()
        if z > CONTRACT_SCALE_ERROR_Z:
            scale_cols.append(col)
        elif z > CONTRACT_SCALE_WARNING_Z:
            warnings.append(
                f"La columna '{col}' se aleja mucho de los valores de entrenamiento "
                f"(mediana de |z| = {z:.1f})."
            )

    errors: list[str] = []
    if unknown_cols:
        errors.append(f"Valores desconocidos en: {_names(unknown_cols, 3)}.")
    if scale_cols:
        errors.append(f"Otra escala (¿porcentaje en vez de fracción?) en: {_names(scale_cols, 2)}.")
    return errors, warnings


def enforce_data_contract(df: pd.DataFrame, preprocessor, present_columns, logger) -> list[str]:
    """Raise DataContractError on contract errors; log and return the non-blocking warnings."""
    errors, warnings = check_data_contract(df, preprocessor, present_columns)
    for warning in warnings:
        logger.warning("ml30 data contract: %s", warning)
    if errors:
        detail = ("El CSV no sigue el formato del modelo. " + " ".join(errors)
                  + " Revisa la estructura esperada del CSV.")
        logger.error("ml30 data contract rejected input: %s", detail)
        raise DataContractError(detail if len(detail) <= _MAX_DETAIL_LEN else detail[:_MAX_DETAIL_LEN - 1] + "…")
    return warnings
