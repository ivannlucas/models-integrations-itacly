"""Input preparation for ml26: raw lot + readings -> (n_lots, 24, 22) model windows.

Mirrors the AI team's raw inference path (src/predict/raw_inputs.py::load_raw_inference_frame +
prepare_raw_model_input) on top of the vendored build_feature_frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientSequenceHistoryError
from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.feature_engineering import (
    build_feature_frame,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import (
    LOT_COLUMNS,
    PIPELINE_CONFIG,
    READING_REQUIRED_COLUMNS,
    STAGES,
    WINDOW,
    WINE_TYPES,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.exceptions import InvalidWineryInputError


def _check_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise InvalidWineryInputError(f"{label}: faltan columnas obligatorias {missing}")


def _check_categories(frame: pd.DataFrame, column: str, allowed: list[str], label: str) -> None:
    values = set(frame[column].dropna().astype(str).unique())
    unknown = sorted(values - set(allowed))
    if unknown or frame[column].isna().any():
        raise InvalidWineryInputError(
            f"{label}: valores no soportados en '{column}': {unknown or ['<vacío>']} (válidos: "
            f"{allowed})"
        )


def _check_stage_progress(frame: pd.DataFrame, label: str) -> None:
    """stage_progress is mandatory on every reading and must lie in [0, 1] (manifest KI-01)."""
    values = pd.to_numeric(frame["stage_progress"], errors="coerce")
    if values.isna().any():
        lots = sorted(frame.loc[values.isna(), "lot_id"].astype(str).unique().tolist())
        raise InvalidWineryInputError(
            f"{label}: stage_progress es obligatorio en todas las lecturas; "
            f"falta en los lotes {lots}"
        )
    if ((values < 0.0) | (values > 1.0)).any():
        raise InvalidWineryInputError(f"{label}: stage_progress debe estar en [0, 1]")


def build_raw_frame(lots: pd.DataFrame, readings: pd.DataFrame) -> pd.DataFrame:
    """Validate and merge lote/lecturas tables like load_raw_inference_frame does."""
    _check_columns(lots, LOT_COLUMNS, "lote")
    _check_columns(readings, READING_REQUIRED_COLUMNS, "lecturas")
    _check_categories(lots, "wine_type", WINE_TYPES, "lote")
    _check_categories(readings, "stage", STAGES, "lecturas")
    _check_stage_progress(readings, "lecturas")
    if lots["lot_id"].duplicated().any():
        raise InvalidWineryInputError("lote: lot_id duplicado — se espera una fila por depósito")

    readings = readings.copy()
    if "timestamp_index" not in readings.columns:
        if "timestamp" in readings.columns:
            readings = readings.sort_values(["lot_id", "timestamp"]).reset_index(drop=True)
        else:
            readings = readings.reset_index(drop=True)
        readings["timestamp_index"] = readings.groupby("lot_id", sort=False).cumcount()
    if "timestamp" not in readings.columns:
        readings["timestamp"] = readings["timestamp_index"]

    merged = readings.merge(lots[LOT_COLUMNS], on="lot_id", how="left", validate="many_to_one")
    if merged[["wine_type", "volume_l", "ambient_temp_c"]].isna().any().any():
        missing_lots = sorted(
            merged.loc[merged["wine_type"].isna(), "lot_id"].astype(str).unique().tolist()
        )
        raise InvalidWineryInputError(f"lecturas contiene lot_id sin fila en lote: {missing_lots}")
    return merged.sort_values(["lot_id", "timestamp_index"]).reset_index(drop=True)


def frames_from_inline(features: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build lote/lecturas DataFrames from the inline request payload (single lot)."""
    lot = dict(features["lot"])
    readings = pd.DataFrame(
        [{k: v for k, v in r.items() if v is not None} for r in features["readings"]]
    )
    if "lot_id" not in readings.columns:
        readings["lot_id"] = lot["lot_id"]
    readings["lot_id"] = readings["lot_id"].fillna(lot["lot_id"])
    other = sorted(set(readings["lot_id"].astype(str)) - {str(lot["lot_id"])})
    if other:
        raise InvalidWineryInputError(
            f"readings contiene lot_id distinto del lote enviado: {other}"
        )
    return pd.DataFrame([lot]), readings


def frames_from_batch_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single batch CSV (readings with the lot columns repeated per row) into
    lote/lecturas.
    """
    _check_columns(df, LOT_COLUMNS + READING_REQUIRED_COLUMNS[1:], "CSV batch")
    lots = df[LOT_COLUMNS].drop_duplicates()
    if lots["lot_id"].duplicated().any():
        bad = sorted(lots.loc[lots["lot_id"].duplicated(), "lot_id"].astype(str).unique().tolist())
        raise InvalidWineryInputError(
            f"CSV batch: wine_type/volume_l/ambient_temp_c no constantes dentro del lote {bad}"
        )
    readings = df.drop(columns=["wine_type", "volume_l", "ambient_temp_c"])
    return lots.reset_index(drop=True), readings


def prepare_windows(
    raw_frame: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return (X (n_lots, 24, n_features), metadata[lot_id,timestamp,timestamp_index], last-step
    feature rows).
    """
    counts = raw_frame.groupby("lot_id").size()
    short = counts[counts < WINDOW]
    if len(short):
        raise InsufficientSequenceHistoryError(
            f"Se requieren al menos {WINDOW} lecturas (48 h a paso de 2 h) por lote; "
            f"lotes con historial insuficiente: {dict((str(k), int(v)) for k, v in short.items())}"
        )

    feature_frame, _, _ = build_feature_frame(raw_frame, PIPELINE_CONFIG, include_targets=False)
    aligned = feature_frame.copy()
    for column in feature_names:
        if column not in aligned.columns:
            aligned[column] = 0.0
    feature_frame = pd.concat(
        [
            feature_frame[["lot_id", "timestamp", "timestamp_index"]].reset_index(drop=True),
            aligned[feature_names].reset_index(drop=True),
        ],
        axis=1,
    )

    windows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    last_rows: list[pd.Series] = []
    ordered = feature_frame.sort_values(["lot_id", "timestamp_index"]).reset_index(drop=True)
    for lot_id, lot_frame in ordered.groupby("lot_id", sort=False):
        window_frame = lot_frame.tail(WINDOW)
        windows.append(window_frame[feature_names].to_numpy(dtype=np.float32))
        last = window_frame.iloc[-1]
        meta_rows.append(
            {
                "lot_id": str(lot_id),
                "timestamp": str(last["timestamp"]),
                "timestamp_index": int(last["timestamp_index"]),
            }
        )
        last_rows.append(last[feature_names])
    return (
        np.stack(windows).astype(np.float32),
        pd.DataFrame(meta_rows),
        pd.DataFrame(last_rows).reset_index(drop=True),
    )


def validate_feature_window(window: list[list[float]], n_features: int) -> np.ndarray:
    """Validate a pre-processed (24, n_features) window (reproducibility mode, memoria §4.2)."""
    values = np.asarray(window, dtype=np.float32)
    if values.shape != (WINDOW, n_features):
        raise InvalidWineryInputError(
            f"feature_window debe tener forma ({WINDOW}, {n_features}); recibido {values.shape}"
        )
    if not np.isfinite(values).all():
        raise InvalidWineryInputError("feature_window contiene valores no finitos")
    return values[np.newaxis, ...]
