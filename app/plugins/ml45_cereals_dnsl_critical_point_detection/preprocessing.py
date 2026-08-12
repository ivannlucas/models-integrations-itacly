"""Builds scaled windows (+ per-window stats) from a raw sensor CSV, mirroring the original
src/predict/xai_predictor.py::run_xai_prediction data-prep steps (validate -> prepare -> impute
-> window -> stats -> scale), using the vendored functions in _vendor/.
"""
from __future__ import annotations

import pandas as pd

from app.domain.services.exceptions import InsufficientWindowHistoryError
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.input_validation import (
    prepare_model_input_dataframe,
    temporal_impute_partial_nulls,
    validate_model_input_data,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.preprocess import (
    create_sequences,
    stats_windows,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
    ID_COLUMN,
    NORMAL_TOKENS,
    PARTIAL_NULL_MAX_RATIO,
    SENSOR_COLUMNS,
    SEQUENCE_LENGTH,
    SOLAPAMIENTO_BETA,
    STATS_CREATION,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)


def build_config_dict() -> dict:
    """Config dict shim matching the shape the vendored functions expect (config.yaml equivalent)."""
    return {
        "data_generation": {"sensors": [{"name": name} for name in SENSOR_COLUMNS]},
        "data_processing": {
            "target_column": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "timestamp_column": TIMESTAMP_COLUMN,
            "normal_tokens": NORMAL_TOKENS,
            "sequence_length": SEQUENCE_LENGTH,
            "solapamiento_beta": SOLAPAMIENTO_BETA,
            "partial_null_max_ratio": PARTIAL_NULL_MAX_RATIO,
            "fuzzy_processing": {"stats_creation": STATS_CREATION},
        },
    }


def build_windows_from_dataframe(
    df: pd.DataFrame,
    scaler_x,
    scaler_num,
    *,
    require_target: bool = False,
):
    """Validate *df* and build scaled (sequences, stats) windows.

    Returns (sequences_scaled [N,T,F], stats_scaled [N,S], stats_columns, timestamp_windows,
    entity_ids, y_seq). Raises InsufficientWindowHistoryError if no window of SEQUENCE_LENGTH
    consecutive rows can be built (e.g. cycle/file shorter than 240 rows).
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    config = build_config_dict()
    validation_report = validate_model_input_data(
        df=df, config=config, context="inferencia", require_target=require_target,
    )
    df = prepare_model_input_dataframe(df, config, context="inferencia")

    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = 0

    df = temporal_impute_partial_nulls(
        df=df,
        partial_null_stats=validation_report.get("partial_null_stats", {}),
        id_column=ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
    )

    sequences, y_seq, timestamp_windows, entity_ids = create_sequences(
        df,
        target_column=TARGET_COLUMN,
        seq_length=SEQUENCE_LENGTH,
        solapamiento_beta=SOLAPAMIENTO_BETA,
        id_column=ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        normal_tokens=NORMAL_TOKENS,
    )
    if len(sequences) == 0:
        raise InsufficientWindowHistoryError(
            f"No se generaron ventanas: se necesitan al menos {SEQUENCE_LENGTH} filas "
            "consecutivas (por ciclo, si hay columna cycle_id) para construir una ventana."
        )

    stats_df = stats_windows(sequences, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)

    n_features = sequences.shape[-1]
    sequences_scaled = scaler_x.transform(sequences.reshape(-1, n_features)).reshape(sequences.shape)
    stats_scaled = scaler_num.transform(stats_df)

    return sequences_scaled, stats_scaled, list(stats_df.columns), timestamp_windows, entity_ids, y_seq


def build_windows_from_csv(
    csv_path: str,
    scaler_x,
    scaler_num,
    *,
    require_target: bool = False,
):
    """Read *csv_path* and delegate to build_windows_from_dataframe()."""
    df = pd.read_csv(csv_path)
    return build_windows_from_dataframe(
        df, scaler_x, scaler_num, require_target=require_target,
    )


def build_windows_from_sensor_arrays(
    sensor_arrays: dict[str, list[float]],
    timestamps: list,
    cycle_id,
    scaler_x,
    scaler_num,
):
    """Build a single-cycle DataFrame from inline sensor arrays (predict_inline without data_path)."""
    n_rows = len(timestamps)
    data = {TIMESTAMP_COLUMN: timestamps}
    for sensor in SENSOR_COLUMNS:
        values = sensor_arrays.get(sensor)
        if values is None or len(values) != n_rows:
            raise ValueError(
                f"'{sensor}' debe ser una lista de {n_rows} valores (misma longitud que timestamp)."
            )
        data[sensor] = values
    if cycle_id is not None:
        data[ID_COLUMN] = [cycle_id] * n_rows

    df = pd.DataFrame(data)
    return build_windows_from_dataframe(df, scaler_x, scaler_num, require_target=False)
