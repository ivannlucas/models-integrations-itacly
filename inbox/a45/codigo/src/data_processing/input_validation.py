"""Validacion comun de datos tabulares para entrenamiento e inferencia."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def expected_sensor_columns(config: dict) -> list[str]:
    sensors = config.get("data_generation", {}).get("sensors", [])
    expected = [
        str(sensor["name"]).strip().lower()
        for sensor in sensors
        if sensor.get("name")
    ]
    if not expected:
        raise ValueError(
            "No se han definido variables de entrada en data_generation.sensors."
        )
    if len(expected) != len(set(expected)):
        raise ValueError(
            "data_generation.sensors contiene nombres de variables duplicados."
        )
    return expected


def drop_fully_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.Series]:
    csv_lines = pd.Series(np.arange(2, len(df) + 2, dtype=int), index=df.index)
    empty_mask = (
        df.replace(r"^\s*$", np.nan, regex=True)
        .isna()
        .all(axis=1)
    )
    empty_count = int(empty_mask.sum())
    if empty_count == 0:
        return df.copy().reset_index(drop=True), 0, csv_lines.reset_index(drop=True)
    
    return (
        df.loc[~empty_mask].copy().reset_index(drop=True), 
        empty_count,
        csv_lines[~empty_mask].reset_index(drop=True)
    )


def parse_timestamp_column(
    series: pd.Series,
) -> tuple[pd.Series, str]:
    """Parsea timestamp como datetime u ordinal, sin permitir mezcla."""

    clean_series = series.astype("object").replace(
        r"^\s*$",
        np.nan,
        regex=True,
    )
    non_null = clean_series.notna()

    numeric_values = pd.to_numeric(clean_series, errors="coerce")
    numeric_mask = numeric_values.notna()

    if bool(non_null.any()) and bool((numeric_mask == non_null).all()):
        return numeric_values, "ordinal numerico"

    if bool((numeric_mask & non_null).any()):
        raise ValueError(
            "La columna timestamp mezcla formato ordinal numerico y formato "
            "datetime. Use un unico formato en todos los registros."
        )

    return pd.to_datetime(clean_series, errors="coerce", format="mixed"), "datetime"

def _duplicate_counts(
    df: pd.DataFrame,
    columns: list[str],
    max_examples: int = 20,
) -> list[dict[str, Any]]:
    counts = (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="count")
    )
    counts = counts[counts["count"] > 1].head(max_examples)

    return [
        {
            **{column: str(row[column]) for column in columns},
            "count": int(row["count"]),
        }
        for _, row in counts.iterrows()
    ]

def _format_examples(
    examples: list[dict[str, Any]],
) -> str:
    if not examples:
        return "ninguno"

    lines = []
    for example in examples:
        fields = [
            f"{key}={value}"
            for key, value in example.items()
            if key != "count"
        ]
        lines.append(
            f"\n\t- En {', '.join(fields)} -> {example['count']} apariciones"
        )

    return "".join(lines)

def validate_model_input_data(
    df: pd.DataFrame,
    config: dict,
    context: str,
    require_target: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data_processing"]
    df_work = df.copy()
    df_work.columns = df_work.columns.astype(str).str.strip().str.lower()

    df_work, empty_count, csv_lines = drop_fully_empty_rows(df_work)
    if empty_count > 0:
        logger.warning(
            "WARNING: Se ignoraron %d filas completamente vacias en %s.",
            empty_count,
            context,
        )

    expected_sensors = expected_sensor_columns(config)
    df_columns = set(df_work.columns)
    errors: list[str] = []

    duplicated = df_work.columns[df_work.columns.duplicated()].tolist()
    if duplicated:
        errors.append(
            f"Columnas duplicadas tras normalizar sus nombres: {duplicated}."
        )

    target_column = str(data_cfg["target_column"]).strip().lower()
    if require_target and target_column not in df_columns:
        errors.append(
            f"Columna objetivo ausente en {context}: '{target_column}'."
        )

    timestamp_column = str(
        data_cfg.get("timestamp_column", "timestamp")
    ).strip().lower()
    id_column_cfg = data_cfg.get("id_column")
    id_column = (
        str(id_column_cfg).strip().lower()
        if id_column_cfg
        else None
    )
    seq_length = int(data_cfg["sequence_length"])
    has_id_column = bool(id_column) and id_column in df_work.columns
    id_column_display = id_column or "cycle_id"
    
    if not id_column:
        logger.warning(
            "WARNING: No se definio data_processing.id_column. "
            "Se asumira una serie temporal global en %s.",
            context,
        )
    elif id_column not in df_columns:
        logger.warning(
            "WARNING: Columna de identificador de ciclo ausente en %s: '%s'."
            " Se asumira una serie temporal global.",
            context,
            id_column,
        )
    else:
        logger.info(
            "Columna de identificador de ciclo '%s' detectada en %s.",
            id_column,
            context,
        )

    if empty_count > 0 and not has_id_column:
        logger.warning(
            "WARNING: Se detectaron %d filas completamente vacias cargadas "
            "en %s, pero no se recibio una columna de identificador de ciclo "
            "('%s'). Estas filas no se usan como separadores de ciclo: los "
            "registros validos se ordenaran por '%s' y se trataran como una "
            "unica serie temporal global.",
            empty_count,
            context,
            id_column_display,
            timestamp_column,
        )

    if not timestamp_column:
        errors.append(
            "data_processing.timestamp_column no puede estar vacia."
        )
    elif timestamp_column not in df_columns:
        errors.append(
            f"Columna temporal obligatoria ausente en {context}: "
            f"'{timestamp_column}'."
        )
    else:
        try:
            parsed_timestamps, detected_timestamp_format = (
                parse_timestamp_column(
                    df_work[timestamp_column],
                )
            )
            logger.info(
                "Columna temporal '%s' detectada como '%s' en %s.",
                timestamp_column,
                detected_timestamp_format,
                context,
            )

        except ValueError as exc:
            errors.append(str(exc))
        else:
            invalid_count = int(parsed_timestamps.isna().sum())

            if invalid_count > 0:
                examples = (
                    df_work.loc[parsed_timestamps.isna(), timestamp_column]
                    .head(20)
                    .astype(str)
                    .tolist()
                )
                examples_lines = (
                    csv_lines.loc[parsed_timestamps.isna()]
                    .head(20)
                    .astype(int)
                    .tolist()
                )
                examples_text = "".join(
                    f"\n\t- {example} -> Linea CSV: {line}"
                    for example, line in zip(examples, examples_lines)
                )
                errors.append(
                    f"La columna temporal '{timestamp_column}' contiene "
                    f"{invalid_count} valores vacios o no interpretables en "
                    f"{context}. Use datetime valido u ordinal numerico. "
                    f"Ejemplos problematicos: {examples_text}."
                )
            else:
                df_temporal = df_work.copy()
                df_temporal[timestamp_column] = parsed_timestamps

                if id_column and id_column in df_temporal.columns:
                    duplicate_pair_examples = _duplicate_counts(
                        df_temporal,
                        [id_column, timestamp_column],
                    )
                    if duplicate_pair_examples:
                        duplicate_rows = df_temporal[
                            df_temporal.duplicated(
                                subset=[id_column, timestamp_column],
                                keep=False,
                            )
                        ]
                        duplicate_timestamp_counts = _duplicate_counts(
                            duplicate_rows,
                            [timestamp_column],
                        )
                        errors.append(
                            "Se detectaron timestamps duplicados para el mismo identificador de ciclo "
                            f"en {context}.\n"
                            "Pares duplicados:"
                            f"{_format_examples(duplicate_pair_examples)}"
                        )

                else:
                    duplicate_timestamp_counts = _duplicate_counts(
                        df_temporal,
                        [timestamp_column],
                    )
                    if duplicate_timestamp_counts:
                        errors.append(
                            "Se detectaron timestamps duplicados en "
                            f"{context}.\n"
                            "Timestamps duplicados:"
                            f"{_format_examples(duplicate_timestamp_counts)}\n"
                            "Sin un identificador de ciclo, cada timestamp debe identificar "
                            "una posicion temporal inequivoca."
                        )
    
    if id_column and id_column in df_work.columns:
        short_cycles = [
            f"ciclo={cycle_id}: {len(group)} filas "
            f"(< {seq_length} requeridas)"
            for cycle_id, group in df_work.groupby(id_column)
            if len(group) < seq_length
        ]
        if short_cycles:
            errors.append(
                "Ciclos con longitud insuficiente para formar ventanas "
                f"temporales: {short_cycles}.\n"
                f"Se requieren al menos {seq_length} filas por ciclo."
            )
    elif len(df_work) < seq_length:
        errors.append(
            f"El archivo tiene {len(df_work)} filas, pero se necesitan al menos "
            f"{seq_length} registros para generar una ventana.\n"
            f"Si los datos contienen varios ciclos, incluya la columna "
            f"'{id_column or 'cycle_id'}'."
        )

    missing = [
        sensor for sensor in expected_sensors if sensor not in df_columns
    ]
    if missing:
        errors.append(
            "Columnas de sensores faltantes: "
            f"{missing}. Columnas requeridas: {expected_sensors}."
        )

    non_numeric = [
        f"{sensor} ({df_work[sensor].dtype})"
        for sensor in expected_sensors
        if sensor not in missing
        and sensor in df_work.columns
        and not pd.api.types.is_numeric_dtype(df_work[sensor])
    ]
    if non_numeric:
        errors.append(
            f"Columnas de sensores con tipo no numerico: {non_numeric}."
        )

    null_columns = [
        sensor
        for sensor in expected_sensors
        if sensor not in missing
        and sensor in df_work.columns
        and df_work[sensor].isnull().all()
    ]
    if null_columns:
        errors.append(
            f"Columnas de sensores completamente vacias: {null_columns}."
        )

    partial_null_max_ratio = float(
        data_cfg.get("partial_null_max_ratio", 0.10)
    )
    partial_null_stats: dict[str, dict[str, float]] = {}
    partial_null_allowed: list[str] = []
    partial_null_exceeded: list[str] = []

    for sensor in expected_sensors:
        if sensor in missing or sensor in null_columns or sensor not in df_work.columns:
            continue

        sensor_series = pd.to_numeric(
            df_work[sensor].replace(r"^\s*$", np.nan, regex=True),
            errors="coerce",
        )
        null_count = int(sensor_series.isna().sum())
        if null_count == 0:
            continue

        null_ratio = float(sensor_series.isna().mean())
        partial_null_stats[sensor] = {
            "null_count": float(null_count),
            "null_ratio": null_ratio,
        }
        description = f"{sensor} ({null_count} nulos, {null_ratio:.2%})"

        if null_ratio > partial_null_max_ratio:
            partial_null_exceeded.append(description)
        else:
            partial_null_allowed.append(description)

    if partial_null_exceeded:
        errors.append(
            "Columnas con nulos parciales por encima del umbral permitido "
            f"({partial_null_max_ratio:.0%}): {partial_null_exceeded}."
        )

    if errors:
        raise ValueError(
            "Validacion de datos fallida en "
            f"{context}.\n- " + "\n- ".join(errors)
        )

    if partial_null_allowed:
        logger.warning(
            "WARNING: Nulos parciales dentro del umbral permitido en %s (%s): %s",
            context,
            f"{partial_null_max_ratio:.0%}",
            partial_null_allowed,
        )

        logger.warning(
            "Estos nulos se imputan con criterio temporal (interpolacion + arrastre).\n"
            " Para minimizar sesgos, conviene completar los datos faltantes\n"
            " en el dataset de origen (data/input), y volver a ejecutar el script."
        )
        

    logger.info(
        "Validacion completa superada en %s: %d filas validas, "
        "%d sensores requeridos presentes, timestamp obligatorio '%s' valido.",
        context,
        len(df_work),
        len(expected_sensors),
        timestamp_column,
    )

    return {
        "partial_null_max_ratio": partial_null_max_ratio,
        "partial_null_stats": partial_null_stats,
    }


def prepare_model_input_dataframe(
    df: pd.DataFrame,
    config: dict,
    context: str,
) -> pd.DataFrame:
    data_cfg = config["data_processing"]
    df_work = df.copy()
    df_work.columns = df_work.columns.astype(str).str.strip().str.lower()
    df_work, _, _ = drop_fully_empty_rows(df_work)

    expected_sensors = expected_sensor_columns(config)
    timestamp_column = str(
        data_cfg.get("timestamp_column", "timestamp")
    ).strip().lower()
    id_column_cfg = data_cfg.get("id_column")
    id_column = (
        str(id_column_cfg).strip().lower()
        if id_column_cfg
        else None
    )

    parsed_timestamps, _ = parse_timestamp_column(
        df_work[timestamp_column]
    )
    df_work[timestamp_column] = parsed_timestamps

    allowed_metadata = {
        timestamp_column,
        "fault_name",
        "fault_label",
        "grain_type",
        "phase",
    }

    if id_column:
        allowed_metadata.add(id_column)

    unexpected = [
        column
        for column in df_work.columns
        if column not in expected_sensors
        and column not in allowed_metadata
    ]
    if unexpected:
        logger.warning(
            "WARNING: Se omitiran columnas no utilizadas por el modelo en %s: %s",
            context,
            unexpected,
        )

    metadata_columns = [
        column
        for column in df_work.columns
        if column in allowed_metadata
        and column not in expected_sensors
    ]

    sort_columns = (
        [id_column, timestamp_column]
        if id_column and id_column in df_work.columns
        else [timestamp_column]
    )

    return (
        df_work[expected_sensors + metadata_columns]
        .sort_values(by=sort_columns)
        .reset_index(drop=True)
        .copy()
    )


def temporal_impute_partial_nulls(
    df: pd.DataFrame,
    partial_null_stats: dict[str, dict[str, float]],
    id_column: Optional[str],
    timestamp_column: str,
) -> pd.DataFrame:
    columns_to_impute = [
        column
        for column, stats in partial_null_stats.items()
        if stats.get("null_count", 0.0) > 0 and column in df.columns
    ]
    if not columns_to_impute:
        return df

    df_work = df.copy()
    has_id = bool(id_column) and id_column in df_work.columns
    has_timestamp = timestamp_column in df_work.columns

    def _fill_series(series: pd.Series) -> pd.Series:
        filled = series.interpolate(method="linear", limit_direction="both")
        return filled.ffill().bfill()

    if has_id and has_timestamp:
        df_work = df_work.sort_values([id_column, timestamp_column]).copy()
        for column in columns_to_impute:
            df_work[column] = (
                df_work.groupby(id_column, sort=False)[column]
                .transform(_fill_series)
            )
    elif has_timestamp:
        df_work = df_work.sort_values(timestamp_column).copy()
        for column in columns_to_impute:
            df_work[column] = _fill_series(df_work[column])
    else:
        for column in columns_to_impute:
            df_work[column] = _fill_series(df_work[column])

    for column in columns_to_impute:
        if df_work[column].isna().any():
            median_value = df_work[column].median()
            if pd.isna(median_value):
                median_value = 0.0
            df_work[column] = df_work[column].fillna(median_value)

    if int(df_work[columns_to_impute].isna().sum().sum()) > 0:
        raise ValueError(
            "Persisten nulos tras la imputacion temporal."
        )

    logger.info(
        "Imputacion temporal aplicada en %s columnas: %s",
        len(columns_to_impute),
        columns_to_impute,
    )
    return df_work