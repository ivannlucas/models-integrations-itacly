"""Pipeline de predicción autónomo end-to-end."""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from src.data_processing.load_data import load_raw_data
from src.data_processing.preprocess import create_sequences, stats_windows
from src.predict.postprocess import decode_predictions
from src.training.model import ParallelDeepNeuroFuzzyModel
from src.utils.common import ensure_dir, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_input_data(df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Valida que los datos del cliente cumplan los requisitos del modelo.

    Comprueba timestamp obligatorio, columnas de sensores, filas minimas
    para la ventana LSTM, tipos numericos y valores nulos.

    Args:
        df: DataFrame cargado del CSV del cliente.
        config: Diccionario de configuracion completo.

    Returns:
        Reporte de validacion con columnas que tienen nulos parciales dentro
        del umbral permitido.

    Raises:
        ValueError: Si los datos no cumplen los requisitos.
    """
    data_cfg = config["data_processing"]
    gen_cfg = config.get("data_generation", {})
    partial_null_max_ratio = float(data_cfg.get("partial_null_max_ratio", 0.10))

    errors: list[str] = []

    df_cols = set(df.columns.astype(str).str.strip().str.lower())

    # 0. Columna temporal obligatoria
    timestamp_column = str(data_cfg.get("timestamp_column", "timestamp")).strip().lower()
    if not timestamp_column:
        errors.append(
            "La configuración data_processing.timestamp_column no puede estar vacía."
        )
    elif timestamp_column not in df_cols:
        errors.append(
            f"Columna temporal obligatoria ausente: '{timestamp_column}'."
            " Incluya esta columna para ordenar cronológicamente los registros."
        )

    # 1. Columnas de sensores requeridas
    expected_sensors = [s["name"].lower() for s in gen_cfg.get("sensors", [])]
    missing: list[str] = []
    if expected_sensors:
        missing = [s for s in expected_sensors if s not in df_cols]
        if missing:
            errors.append(
                "Columnas de sensores faltantes: "
                f"{missing}.\n"
                f"Columnas requeridas (13 sensores): {expected_sensors}.\n"
                f"Columnas encontradas: {sorted(df.columns.tolist())}."
            )

    # 2. Filas minimas para la ventana LSTM
    seq_length = data_cfg["sequence_length"]
    id_column = data_cfg.get("id_column", "")
    id_col_lower = id_column.lower() if id_column else None

    if id_col_lower and id_col_lower in df.columns:
        short_cycles: list[str] = []
        for cid, group in df.groupby(id_col_lower):
            if len(group) < seq_length:
                short_cycles.append(
                    f"ciclo={cid}: {len(group)} filas (< {seq_length} requeridas)"
                )
        if short_cycles:
            errors.append(
                "Ciclos con longitud insuficiente para formar ventanas temporales: "
                f"{short_cycles}.\n"
                f"Se requieren al menos {seq_length} filas consecutivas por ciclo "
                "para construir una ventana completa."
            )
    else:
        if len(df) < seq_length:
            errors.append(
                f"El archivo tiene {len(df)} filas, pero el modelo LSTM necesita al menos "
                f"{seq_length} registros temporales consecutivos para generar una ventana de analisis.\n"
                f"Si sus datos tienen multiples ciclos, incluya una columna "
                f"'{id_column or 'cycle_id'}' para identificarlos."
            )

    # 3. Columnas de sensores deben ser numericas
    non_numeric_cols: list[str] = []
    for sensor in expected_sensors:
        col = sensor.lower()
        if col in missing:
            continue
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(f"{col} ({df[col].dtype})")
    if non_numeric_cols:
        errors.append(
            "Columnas de sensores con tipo no numerico: "
            f"{non_numeric_cols}.\n"
            "Verifique que las lecturas de sensores sean valores numericos."
        )

    # 4. Columnas completamente nulas
    null_cols = [
        s for s in expected_sensors
        if s not in missing and s in df.columns and df[s].isnull().all()
    ]
    if null_cols:
        errors.append(
            "Columnas de sensores completamente vacias: "
            f"{null_cols}.\n"
            "Todas las columnas de sensores deben contener datos validos."
        )

    # 5. Nulos parciales por columna (permitidos hasta umbral)
    partial_null_cols_allowed: list[str] = []
    partial_null_cols_exceeded: list[str] = []
    partial_null_stats: dict[str, dict[str, float]] = {}

    for sensor in expected_sensors:
        if sensor in missing or sensor in null_cols or sensor not in df.columns:
            continue

        # Trata vacios de texto como nulos y fuerza conversion numerica para medir faltantes reales.
        sensor_series = pd.to_numeric(
            df[sensor].replace(r"^\s*$", np.nan, regex=True),
            errors="coerce",
        )

        null_ratio = float(sensor_series.isna().mean())
        null_count = int(sensor_series.isna().sum())

        if null_count <= 0:
            continue

        partial_null_stats[sensor] = {
            "null_count": float(null_count),
            "null_ratio": null_ratio,
        }

        if null_ratio > partial_null_max_ratio:
            partial_null_cols_exceeded.append(
                f"{sensor} ({null_count} nulos, {null_ratio:.2%})"
            )
        else:
            partial_null_cols_allowed.append(
                f"{sensor} ({null_count} nulos, {null_ratio:.2%})"
            )

    if partial_null_cols_exceeded:
        errors.append(
            "Columnas con nulos parciales por encima del umbral permitido "
            f"({partial_null_max_ratio:.0%}): {partial_null_cols_exceeded}."
        )

    if errors:
        error_lines = ["ERROR: Validacion de datos de entrada fallida."]

        for message in errors:
            message_lines = str(message).splitlines()
            if not message_lines:
                continue
            error_lines.append(f"- {message_lines[0]}")
            for continuation_line in message_lines[1:]:
                error_lines.append(f"\t{continuation_line}")

        if partial_null_cols_allowed:
            error_lines.append(
                "- Aviso adicional: se detectaron nulos parciales dentro del umbral "
                f"({partial_null_max_ratio:.0%}): {partial_null_cols_allowed}."
            )
            error_lines.append(
                "\tEn inferencia esos nulos se imputan con criterio temporal "
                "(interpolacion + arrastre). Para minimizar sesgos, conviene "
                "completar los datos faltantes en el dataset de origen "
                "(situado en data/input), y volver a ejecutar python scripts/predict.py."
            )
        raise ValueError("\n".join(error_lines))

    if partial_null_cols_allowed:
        logger.warning(
            "Se detectaron nulos parciales dentro del umbral permitido (%s): %s",
            f"{partial_null_max_ratio:.0%}",
            partial_null_cols_allowed,
        )
        logger.warning(
            "En inferencia esos nulos se imputan con criterio temporal "
            "(interpolacion + arrastre). Para minimizar sesgos, conviene completar "
            "los datos faltantes en el dataset de origen (data/input), y volver a "
            "ejecutar python scripts/predict.py."
        )

    logger.info(
        "Validacion de datos de entrada superada correctamente: "
        "variables presentes, tipo numerico, sin columnas completamente nulas, "
        "cantidad suficiente de datos."
    )
    logger.info("=" * 30)

    return {
        "partial_null_max_ratio": partial_null_max_ratio,
        "partial_null_stats": partial_null_stats,
    }


def _temporal_impute_partial_nulls(
    df: pd.DataFrame,
    partial_null_stats: dict[str, dict[str, float]],
    id_column: Optional[str],
    timestamp_column: str,
) -> pd.DataFrame:
    """Imputa nulos parciales con criterio temporal antes de inferencia."""
    if not partial_null_stats:
        return df

    columns_to_impute = [
        col for col, stats in partial_null_stats.items()
        if stats.get("null_count", 0.0) > 0 and col in df.columns
    ]
    if not columns_to_impute:
        return df

    df_work = df.copy()
    has_id = bool(id_column) and id_column in df_work.columns
    has_ts = timestamp_column in df_work.columns

    def _fill_series(series: pd.Series) -> pd.Series:
        # Interpolacion lineal temporal + arrastre para mantener continuidad.
        filled = series.interpolate(method="linear", limit_direction="both")
        return filled.ffill().bfill()

    if has_id and has_ts:
        df_work = df_work.sort_values([id_column, timestamp_column]).copy()
        for col in columns_to_impute:
            df_work[col] = (
                df_work.groupby(id_column, sort=False)[col]
                .transform(_fill_series)
            )
    elif has_ts:
        df_work = df_work.sort_values([timestamp_column]).copy()
        for col in columns_to_impute:
            df_work[col] = _fill_series(df_work[col])
    elif has_id:
        for col in columns_to_impute:
            df_work[col] = (
                df_work.groupby(id_column, sort=False)[col]
                .transform(_fill_series)
            )
    else:
        for col in columns_to_impute:
            df_work[col] = _fill_series(df_work[col])

    # Fallback robusto: cualquier remanente se rellena con mediana por columna.
    for col in columns_to_impute:
        remaining = int(df_work[col].isna().sum())
        if remaining > 0:
            median_val = df_work[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_work[col] = df_work[col].fillna(median_val)

    total_remaining = int(df_work[columns_to_impute].isna().sum().sum())
    if total_remaining > 0:
        raise ValueError(
            "ERROR: Persisten nulos tras la imputacion temporal en inferencia. "
            "Revise la calidad del CSV de entrada."
        )

    logger.info(
        "Imputacion temporal aplicada en inferencia para columnas con nulos parciales: %s",
        columns_to_impute,
    )
    return df_work


def load_model_artifacts(
    config: dict,
) -> tuple[ParallelDeepNeuroFuzzyModel, dict, object, float]:
    """Carga modelo entrenado y todos los artefactos necesarios para inferencia.

    Args:
        config: Diccionario de configuración.

    Returns:
        Tupla con (modelo, model_cfg, scalers, threshold).

    Raises:
        FileNotFoundError: Si algún artefacto no existe.
    """
    paths = config["paths"]
    artifacts_dir = paths["model_artifacts"]
    metrics_dir = paths["model_metrics"]
    checkpoint_dir = paths.get("model_checkpoint_dir", artifacts_dir)

    # Cargar checkpoint del modelo
    model_path = os.path.join(checkpoint_dir, paths["model_filename"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_cfg = checkpoint["model_cfg"]
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        model_cfg = None

    # Cargar scaler
    scaler_path = os.path.join(artifacts_dir, paths["scaler_filename"])
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado en {scaler_path}")

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    if not isinstance(scalers, dict):
        raise ValueError(
            "Artifact de scalers inválido: se esperaba un dict con 'scaler_x' y 'scaler_num'."
        )
    if "scaler_x" not in scalers or "scaler_num" not in scalers:
        raise ValueError(
            "Artifact de scalers incompleto:“faltan scalers compatibles en models/artifacts/scaler.pkl. "
            "Restaura artefactos del modelo o reentrena."
        )

    # Cargar threshold (prioridad: results.json > checkpoint/model_cfg > default)
    results_path = os.path.join(metrics_dir, paths["results_filename"])
    threshold = 0.5
    threshold_source = "default"

    # 1) Fallback desde checkpoint/model_cfg (siempre disponible tras entrenamiento correcto)
    if isinstance(checkpoint, dict):
        for key in ("best_threshold", "selected_threshold", "threshold", "decision_threshold"):
            if key in checkpoint:
                try:
                    threshold = float(checkpoint[key])
                    threshold_source = f"checkpoint.{key}"
                    break
                except Exception:
                    continue

    if model_cfg is not None and isinstance(model_cfg, dict):
        training_kwargs = model_cfg.get("training_kwargs", {})
        if isinstance(training_kwargs, dict) and "threshold" in training_kwargs:
            try:
                threshold = float(training_kwargs["threshold"])
                threshold_source = "checkpoint.model_cfg.training_kwargs.threshold"
            except Exception:
                pass

    # 2) Umbral oficial desde results.json (si existe) sobrescribe al fallback
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Posibles ubicaciones del umbral en el JSON de métricas
        if "best_threshold" in results:
            threshold = float(results["best_threshold"])
            threshold_source = "best_threshold"
        elif "selected_threshold" in results:
            threshold = float(results["selected_threshold"])
            threshold_source = "selected_threshold"
        elif "test_metrics" in results and isinstance(results["test_metrics"], dict) and "selected_threshold" in results["test_metrics"]:
            threshold = float(results["test_metrics"]["selected_threshold"])
            threshold_source = "test_metrics.selected_threshold"
        elif "training_kwargs" in results and isinstance(results["training_kwargs"], dict) and "threshold" in results["training_kwargs"]:
            threshold = float(results["training_kwargs"]["threshold"])
            threshold_source = "training_kwargs.threshold"
        else:
            # Fallback: intentar leer keys genéricas
            for key in ("threshold", "decision_threshold"):
                if key in results:
                    try:
                        threshold = float(results[key])
                        threshold_source = key
                        break
                    except Exception:
                        continue

    logger.info("Umbral cargado: %.3f (fuente=%s)", float(threshold), threshold_source)

    # Reconstruir modelo
    if model_cfg is not None:
        model = ParallelDeepNeuroFuzzyModel(model_cfg)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(
            "El checkpoint no contiene model_cfg. "
            "Asegúrate de guardar el modelo con save_model_artifacts()."
        )

    model.eval()
    logger.info("Modelo cargado desde %s", model_path)

    return model, model_cfg, scalers, threshold


def run_prediction(
    config_path: str = "config/config.yaml",
    input_path: Optional[str] = None,
) -> pd.DataFrame:
    """Pipeline completo de predicción: carga datos, preprocesa, infiere, guarda.

    Args:
        config_path: Ruta al archivo de configuración.
        input_path: Ruta al archivo de datos (sobreescribe config si se provee).

    Returns:
        DataFrame con predicciones.
    """
    config = load_config(config_path)
    data_cfg = config["data_processing"]
    paths = config["paths"]

    # Cargar artefactos
    model, model_cfg, scalers, threshold = load_model_artifacts(config)

    scaler_x = scalers["scaler_x"]
    scaler_num = scalers["scaler_num"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Cargar datos nuevos
    data_path = input_path or paths.get("input_data", paths["raw_data"])

    logger.info("=" * 30)

    # Si data_path es un directorio, buscar el primer CSV dentro
    if os.path.isdir(data_path):
        csv_files = sorted(
            [f for f in os.listdir(data_path) if f.endswith(".csv")]
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {data_path}"
            )
        data_path = os.path.join(data_path, csv_files[0])
        logger.info("Archivo detectado en directorio de entrada: %s", data_path)

    df = load_raw_data(data_path)
    # Refuerzo de normalizacion por si la fuente cambia en el futuro.
    df.columns = df.columns.astype(str).str.strip().str.lower()
    logger.info("=" * 30)

    # Validar datos de entrada
    validation_report = validate_input_data(df, config)

    # Reordenar columnas de sensores al orden esperado por el modelo/scaler.
    # Esto garantiza robustez incluso si el CSV del cliente tiene las columnas
    # en un orden distinto al definido en data_generation.sensors.
    expected_sensors = [s["name"].lower() for s in config.get("data_generation", {}).get("sensors", [])]
    if expected_sensors:
        non_sensor_cols = [c for c in df.columns if c not in expected_sensors]
        desired_order = expected_sensors + non_sensor_cols
        # Filtra solo columnas que realmente existen en el DataFrame.
        desired_order = [c for c in desired_order if c in df.columns]
        current_sensor_order = [c for c in df.columns if c in expected_sensors]
        if current_sensor_order != expected_sensors:
            logger.info(
                "Reordenando columnas de sensores: %s -> %s",
                current_sensor_order, expected_sensors,
            )
            df = df[desired_order]

    # Preparar datos para inferencia (sin target real)
    target_column = data_cfg["target_column"].lower()
    id_column = data_cfg.get("id_column")
    id_column = id_column.lower() if id_column else None
    timestamp_column = data_cfg.get("timestamp_column", "timestamp")
    stats_creation = data_cfg.get("fuzzy_processing", {}).get(
        "stats_creation", ["mean", "std", "slope", "max", "min"]
    )
    solapamiento_beta = float(config.get("inference", {}).get("solapamiento_beta"))

    # Si no hay columna target, crear una dummy para compatibilidad
    has_target = target_column in df.columns
    if not has_target:
        df[target_column] = 0

    # Imputar nulos parciales permitidos antes de crear secuencias y escalar.
    df = _temporal_impute_partial_nulls(
        df=df,
        partial_null_stats=validation_report.get("partial_null_stats", {}),
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    # Crear secuencias
    X_seq, _, timestamp_windows_labels = create_sequences(
        df,
        target_column=target_column,
        seq_length=data_cfg["sequence_length"],
        solapamiento_beta=solapamiento_beta,
        id_column=id_column,
        timestamp_column=timestamp_column,
        normal_tokens=data_cfg.get("normal_tokens"),
    )

    if len(X_seq) == 0:
        raise ValueError(
            "ERROR: No se generaron secuencias de inferencia con los datos de entrada.\n"
            f"Revise sequence_length={data_cfg['sequence_length']}, "
            f"solapamiento_beta={solapamiento_beta} y la estructura temporal del CSV."
        )

    # Calcular estadísticas por ventana
    exclude_cols = {target_column, timestamp_column}
    if id_column:
        exclude_cols.add(id_column)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_stats = stats_windows(
        X_seq,
        feature_names=feature_cols,
        stats_creation=stats_creation,
    )

    # Escalar
    n_features = X_seq.shape[-1]
    X_scaled = scaler_x.transform(
        X_seq.reshape(-1, n_features)
    ).reshape(X_seq.shape)

    stats_scaled = scaler_num.transform(df_stats)

    logger.info(
        "Ventanas creadas: %s",
        X_scaled.shape,
    )
    logger.info(
        "Ventanas totales: %d, con %d registros cada una y un solapamiento de %s (%d registros).",
        len(X_scaled),
        X_scaled.shape[1],
        solapamiento_beta,
        solapamiento_beta * X_scaled.shape[1],
    )
    logger.info("=" * 30)

    # Inferencia
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    S_tensor = torch.tensor(stats_scaled, dtype=torch.float32).to(device)

    model.eval()
    all_preds = []
    all_probs = []
    inference_cfg = config.get("inference", {})
    training_cfg = config.get("training", {})
    batch_size = int(inference_cfg.get("batch_size", training_cfg.get("batch_size", 256)))
    if batch_size <= 0:
        raise ValueError("inference.batch_size debe ser un entero > 0")

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            x_batch = X_tensor[i:i + batch_size]
            s_batch = S_tensor[i:i + batch_size]

            out = model(x_batch, s_batch)

            anomaly_score = out["anomaly_score"]

            probs = torch.sigmoid(anomaly_score.view(-1))
            preds = (probs >= threshold).long()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    pred_classes = np.concatenate(all_preds)
    anomaly_probs = np.concatenate(all_probs)

    # Postprocesar
    window_indices = np.arange(len(pred_classes)) + 1  # +1 para que el índice sea 1-based
    results = decode_predictions(
        pred_classes=pred_classes,
        anomaly_scores=anomaly_probs,
        threshold=threshold,
        timestamp_index=timestamp_windows_labels,
        window_index=window_indices,
    )

    # Guardar
    output_dir = ensure_dir(paths["predictions"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    results.to_csv(output_path, index=False)

    summary_compact = {
        "n_procesados": int(len(pred_classes)),
        "n_errores": 0,
        "Anomalias_Detectadas": int(np.sum(pred_classes)),
        "Umbral_Utilizado": round(float(threshold), 2),
    }
    logger.info("Predicción completada: %s", summary_compact)

    logger.info(
        "Predicciones guardadas en %s",
        output_path,
    )
    return results
