"""Pipeline de inferencia con XAI y monitorización PCC."""

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
from src.predict.postprocess import decode_predictions, _json_safe
from src.data_processing.input_validation import (
    expected_sensor_columns,
    prepare_model_input_dataframe,
    temporal_impute_partial_nulls,
    validate_model_input_data,
)
from src.training.model import ParallelDeepNeuroFuzzyModel
from src.utils.common import ensure_dir, load_config
from src.utils.logging import get_logger
from src.xai import DNFLExplainer

logger = get_logger(__name__)


def load_model_artifacts(
    config: dict,
) -> tuple[ParallelDeepNeuroFuzzyModel, dict, dict, float]:
    """Carga modelo, escaladores y umbral de inferencia."""
    paths = config["paths"]
    artifacts_dir = paths["model_artifacts"]
    metrics_dir = paths["model_metrics"]
    checkpoint_dir = paths.get(
        "model_checkpoint_dir",
        artifacts_dir,
    )

    model_path = os.path.join(
        checkpoint_dir,
        paths["model_filename"],
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
    ):
        raise ValueError(
            "El checkpoint debe contener model_state_dict y model_cfg."
        )

    model_cfg = checkpoint.get("model_cfg")
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "El checkpoint no contiene un model_cfg válido."
        )

    scaler_path = os.path.join(
        artifacts_dir,
        paths["scaler_filename"],
    )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler no encontrado en {scaler_path}"
        )

    with open(scaler_path, "rb") as file:
        scalers = pickle.load(file)
    if (
        not isinstance(scalers, dict)
        or "scaler_x" not in scalers
        or "scaler_num" not in scalers
    ):
        raise ValueError(
            "scaler.pkl debe contener scaler_x y scaler_num."
        )

    expected_sensors = expected_sensor_columns(config)
    model_input_features = int(
        model_cfg.get("input_features", len(expected_sensors))
    )
    if model_input_features != len(expected_sensors):
        raise ValueError(
            "El checkpoint no coincide con data_generation.sensors: "
            f"{model_input_features} frente a "
            f"{len(expected_sensors)} variables."
        )

    scaler_input_features = getattr(
        scalers["scaler_x"],
        "n_features_in_",
        None,
    )
    if (
        scaler_input_features is not None
        and int(scaler_input_features) != len(expected_sensors)
    ):
        raise ValueError(
            "scaler_x no es compatible con las variables configuradas: "
            f"{scaler_input_features} frente a "
            f"{len(expected_sensors)}."
        )

    model_stats_features = int(
        model_cfg.get("n_stats_features", 0)
    )
    scaler_stats_features = getattr(
        scalers["scaler_num"],
        "n_features_in_",
        None,
    )
    if (
        scaler_stats_features is not None
        and model_stats_features > 0
        and int(scaler_stats_features) != model_stats_features
    ):
        raise ValueError(
            "scaler_num no es compatible con el checkpoint: "
            f"{scaler_stats_features} frente a "
            f"{model_stats_features} estadísticas."
        )

    threshold = 0.5
    threshold_source = "default"
    for key in (
        "best_threshold",
        "selected_threshold",
        "threshold",
        "decision_threshold",
    ):
        if key not in checkpoint:
            continue
        try:
            threshold = float(checkpoint[key])
            threshold_source = f"checkpoint.{key}"
            break
        except (TypeError, ValueError):
            continue

    training_kwargs = model_cfg.get("training_kwargs", {})
    if (
        isinstance(training_kwargs, dict)
        and "threshold" in training_kwargs
    ):
        try:
            threshold = float(training_kwargs["threshold"])
            threshold_source = (
                "checkpoint.model_cfg.training_kwargs.threshold"
            )
        except (TypeError, ValueError):
            pass

    results_path = os.path.join(
        metrics_dir,
        paths["results_filename"],
    )
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as file:
            results = json.load(file)

        test_metrics = results.get("test_metrics", {})
        threshold_candidates = [
            ("best_threshold", results.get("best_threshold")),
            (
                "selected_threshold",
                results.get("selected_threshold"),
            ),
            (
                "test_metrics.selected_threshold",
                (
                    test_metrics.get("selected_threshold")
                    if isinstance(test_metrics, dict)
                    else None
                ),
            ),
            (
                "training_kwargs.threshold",
                (
                    results.get("training_kwargs", {}).get("threshold")
                    if isinstance(results.get("training_kwargs"), dict)
                    else None
                ),
            ),
            ("threshold", results.get("threshold")),
            (
                "decision_threshold",
                results.get("decision_threshold"),
            ),
        ]
        for source, value in threshold_candidates:
            if value is None:
                continue
            try:
                threshold = float(value)
                threshold_source = source
                break
            except (TypeError, ValueError):
                continue

    model = ParallelDeepNeuroFuzzyModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Umbral cargado: %.3f (fuente=%s)",
        threshold,
        threshold_source,
    )
    logger.info("Modelo cargado desde %s", model_path)
    return model, model_cfg, scalers, threshold


def _resolve_xai_runtime_config(config: dict) -> dict:
    """Resuelve los parámetros XAI de ejecución."""
    xai_cfg = config.get("xai", {}) or {}
    return {
        "n_background": int(xai_cfg.get("n_background", 64)),
        "top_rules": int(xai_cfg.get("top_rules", 5)),
        "top_variables": int(xai_cfg.get("top_variables", 8)),
        "background_cfg": xai_cfg.get("background", {}) or {},
        "pcc_cfg": xai_cfg.get("pcc", {}) or {},
    }


def _resolve_background_windows(
    scaler_x,
    data_cfg: dict,
    config: dict,
) -> np.ndarray:
    """Resuelve el background SHAP: NPY o CSV."""
    paths_cfg = config.get("paths", {})
    artifacts_dir = str(
        paths_cfg.get("model_artifacts", "models/artifacts")
    )
    raw_data_path = str(
        paths_cfg.get("raw_data", "data/raw/")
    ).rstrip("/\\")
    if (
        os.path.isdir(raw_data_path)
        or not os.path.splitext(raw_data_path)[1]
    ):
        raw_dir = raw_data_path or "data/raw/"
    else:
        raw_dir = os.path.dirname(raw_data_path) or "data/raw/"
    raw_dir = ensure_dir(raw_dir)

    npy_path = os.path.join(
        artifacts_dir,
        "xai_background.npy",
    )
    csv_path = os.path.join(raw_dir, "xai_background.csv")

    if os.path.exists(csv_path):
        logger.info(
            "Procesando background SHAP...",
        )
        background_df = pd.read_csv(csv_path)
        background_df = prepare_model_input_dataframe(
            background_df,
            config,
            context="background SHAP",
        )

        target_column = str(
            data_cfg["target_column"]
        ).strip().lower()
        id_column_cfg = data_cfg.get("id_column")
        id_column = (
            str(id_column_cfg).strip().lower()
            if id_column_cfg
            else None
        )
        solapamiento_beta = float(data_cfg.get("solapamiento_beta", 0.5))
        timestamp_column = str(
            data_cfg.get("timestamp_column", "timestamp")
        ).strip().lower()

        if target_column not in background_df.columns:
            background_df[target_column] = 0

        background_sequences, _, _, _ = create_sequences(
            background_df,
            target_column=target_column,
            seq_length=int(data_cfg["sequence_length"]),
            solapamiento_beta=solapamiento_beta,
            id_column=id_column,
            timestamp_column=timestamp_column,
            normal_tokens=data_cfg.get("normal_tokens"),
        )
        if len(background_sequences) == 0:
            raise ValueError(
                "No se pudieron generar secuencias desde "
                f"{csv_path}."
            )

        n_features = background_sequences.shape[-1]
        background_scaled = scaler_x.transform(
            background_sequences.reshape(-1, n_features)
        ).reshape(background_sequences.shape)
        ensure_dir(artifacts_dir)
        np.save(npy_path, background_scaled)
        logger.info(
            "Background SHAP en %s (shape=%s)",
            npy_path,
            background_scaled.shape,
        )
        return background_scaled

    raise FileNotFoundError(
        "No se encontró un background válido para SHAP. "
        f"Se requiere {csv_path}."
    )


def run_xai_prediction(
    config_path: str = "config/config.yaml",
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Ejecuta predicción, explicabilidad XAI y monitorización PCC."""
    config = load_config(config_path)
    paths = config["paths"]
    data_cfg = config["data_processing"]
    xai_runtime_cfg = _resolve_xai_runtime_config(config)

    output_dir = ensure_dir(
        output_dir if output_dir is not None else paths["predictions"]
    )

    model, _, scalers, threshold = load_model_artifacts(config)
    scaler_x = scalers["scaler_x"]
    scaler_num = scalers["scaler_num"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    data_path = input_path or paths.get(
        "input_data",
        paths["raw_data"],
    )
    if os.path.isdir(data_path):
        csv_files = sorted(
            filename
            for filename in os.listdir(data_path)
            if filename.lower().endswith(".csv")
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {data_path}"
            )
        data_path = os.path.join(data_path, csv_files[0])
        logger.info(
            "Archivo detectado en directorio de entrada: %s",
            data_path,
        )

    input_df = load_raw_data(data_path)
    validation_report = validate_model_input_data(
        df=input_df,
        config=config,
        context="inferencia",
        require_target=False,
    )
    input_df = prepare_model_input_dataframe(
        input_df,
        config,
        context="inferencia",
    )

    target_column = str(
        data_cfg["target_column"]
    ).strip().lower()
    id_column_cfg = data_cfg.get("id_column")
    id_column = (
        str(id_column_cfg).strip().lower()
        if id_column_cfg
        else None
    )
    timestamp_column = str(
        data_cfg.get("timestamp_column", "timestamp")
    ).strip().lower()
    stats_creation = (
        data_cfg.get("fuzzy_processing", {})
        .get("stats_creation")
    )
    solapamiento_beta = float(data_cfg.get("solapamiento_beta", 0.5))

    if not stats_creation:
        raise ValueError(
            "data_processing.fuzzy_processing.stats_creation "
            "no está definido."
        )

    if target_column not in input_df.columns:
        input_df[target_column] = 0

    input_df = temporal_impute_partial_nulls(
        df=input_df,
        partial_null_stats=validation_report.get(
            "partial_null_stats",
            {},
        ),
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    sequences, _, timestamp_windows_labels, entity_ids = create_sequences(
        input_df,
        target_column=target_column,
        seq_length=int(data_cfg["sequence_length"]),
        solapamiento_beta=solapamiento_beta,
        id_column=id_column,
        timestamp_column=timestamp_column,
        normal_tokens=data_cfg.get("normal_tokens"),
    )
    if len(sequences) == 0:
        raise ValueError(
            "No se generaron secuencias para inferencia. Registros insuficientes, se necesitan al menos %d registros."
            % (
                int(data_cfg["sequence_length"])
            )
        )

    feature_columns = expected_sensor_columns(config)
    stats_df = stats_windows(
        sequences,
        feature_names=feature_columns,
        stats_creation=stats_creation,
    )

    n_features = sequences.shape[-1]
    sequences_scaled = scaler_x.transform(
        sequences.reshape(-1, n_features)
    ).reshape(sequences.shape)
    stats_scaled = scaler_num.transform(stats_df)

    logger.info(
        "Ventanas creadas: %s",
        sequences_scaled.shape,
    )
    logger.info(
        "Ventanas totales: %d, con %d registros cada una y un solapamiento de %s (%d registros).",
        len(sequences_scaled),
        sequences_scaled.shape[1],
        solapamiento_beta,
        solapamiento_beta * sequences_scaled.shape[1],
    )
    logger.info("=" * 30)

    background_windows = _resolve_background_windows(
        scaler_x=scaler_x,
        data_cfg=data_cfg,
        config=config,
    )

    explainer = DNFLExplainer(
        model=model,
        feature_names_stats=stats_df.columns.tolist(),
        feature_names_original=feature_columns,
        pcc_cfg=xai_runtime_cfg["pcc_cfg"],
        stats_creation=stats_creation,
        model_cfg=config.get("model", {}),
    )

    logger.info(
        "Procesando %d ventanas para XAI...",
        len(sequences_scaled),
    )
    logger.info("=" * 30)

    reports: list[dict[str, Any]] = []
    predicted_classes: list[int] = []
    anomaly_probabilities: list[float] = []
    processed_indices: list[int] = []
    processed_timestamps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    processed_entity_ids: list[Any] = []

    for sample_index in range(len(sequences_scaled)):
        try:
            timestamp_window = timestamp_windows_labels[sample_index] if timestamp_windows_labels else None
            entity_id = entity_ids[sample_index] if entity_ids else None
            xai_result = explainer.explain(
                x_window=sequences_scaled[sample_index],
                s_stats=stats_scaled[sample_index],
                background_windows=background_windows,
                anomaly_threshold=float(threshold),
                n_background=xai_runtime_cfg["n_background"],
                random_state=int(
                    config.get("project", {}).get("seed", 42)
                ),
                top_rules=xai_runtime_cfg["top_rules"],
                top_variables=xai_runtime_cfg["top_variables"],
            )

            anomaly_probability = float(
                xai_result["prediction"]["anomaly_probability"]
            )
            if (
                not np.isfinite(anomaly_probability)
                or not 0.0 <= anomaly_probability <= 1.0
            ):
                raise ValueError(
                    "Probabilidad de anomalía inválida: "
                    f"{anomaly_probability}"
                )

            predicted_class = int(
                anomaly_probability >= float(threshold)
            )
            predicted_classes.append(predicted_class)
            anomaly_probabilities.append(anomaly_probability)
            processed_indices.append(int(sample_index + 1))
            if timestamp_window is not None:
                processed_timestamps.append(timestamp_window)

                if entity_id is not None:
                    processed_entity_ids.append(entity_id)
                    report = {
                    "Índice de ventana": int(sample_index + 1),
                    "ID de ciclo": entity_id,
                    "Timestamp cubierto por ventana": f"{timestamp_window[0]} / {timestamp_window[1]}",
                }
                else:
                    report = {
                        "Índice de ventana": int(sample_index + 1),
                        "Timestamp cubierto por ventana": f"{timestamp_window[0]} / {timestamp_window[1]}",
                    }
                
            else:
                raise ValueError("Debe proporcionar los timestamps de las ventanas.")
            
            report.update(dict(xai_result["final_report"]))
            reports.append(report)
        except Exception as exc:
            logger.error(
                "Error en muestra %d: %s",
                sample_index,
                str(exc),
            )
            reports.append(
                {
                    "Índice de ventana": int(sample_index + 1),
                    "error": str(exc),
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(
        output_dir,
        f"PCC_monitor_{timestamp}.json",
    )
    output_csv = os.path.join(
        output_dir,
        f"predictions_{timestamp}.csv",
    )

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(
            _json_safe(reports),
            file,
            indent=2,
            ensure_ascii=False,
        )

    if predicted_classes:
        predictions_df = decode_predictions(
            pred_classes=np.asarray(predicted_classes, dtype=int),
            anomaly_scores=np.asarray(anomaly_probabilities, dtype=float),
            threshold=float(threshold),
            timestamp_index=processed_timestamps if timestamp_windows_labels else None,
            window_index=processed_indices,
            cycle_id=processed_entity_ids if processed_entity_ids else None
        )

        predictions_df.to_csv(output_csv, index=False)
    else:
        raise ValueError("No se generaron predicciones válidas.")
    

    logger.info(
        "Resultados guardados en:\n- %s\n- %s",
        output_json,
        output_csv,
    )

    summary = {
        "n_processed": len(processed_indices),
        "n_errors": len(reports) - len(processed_indices),
        "anomalies_detected": int(sum(predicted_classes)),
        "threshold_used": float(threshold),
    }
    return {
        "results": reports,
        "summary": summary,
        "config": {
            "n_background": xai_runtime_cfg["n_background"],
            "top_rules": xai_runtime_cfg["top_rules"],
            "top_variables": xai_runtime_cfg["top_variables"],
        },
    }
