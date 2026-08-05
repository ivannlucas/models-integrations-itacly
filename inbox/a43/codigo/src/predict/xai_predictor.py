"""Pipeline de predicción con XAI integrado y consistente con inferencia estándar."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

from src.data_processing.load_data import load_raw_data
from src.data_processing.preprocess import create_sequences, stats_windows
from src.predict.postprocess import decode_predictions
from src.predict.predictor import (
    _temporal_impute_partial_nulls,
    load_model_artifacts,
    validate_input_data,
)
from src.xai import DNFLExplainer
from src.utils.common import ensure_dir, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_xai_runtime_config(
    config: dict,
    n_background: Optional[int],
    top_rules: Optional[int],
    top_variables: Optional[int],
    top_blocks: Optional[int],
) -> dict:
    """Resuelve parámetros XAI combinando argumentos y config.yaml."""
    xai_cfg = config.get("xai", {}) or {}
    return {
        "n_background": int(
            n_background if n_background is not None else xai_cfg.get("n_background", 64)
        ),
        "top_rules": int(
            top_rules if top_rules is not None else xai_cfg.get("top_rules", 5)
        ),
        "top_variables": int(
            top_variables if top_variables is not None else xai_cfg.get("top_variables", 8)
        ),
        "top_blocks": int(
            top_blocks if top_blocks is not None else xai_cfg.get("top_blocks", 3)
        ),
        "background_cfg": xai_cfg.get("background", {}) or {},
        "action_cfg": xai_cfg.get("action_config"),
        "corrective_actions_cfg": xai_cfg.get("corrective_actions", {}) or {},
    }


def _prepare_inference_dataframe(
    df: pd.DataFrame,
    config: dict,
    data_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Aplica la misma normalizacion robusta que la inferencia estandar."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    validation_report = validate_input_data(df, config)

    expected_sensors = [
        s["name"].lower()
        for s in config.get("data_generation", {}).get("sensors", [])
    ]
    if expected_sensors:
        non_sensor_cols = [c for c in df.columns if c not in expected_sensors]
        desired_order = [
            c for c in expected_sensors + non_sensor_cols
            if c in df.columns
        ]
        current_sensor_order = [c for c in df.columns if c in expected_sensors]
        if current_sensor_order != expected_sensors:
            logger.info(
                "Reordenando columnas de sensores: %s -> %s",
                current_sensor_order,
                expected_sensors,
            )
            df = df[desired_order]

    target_column = data_cfg["target_column"].lower()
    id_column = data_cfg.get("id_column")
    id_column = id_column.lower() if id_column else None
    timestamp_column = data_cfg.get("timestamp_column", "timestamp")

    if target_column not in df.columns:
        df[target_column] = 0

    df = _temporal_impute_partial_nulls(
        df=df,
        partial_null_stats=validation_report.get("partial_null_stats", {}),
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    return df, validation_report


def _resolve_background_windows(
    scaler_x,
    data_cfg: dict,
    config: dict,
) -> np.ndarray:
    """Resuelve el background SHAP: CSV de entrada > error si no se encuentra ninguno válido."""

    paths_cfg = config.get("paths", {})
    artifacts_dir = Path(paths_cfg.get("model_artifacts", "models/artifacts"))
    raw_data_path = Path(paths_cfg.get("raw_data", "data/raw/oven_full_dataset.csv"))

    raw_dir = raw_data_path.parent
    if str(raw_dir) in {"", "."}:
        raw_dir = Path("data/raw")

    npy_path = (artifacts_dir / "xai_background.npy").as_posix()
    csv_path = (raw_dir / "xai_background.csv").as_posix()
    
    if os.path.exists(csv_path):
        logger.info("Archivo CSV de background detectado en: %s", csv_path)
        df_bg = pd.read_csv(csv_path)
        df_bg, _ = _prepare_inference_dataframe(
            df=df_bg,
            config=config,
            data_cfg=data_cfg,
        )
        target_column = str(data_cfg.get("target_column", "")).lower().strip()
        if not target_column:
            raise ValueError("data_cfg['target_column'] es obligatorio.")

        id_column = data_cfg.get("id_column")
        id_column = str(id_column).lower().strip() if id_column else None
        timestamp_column = str(data_cfg.get("timestamp_column", "timestamp")).lower().strip()

        seq_length = int(data_cfg.get("sequence_length", 0))
        if seq_length <= 0:
            raise ValueError("data_cfg['sequence_length'] debe ser > 0.")

        overlap_beta = float(data_cfg.get("solapamiento_beta", 0.0))

        x_bg_sequences, _, _ = create_sequences(
            df_bg,
            target_column=target_column,
            seq_length=seq_length,
            solapamiento_beta=overlap_beta,
            id_column=id_column,
            timestamp_column=timestamp_column,
            normal_tokens=data_cfg.get("normal_tokens"),
        )

        if len(x_bg_sequences) == 0:
            raise ValueError(f"No se pudieron generar secuencias desde CSV de background: {csv_path}")

        n_features = x_bg_sequences.shape[-1]
        x_bg_scaled = scaler_x.transform(
            x_bg_sequences.reshape(-1, n_features)
        ).reshape(x_bg_sequences.shape)

        out_dir = os.path.dirname(npy_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(npy_path, x_bg_scaled)

        return x_bg_scaled

    # Si no se encuentra ningún background válido, lanzar error
    else:
        raise FileNotFoundError(
            f"No se encontró un background válido para SHAP. "
            f"Se requiere un CSV en {csv_path}"
            "Genere un background ejecutando: python scripts/generate_dataset.py --xai"
        )


def run_xai_prediction(
    config_path: str = "config/config.yaml",
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_background: Optional[int] = None,
    top_rules: Optional[int] = None,
    top_variables: Optional[int] = None,
    top_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta predicción con explicabilidad XAI completa.

    Args:
        config_path: Ruta a config YAML
        input_path: Ruta a datos de entrada
        output_dir: Directorio para guardar reportes XAI
        n_background: Muestras de background para SHAP
        top_rules: Reglas fuzzy principales
        top_variables: Variables temporales principales
        top_blocks: Bloques de acción principales

    Returns:
        Diccionario con predicciones y explicaciones
    """
    config = load_config(config_path)
    paths = config["paths"]
    data_cfg = config["data_processing"]
    xai_runtime_cfg = _resolve_xai_runtime_config(
        config=config,
        n_background=n_background,
        top_rules=top_rules,
        top_variables=top_variables,
        top_blocks=top_blocks,
    )

    # 1) Salida
    if output_dir is None:
        output_dir = ensure_dir(os.path.join(paths["predictions"], "xai"))
    else:
        output_dir = ensure_dir(output_dir)

    # 2) Modelo + artefactos + umbral oficial
    model, _, scalers, threshold = load_model_artifacts(config)
    scaler_x = scalers["scaler_x"]
    scaler_num = scalers["scaler_num"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 3) Resolver input
    data_path = input_path or paths.get("input_data", paths["raw_data"])
    if os.path.isdir(data_path):
        csv_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])
        if not csv_files:
            raise FileNotFoundError(f"No se encontraron archivos CSV en {data_path}")
        data_path = os.path.join(data_path, csv_files[0])
        logger.info("Archivo detectado en directorio de entrada: %s", data_path)

    logger.info("=" * 30)
    df_input = load_raw_data(data_path)
    df_input, _ = _prepare_inference_dataframe(
        df=df_input,
        config=config,
        data_cfg=data_cfg,
    )

    # 4) Preparación de secuencias (misma lógica de inferencia estándar)
    target_column = data_cfg["target_column"].lower()
    id_column = data_cfg.get("id_column")
    id_column = id_column.lower() if id_column else None
    timestamp_column = data_cfg.get("timestamp_column", "timestamp")
    stats_creation = data_cfg.get("fuzzy_processing", {}).get(
        "stats_creation", ["mean", "std", "slope", "max", "min"]
    )
    solapamiento_beta = float(config.get("inference", {}).get("solapamiento_beta", 0.5))

    if target_column not in df_input.columns:
        df_input[target_column] = 0

    x_sequences, _, timestamp_windows_labels = create_sequences(
        df_input,
        target_column=target_column,
        seq_length=data_cfg["sequence_length"],
        solapamiento_beta=solapamiento_beta,
        id_column=id_column,
        timestamp_column=timestamp_column,
        normal_tokens=data_cfg.get("normal_tokens"),
    )

    if len(x_sequences) == 0:
        raise ValueError("No se generaron secuencias para explicación XAI.")

    exclude_cols = {target_column, timestamp_column}
    if id_column:
        exclude_cols.add(id_column)
    feature_cols = [c for c in df_input.columns if c not in exclude_cols]

    df_stats = stats_windows(
        x_sequences,
        feature_names=feature_cols,
        stats_creation=stats_creation,
    )

    n_features = x_sequences.shape[-1]
    x_sequences_scaled = scaler_x.transform(x_sequences.reshape(-1, n_features)).reshape(x_sequences.shape)
    stats_scaled = scaler_num.transform(df_stats)

    logger.info(
        "Ventanas creadas: %s",
        x_sequences_scaled.shape,
    )
    logger.info(
        "Ventanas totales: %d, con %d registros cada una y un solapamiento de %s (%d registros).",
        len(x_sequences_scaled),
        x_sequences_scaled.shape[1],
        solapamiento_beta,
        solapamiento_beta * x_sequences_scaled.shape[1],
    )
    logger.info("=" * 30)

    # 5) Background para SHAP: artifact cacheado > CSV de entrada > error si no se encuentra ninguno válido
    background_windows = _resolve_background_windows(
        scaler_x=scaler_x,
        data_cfg=data_cfg,
        config=config,
    )

    # 6) Explainer
    explainer = DNFLExplainer(
        model=model,
        feature_names_stats=df_stats.columns.tolist(),
        feature_names_original=feature_cols,
        action_config=xai_runtime_cfg["action_cfg"],
        corrective_actions_cfg=xai_runtime_cfg["corrective_actions_cfg"],
        stats_creation=stats_creation,
        model_cfg=config.get("model", {}),
    )

    # 7) Procesar todas las muestras
    sample_indices = list(range(len(x_sequences_scaled)))
    logger.info("Procesando %d ventanas para XAI... (espere)", len(sample_indices))
    logger.info("=" * 30)

    results = []
    pred_classes = []
    anomaly_probs = []
    processed_idx = []
    processed_timestamps = []

    for idx_in_list, idx_global in enumerate(sample_indices):
        try:
            x_window = x_sequences_scaled[idx_global]
            s_stats_sample = stats_scaled[idx_global]
            timestamp_window = timestamp_windows_labels[idx_global] if timestamp_windows_labels else None

            xai_result = explainer.explain(
                x_window=x_window,
                s_stats=s_stats_sample,
                background_windows=background_windows,
                anomaly_threshold=float(threshold),
                n_background=xai_runtime_cfg["n_background"],
                random_state=int(config.get("project", {}).get("seed", 42)),
                top_rules=xai_runtime_cfg["top_rules"],
                top_variables=xai_runtime_cfg["top_variables"],
                top_blocks=xai_runtime_cfg["top_blocks"],
            )

            p_anom = float(xai_result["prediction"]["anomaly_probability"])
            if (
                not np.isfinite(p_anom)
                or not 0.0 <= p_anom <= 1.0
            ):
                raise ValueError(
                    "Probabilidad de anomalía inválida: "
                    f"{p_anom}"
                )
            pred_bin = int(p_anom >= float(threshold))

            pred_classes.append(pred_bin)
            anomaly_probs.append(p_anom)
            if timestamp_window is not None:
                processed_timestamps.append(timestamp_window)
            processed_idx.append(int(idx_global + 1))  # +1 para que el índice sea 1-based

            if timestamp_window is not None:
                final_report = {
                    "Índice_de_ventana": int(idx_global + 1),  
                    "Timestamp_cubierto_por_ventana": f"{timestamp_window[0]} / {timestamp_window[1]}",
                }
            else:
                raise ValueError("Debe proporcionar los timestamps de las ventanas.")
            final_report.update(dict(xai_result["final_report"]))
            results.append(final_report)
        except Exception as e:
            logger.error("Error en muestra %d: %s", idx_global, str(e))
            results.append({"Índice_de_ventana": int(idx_global + 1), "error": str(e)})

    # 8) Guardar salidas
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(output_dir, f"xai_predictions_{ts}.json")
    output_csv = os.path.join(output_dir, f"xai_predictions_{ts}.csv")

    def _json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2, ensure_ascii=False)

    if pred_classes:
        pred_df = decode_predictions(
            pred_classes=np.asarray(pred_classes, dtype=int),
            anomaly_scores=np.asarray(anomaly_probs, dtype=float),
            threshold=float(threshold),
            timestamp_index=processed_timestamps if timestamp_windows_labels else None,
            window_index=processed_idx
        )

        pred_df.to_csv(output_csv, index=False)
    else:
        pd.DataFrame(columns=[
            "predicted_anomaly_class",
            "predicted_anomaly_label",
            "anomaly_probability",
        ]).to_csv(output_csv, index=False)

    logger.info(
        "Resultados XAI guardados en:\n- %s\n- %s",
        output_json,
        output_csv,
    )

    # Resumen
    summary = {
        "n_processed": len([r for r in results if "error" not in r]),
        "n_errors": len([r for r in results if "error" in r]),
        "anomalies_detected": int(sum(pred_classes)) if pred_classes else 0,
        "threshold_used": round(float(threshold), 2),
        "output_file_json": output_json,
        "output_file_csv": output_csv,
    }

    summary_compact = {
        "n_procesados": summary["n_processed"],
        "n_errores": summary["n_errors"],
        "Anomalias_Detectadas": summary["anomalies_detected"],
        "Umbral_Utilizado": summary["threshold_used"],
    }
    logger.info("Predicción XAI completada: %s", summary_compact)

    return {
        "results": results,
        "summary": summary,
        "config": {
            "n_background": xai_runtime_cfg["n_background"],
            "top_rules": xai_runtime_cfg["top_rules"],
            "top_variables": xai_runtime_cfg["top_variables"],
            "top_blocks": xai_runtime_cfg["top_blocks"],
        },
    }