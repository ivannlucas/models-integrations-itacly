"""Orquestador principal del proyecto DatagIA.

Punto de entrada lógico con las 5 funciones principales:
- generate_dataset(): Generación del dataset sintético.
- data_processing(): Pipeline de procesamiento de datos.
- train(): Entrenamiento del modelo.
- predict(): Inferencia sobre datos nuevos.
- get_stats(): Generación de estadísticas y métricas.
"""

import json
import os
import pickle
from typing import Optional

from src.data_processing.generate_dataset import run_dataset_generation, run_xai_datasets_generation
from src.utils.common import ensure_dir, load_config, seed_everything
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _apply_seed_from_config(config: dict) -> None:
    """Aplica semilla global con opciones de determinismo desde config.project."""
    project_cfg = config.get("project", {})
    seed_everything(
        seed=int(project_cfg.get("seed", 42)),
        deterministic_strict=bool(project_cfg.get("deterministic_strict", True)),
        deterministic_warn_only=bool(project_cfg.get("deterministic_warn_only", False)),
    )


def generate_dataset(config_path: str = "config/config.yaml") -> None:
    """Genera el dataset sintético de secadora de grano de flujo mixto.

    Simula ciclos operacionales con 11 sensores IIoT, incluyendo
    modos de operación normal y 5 tipos de falla. Guarda el dataset
    completo en data/raw/ y los splits en data/processed/.

    Args:
        config_path: Ruta al archivo de configuración YAML.
    """
    config = load_config(config_path)
    _apply_seed_from_config(config)

    

    df = run_dataset_generation(config)
    logger.info(
        "Dataset generado: %d filas, %d ciclos.",
        len(df),
        df["cycle_id"].nunique(),
    )

def generate_dataset_for_xai(config_path: str = "config/config.yaml") -> None:
    """Crea 3 datasets con características controladas para validar explicaciones:

    - Dataset de validación/calibración interpretativa (interpretability_val): dataset balanceado.
    - Dataset de evaluación del sistema de detección de Puntos Críticos de Control (pcc_system_eval): dataset desbalanceado (más realista)
    - Dataset para background de SHAP (xai_background): dataset pequeño y balanceado.

    Todos los datasets se generan con semillas específicas para reproducibilidad y se guardan en data/raw/ en formato CSV.

    Args:
        config_path: Ruta al archivo de configuración YAML.
    """
    config = load_config(config_path)
    run_xai_datasets_generation(config=config)
    
def data_processing(config_path: str = "config/config.yaml") -> None:
    """Ejecuta el pipeline de procesamiento: carga splits, secuencias, escala y guarda.

    Lee los splits pre-generados desde data/splits/ (generados por
    generate_dataset), aplica secuencias temporales, escalado y guarda
    artefactos en data/processed/ y models/artifacts/.

    Args:
        config_path: Ruta al archivo de configuración YAML.
    """
    config = load_config(config_path)
    _apply_seed_from_config(config)

    import numpy as np
    import pandas as pd

    from src.data_processing.preprocess import create_sequences, run_data_processing_pipeline

    paths = config["paths"]
    data_cfg = config["data_processing"]
    splits_dir = paths["splits"]

    # Cargar splits pre-generados por generate_dataset
    pre_split = {}
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"No se encontró {split_path}. "
                f"Ejecuta generate_dataset() primero."
            )
        df_split = pd.read_csv(split_path)
        df_split.columns = df_split.columns.str.lower()
        pre_split[split_name] = df_split
        logger.info("Split '%s' cargado: %d filas", split_name, len(df_split))

    split_data, split_data_df_stats, scaler_x, scaler_num = (
        run_data_processing_pipeline(
            data_cfg=data_cfg,
            pre_split=pre_split,
            config=config,
        )
    )

    # Guardar artefactos de procesamiento
    artifacts_dir = ensure_dir(paths["model_artifacts"])

    scalers = {"scaler_x": scaler_x, "scaler_num": scaler_num}
    with open(os.path.join(artifacts_dir, paths["scaler_filename"]), "wb") as f:
        pickle.dump(scalers, f)

    # Guardar arrays procesados
    splits_dir = ensure_dir(paths["processed_data"])
    for split_name in ["train", "val", "test"]:
        np.save(
            os.path.join(splits_dir, f"X_{split_name}.npy"),
            split_data["X"][split_name],
        )
        np.save(
            os.path.join(splits_dir, f"y_{split_name}.npy"),
            split_data["y"][split_name],
        )
        split_data_df_stats[split_name].to_csv(
            os.path.join(splits_dir, f"stats_{split_name}.csv"), index=False
        )

    # Construir background SHAP desde xai_background.csv (si existe)
    raw_dir = paths.get("raw_data", "data/raw/")
    bg_csv_path = os.path.join(raw_dir, "xai_background.csv")

    
    target_column = data_cfg["target_column"].lower()
    id_column = data_cfg.get("id_column")
    id_column = id_column.lower().strip() if id_column else None
    seq_length = int(data_cfg["sequence_length"])
    overlap_beta = float(data_cfg.get("solapamiento_beta", 0.5))
    normal_tokens = data_cfg.get("normal_tokens")

    if os.path.exists(bg_csv_path):
        logger.info("Generando background SHAP desde %s", bg_csv_path)
        df_bg = pd.read_csv(bg_csv_path)
        df_bg.columns = df_bg.columns.astype(str).str.strip().str.lower()

        # Asegurar columna target para create_sequences
        if target_column not in df_bg.columns:
            df_bg[target_column] = 0

        x_bg_seq, _, _, _ = create_sequences(
            df_bg,
            target_column=target_column,
            seq_length=seq_length,
            solapamiento_beta=overlap_beta,
            id_column=id_column,
            normal_tokens=normal_tokens,
        )

        if len(x_bg_seq) > 0:
            n_features = x_bg_seq.shape[-1]
            x_bg_scaled = scaler_x.transform(
                x_bg_seq.reshape(-1, n_features)
            ).reshape(x_bg_seq.shape)

            bg_npy_path = os.path.join(artifacts_dir, "xai_background.npy")
            np.save(bg_npy_path, x_bg_scaled)
            logger.info(
                "Background SHAP guardado: %s (shape=%s)",
                bg_npy_path, x_bg_scaled.shape,
            )
        else:
            logger.warning(
                "No se pudieron generar secuencias desde xai_background.csv."
            )
    else:
        logger.info(
            "Archivo xai_background.csv no encontrado en %s. "
            "El background SHAP se generará en tiempo de inferencia.",
            bg_csv_path,
        )

    logger.info("Pipeline de procesamiento de datos completado.")


def train(config_path: str = "config/config.yaml") -> dict:
    """Entrena el modelo, guarda artefactos y métricas.

    Lee los arrays procesados desde data/processed/ (generados por data_processing)
    y ejecuta el pipeline de entrenamiento completo.

    Args:
        config_path: Ruta al archivo de configuración YAML.

    Returns:
        Diccionario con resultados del entrenamiento.
    """
    config = load_config(config_path)
    _apply_seed_from_config(config)

    import numpy as np
    import pandas as pd

    from src.training.trainer import run_train_pipeline

    paths = config["paths"]
    data_cfg = config["data_processing"]
    processed_dir = paths["processed_data"]

    # Cargar arrays procesados por data_processing
    logger.info("=" * 30)
    logger.info("INICIO DEL ENTRENAMIENTO")
    logger.info("=" * 30)
    logger.info("Datos de entrenamiento:")
    logger.info("=" * 30)
    split_data = {"X": {}, "y": {}}
    split_data_df_stats = {}
    for split_name in ["train", "val", "test"]:
        x_path = os.path.join(processed_dir, f"X_{split_name}.npy")
        y_path = os.path.join(processed_dir, f"y_{split_name}.npy")
        stats_path = os.path.join(processed_dir, f"stats_{split_name}.csv")
        if not os.path.exists(x_path):
            raise FileNotFoundError(
                f"No se encontró {x_path}. "
                f"Ejecuta data_processing() primero."
            )
        split_data["X"][split_name] = np.load(x_path)
        split_data["y"][split_name] = np.load(y_path)
        split_data_df_stats[split_name] = pd.read_csv(stats_path)
        logger.info(
            "Split '%s' cargado: X=%s, y=%s",
            split_name,
            split_data["X"][split_name].shape,
            split_data["y"][split_name].shape,
        )

    # Cargar scalers (guardados por data_processing)
    scaler_path = os.path.join(paths["model_artifacts"], paths["scaler_filename"])
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No se encontró {scaler_path}. "
            f"Ejecuta data_processing() primero."
        )
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_x = scalers["scaler_x"]
    scaler_num = scalers["scaler_num"]

    # Construir configuración del modelo
    model_cfg = {
        "seed": int(config.get("project", {}).get("seed", 42)),
        "data_processing": data_cfg,
        "lstm": config["model"]["lstm"],
        "fuzzy": config["model"]["fuzzy"],
        "training_kwargs": config["training"],
    }

    # Entrenar
    artifacts_dir = ensure_dir(paths["model_artifacts"])
    metrics_dir = ensure_dir(paths["model_metrics"])
    checkpoint_dir = ensure_dir(paths.get("model_checkpoint_dir", artifacts_dir))
    train_out = run_train_pipeline(
        split_data=split_data,
        model_cfg=model_cfg,
        split_data_df_stats=split_data_df_stats,
        save_prefix="dnf_model",
        save_dir=checkpoint_dir,
    )

    # Guardar artefactos
    import torch

    # Modelo con metadata
    model = train_out["model"]
    model_save_path = os.path.join(checkpoint_dir, paths["model_filename"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": train_out["model_cfg"],
        },
        model_save_path,
    )

    # Resultados (threshold, best_epoch y épocas entrenadas)
    results = {
        "best_threshold": train_out["best_threshold"],
        "best_epoch": train_out.get("best_epoch"),
        "n_epochs_trained": len(train_out["history"]["train_loss"]),
    }
    with open(
        os.path.join(metrics_dir, paths["results_filename"]),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Historial
    history_serializable = {}
    for k, v in train_out["history"].items():
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], np.ndarray):
                history_serializable[k] = [arr.tolist() for arr in v]
            else:
                history_serializable[k] = v
        else:
            history_serializable[k] = v

    with open(
        os.path.join(metrics_dir, "training_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history_serializable, f, indent=2)

    logger.info("=" * 30)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 30)
    if train_out.get("best_epoch") is not None and train_out.get("best_epoch_summary") is not None:
        logger.info(f"MEJOR EPOCH SELECCIONADO: {train_out['best_epoch']:03d}/{train_out['num_epochs']}")
        logger.info(train_out["best_epoch_summary"])
        logger.info("=" * 30)
    logger.info("Artefactos guardados:")
    logger.info("  - %s", model_save_path)
    logger.info("  - %s", os.path.join(metrics_dir, paths["results_filename"]))
    logger.info("  - %s", os.path.join(metrics_dir, "training_history.json"))
    logger.info("=" * 30)
    logger.info("Ejecuta python scripts/get_stats.py para obtener las métricas de test.")

    return train_out


def predict(
    config_path: str = "config/config.yaml",
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Predice anomalías con explicabilidad XAI integrada.

    La inferencia pública del proyecto ejecuta siempre el flujo completo:
    predicción + explicaciones fuzzy + explicaciones temporales +
    detección PCC.

    Args:
        config_path: Ruta al archivo de configuración YAML.
        input_path: Ruta al archivo de datos de entrada (sobreescribe config).
        output_dir: Directorio para guardar reportes XAI.

    Returns:
        Diccionario con predicciones, explicaciones y reporte PCC.
    """
    from src.predict.xai_predictor import run_xai_prediction

    config = load_config(config_path)
    _apply_seed_from_config(config)

    results = run_xai_prediction(
        config_path=config_path,
        input_path=input_path,
        output_dir=output_dir,
    )

    logger.info(
        "Predicción completada: %d ventanas procesadas, %d anomalías detectadas, %d errores. Umbral de anomalía usado: %.2f",
        results["summary"]["n_processed"],
        results["summary"]["anomalies_detected"],
        results["summary"]["n_errors"],
        results["summary"]["threshold_used"],
    )

    return results


def get_stats(config_path: str = "config/config.yaml") -> None:
    """Genera visualizaciones y métricas de evaluación.

    Args:
        config_path: Ruta al archivo de configuración YAML.
    """
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.training.model import ParallelDeepNeuroFuzzyModel
    from src.get_stats.stats import run_metrics_pipeline

    config = load_config(config_path)
    paths = config["paths"]
    metrics_dir = paths["model_metrics"]
    checkpoint_dir = paths.get("model_checkpoint_dir", paths["model_artifacts"])
    processed_dir = paths["processed_data"]

    results_path = os.path.join(metrics_dir, paths["results_filename"])
    history_path = os.path.join(metrics_dir, "training_history.json")

    if not os.path.exists(results_path):
        logger.error(
            "No se encontraron resultados en %s. Ejecuta train() primero.",
            results_path,
        )
        return

    if not os.path.exists(history_path):
        logger.error("No se encontró el historial en %s.", history_path)
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Cargar modelo desde checkpoint
    model_path = os.path.join(checkpoint_dir, paths["model_filename"])
    if not os.path.exists(model_path):
        logger.error("No se encontró el modelo en %s.", model_path)
        return
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["model_cfg"]
    model = ParallelDeepNeuroFuzzyModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Reconstruir test_loader para evaluación
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy")).reshape(-1)
    stats_test = pd.read_csv(os.path.join(processed_dir, "stats_test.csv")).values
    test_dataset = TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(stats_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long),
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    run_metrics_pipeline(
        model=model,
        history=history,
        test_loader=test_loader,
        model_cfg=model_cfg,
        model_path=None,
        default_threshold=float(results.get("best_threshold", 0.5)),
        output_path=results_path,
        train_out=None,
        auto_use_train_outputs=False,
        auto_use_train_threshold=True,
        output_dir=metrics_dir,
        best_epoch=results.get("best_epoch"),
    )