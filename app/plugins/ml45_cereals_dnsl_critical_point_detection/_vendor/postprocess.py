"""Vendored verbatim from inbox/a45/codigo/src/predict/postprocess.py."""
from __future__ import annotations

from typing import Optional, Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    """Convierte valores NumPy anidados a tipos serializables en JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def decode_predictions(
    pred_classes: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float = 0.5,
    timestamp_index: Optional[list[tuple[pd.Timestamp, pd.Timestamp]]] = None,
    window_index: Optional[list[int]] = None,
    cycle_id: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Convierte arrays de predicción binaria a DataFrame legible.

    Args:
        pred_classes: Array binario predicho (0/1) [N].
        anomaly_scores: Array de probabilidades de anomalía [N].
        threshold: Umbral para clasificación binaria de anomalía.
        timestamp_index: Lista de tuplas con timestamps de inicio y fin de cada ventana de predicción.
        window_index: Lista de índices de las ventanas de predicción.
        cycle_id: Lista de IDs de los ciclos asociados a cada ventana.
    Returns:
        DataFrame con columnas de predicción.
    """
    is_anomaly = (anomaly_scores >= threshold).astype(int)
    anomaly_labels = np.where(is_anomaly == 1, "Fallo", "No Fallo")

    if timestamp_index is not None:

        result = pd.DataFrame({
            "window_index": window_index,  # +1 para que el índice sea 1-based
            "cycle_id": cycle_id,
            "timestamp_init": [ts[0] for ts in timestamp_index],
            "timestamp_end": [ts[1] for ts in timestamp_index],
            "predicted_anomaly_class": pred_classes.astype(int),
            "predicted_anomaly_label": anomaly_labels,
            "anomaly_probability": np.round(anomaly_scores, 4),
        })

        if cycle_id is None:
            result.drop(columns=["cycle_id"], inplace=True)

    else:
        raise ValueError("Debe proporcionar los timestamps de las ventanas.")

    # Añadir columna con el umbral usado para que quede registrada en el CSV
    result["decision_threshold"] = float(threshold)

    return result
