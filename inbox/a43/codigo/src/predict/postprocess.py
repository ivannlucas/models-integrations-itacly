"""Postprocesamiento de predicciones."""

from typing import Optional

import numpy as np
import pandas as pd


def decode_predictions(
    pred_classes: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float = 0.5,
    timestamp_index: Optional[list[tuple[pd.Timestamp, pd.Timestamp]]] = None,
    window_index: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Convierte arrays de predicción binaria a DataFrame legible.

    Args:
        pred_classes: Array binario predicho (0/1) [N].
        anomaly_scores: Array de probabilidades de anomalía [N].
        threshold: Umbral para clasificación binaria de anomalía.
        timestamp_index: Lista de tuplas con timestamps de inicio y fin de cada ventana de predicción.
    Returns:
        DataFrame con columnas de predicción.
    """
    is_anomaly = (anomaly_scores >= threshold).astype(int)
    anomaly_labels = np.where(is_anomaly == 1, "Fallo", "No Fallo")

    if timestamp_index is not None:
        result = pd.DataFrame({
            "window_index": window_index,  # +1 para que el índice sea 1-based
            "timestamp_init": [ts[0] for ts in timestamp_index],
            "timestamp_end": [ts[1] for ts in timestamp_index],
            "predicted_anomaly_class": pred_classes.astype(int),
            "predicted_anomaly_label": anomaly_labels,
            "anomaly_probability": np.round(anomaly_scores, 4),
        })

    else:
        raise ValueError("Debe proporcionar los timestamps de las ventanas.")

    # Añadir columna con el umbral usado para que quede registrada en el CSV
    result["decision_threshold"] = round(float(threshold), 2)

    return result
