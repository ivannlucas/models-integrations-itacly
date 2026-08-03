"""Métricas de evaluación para el modelo Deep Neuro-Fuzzy."""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def binary_metrics_np(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calcula métricas binarias (accuracy y F1).

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.

    Returns:
        Dict con 'acc' y 'f1'.
    """
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_class_weights(
    y_train: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Calcula pos_weight para BCE de anomalía.

    Args:
        y_train: Etiquetas de entrenamiento.
        device: Dispositivo para los tensores.

    Returns:
        Tensor de pos_weight para BCEWithLogitsLoss.
    """
    y_np = np.asarray(y_train).astype(int)

    if not np.all(np.isin(y_np, [0, 1])):
        raise ValueError("y_train debe contener únicamente etiquetas binarias {0,1}.")

    n_neg = float((y_np == 0).sum())
    n_pos = float((y_np == 1).sum())

    return torch.tensor(
        [n_neg / max(n_pos, 1.0)],
        dtype=torch.float32,
        device=device,
    )


def optimize_threshold_by_f1(
    y_true: np.ndarray,
    prob_anomaly: np.ndarray,
    n_points: int = 101,
) -> Dict[str, float]:
    """Optimiza umbral binario por F1 sobre probabilidades de anomalía.

    Args:
        y_true: Etiquetas binarias verdaderas.
        prob_anomaly: Probabilidades de anomalía.
        n_points: Número de puntos para la búsqueda de umbral.

    Returns:
        Dict con 'best_threshold' y 'best_f1'.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_anomaly = np.asarray(prob_anomaly).astype(float)

    thresholds = np.linspace(0.0, 1.0, n_points)
    best_threshold = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        pred = (prob_anomaly >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thr

    return {
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
    }
