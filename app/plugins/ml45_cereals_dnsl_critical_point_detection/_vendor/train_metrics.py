"""Training-time metrics for the Deep Neuro-Fuzzy model.

Copied verbatim (module-path/logger imports adjusted) from
a45-dnsl-cereals-deteccion-puntos-criticos/src/training/metrics.py — confirmed
byte-identical to a43-44-neurofuzzy-anomalias-fallas/src/training/metrics.py — and the
two epoch-level helpers from src/training/trainer.py (alpha_entropy_mean,
rule_corr_mean). Used by plugin.py::train() to reproduce the real training repo's
class-weighting, checkpoint-selection monitor score, and threshold calibration exactly
(modelo 43-44-45 audit, Fase 5: decision_threshold parity with local training).
"""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def binary_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas binarias (accuracy y F1)."""
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_class_weights(
    y_train: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Calcula pos_weight para BCE de anomalía."""
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

    Uniform 101-point grid search over [0, 1] — deliberately NOT
    sklearn.precision_recall_curve (whose candidate thresholds are the observed
    probabilities themselves): must match the real repo's search space exactly so a
    calibrated threshold trained here is comparable to one trained locally on the
    same data.
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


def alpha_entropy_mean(
    alpha_probs: Optional[torch.Tensor], eps: float = 1e-8
) -> float:
    """Entropía media de alpha_probs [R, F, M]."""
    if alpha_probs is None:
        return 0.0
    p = alpha_probs.clamp_min(eps)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    return float(entropy.mean().detach().cpu())


def rule_corr_mean(
    rule_activations: Optional[torch.Tensor], eps: float = 1e-8
) -> float:
    """Correlación media absoluta fuera de diagonal entre reglas."""
    if rule_activations is None or rule_activations.dim() != 2:
        return 0.0

    B, R = rule_activations.shape
    if B < 2 or R < 2:
        return 0.0

    x = rule_activations - rule_activations.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    x = x / std
    corr = (x.T @ x) / B

    mask = ~torch.eye(R, dtype=torch.bool, device=corr.device)
    return float(corr[mask].abs().mean().detach().cpu())
