"""Model forward pass + output formatting for ml26.

Mirrors src/training/models/inference.py::predict_sequence (z-score, forward, de-normalize, clip
risk
to [0, 1]) and src/predict/postprocess.py::format_output (round to 6 decimals).
"""

from __future__ import annotations

import numpy as np
import torch

from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import (
    RISK_BAND_LOW_MAX,
    RISK_BAND_MEDIUM_MAX,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import LoadedModel


def constrain_predictions(preds: np.ndarray, target_names: list[str]) -> np.ndarray:
    """Clip underprotection_risk_* targets to [0, 1] (src/utils/target_constraints.py)."""
    constrained = np.asarray(preds, dtype=float).copy()
    for index, target in enumerate(target_names):
        if target.startswith("underprotection_risk_") and target.endswith("h"):
            constrained[:, index] = np.clip(constrained[:, index], 0.0, 1.0)
    return constrained


def predict_windows(model: LoadedModel, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return (n, n_targets) de-normalized, constrained and 6-decimal-rounded predictions."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != len(model.feature_names):
        raise ValueError(
            f"Sequence inference requires (samples, window, {len(model.feature_names)}); got "
            f"{values.shape}"
        )
    normalized = ((values - model.mean) / model.std).astype(np.float32)
    outputs: list[np.ndarray] = []
    model.network.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.tensor(normalized[start:start + batch_size], dtype=torch.float32)
            outputs.append(model.network(batch).cpu().numpy())
    preds = (
        np.concatenate(outputs, axis=0)
        if outputs
        else np.empty((0, len(model.target_names)), dtype=np.float32)
    )
    preds = np.asarray(preds * model.y_std + model.y_mean, dtype=np.float32)
    return np.round(constrain_predictions(preds, model.target_names), 6)


def risk_band(score: float) -> str:
    """Operational band used by the AI team's KPI (notebooks/EDA/eda.ipynb::risk_band — manifest
    KI-07).
    """
    if score < RISK_BAND_LOW_MAX:
        return "bajo"
    if score < RISK_BAND_MEDIUM_MAX:
        return "medio"
    return "alto"
