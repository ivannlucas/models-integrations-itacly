"""Postprocessing utilities for meat-traceability predictions."""
from __future__ import annotations

import numpy as np
import torch


def run_inference(preprocessor, mlp, x_df) -> tuple[np.ndarray, np.ndarray]:
    """Apply the preprocessor then the MLP; return (y_pred, y_score)."""
    x_tensor = torch.tensor(preprocessor.transform(x_df), dtype=torch.float32)
    with torch.no_grad():
        y_score = torch.sigmoid(mlp(x_tensor).reshape(-1)).numpy()
    return (y_score >= 0.5).astype(int), y_score
