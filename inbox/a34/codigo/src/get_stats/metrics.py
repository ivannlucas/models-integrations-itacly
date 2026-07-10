"""
Regression metrics computation for model evaluation.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None,
) -> Dict:
    """
    Compute RMSE, MAE and R2 for each target variable.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, n_targets)
        Ground truth values in original units.
    y_pred : np.ndarray, shape (N, n_targets)
        Predictions in original units.
    target_names : list, optional
        Target names (default: ['E_consumo', 'T_out_leche']).

    Returns
    -------
    dict
        Nested dictionary with metrics per target.
        Example: {'E_consumo': {'RMSE': ..., 'MAE': ..., 'R2': ...}, ...}
    """
    if target_names is None:
        target_names = ["E_consumo", "T_out_leche"]

    results = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        results[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
        }

    return results
