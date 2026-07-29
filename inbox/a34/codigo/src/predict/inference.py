"""
Inference with the trained MLP model.

Provides predict_with_model() as a wrapper around the MLP model: given a
complete operational scenario, returns (E_consumo, T_out_leche) in real units.
Functional approach (receives model and scalers as parameters) for
flexibility and testability.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from src.utils.paths import PREDICTIONS_DIR


def predict_with_model(
    F_flow: float,
    T_servicio: float,
    T_in_leche: float,
    t_ciclo: float,
    Delta_P: float,
    model: nn.Module = None,
    scaler_X: MinMaxScaler = None,
    scaler_y: MinMaxScaler = None,
) -> Tuple[float, float]:
    """
    Query the MLP and return (E_consumo, T_out_leche) in real units.

    The feature array order follows the one defined in
    model_config.json: [T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P].

    Parameters
    ----------
    F_flow : float
        Milk flow rate (L/h).
    T_servicio : float
        Service water temperature (C).
    T_in_leche : float
        Milk inlet temperature (C).
    t_ciclo : float
        Time since last CIP cleaning (min).
    Delta_P : float
        Pressure drop across the heat exchanger (bar).
    model : nn.Module
        Loaded MLP model (in eval mode).
    scaler_X : MinMaxScaler
        Feature scaler.
    scaler_y : MinMaxScaler
        Target scaler.

    Returns
    -------
    tuple[float, float]
        (E_consumo in kW, T_out_leche in C).

    Raises
    ------
    ValueError
        If model, scaler_X or scaler_y are not provided.
    """
    if model is None or scaler_X is None or scaler_y is None:
        raise ValueError(
            "Must provide model, scaler_X and scaler_y. "
            "Use load_artifacts() from src.training.artifacts to load them."
        )

    # Order aligned with model_config.json: features_in_order
    x_raw = np.array([[T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P]])
    x_scaled = scaler_X.transform(x_raw)
    x_tensor = torch.FloatTensor(x_scaled)

    with torch.no_grad():
        y_scaled = model(x_tensor).numpy()

    y_real = scaler_y.inverse_transform(y_scaled)[0]
    return float(y_real[0]), float(y_real[1])  # E_consumo, T_out_leche


def predict_batch(
    X_raw: np.ndarray,
    model: nn.Module,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
) -> np.ndarray:
    """
    Batch prediction: receives an (N, 5) array and returns (N, 2) in real units.

    Parameters
    ----------
    X_raw : np.ndarray, shape (N, 5)
        Features in original units, order: [T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P].
    model : nn.Module
    scaler_X, scaler_y : MinMaxScaler

    Returns
    -------
    np.ndarray, shape (N, 2)
        Predictions [E_consumo, T_out_leche] in real units.
    """
    x_scaled = scaler_X.transform(X_raw)
    device = next(model.parameters()).device
    x_tensor = torch.FloatTensor(x_scaled).to(device)

    with torch.no_grad():
        y_scaled = model(x_tensor).cpu().numpy()

    return scaler_y.inverse_transform(y_scaled)


def save_predictions(
    df_pred: pd.DataFrame,
    out_path: str = None,
    filename: str = "predictions.csv",
) -> str:
    """
    Save a predictions DataFrame to data/predictions/.

    Parameters
    ----------
    df_pred : pd.DataFrame
        DataFrame containing prediction results.
    out_path : str, optional
        Full destination path. If None, saves to
        ``data/predictions/{filename}``.
    filename : str
        Filename when ``out_path`` is not specified.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    if out_path is None:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(PREDICTIONS_DIR / filename)
    df_pred.to_csv(out_path, index=False)
    return out_path
