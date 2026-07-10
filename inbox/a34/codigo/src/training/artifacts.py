"""
Model artifact management: save and load weights, scalers and configuration.
"""

import json
from typing import Dict, Tuple

import joblib
import torch
from sklearn.preprocessing import MinMaxScaler

from src.training.model import DynamicMLP
from src.utils.constants import FEATURES, TARGETS
from src.utils.paths import (
    ARTIFACTS_DIR, MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH,
    SCALER_X_PATH, SCALER_Y_PATH, TRAIN_METRICS_PATH, METRICS_DIR,
)


def save_artifacts(
    model: DynamicMLP,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    best_params: Dict,
    metrics: Dict = None,
    features: list = None,
    targets: list = None,
) -> None:
    """
    Save all production artifacts: model, scalers, config and metrics.

    Parameters
    ----------
    model : DynamicMLP
        Trained model.
    scaler_X, scaler_y : MinMaxScaler
        Fitted scalers.
    best_params : dict
        Best model hyperparameters (num_layers, neurons, lr, activation).
    metrics : dict, optional
        Evaluation metrics to save as JSON.
    features, targets : list, optional
        Lists of feature and target names.
    """
    features = features or FEATURES
    targets = targets or TARGETS
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Model weights
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

    # Scalers
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    # Configuration with explicit feature and target names
    config = {
        "input_size": len(features),
        "output_size": len(targets),
        "features_in_order": features,
        "targets_in_order": targets,
        **best_params,
    }
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    # Metrics
    if metrics is not None:
        with open(TRAIN_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)


def load_artifacts() -> Tuple[DynamicMLP, MinMaxScaler, MinMaxScaler, Dict]:
    """
    Load production artifacts: model, scalers and configuration.

    Returns
    -------
    tuple
        (model in eval mode, scaler_X, scaler_y, config_dict)
    """
    # Configuration
    with open(MODEL_CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Reconstruct and instantiate model
    model = DynamicMLP(
        input_size=config["input_size"],
        output_size=config["output_size"],
        num_layers=config["num_layers"],
        neurons=config["neurons"],
        activation=config["activation"],
    )
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    # Scalers
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    return model, scaler_X, scaler_y, config
