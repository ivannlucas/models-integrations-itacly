"""MLflow helper for ml34 dairy pasteurization — download user-trained model from MLflow."""
from __future__ import annotations

import json
import os

import joblib
import torch

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml34_dairy_pasteurization_energy_ga.constants import (
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.model_loader import build_model_from_config


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned MLP from MLflow.

    Returns (model, scaler_X, scaler_y, config, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).

    The architecture is rebuilt dynamically from the downloaded
    model_config.json, so a retrained model with a different topology
    still loads correctly.
    """
    loaded = download_run_artifacts(run_id, _load, prefix="mlflow_ml34_")
    if loaded is None:
        return None
    (model, scaler_x, scaler_y, config), tmp = loaded
    return model, scaler_x, scaler_y, config, tmp


def _load(local_path: str):
    """Rebuild the MLP (from its config), its weights and both scalers from ``local_path``."""
    with open(os.path.join(local_path, MODEL_CONFIG_FILENAME), "r", encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config)
    model.load_state_dict(
        torch.load(
            os.path.join(local_path, MODEL_FILENAME),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    scaler_x = joblib.load(os.path.join(local_path, SCALER_X_FILENAME))
    scaler_y = joblib.load(os.path.join(local_path, SCALER_Y_FILENAME))
    return model, scaler_x, scaler_y, config
