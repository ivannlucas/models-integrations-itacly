"""MLflow helper for ml34 dairy pasteurization — download user-trained model from MLflow."""
from __future__ import annotations

import json
import logging
import os

import joblib
import torch

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.ml34_dairy_pasteurization_energy_ga.constants import (
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.model_loader import build_model_from_config

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned MLP from MLflow.

    Returns (model, scaler_X, scaler_y, config, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.

    The architecture is rebuilt dynamically from the downloaded
    model_config.json, so a retrained model with a different topology
    still loads correctly.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="model", prefix="mlflow_ml34_")
    if result is None:
        return None
    tmp, local_path = result

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

    scaler_X = joblib.load(os.path.join(local_path, SCALER_X_FILENAME))  # pylint: disable=invalid-name
    scaler_y = joblib.load(os.path.join(local_path, SCALER_Y_FILENAME))

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, scaler_X, scaler_y, config, tmp
