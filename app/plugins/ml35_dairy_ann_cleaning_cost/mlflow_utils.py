"""MLflow helper for ml35 dairy ANN — download user-trained model from MLflow."""
from __future__ import annotations

import logging
import os

import joblib
import torch

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import (
    FEATURES,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.model_loader import PasteurizationANN

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned ANN from MLflow.

    Returns (model, scaler_X, scaler_y, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="model", prefix="mlflow_ml35_")
    if result is None:
        return None
    tmp, local_path = result

    model = PasteurizationANN(input_size=len(FEATURES))
    model.load_state_dict(
        torch.load(
            os.path.join(local_path, MODEL_FILENAME),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    scaler_X = joblib.load(os.path.join(local_path, SCALER_X_FILENAME))
    scaler_y = joblib.load(os.path.join(local_path, SCALER_Y_FILENAME))

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, scaler_X, scaler_y, tmp
