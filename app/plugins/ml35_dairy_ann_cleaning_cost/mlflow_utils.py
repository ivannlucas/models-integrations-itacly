"""MLflow helper for ml35 dairy ANN — download user-trained model from MLflow."""
from __future__ import annotations

import os

import joblib
import torch

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import (
    FEATURES,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.model_loader import PasteurizationANN


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned ANN from MLflow.

    Returns (model, scaler_X, scaler_y, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(run_id, _load, prefix="mlflow_ml35_")
    if loaded is None:
        return None
    (model, scaler_x, scaler_y), tmp = loaded
    return model, scaler_x, scaler_y, tmp


def _load(local_path: str):
    """Rebuild the ANN and its two scalers from a downloaded ``model`` folder."""
    model = PasteurizationANN(input_size=len(FEATURES))
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
    return model, scaler_x, scaler_y
