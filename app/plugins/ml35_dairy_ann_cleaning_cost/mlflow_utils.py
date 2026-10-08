"""MLflow helper for ml35 dairy ANN — download user-trained model from MLflow."""
from __future__ import annotations

import logging
import os

import joblib
import torch
import shutil

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import (
    FEATURES,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.model_loader import PasteurizationANN

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned ANN from MLflow.

    Returns (model, scaler_X, scaler_y, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mlflow_ml35_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
        if not local_path:
            shutil.rmtree(tmp, ignore_errors=True)
            return None

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
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
