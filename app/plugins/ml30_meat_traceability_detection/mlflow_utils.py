"""MLflow helper for ml30 traceability — download user-trained model from MLflow."""
from __future__ import annotations

import logging
import os
import pickle

import torch

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.ml30_meat_traceability_detection.constants import (
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download user-trained MLP + preprocessor from MLflow.

    Returns (preprocessor, mlp, feature_columns, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="model", prefix="mlflow_ml30_")
    if result is None:
        return None
    tmp, local_path = result

    from app.plugins.ml30_meat_traceability_detection.model_loader import build_torch_mlp, load_payload
    from app.plugins.ml30_meat_traceability_detection.constants import FEATURE_COLUMNS

    preprocessor_path = os.path.join(local_path, PREPROCESSOR_FILENAME)
    state_dict_path = os.path.join(local_path, MODEL_FILENAME)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    mlp = build_torch_mlp(load_payload())
    mlp.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=False))
    mlp.eval()

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return preprocessor, mlp, list(FEATURE_COLUMNS), tmp
