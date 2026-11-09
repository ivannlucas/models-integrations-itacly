"""MLflow helper for ml30 traceability — download user-trained model from MLflow."""
from __future__ import annotations

import os
import pickle

import torch

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml30_meat_traceability_detection.constants import (
    FEATURE_COLUMNS,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
)
from app.plugins.ml30_meat_traceability_detection.model_loader import (
    build_torch_mlp,
    load_payload,
)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download user-trained MLP + preprocessor from MLflow.

    Returns (preprocessor, mlp, feature_columns, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(run_id, _load, prefix="mlflow_ml30_")
    if loaded is None:
        return None
    (preprocessor, mlp), tmp = loaded
    return preprocessor, mlp, list(FEATURE_COLUMNS), tmp


def _load(local_path: str):
    """Rebuild the fitted preprocessor and the MLP from a downloaded ``model`` folder."""
    preprocessor_path = os.path.join(local_path, PREPROCESSOR_FILENAME)
    state_dict_path = os.path.join(local_path, MODEL_FILENAME)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    mlp = build_torch_mlp(load_payload())
    mlp.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=False))
    mlp.eval()
    return preprocessor, mlp
