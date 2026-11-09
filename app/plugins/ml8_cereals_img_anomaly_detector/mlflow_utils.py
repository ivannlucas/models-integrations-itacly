"""MLflow helper for ml8 cereals — download user-trained model from MLflow."""
from __future__ import annotations

import os

import torch

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml8_cereals_img_anomaly_detector.constants import (
    MODEL_FILENAME,
)
from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import (
    MultiTaskMobileNetV3Large,
    _safe_device,
)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-trained MultiTaskMobileNetV3Large from MLflow.

    Returns (model_bundle_dict, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    return download_run_artifacts(run_id, _load, prefix="mlflow_ml8_")


def _load(local_path: str) -> dict:
    """Rebuild the model bundle from a downloaded ``model`` folder."""
    checkpoint_path = os.path.join(local_path, MODEL_FILENAME)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    device = _safe_device()
    model = MultiTaskMobileNetV3Large(
        num_classes=len(checkpoint["idx_to_class"]),
        num_cereals=len(checkpoint["idx_to_cereal"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "image_size": 224,
        "idx_to_class": checkpoint["idx_to_class"],
        "idx_to_cereal": checkpoint["idx_to_cereal"],
        "model_id": "ml8-cereals-img-anomaly-detector",
    }
