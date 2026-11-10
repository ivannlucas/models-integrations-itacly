"""MLflow helper for Ml10DairyDiseaseVectorDetection.

Downloads the user-trained classifier from MLflow.
"""
from __future__ import annotations

import json
import logging
import os

import torch
from torch import nn
from torchvision import models

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml10_dairy_disease_vector_detection.constants import (
    CLASSIFIER_FILENAME,
    CLASS_NAMES_FILENAME,
)

logger = logging.getLogger(__name__)


@require_user_model
def download_user_classifier_from_mlflow(run_id: str):
    """Download a user-trained MobileNetV3 classifier from MLflow.

    Returns (model, class_names, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(
        run_id, _load, prefix="mlflow_modelo10_", artifact_path="classifier",
    )
    if loaded is None:
        return None
    (model, class_names), tmp = loaded
    logger.info(
        "Downloaded user classifier from MLflow run_id=%s, classes=%s", run_id, class_names
    )
    return model, class_names, tmp


def _load(local_path: str):
    """Rebuild ``(classifier, class_names)`` from a downloaded ``classifier`` folder."""
    state_dict_path = os.path.join(local_path, CLASSIFIER_FILENAME)
    class_names_path = os.path.join(local_path, CLASS_NAMES_FILENAME)

    with open(class_names_path, encoding="utf-8") as f:
        class_names = json.load(f)

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names
