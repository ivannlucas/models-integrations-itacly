"""MLflow helper for Modelo10Lacteo — download user-trained classifier from MLflow."""
from __future__ import annotations

import json
import logging
import os

import torch
from torch import nn
from torchvision import models

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.modelo10_lacteo.constants import (
    CLASSIFIER_FILENAME,
    CLASS_NAMES_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_classifier_from_mlflow(run_id: str):
    """Download a user-trained MobileNetV3 classifier from MLflow.

    Returns (model, class_names, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="classifier", prefix="mlflow_modelo10_")
    if result is None:
        return None
    tmp, local_path = result

    state_dict_path = os.path.join(local_path, CLASSIFIER_FILENAME)
    class_names_path = os.path.join(local_path, CLASS_NAMES_FILENAME)

    with open(class_names_path) as f:
        class_names = json.load(f)

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Downloaded user classifier from MLflow run_id=%s, classes=%s", run_id, class_names)
    return model, class_names, tmp
