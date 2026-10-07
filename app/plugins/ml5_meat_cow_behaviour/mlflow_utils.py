"""MLflow helper for ml5 cow-behaviour — download a user-retrained SlowFast classifier.

Only the SlowFast behaviour CLASSIFIER supports user retraining via ``/train`` (see
training.py / plugin.py::train and inbox/a05/manifest.yaml training section). The
Faster R-CNN DETECTOR always uses the fixed served artifact — there is no user-trained
detector to fetch from MLflow, so this module only ever deals with the classifier.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml5_meat_cow_behaviour.constants import CLASSIFIER_FILENAME

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str) -> tuple[dict, str] | None:
    """Download a user-retrained SlowFast classifier checkpoint from an MLflow run.

    Returns ``(model_state_dict, tmp_dir)`` on success. With an empty ``run_id`` returns
    ``None`` (the caller serves the base model); if the run has no usable ``classifier``
    artifact, ``require_user_model`` raises ``UserModelUnavailableError`` — the base model
    is never served in its place. Caller MUST ``shutil.rmtree(tmp_dir)`` once done with it.

    The behaviour-class mapping is never user-trained — it is always
    ``constants.TRAINING_BEHAVIOR_TO_IDX`` (the classifier's 12 classes are fixed by the
    delivered dataset/contract, unlike e.g. modelo10's variable ImageFolder classes) —
    so, unlike ml10_dairy_disease_vector_detection's equivalent helper, there is no separate class-names file
    to download here.
    """
    if not run_id:
        return None

    tmp_dir = tempfile.mkdtemp(prefix="mlflow_ml5_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp_dir, artifact_path="classifier")
    if not local_path:
        logger.warning(
            "mlflow_run_id=%s no tiene artefacto 'classifier'.", run_id
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    checkpoint_path = os.path.join(local_path, CLASSIFIER_FILENAME)
    if not os.path.exists(checkpoint_path):
        logger.warning(
            "mlflow_run_id=%s: falta %s en el artefacto 'classifier'.", run_id, CLASSIFIER_FILENAME,
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    logger.info("Downloaded user-retrained SlowFast classifier from MLflow run_id=%s", run_id)
    return state_dict, tmp_dir
