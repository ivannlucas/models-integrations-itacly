"""MLflow helper for ml5 cow-behaviour — download a user-retrained SlowFast classifier.

Only the SlowFast behaviour CLASSIFIER supports user retraining via ``/train`` (see
training.py / plugin.py::train and inbox/a05/manifest.yaml training section). The
Faster R-CNN DETECTOR always uses the fixed served artifact — there is no user-trained
detector to fetch from MLflow, so this module only ever deals with the classifier.
"""
from __future__ import annotations

import logging
import os

import torch

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml5_meat_cow_behaviour.constants import CLASSIFIER_FILENAME

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str) -> tuple[dict, str] | None:
    """Download a user-retrained SlowFast classifier checkpoint from an MLflow run.

    Returns ``(model_state_dict, tmp_dir)`` on success. With an empty ``run_id`` returns
    ``None`` (the caller serves the base model); if the run has no usable ``classifier``
    artifact, ``require_user_model`` raises ``UserModelUnavailableError`` — the base model
    is never served in its place. Caller MUST remove ``tmp_dir`` once done with it.

    The behaviour-class mapping is never user-trained — it is always
    ``constants.TRAINING_BEHAVIOR_TO_IDX`` (the classifier's 12 classes are fixed by the
    delivered dataset/contract, unlike e.g. modelo10's variable ImageFolder classes) —
    so, unlike ml10_dairy_disease_vector_detection's equivalent helper, there is no separate
    class-names file to download here.
    """

    if not run_id:
        return None

    return download_run_artifacts(
        run_id,
        lambda local_path: _load(local_path, run_id),
        prefix="mlflow_ml5_",
        artifact_path="classifier",
    )


def _load(local_path: str, run_id: str) -> dict | None:
    """Read the classifier state_dict from a downloaded ``classifier`` folder (None if missing)."""
    checkpoint_path = os.path.join(local_path, CLASSIFIER_FILENAME)
    if not os.path.exists(checkpoint_path):
        logger.warning(
            "mlflow_run_id=%s: falta %s en el artefacto 'classifier'.",
            run_id,
            CLASSIFIER_FILENAME,
        )
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return checkpoint.get("model_state_dict", checkpoint)
