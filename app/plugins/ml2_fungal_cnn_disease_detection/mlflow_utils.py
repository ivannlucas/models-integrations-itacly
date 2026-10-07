"""MLflow helper for the fungal leaf-disease CNN (LeafCNN) — download a user-trained
model from MLflow.

Mandatory by repo convention. Now that ``train()`` is real (see ``plugin.py``), this is
actually exercised: every retrain is uploaded only to its own MLflow run (never to the
fixed S3/local base artifact), so serving a user-trained model means downloading it back
from here. The number of classes can differ between retrains (a user ZIP may not cover
all 5 original classes, or may add new ones), so the architecture is reconstructed from the
checkpoint's own class list instead of assuming ``constants.CLASS_NAMES``.
"""
from __future__ import annotations

import logging
import os
import tempfile

import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml2_fungal_cnn_disease_detection.constants import IMAGE_SIZE, MODEL_FILENAME

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str) -> tuple[dict, str] | None:
    """Download a user-trained LeafCNN checkpoint from MLflow run ``run_id``.

    Returns ``(bundle, temp_dir)`` — the same shape as ``model_loader.load_model_bundle()`` —
    or ``None`` if the run has no ``model`` artifacts (not found / not yet trained).
    Caller MUST ``shutil.rmtree(temp_dir)`` after use (see ``plugin.py``'s try/finally).
    """
    # pylint: disable=import-outside-toplevel
    from app.plugins.ml2_fungal_cnn_disease_detection.model_loader import LeafCNN, _safe_device
    # pylint: enable=import-outside-toplevel

    tmp = tempfile.mkdtemp(prefix="mlflow_ml2_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        logger.warning("No model artifacts found in MLflow run_id=%s", run_id)
        return None

    checkpoint_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(checkpoint_path):
        logger.warning(
            "MLflow run_id=%s has no %s under its model artifacts", run_id, MODEL_FILENAME
        )
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes: list[str] = checkpoint["classes"]

    device = _safe_device()
    model = LeafCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bundle = {
        "model_id": checkpoint.get("model_id", "ml2-fungal-cnn-disease-detection"),
        "model": model,
        "device": device,
        "image_size": checkpoint.get("image_size", IMAGE_SIZE),
        "classes": classes,
    }

    logger.info("Downloaded user model from MLflow run_id=%s (classes=%s)", run_id, classes)
    return bundle, tmp
