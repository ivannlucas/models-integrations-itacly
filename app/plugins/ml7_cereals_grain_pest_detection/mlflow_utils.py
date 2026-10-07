"""MLflow helper for ml7 cereal grain pest-detection YOLO — download a user fine-tuned
checkpoint from MLflow / upload a freshly fine-tuned one.

The served checkpoint in artifacts/ml7_cereals_grain_pest_detection/best.pt (mirrored
from S3) is the fixed base model trained by the AI team and is NEVER overwritten by a
user retrain. train() always persists the fine-tuned weights to MLflow instead — predict/
stats then pass that same mlflow_run_id to use the retrained checkpoint.
"""
from __future__ import annotations

import logging
import os

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml7_cereals_grain_pest_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
)

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned YOLO checkpoint from MLflow.

    Returns (model, temp_dir) or None. Caller MUST shutil.rmtree(temp_dir) after
    inference — use try/finally.
    """
    import tempfile  # noqa: PLC0415

    from ultralytics import YOLO  # noqa: PLC0415 — heavy import kept lazy

    tmp = tempfile.mkdtemp(prefix="mlflow_ml7_grain_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    checkpoint_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(checkpoint_path):
        logger.warning(
            "MLflow run_id=%s did not contain %s under artifact_path='model'",
            run_id, MODEL_FILENAME,
        )
        return None

    model = YOLO(checkpoint_path)
    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, tmp


def upload_artifacts_to_mlflow(
    artifact_dir: str, mlflow_run_id: str, metrics: dict | None = None
) -> str:
    """Upload a fine-tuned YOLO checkpoint directory to an existing MLflow run."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)

    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})

    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow run_id=%s", mlflow_run_id)
    return mlflow_run_id
