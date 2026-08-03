"""MLflow helper for m45 — download user-trained model from MLflow / upload fine-tuned artifacts."""
from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.model import (
    ParallelDeepNeuroFuzzyModel,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    DEFAULT_THRESHOLD,
    MODEL_FILENAME,
    SCALER_FILENAME,
    XAI_BACKGROUND_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned m45 model from MLflow.

    Returns (model, model_cfg, scaler_x, scaler_num, xai_background, threshold, temp_dir) or None.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    import tempfile

    tmp = tempfile.mkdtemp(prefix="mlflow_ml45_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    checkpoint_path = os.path.join(local_path, MODEL_FILENAME)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["model_cfg"]

    model = ParallelDeepNeuroFuzzyModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scalers = joblib.load(os.path.join(local_path, SCALER_FILENAME))

    bg_path = os.path.join(local_path, XAI_BACKGROUND_FILENAME)
    xai_background = np.load(bg_path) if os.path.exists(bg_path) else None

    threshold = float(
        model_cfg.get("training_kwargs", {}).get("threshold", DEFAULT_THRESHOLD)
    )

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, model_cfg, scalers["scaler_x"], scalers["scaler_num"], xai_background, threshold, tmp


def upload_artifacts_to_mlflow(
    artifact_dir: str,
    mlflow_run_id: str = "",
    metrics: dict | None = None,
) -> str:
    """Upload fine-tuned artifacts to MLflow and return the run_id."""
    from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import MODEL_ID

    if mlflow_run_id:
        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.connect(mlflow_run_id)
    else:
        import mlflow

        mlflow.set_tracking_uri(BaseMLflowTracker.TRACKING_URI)
        mlflow.set_experiment(ARTIFACT_FOLDER_NAME)
        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id
        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.connect(mlflow_run_id)

    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": MODEL_ID})

    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow run_id=%s", mlflow_run_id)
    return mlflow_run_id
