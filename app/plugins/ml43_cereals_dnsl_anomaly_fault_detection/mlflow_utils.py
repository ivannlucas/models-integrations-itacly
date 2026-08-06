"""MLflow helpers for ml43 — download/upload user-trained model bundles."""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
    MODEL_ID,
    SCALER_FILENAME,
    XAI_BACKGROUND_FILENAME,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.model_loader import (
    build_explainer,
    build_model,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-trained model bundle from MLflow.

    Returns (model, model_cfg, scaler_x, scaler_num, xai_background, explainer, temp_dir),
    or None if the run has no artifacts. Caller MUST shutil.rmtree(temp_dir) after
    inference — use try/finally.
    """
    import tempfile

    tmp = tempfile.mkdtemp(prefix="mlflow_ml43_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    checkpoint = torch.load(
        os.path.join(local_path, MODEL_FILENAME), map_location="cpu", weights_only=False,
    )
    model_cfg = checkpoint["model_cfg"]
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(os.path.join(local_path, SCALER_FILENAME), "rb") as f:
        scaler_dict = pickle.load(f)

    xai_background = None
    bg_path = os.path.join(local_path, XAI_BACKGROUND_FILENAME)
    if os.path.exists(bg_path):
        xai_background = np.load(bg_path)

    explainer = build_explainer(model, model_cfg)

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, model_cfg, scaler_dict.get("scaler_x"), scaler_dict.get("scaler_num"), xai_background, explainer, tmp


def upload_artifacts_to_mlflow(artifact_dir: str, mlflow_run_id: str = "", metrics: dict | None = None) -> str:
    """Upload training artifacts to MLflow and return the run_id.

    If mlflow_run_id is provided, logs to that existing run. Otherwise starts a new
    run under the ml43 experiment.
    """
    import mlflow

    mlflow.set_tracking_uri(BaseMLflowTracker.TRACKING_URI)

    run_id = mlflow_run_id
    if not run_id:
        mlflow.set_experiment(ARTIFACT_FOLDER_NAME)
        with mlflow.start_run() as run:
            run_id = run.info.run_id

    tracker = BaseMLflowTracker(run_id)
    tracker.connect(run_id)

    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": MODEL_ID})
    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow run_id=%s", run_id)
    return run_id
