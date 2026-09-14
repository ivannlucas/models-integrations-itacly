"""MLflow helper for ml15 — download a user-retrained Ridge artifact from MLflow."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

import joblib

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml15_wine_ipi_price_forecast.constants import MODEL_FILENAME

logger = logging.getLogger(__name__)

_REQUIRED_PAYLOAD_KEYS = ("model", "feature_columns", "target_column", "anchor_column")


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained artifact payload from MLflow.

    Returns (payload, temp_dir), or None if the download fails or the artifact is incomplete.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml15_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    artifact_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(artifact_path):
        logger.error("MLflow run %s does not contain %s", run_id, MODEL_FILENAME)
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    payload = joblib.load(artifact_path)
    missing = [key for key in _REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        logger.error("MLflow run %s artifact is missing keys: %s", run_id, missing)
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return payload, tmp
