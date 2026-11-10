"""MLflow helper for ml15 — download a user-retrained Ridge artifact from MLflow."""
from __future__ import annotations

import logging
import os
from functools import partial

import joblib

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml15_wine_ipi_price_forecast.constants import MODEL_FILENAME

logger = logging.getLogger(__name__)

_REQUIRED_PAYLOAD_KEYS = ("model", "feature_columns", "target_column", "anchor_column")


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained artifact payload from MLflow.

    Returns (payload, temp_dir), or None if the download fails or the artifact is incomplete.
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    return download_run_artifacts(run_id, partial(_load, run_id=run_id), prefix="mlflow_ml15_")


def _load(local_path: str, *, run_id: str) -> dict | None:
    """Load the payload from a downloaded ``model`` folder, or None if it is incomplete."""
    artifact_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(artifact_path):
        logger.error("MLflow run %s does not contain %s", run_id, MODEL_FILENAME)
        return None

    payload = joblib.load(artifact_path)
    missing = [key for key in _REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        logger.error("MLflow run %s artifact is missing keys: %s", run_id, missing)
        return None
    return payload
