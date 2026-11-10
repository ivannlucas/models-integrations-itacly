"""MLflow helper for ml3 — download a user-retrained ensemble bundle from MLflow."""
from __future__ import annotations

import logging

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml3_wine_disease_pest_forecast.model_loader import load_user_artifacts

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained LSTM/CNN/BiGRU bundle plus scaler and label_encoder.

    Returns (models, scaler, label_encoder, class_names, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(run_id, load_user_artifacts, prefix="mlflow_ml3_")
    if loaded is None:
        return None
    (models, scaler, le, class_names), tmp = loaded
    logger.info("User model from MLflow run_id=%s has %d classes", run_id, len(class_names))
    return models, scaler, le, class_names, tmp
