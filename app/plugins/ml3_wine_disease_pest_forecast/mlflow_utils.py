"""MLflow helper for ml3 — download a user-retrained ensemble bundle from MLflow."""
from __future__ import annotations

import logging
import tempfile

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml3_wine_disease_pest_forecast.model_loader import load_user_artifacts

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained LSTM/CNN/BiGRU bundle plus scaler and label_encoder.

    Returns (models, scaler, label_encoder, class_names, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml3_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    models, scaler, le, class_names = load_user_artifacts(local_path)
    logger.info(
        "Downloaded user model from MLflow run_id=%s (%d classes)", run_id, len(class_names)
    )
    return models, scaler, le, class_names, tmp
