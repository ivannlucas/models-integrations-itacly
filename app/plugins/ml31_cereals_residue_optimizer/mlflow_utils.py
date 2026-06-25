"""MLflow helper for ml31 residue optimizer — download user-trained pipeline from MLflow."""
from __future__ import annotations

import logging

import joblib

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.ml31_cereals_residue_optimizer.constants import MODEL_FILENAME

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download user-trained sklearn Pipeline from MLflow.

    Returns (pipeline, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="model", prefix="mlflow_ml31_")
    if result is None:
        return None
    tmp, local_path = result

    import os  # pylint: disable=import-outside-toplevel
    pipe = joblib.load(os.path.join(local_path, MODEL_FILENAME))

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return pipe, tmp
