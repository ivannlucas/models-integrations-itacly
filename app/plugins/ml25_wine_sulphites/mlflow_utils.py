"""MLflow helper for wine sulphite — download user-trained models from MLflow."""
from __future__ import annotations

import json
import logging
import os
import pickle

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.ml25_wine_sulphites.constants import (
    BOUND_RF_MODEL_FILENAME,
    METADATA_FILENAME,
    QUALITY_RF_MODEL_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_predictor_from_mlflow(run_id: str):
    """Download user-trained RandomForest models from MLflow.

    Returns (model_qual, model_bound, metadata, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="model", prefix="mlflow_wine_")
    if result is None:
        return None
    tmp, local_path = result

    qual_path = os.path.join(local_path, QUALITY_RF_MODEL_FILENAME)
    bound_path = os.path.join(local_path, BOUND_RF_MODEL_FILENAME)
    meta_path = os.path.join(local_path, METADATA_FILENAME)

    with open(qual_path, "rb") as f:
        model_qual = pickle.load(f)
    with open(bound_path, "rb") as f:
        model_bound = pickle.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info("Downloaded user predictor from MLflow run_id=%s", run_id)
    return model_qual, model_bound, metadata, tmp
