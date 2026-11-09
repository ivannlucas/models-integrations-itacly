"""MLflow helper for wine sulphite — download user-trained models from MLflow."""
from __future__ import annotations

import json
import os

import joblib

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml25_wine_sulphites.constants import (
    BOUND_RF_MODEL_FILENAME,
    METADATA_FILENAME,
    QUALITY_RF_MODEL_FILENAME,
)


@require_user_model
def download_user_predictor_from_mlflow(run_id: str):
    """Download user-trained RandomForest models from MLflow.

    Returns (model_qual, model_bound, metadata, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(run_id, _load, prefix="mlflow_wine_")
    if loaded is None:
        return None
    (model_qual, model_bound, metadata), tmp = loaded
    return model_qual, model_bound, metadata, tmp


def _load(local_path: str):
    """Rebuild the two RandomForests and their metadata from a downloaded ``model`` folder."""
    qual_path = os.path.join(local_path, QUALITY_RF_MODEL_FILENAME)
    bound_path = os.path.join(local_path, BOUND_RF_MODEL_FILENAME)
    meta_path = os.path.join(local_path, METADATA_FILENAME)

    # joblib.load, not pickle.load: train() writes these files with joblib.dump (same as the
    # base artifacts), whose numpy arrays pickle.load cannot read
    # ("UnpicklingError: STACK_GLOBAL requires str").
    model_qual = joblib.load(qual_path)
    model_bound = joblib.load(bound_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return model_qual, model_bound, metadata
