"""MLflow helper for ml16 — download a user-retrained model bundle from MLflow."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile

import joblib

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    BAGGING_FILENAMES,
    MODEL_FILENAMES,
    SCALER_FILENAMES,
    TARGETS,
    TRAIN_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained bundle (both targets) from MLflow.

    Returns (models, scalers, bagging_models, train_config, temp_dir), or None if the download
    fails or the bundle is incomplete. Caller MUST shutil.rmtree(temp_dir) after inference —
    use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml16_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    config_path = os.path.join(local_path, TRAIN_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        logger.error("MLflow run %s does not contain %s", run_id, TRAIN_CONFIG_FILENAME)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    with open(config_path, encoding="utf-8") as fh:
        train_config = json.load(fh)

    models: dict = {}
    scalers: dict = {}
    bagging_models: dict = {}
    for target in TARGETS:
        model_path = os.path.join(local_path, MODEL_FILENAMES[target])
        scaler_path = os.path.join(local_path, SCALER_FILENAMES[target])
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            logger.error("MLflow run %s is missing model/scaler for %s", run_id, target)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        models[target] = joblib.load(model_path)
        scalers[target] = joblib.load(scaler_path)
        bagging_path = os.path.join(local_path, BAGGING_FILENAMES[target])
        bagging_models[target] = joblib.load(bagging_path) if os.path.exists(bagging_path) else []

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return models, scalers, bagging_models, train_config, tmp
