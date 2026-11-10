"""MLflow helper for ml21 ESP-CEREAL — download user-trained models from MLflow."""
from __future__ import annotations

import json
import logging
import os

import joblib

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml21_cereals_price_spatial.constants import (
    METADATA_FILENAME,
    MODEL_H1_CLF,
    MODEL_H1_REG,
    MODEL_H2_CLF,
    MODEL_H2_REG,
    MODEL_H3_CLF,
    MODEL_H3_REG,
)

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download user-trained cereal models from MLflow.

    Returns (models_by_horizon, metadata, temp_dir).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(run_id, _load, prefix="mlflow_ml21_")
    if loaded is None:
        return None
    (models, metadata), tmp = loaded
    return models, metadata, tmp


def _load(local_path: str):
    """Read model_metadata.json and the 6 models; None if a horizon's pair is missing."""
    with open(os.path.join(local_path, METADATA_FILENAME), encoding="utf-8") as fh:
        metadata = json.load(fh)

    model_files = {
        "H1": {"reg": MODEL_H1_REG, "clf": MODEL_H1_CLF},
        "H2": {"reg": MODEL_H2_REG, "clf": MODEL_H2_CLF},
        "H3": {"reg": MODEL_H3_REG, "clf": MODEL_H3_CLF},
    }

    models = {}
    for h_key, files in model_files.items():
        reg_path = os.path.join(local_path, files["reg"])
        clf_path = os.path.join(local_path, files["clf"])

        if not os.path.exists(reg_path) or not os.path.exists(clf_path):
            logger.warning("Modelos user no encontrados para %s en %s", h_key, local_path)
            return None

        models[h_key] = {
            "reg": joblib.load(reg_path),
            "clf": joblib.load(clf_path),
        }
    return models, metadata
