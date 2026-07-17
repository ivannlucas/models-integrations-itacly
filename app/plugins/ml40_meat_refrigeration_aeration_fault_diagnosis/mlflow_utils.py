"""MLflow helper for ml40 — download a user-retrained model bundle from MLflow."""
from __future__ import annotations

import logging
import os

import joblib
import yaml

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    MODEL_FILENAMES,
    SCALER_FILENAMES,
    STATS_FILENAMES,
    SYSTEMS,
    THRESHOLDS_FILENAMES,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained RandomForest bundle (one system) from MLflow.

    The bundle uploaded by train() uses the canonical artifact filenames
    ({system}_model.pkl, plus scaler/thresholds/stats when they apply), so the system is
    inferred from which model file is present.

    Returns (system, bundle_dict, temp_dir) or None if the download fails.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mlflow_ml40_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    system = next(
        (s for s in SYSTEMS if os.path.exists(os.path.join(local_path, MODEL_FILENAMES[s]))),
        None,
    )
    if system is None:
        logger.error("MLflow run %s does not contain any ml40 model file", run_id)
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        return None

    def _yaml(filename: str) -> dict:
        path = os.path.join(local_path, filename)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    bundle = {
        "model": joblib.load(os.path.join(local_path, MODEL_FILENAMES[system])),
        "scaler": None,
        "thresholds": _yaml(THRESHOLDS_FILENAMES[system]),
        "stats": _yaml(STATS_FILENAMES[system]),
    }
    scaler_name = SCALER_FILENAMES.get(system)
    if scaler_name and os.path.exists(os.path.join(local_path, scaler_name)):
        bundle["scaler"] = joblib.load(os.path.join(local_path, scaler_name))

    logger.info("Downloaded user model from MLflow run_id=%s (system=%s)", run_id, system)
    return system, bundle, tmp
