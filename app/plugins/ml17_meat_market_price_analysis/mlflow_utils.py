"""MLflow helper for ml17 meat market price analysis.

Mandatory by repo convention (plugin-integration SKILL.md). ml17's train() now has a real
reentrenamiento (Ridge pipeline refit, see plugin.py::train) -- the retrained artifact is
persisted exclusively to the caller's MLflow run (never overwriting the fixed S3 artifact
served by model_loader.py), so this module downloads it back for predict_inline/predict_batch/
stats when the caller passes mlflow_run_id.
"""
from __future__ import annotations

import logging
import os
import tempfile

import joblib

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml17_meat_market_price_analysis.constants import MODEL_FILENAME

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained Ridge pipeline from MLflow.

    Returns (model, temp_dir) or None if the run has no persisted artifact / download fails.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    if not run_id:
        return None
    tmp = tempfile.mkdtemp(prefix="mlflow_ml17_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        logger.warning("No user-trained artifact found in MLflow run_id=%s", run_id)
        return None

    model_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(model_path):
        logger.warning(
            "MLflow run_id=%s has no %s under artifact_path='model'", run_id, MODEL_FILENAME
        )
        return None

    model = joblib.load(model_path)
    logger.info("Downloaded user-trained Ridge pipeline from MLflow run_id=%s", run_id)
    return model, tmp
