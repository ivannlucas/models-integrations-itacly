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
from functools import partial

import joblib

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml17_meat_market_price_analysis.constants import MODEL_FILENAME

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained Ridge pipeline from MLflow.

    Returns (model, temp_dir) or None if the run has no persisted artifact / download fails.
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    if not run_id:
        return None
    return download_run_artifacts(run_id, partial(_load, run_id=run_id), prefix="mlflow_ml17_")


def _load(local_path: str, *, run_id: str):
    """Load the Ridge pipeline from a downloaded ``model`` folder, or None if it is missing."""
    model_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(model_path):
        logger.warning(
            "MLflow run_id=%s has no %s under artifact_path='model'", run_id, MODEL_FILENAME
        )
        return None
    return joblib.load(model_path)
