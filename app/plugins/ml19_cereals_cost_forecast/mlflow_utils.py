"""MLflow helpers for ml19 — a retraining run keeps the full bundle under artifact_path "model".

/train publishes every retraining as the new base version (see plugin.train); when the caller
also passes mlflow_run_id, the same bundle (the 6 .pkl, training_meta.json and the
dataset_v7_fe.csv it was trained on) is uploaded to that run, so a prediction with that
mlflow_run_id is served by that model on its own dataset.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml19_cereals_cost_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    MLFLOW_ARTIFACT_PATH,
    TRAINING_META_FILENAME,
)
from app.plugins.ml19_cereals_cost_forecast.model_loader import load_bundle_from_dir

logger = logging.getLogger(__name__)


@require_user_model   # None or an artifact load error, with run_id → UserModelUnavailableError (422)
def download_user_model_from_mlflow(run_id: str):
    """Download a retrained bundle and return (bundle, temp_dir), or None if the run has none.

    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml19_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
        if not local_path or not os.path.exists(os.path.join(local_path, TRAINING_META_FILENAME)):
            logger.warning("MLflow run_id=%s sin artefacto completo en '%s'", run_id, MLFLOW_ARTIFACT_PATH)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        bundle = load_bundle_from_dir(local_path)
        logger.info("Downloaded user model from MLflow run_id=%s", run_id)
        return bundle, tmp
    except BaseException:   # e.g. a .pkl missing: never leave the temp dir behind
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def upload_artifacts_to_mlflow(artifact_dir: str, mlflow_run_id: str, metrics: dict, params: dict) -> None:
    """Log params and the walk-forward metrics of the selected models, then upload to "model"."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    tracker.log_params(params)
    tracker.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
    tracker.upload_artifacts(artifact_dir, artifact_path=MLFLOW_ARTIFACT_PATH)
