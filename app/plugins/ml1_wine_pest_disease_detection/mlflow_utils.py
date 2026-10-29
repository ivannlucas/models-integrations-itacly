"""MLflow helpers for ml1 — only the 2 RandomForest tabular experts are retrainable (see
inbox/a01/manifest.yaml training.scope_note): the CNN Deep Ensemble and YOLO detector are served
as-is from the fixed base artifact and are never touched by train()/mlflow_run_id.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

import joblib

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model

logger = logging.getLogger(__name__)

RF_RIESGO_MLFLOW_FILENAME = "rf_riesgo_expert.joblib"


@require_user_model  # None or an artifact load error, with run_id -> UserModelUnavailableError (422)
def download_user_model_from_mlflow(run_id: str):
    """Download the user-retrained RandomForest risk regressor from MLflow (the only RF the
    decoupled production mode uses — runs from before this change also carry an RF classifier,
    which is ignored).

    Returns ({"rf_riesgo": ...}, temp_dir), or None if the run has no regressor. Caller MUST
    shutil.rmtree(temp_dir) after use — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml1_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
        if not local_path:
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        riesgo_path = os.path.join(local_path, RF_RIESGO_MLFLOW_FILENAME)
        if not os.path.exists(riesgo_path):
            logger.warning("MLflow run_id=%s sin %s en 'model'", run_id, RF_RIESGO_MLFLOW_FILENAME)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        return {"rf_riesgo": joblib.load(riesgo_path)}, tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def upload_artifacts_to_mlflow(artifact_dir: str, mlflow_run_id: str, metrics: dict) -> None:
    """Log metrics and upload the retrained RF risk regressor to MLflow under artifact_path="model"."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    tracker.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.set_tags({"model_id": "ml1-wine-pest-disease-detection"})
    tracker.upload_artifacts(artifact_dir, artifact_path="model")
