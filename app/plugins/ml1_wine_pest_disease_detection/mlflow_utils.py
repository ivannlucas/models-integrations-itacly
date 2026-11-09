"""MLflow helpers for ml1 — only the 2 RandomForest tabular experts are retrainable (see
inbox/a01/manifest.yaml training.scope_note): the CNN Deep Ensemble and YOLO detector are served
as-is from the fixed base artifact and are never touched by train()/mlflow_run_id.
"""
from __future__ import annotations

import logging
import os

import joblib

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model

logger = logging.getLogger(__name__)

RF_RIESGO_MLFLOW_FILENAME = "rf_riesgo_expert.joblib"


# None or an artifact load error, with run_id -> UserModelUnavailableError (422)
@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download the user-retrained RandomForest risk regressor from MLflow (the only RF the
    decoupled production mode uses — runs from before this change also carry an RF classifier,
    which is ignored).

    Returns ({"rf_riesgo": ...}, temp_dir), or None if the run has no regressor. Caller MUST
    remove temp_dir after use (``removing_dir``).
    """
    return download_run_artifacts(
        run_id, lambda local_path: _load(local_path, run_id), prefix="mlflow_ml1_"
    )


def _load(local_path: str, run_id: str) -> dict | None:
    """Load the RF risk regressor from a downloaded ``model`` folder (None if it is missing)."""
    riesgo_path = os.path.join(local_path, RF_RIESGO_MLFLOW_FILENAME)
    if not os.path.exists(riesgo_path):
        logger.warning("MLflow run_id=%s sin %s en 'model'", run_id, RF_RIESGO_MLFLOW_FILENAME)
        return None
    return {"rf_riesgo": joblib.load(riesgo_path)}


def log_run_metadata(tracker: BaseMLflowTracker, metrics: dict) -> None:
    """Log the numeric training metrics and the model_id tag of a retrain run."""
    tracker.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.set_tags({"model_id": "ml1-wine-pest-disease-detection"})


def upload_artifacts_to_mlflow(artifact_dir: str, mlflow_run_id: str, metrics: dict) -> None:
    """Log metrics and upload the retrained RF risk regressor to MLflow.

    Uploaded under artifact_path="model".
    """
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    log_run_metadata(tracker, metrics)
    tracker.upload_artifacts(artifact_dir, artifact_path="model")
