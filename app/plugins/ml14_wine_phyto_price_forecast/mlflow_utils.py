"""MLflow helpers for ml14 — retrained GRU bundles live only in their MLflow run."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    MLFLOW_ARTIFACT_PATH,
    MODEL_FILENAME,
)
from app.plugins.ml14_wine_phyto_price_forecast.model_loader import load_user_bundle

logger = logging.getLogger(__name__)


@require_user_model   # None or an artifact load error, with run_id → UserModelUnavailableError (422)
def download_user_model_from_mlflow(run_id: str):
    """Download a retrained bundle (gru_model.pt + scalers.json) from MLflow.

    Returns (bundle, temp_dir), or None if the run has no complete model.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml14_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
        if not local_path or not os.path.exists(os.path.join(local_path, MODEL_FILENAME)):
            logger.warning("MLflow run_id=%s sin artefacto completo en '%s'", run_id, MLFLOW_ARTIFACT_PATH)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        bundle = load_user_bundle(local_path)
        logger.info("Downloaded user model from MLflow run_id=%s", run_id)
        return bundle, tmp
    except BaseException:   # e.g. a missing scalers.json: never leave the temp dir behind
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def upload_artifacts_to_mlflow(artifact_dir: str, mlflow_run_id: str, metrics: dict, params: dict) -> None:
    """Log params and final metrics, then upload the retrained bundle under artifact_path="model"."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    tracker.log_params(params)
    tracker.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
    tracker.upload_artifacts(artifact_dir, artifact_path=MLFLOW_ARTIFACT_PATH)
