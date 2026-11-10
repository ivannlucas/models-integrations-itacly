"""MLflow helpers for ml18 — retrained GRU bundles live only in their MLflow run."""
from __future__ import annotations

import functools
import logging
import os

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml18_meat_spatial_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    MLFLOW_ARTIFACT_PATH,
    MODEL_FILENAME,
)
from app.plugins.ml18_meat_spatial_price_forecast.model_loader import load_bundle_from_dir

logger = logging.getLogger(__name__)


# None or an artifact load error, with run_id → UserModelUnavailableError (422)
@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained bundle (model.joblib + x/y scalers) from MLflow.

    Returns (bundle, temp_dir), or None if the run has no complete model.
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    return download_run_artifacts(
        run_id, functools.partial(_load, run_id=run_id),
        prefix="mlflow_ml18_", artifact_path=MLFLOW_ARTIFACT_PATH,
    )


def _load(local_path: str, *, run_id: str) -> dict | None:
    """Rebuild the bundle from a downloaded folder; None if it has no model file.

    A missing scaler raises (the shared helper removes the temp dir and re-raises).
    """
    if not os.path.exists(os.path.join(local_path, MODEL_FILENAME)):
        logger.warning(
            "MLflow run_id=%s sin artefacto completo en '%s'", run_id, MLFLOW_ARTIFACT_PATH
        )
        return None
    return load_bundle_from_dir(local_path)


def upload_artifacts_to_mlflow(
    artifact_dir: str, mlflow_run_id: str, metrics: dict, params: dict,
    history: list[tuple[int, dict]] | None = None,
) -> None:
    """Log params, per-epoch Keras logs and final metrics, then upload the bundle to "model"."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    tracker.log_params(params)
    for epoch, logs in history or []:
        tracker.log_metrics(logs, step=epoch)
    tracker.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
    tracker.upload_artifacts(artifact_dir, artifact_path=MLFLOW_ARTIFACT_PATH)
