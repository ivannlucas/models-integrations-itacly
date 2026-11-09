"""MLflow helpers for ml14 — retrained GRU bundles live only in their MLflow run."""
from __future__ import annotations

import logging
import os
from functools import partial

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    MLFLOW_ARTIFACT_PATH,
    MODEL_FILENAME,
)
from app.plugins.ml14_wine_phyto_price_forecast.model_loader import load_user_bundle

logger = logging.getLogger(__name__)


# None or an artifact load error, with run_id → UserModelUnavailableError (422)
@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a retrained bundle (gru_model.pt + scalers.json) from MLflow.

    Returns (bundle, temp_dir), or None if the run has no complete model.
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    return download_run_artifacts(
        run_id, partial(_load, run_id=run_id), prefix="mlflow_ml14_",
        artifact_path=MLFLOW_ARTIFACT_PATH,
    )


def _load(local_path: str, *, run_id: str) -> dict | None:
    """Load the GRU bundle from a downloaded folder, or None if gru_model.pt is missing.

    A missing scalers.json raises (load_user_bundle); the shared helper removes the temp dir.
    """
    if not os.path.exists(os.path.join(local_path, MODEL_FILENAME)):
        logger.warning(
            "MLflow run_id=%s sin artefacto completo en '%s'", run_id, MLFLOW_ARTIFACT_PATH
        )
        return None
    return load_user_bundle(local_path)


def upload_artifacts_to_mlflow(
    artifact_dir: str, mlflow_run_id: str, metrics: dict, params: dict
) -> None:
    """Log params and final metrics.

    Then upload the retrained bundle under artifact_path="model".
    """
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    tracker.log_params(params)
    tracker.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
    tracker.upload_artifacts(artifact_dir, artifact_path=MLFLOW_ARTIFACT_PATH)
