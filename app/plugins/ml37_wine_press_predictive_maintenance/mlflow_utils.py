"""MLflow helpers for ml37: download a user-retrained LSTM model from MLflow.

A retrained model lives ONLY in its MLflow run, under ``model/``: ``best_model.keras``,
``scaler.json``, ``class_mappings.json``. The base artifacts in
artifacts/ml37_wine_press_predictive_maintenance/ are never written. Uploading a new retrain is
done directly in plugin.py::train() via ``require_train_tracker``/``upload_model_to_mlflow``
(app.domain.services.mlflow_artifacts), same pattern as the rest of the repo's plugins.
"""
from __future__ import annotations

import logging
import os

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASS_MAPPINGS_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Return ``({"model", "scaler", "class_mappings"}, temp_dir)`` for the LSTM saved in
    ``run_id``. ``None`` (→ UserModelUnavailableError, 422) when the run has no complete model.
    Caller MUST remove ``temp_dir`` after use (``removing_dir``)."""
    return download_run_artifacts(
        run_id, lambda local_path: _load(local_path, run_id),
        prefix=f"mlflow_{ARTIFACT_FOLDER_NAME}_",
    )


def _load(local_path: str, run_id: str):
    missing = [
        f for f in (MODEL_FILENAME, SCALER_FILENAME, CLASS_MAPPINGS_FILENAME)
        if not os.path.exists(os.path.join(local_path, f))
    ]
    if missing:
        logger.warning("MLflow run_id=%s sin artefactos completos: faltan %s", run_id, missing)
        return None

    # Imported lazily: the module stays importable (hygiene tests) without building keras.
    # pylint: disable-next=import-outside-toplevel
    from app.plugins.ml37_wine_press_predictive_maintenance import model_loader

    bundle = model_loader.load_model_bundle_from_dir(local_path)
    logger.info("Modelo de usuario descargado de MLflow run_id=%s", run_id)
    return bundle


def log_run_metadata(tracker: BaseMLflowTracker, metrics: dict) -> None:
    """Log the numeric training metrics and the model_id tag of a retrain run."""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update({f"{key}_{k}": v for k, v in value.items() if isinstance(v, (int, float))})
        elif isinstance(value, (int, float)):
            flat[key] = value
    tracker.log_metrics(flat)
    tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
