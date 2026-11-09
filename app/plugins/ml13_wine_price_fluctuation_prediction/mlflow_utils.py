"""MLflow helper for ml13 — download a user-retrained model bundle from MLflow."""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    FEATURE_SCHEMA_FILENAME,
    MLFLOW_ARTIFACT_PATH,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.model_loader import load_bundle_from_paths

logger = logging.getLogger(__name__)


# None or an artifact load error, with run_id → UserModelUnavailableError (422)
@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained bundle (model, scaler, feature_schema, model_config) from MLflow.

    The model may be LogisticRegression or XGBoost — the retraining procedure re-selects the
    family by Smart Score, so model_config.json (model_type) is read, never assumed.

    Returns (bundle, temp_dir), or None if the run has no complete bundle (the decorator turns
    that, and any artifact load error, into UserModelUnavailableError → 422).
    Caller MUST remove temp_dir after inference (``removing_dir``).
    """
    loaded = download_run_artifacts(
        run_id, partial(_load, run_id=run_id), prefix="mlflow_ml13_",
        artifact_path=MLFLOW_ARTIFACT_PATH,
    )
    if loaded is None:
        return None
    bundle, tmp = loaded
    logger.info("User model of MLflow run_id=%s: model_type=%s", run_id, bundle["model_type"])
    return bundle, tmp


def _load(local_path: str, *, run_id: str) -> dict | None:
    """Load the bundle from a downloaded ``model`` folder, or None if a file is missing."""
    root = Path(local_path)
    names = (MODEL_FILENAME, SCALER_FILENAME, FEATURE_SCHEMA_FILENAME, MODEL_CONFIG_FILENAME)
    paths = [root / name for name in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        logger.error("MLflow run %s is missing %s", run_id, missing)
        return None
    return load_bundle_from_paths(*paths)
