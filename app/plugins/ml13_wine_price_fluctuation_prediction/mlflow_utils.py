"""MLflow helper for ml13 — download a user-retrained model bundle from MLflow."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    FEATURE_SCHEMA_FILENAME,
    MLFLOW_ARTIFACT_PATH,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.model_loader import load_bundle_from_paths

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained bundle (model, scaler, feature_schema, model_config) from MLflow.

    The model may be LogisticRegression or XGBoost — the retraining procedure re-selects the
    family by Smart Score, so model_config.json (model_type) is read, never assumed.

    Returns (bundle, temp_dir), or None if the download fails or the bundle is incomplete.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml13_")
    tracker = BaseMLflowTracker(run_id)
    local_path = tracker.download_artifacts(tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
    if not local_path:
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    root = Path(local_path)
    names = (MODEL_FILENAME, SCALER_FILENAME, FEATURE_SCHEMA_FILENAME, MODEL_CONFIG_FILENAME)
    paths = [root / name for name in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        logger.error("MLflow run %s is missing %s", run_id, missing)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    try:
        bundle = load_bundle_from_paths(*paths)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("MLflow run %s bundle could not be loaded: %s", run_id, exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    logger.info(
        "Downloaded user model from MLflow run_id=%s (model_type=%s)", run_id, bundle["model_type"],
    )
    return bundle, tmp
