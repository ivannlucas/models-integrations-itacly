"""MLflow helper for ml13 — download a user-retrained model bundle from MLflow."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    FEATURE_SCHEMA_FILENAME,
    MLFLOW_ARTIFACT_PATH,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.model_loader import load_bundle_from_paths

logger = logging.getLogger(__name__)


@require_user_model   # None or an artifact load error, with run_id → UserModelUnavailableError (422)
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained bundle (model, scaler, feature_schema, model_config) from MLflow.

    The model may be LogisticRegression or XGBoost — the retraining procedure re-selects the
    family by Smart Score, so model_config.json (model_type) is read, never assumed.

    Returns (bundle, temp_dir), or None if the run has no complete bundle (the decorator turns
    that, and any artifact load error, into UserModelUnavailableError → 422).
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml13_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
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
        bundle = load_bundle_from_paths(*paths)
        logger.info(
            "Downloaded user model from MLflow run_id=%s (model_type=%s)", run_id, bundle["model_type"],
        )
        return bundle, tmp
    except BaseException:   # never leave the temp dir behind
        shutil.rmtree(tmp, ignore_errors=True)
        raise
