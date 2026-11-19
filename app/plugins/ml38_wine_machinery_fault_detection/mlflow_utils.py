"""MLflow helper for ml38 — load a user model retrained by ``train()`` from its MLflow run.

The retrained bundle is uploaded under ``model/`` with the same four file names as the delivered
artifacts, so it is rebuilt with the same loader as the base model.
"""
from __future__ import annotations

import os

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml38_wine_machinery_fault_detection.constants import ARTIFACT_FILENAMES
from app.plugins.ml38_wine_machinery_fault_detection.model_loader import load_bundle_from


@require_user_model   # None or an artifact error with a run_id → UserModelUnavailableError (422)
def download_user_model_from_mlflow(run_id: str):
    """Return (bundle, temp_dir). Caller MUST remove temp_dir (``removing_dir``)."""
    return download_run_artifacts(run_id, _load, prefix="mlflow_ml38_")


def _load(local_path: str):
    """Rebuild the retrained bundle from the downloaded ``model`` folder (None if incomplete)."""
    if not all(os.path.exists(os.path.join(local_path, name)) for name in ARTIFACT_FILENAMES):
        return None
    return load_bundle_from(lambda name: os.path.join(local_path, name))
