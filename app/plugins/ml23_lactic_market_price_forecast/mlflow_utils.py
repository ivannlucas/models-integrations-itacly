"""MLflow helper for ml23 — download a user-retrained GRU bundle from MLflow / upload a
freshly refit one.

The served artifact in artifacts/ml23_lactic_market_price_forecast/ (gru_model.pt +
rnn_scaler.npz + manifest.json, mirrored from S3) is the fixed base model trained by the
AI team and is NEVER overwritten by a user retrain. train() always persists the refit
bundle to MLflow instead — predict/stats then pass that same mlflow_run_id to use the
retrained model.
"""
from __future__ import annotations

import functools
import json
import logging
import os

import numpy as np
import torch

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml23_lactic_market_price_forecast.constants import ARTIFACT_FOLDER_NAME
from app.plugins.ml23_lactic_market_price_forecast.rnn_models import GRUModel

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained GRU bundle (model + scaler + manifest) from MLflow.

    Returns (model, scaler_mean, scaler_scale, manifest, temp_dir) or None. Caller MUST
    remove temp_dir after use (``removing_dir``).
    """
    loaded = download_run_artifacts(
        run_id, functools.partial(_load, run_id=run_id), prefix="mlflow_ml23_dairy_"
    )
    if loaded is None:
        return None
    (model, scaler_mean, scaler_scale, manifest), tmp = loaded
    return model, scaler_mean, scaler_scale, manifest, tmp


def _load(local_path: str, *, run_id: str):
    """Rebuild (model, scaler_mean, scaler_scale, manifest); None if a file is missing."""
    manifest_path = os.path.join(local_path, "manifest.json")
    model_path = os.path.join(local_path, "gru_model.pt")
    scaler_path = os.path.join(local_path, "rnn_scaler.npz")
    if not all(os.path.exists(p) for p in (manifest_path, model_path, scaler_path)):
        logger.warning(
            "MLflow run_id=%s did not contain a complete GRU bundle under "
            "artifact_path='model'",
            run_id,
        )
        return None

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    bundle = np.load(scaler_path)
    scaler_mean, scaler_scale = bundle["mean"], bundle["scale"]

    model = GRUModel(
        input_size=len(manifest["feature_cols"]), hidden_size=int(manifest["hidden_size"])
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler_mean, scaler_scale, manifest


def upload_artifacts_to_mlflow(
    artifact_dir: str, mlflow_run_id: str, metrics: dict | None = None,
) -> str:
    """Upload a refit GRU bundle directory to an existing MLflow run."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)

    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})

    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow run_id=%s", mlflow_run_id)
    return mlflow_run_id
