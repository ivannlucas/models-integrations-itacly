"""MLflow helper for ml9 — download a user fine-tuned sequence classifier from MLflow.

Mandatory file for every plugin (CLAUDE.md). Here it is also functional: /train supports real
fine-tuning, so a caller can serve their own retrained GRU by passing its mlflow_run_id.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.sequential import load_checkpoint, load_pickle
from app.plugins.ml9_cereals_infestation_sequence_classifier.constants import (
    BUNDLE_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
    USER_BUNDLE_FILENAME,
    USER_MODEL_FILENAME,
    USER_SCALER_FILENAME,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.model_loader import _load_json

logger = logging.getLogger(__name__)


def _first_existing(local_path: str, *filenames: str) -> str | None:
    """Return the first of *filenames* that exists under *local_path*."""
    for name in filenames:
        candidate = os.path.join(local_path, name)
        if os.path.exists(candidate):
            return candidate
    return None


def download_user_model_from_mlflow(run_id: str) -> tuple[dict, Any, dict, str] | None:
    """Download a user fine-tuned checkpoint + scaler + bundle from MLflow.

    Returns (checkpoint, scaler, bundle, temp_dir) or None if the download failed.
    The caller MUST ``shutil.rmtree(temp_dir, ignore_errors=True)`` afterwards — use try/finally.

    The number of classes and the 65-feature contract are rebuilt from the downloaded artifacts
    (the checkpoint carries its own model_config, the bundle its own feature_columns), so a user
    model retrained on a different class cardinality still loads correctly.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="model", prefix="mlflow_ml9_")
    if result is None:
        return None
    tmp, local_path = result

    model_path = _first_existing(local_path, USER_MODEL_FILENAME, MODEL_FILENAME)
    bundle_path = _first_existing(local_path, USER_BUNDLE_FILENAME, BUNDLE_FILENAME)
    scaler_path = _first_existing(local_path, USER_SCALER_FILENAME, SCALER_FILENAME)
    if not model_path or not bundle_path or not scaler_path:
        logger.error(
            "MLflow run_id=%s does not contain the expected ml9 artifacts (checkpoint/bundle/scaler)",
            run_id,
        )
        return None

    checkpoint = load_checkpoint(model_path, map_location="cpu")
    scaler = load_pickle(scaler_path)
    bundle = _load_json(bundle_path)

    logger.info(
        "Downloaded user model from MLflow run_id=%s — model_type=%s n_features=%d",
        run_id, checkpoint.get("model_type"), len(bundle.get("feature_columns") or []),
    )
    return checkpoint, scaler, bundle, tmp
