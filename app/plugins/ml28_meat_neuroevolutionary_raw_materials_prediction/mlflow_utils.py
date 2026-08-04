"""MLflow helper for ml28 — included for repo convention (every plugin has this file), but this
model has nothing to download from MLflow: training.supported=false (see inbox/a28/manifest.yaml
— there are no trainable weights, the decision engine is a deterministic rules function over
fixed config constants), so no per-user retrained artifact can ever exist in MLflow for this
model_id. download_user_model_from_mlflow() always returns None; callers fall back to the fixed
rules engine, which is the only thing this plugin ever serves.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """No-op: ml28 has no trainable artifact, so there is nothing to fetch from MLflow."""
    logger.warning(
        "mlflow_run_id=%s ignored — ml28 has no trained artifact (training.supported=false), "
        "serving the fixed rules engine.",
        run_id,
    )
    return None
