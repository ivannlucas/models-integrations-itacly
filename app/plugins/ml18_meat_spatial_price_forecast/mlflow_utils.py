"""MLflow helper for ml18 — download user-trained model from MLflow.

ml18 has no retraining path today: train() raises TrainingNotSupportedError. The delivered
scripts/train.py (src.main::train()) does not accept any client-supplied data — it always
reads the fixed dataset_path from config.yaml (see inbox/a18/manifest.yaml::training). This
file exists to satisfy the repo-wide "every plugin ships mlflow_utils.py" convention.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Return None — no user-trained model format exists for ml18 (train() is unsupported)."""
    logger.warning(
        "ml18 has no retraining path — ignoring mlflow_run_id=%s and using the fixed artifact.",
        run_id,
    )
