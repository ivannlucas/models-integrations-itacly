"""MLflow helper for ml19 — download user-trained model from MLflow.

ml19 has no retraining path today: train() raises TrainingNotSupportedError. The delivered
src/training/train.py always reads the fixed data/processed/auto/dataset_v7_fe.csv — it never
accepts client-supplied data (see inbox/a19/manifest.yaml::training). This file exists to
satisfy the repo-wide "every plugin ships mlflow_utils.py" convention.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Return None — no user-trained model format exists for ml19 (train() is unsupported)."""
    logger.warning(
        "ml19 has no retraining path — ignoring mlflow_run_id=%s and using the fixed artifact.",
        run_id,
    )
