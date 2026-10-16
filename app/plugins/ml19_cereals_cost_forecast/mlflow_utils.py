"""MLflow helper for ml19 — download user-trained model from MLflow.

ml19 has no retraining path today: train() raises TrainingNotSupportedError. The delivered
src/training/train.py always reads the fixed data/processed/auto/dataset_v7_fe.csv — it never
accepts client-supplied data (see inbox/a19/manifest.yaml::training). This file exists to
satisfy the repo-wide "every plugin ships mlflow_utils.py" convention.

Repo review finding (PR #20): returning None unconditionally let plugin.py silently ignore any
mlflow_run_id and serve the base model with HTTP 200, even for a run_id that does not exist.
@require_user_model turns that None into UserModelUnavailableError (-> 422) whenever the caller
actually supplied a non-empty run_id — exactly right for ml19, since no run_id is ever valid
here.
"""
from __future__ import annotations

import logging

from app.domain.services.mlflow_tracker import require_user_model

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Return None — no user-trained model format exists for ml19 (train() is unsupported).

    @require_user_model converts that into UserModelUnavailableError when run_id is non-empty.
    """
    logger.warning(
        "ml19 has no retraining path — mlflow_run_id=%s cannot be served.", run_id,
    )
