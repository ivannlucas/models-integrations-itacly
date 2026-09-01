"""MLflow helper for ml23 — download user-trained model from MLflow.

ml23 has no retraining path today: train() raises TrainingNotSupportedError, and the
model is trained externally (a23-rnn-dairy-prediccion, src/training/compare_models.py) —
there is no user-trained-artifact format defined for this plugin to download and load. This
file exists to satisfy the repo-wide "every plugin ships mlflow_utils.py" convention (see
inbox/a23/manifest.yaml known_issues) so a future real fine-tuning implementation has a
consistent place to land, without inventing a download format that has no caller today.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Return None — no user-trained model format exists for ml23 (train() is unsupported).

    Kept for interface consistency with every other plugin's mlflow_utils.py; predict_inline/
    predict_batch never call this because mlflow_run_id has no effect on ml23 (there is no
    user-retrained artifact it could ever point to).
    """
    logger.warning(
        "ml23 has no retraining path — ignoring mlflow_run_id=%s and using the fixed artifact.",
        run_id,
    )
