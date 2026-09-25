"""MLflow helper for ml14 — download user-trained model from MLflow.

ml14 has no retraining path today: train() raises TrainingNotSupportedError. The delivered
scripts/train.py (src/training/compare_models.py::main()) does not accept any client-supplied
data — it always re-reads the fixed bundled dataset and only exposes hyperparameter flags — so
there is no user-trained-artifact format this plugin could ever download and load (see
inbox/a14/manifest.yaml::training). This file exists to satisfy the repo-wide "every plugin
ships mlflow_utils.py" convention, so a future real fine-tuning implementation has a
consistent place to land, without inventing a download format that has no caller today.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Return None — no user-trained model format exists for ml14 (train() is unsupported).

    Kept for interface consistency with every other plugin's mlflow_utils.py; predict_inline/
    predict_batch never call this because mlflow_run_id has no effect on ml14 (there is no
    user-retrained artifact it could ever point to).
    """
    logger.warning(
        "ml14 has no retraining path — ignoring mlflow_run_id=%s and using the fixed artifact.",
        run_id,
    )
