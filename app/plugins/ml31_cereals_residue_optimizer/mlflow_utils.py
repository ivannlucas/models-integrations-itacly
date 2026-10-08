"""MLflow helper for ml31 cereal residue optimizer.

Mandatory by repo convention even though the v2.0 model is a deterministic LP optimizer
with NO user retraining (training.supported=false, train() answers 501) and NO serialized
weights. Nothing in this repo ever writes reference data to an MLflow run for ml31, so
there is nothing to fetch: predict/stats always use the fixed reference data loaded by
model_loader.py. A run id is ignored without touching MLflow — same behaviour as ml28/ml33.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Always returns None: ml31 has no user-trained artifact to fetch from MLflow."""
    if run_id:
        logger.warning(
            "mlflow_run_id=%s provided but ml31 is a deterministic LP optimizer with no "
            "user-trained artifact; ignoring and using the fixed reference data.",
            run_id,
        )
    return None
