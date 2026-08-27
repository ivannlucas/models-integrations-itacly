"""MLflow helper for ml33 cereal reuse-strategy optimizer.

Mandatory by repo convention even though this model has NO user retraining
(training.supported=false, see inbox manifest.yaml) and NO serialized artifact of any
kind — the deployed engine is a deterministic MILP solver (see optimizer.py). There is
nothing to fetch from MLflow, so this always returns None; predict/stats always use the
fixed, self-contained engine. Kept for interface uniformity across the plugins.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Always returns None: ml33 has no trainable/serialized artifact to fetch from MLflow."""
    if run_id:
        logger.warning(
            "mlflow_run_id=%s provided but ml33 is a deterministic MILP optimizer with no "
            "user-trained artifact; ignoring and using the fixed engine.",
            run_id,
        )
    return None
