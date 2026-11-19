"""MLflow helper for ml39 — included for repo convention (every plugin has this file), but this
plugin has nothing to download from MLflow: training.supported=false (see
inbox/a39/manifest.yaml) and, unlike ml2/ml4/ml7's bug history, there is no alternative
retrained model to ignore — a39 explains whichever model ml37 serves (always the base: the
LSTM multitask model, scaler and class mappings are a39's own copy of ml37's artifacts).
download_user_model_from_mlflow() always returns None; callers fall back to the base bundle,
which is the only thing this plugin can ever explain.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """No-op: ml39 has no trainable artifact of its own (training.supported=false), so there is
    nothing to fetch from MLflow — it always explains the base ml37 model."""
    logger.warning(
        "mlflow_run_id=%s ignored — ml39 has no trainable artifact (training.supported=false), "
        "explaining the base model.",
        run_id,
    )
