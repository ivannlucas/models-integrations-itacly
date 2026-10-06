"""MLflow helper for ml4 thermal mastitis-detection CNN — download a user-retrained model.

Now that train() implements a real fine-tuning procedure (see plugin.py and
inbox/a04/manifest.yaml::training), a user-retrained checkpoint lives only under its MLflow
run (never overwriting the fixed base artifact under
artifacts/ml4_lactic_cnn_thermal_early_disease_detection/). predict_inline/predict_batch swap
this model in for the duration of the request when ``mlflow_run_id`` is provided.
"""
from __future__ import annotations

import logging
import os
import tempfile

import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.constants import (
    BACKBONE,
    DROPOUT,
    MODEL_FILENAME,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-retrained BaselineModel checkpoint from MLflow run ``run_id``.

    Returns ``(model, device, temp_dir)`` or ``None`` if the run has no model artifact.
    Caller MUST ``shutil.rmtree(temp_dir)`` after inference — use try/finally.
    """
    if not run_id:
        return None

    tmp = tempfile.mkdtemp(prefix="mlflow_ml4_thermal_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        logger.warning("mlflow run_id=%s has no 'model' artifact path", run_id)
        return None

    checkpoint_path = os.path.join(local_path, MODEL_FILENAME)
    if not os.path.exists(checkpoint_path):
        logger.warning(
            "mlflow run_id=%s: %s not found under %s", run_id, MODEL_FILENAME, local_path
        )
        return None

    # Imported lazily to avoid a hard torch/timm dependency at module import time for callers
    # that never retrain (mirrors ml8_cereals_img_anomaly_detector's mlflow_utils.py).
    from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.model_loader import (
        BaselineModel,
        _safe_device,
    )

    device = _safe_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model = BaselineModel(
        backbone=BACKBONE, pretrained=False, num_classes=NUM_CLASSES, dropout=DROPOUT,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, device, tmp
