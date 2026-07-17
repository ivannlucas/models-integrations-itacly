"""MLflow helper for ml46 — download a user fine-tuned model bundle from MLflow."""
from __future__ import annotations

import logging
import os
from dataclasses import fields

import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import FeatureArtifacts, TrainConfig
from app.plugins.ml46_dairy_fouling_clog_detection.constants import (
    FEATURE_ARTIFACTS_FILENAME,
    MODEL_FILENAME,
    POLICY_THRESHOLDS_FILENAME,
    TRAINING_CONFIG_FILENAME,
)
from app.plugins.ml46_dairy_fouling_clog_detection.model_loader import build_model, _load_json

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user fine-tuned TCN + feature_artifacts + training_config + policy from MLflow.

    Returns (model, train_cfg, feature_artifacts, policy, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mlflow_ml46_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    train_cfg_data = _load_json(os.path.join(local_path, TRAINING_CONFIG_FILENAME))
    train_cfg_data["device"] = "cpu"
    known_cfg = {f.name for f in fields(TrainConfig)}
    train_cfg = TrainConfig(**{k: v for k, v in train_cfg_data.items() if k in known_cfg})

    fa_data = _load_json(os.path.join(local_path, FEATURE_ARTIFACTS_FILENAME))
    known_fa = {f.name for f in fields(FeatureArtifacts)}
    feature_artifacts = FeatureArtifacts(**{k: v for k, v in fa_data.items() if k in known_fa})

    policy = _load_json(os.path.join(local_path, POLICY_THRESHOLDS_FILENAME))

    model = build_model(train_cfg, feature_artifacts)
    state = torch.load(os.path.join(local_path, MODEL_FILENAME), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, train_cfg, feature_artifacts, policy, tmp
