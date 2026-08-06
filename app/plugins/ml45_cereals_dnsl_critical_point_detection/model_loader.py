"""Artifact loading for the m45 Deep Neuro-Fuzzy plugin.

best_dnf_model.pt is a checkpoint dict {model_state_dict, model_cfg} — model_cfg carries
input_features/n_stats_features/sequence_length, so the architecture is reconstructed from the
checkpoint itself rather than from a fixed constant, same as the original src/main.py::get_stats.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.model import (
    ParallelDeepNeuroFuzzyModel,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    DEFAULT_THRESHOLD,
    MODEL_FILENAME,
    SCALER_FILENAME,
    XAI_BACKGROUND_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_artifacts_from_dir(artifact_dir: Path):
    """Load (model, model_cfg, scaler_x, scaler_num, xai_background, threshold) from *artifact_dir*."""
    checkpoint_path = artifact_dir / MODEL_FILENAME
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} must contain model_state_dict and model_cfg.")

    model_cfg = checkpoint.get("model_cfg")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{checkpoint_path} does not contain a valid model_cfg.")

    model = ParallelDeepNeuroFuzzyModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scalers = joblib.load(artifact_dir / SCALER_FILENAME)
    if not isinstance(scalers, dict) or "scaler_x" not in scalers or "scaler_num" not in scalers:
        raise ValueError(f"{artifact_dir / SCALER_FILENAME} must contain scaler_x and scaler_num.")

    bg_path = artifact_dir / XAI_BACKGROUND_FILENAME
    xai_background = np.load(bg_path) if bg_path.exists() else None

    threshold = DEFAULT_THRESHOLD
    training_kwargs = model_cfg.get("training_kwargs", {})
    if isinstance(training_kwargs, dict) and "threshold" in training_kwargs:
        try:
            threshold = float(training_kwargs["threshold"])
        except (TypeError, ValueError):
            pass

    logger.info(
        "m45 artifacts loaded from %s — input_features=%s n_stats_features=%s threshold=%.3f",
        artifact_dir, model_cfg.get("input_features"), model_cfg.get("n_stats_features"), threshold,
    )
    return model, model_cfg, scalers["scaler_x"], scalers["scaler_num"], xai_background, threshold


def load_artifacts():
    return load_artifacts_from_dir(_store.local_dir)
