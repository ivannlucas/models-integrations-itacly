"""Loads ml43 (cereal dryer Deep Neuro-Fuzzy anomaly/fault detector) artifacts via ArtifactStore."""
from __future__ import annotations

import logging
import pickle

import numpy as np
import torch

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.model_arch import (
    ParallelDeepNeuroFuzzyModel,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.preprocess import (
    SENSOR_COLUMNS,
    STATS_CREATION,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.xai.explainer import DNFLExplainer
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    DEFAULT_MODEL_CFG,
    MODEL_FILENAME,
    SCALER_FILENAME,
    XAI_BACKGROUND_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)

STATS_FEATURE_NAMES = [f"{stat}_{col}" for stat in STATS_CREATION for col in SENSOR_COLUMNS]


def build_model(model_cfg: dict) -> ParallelDeepNeuroFuzzyModel:
    """Instantiate the Deep Neuro-Fuzzy architecture for the given config."""
    return ParallelDeepNeuroFuzzyModel(model_cfg=model_cfg)


def build_explainer(model: ParallelDeepNeuroFuzzyModel, model_cfg: dict) -> DNFLExplainer:
    """Instantiate the CU44 XAI orchestrator (fuzzy + SHAP + corrective actions) for a loaded model."""
    return DNFLExplainer(
        model=model,
        feature_names_stats=STATS_FEATURE_NAMES,
        feature_names_original=SENSOR_COLUMNS,
        stats_creation=STATS_CREATION,
        model_cfg=model_cfg,
    )


def load_artifacts():
    """Load the trained checkpoint, scalers, SHAP background windows and XAI explainer.

    Returns (model, model_cfg, scaler_x, scaler_num, xai_background, explainer).

    Raises FileNotFoundError if the model checkpoint or scaler is missing locally and
    STORAGE_BUCKET is not configured to download them.
    """
    checkpoint = torch.load(str(_store.path(MODEL_FILENAME)), map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("model_cfg", DEFAULT_MODEL_CFG)

    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(_store.path(SCALER_FILENAME), "rb") as f:
        scaler_dict = pickle.load(f)
    scaler_x = scaler_dict.get("scaler_x")
    scaler_num = scaler_dict.get("scaler_num")

    xai_background = None
    try:
        xai_background = np.load(str(_store.path(XAI_BACKGROUND_FILENAME)))
    except FileNotFoundError:
        logger.warning("xai_background.npy not found — XAI explanations will be unavailable.")

    explainer = build_explainer(model, model_cfg)

    logger.info(
        "ml43 artifacts loaded (n_rules=%d, hidden=%d, bidir=%s)",
        model_cfg["fuzzy"]["n_rules"], model_cfg["lstm"]["hidden_size"], model_cfg["lstm"]["bidirectional"],
    )
    return model, model_cfg, scaler_x, scaler_num, xai_background, explainer
