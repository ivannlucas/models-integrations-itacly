"""Loads ml16 artifacts (per-target XGBoost/LogReg model + scaler + optional bagging +
train_config) via ArtifactStore.
"""
from __future__ import annotations

import json
import logging

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    ARTIFACT_FOLDER_NAME,
    BAGGING_FILENAMES,
    MODEL_FILENAMES,
    SCALER_FILENAMES,
    TARGETS,
    TRAIN_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_artifacts() -> dict:
    """Load models, scalers, optional bagging and train_config for both targets.

    Bagging is optional at inference time (predictor.py::load_model degrades gracefully): if a
    bagging_target_*.joblib is missing or fails to load, that target's uncertainty range
    collapses to proba_low == proba_high == proba.
    """
    with open(_store.path(TRAIN_CONFIG_FILENAME), encoding="utf-8") as fh:
        train_config = json.load(fh)

    models: dict = {}
    scalers: dict = {}
    bagging_models: dict = {}
    for target in TARGETS:
        models[target] = joblib.load(_store.path(MODEL_FILENAMES[target]))
        scalers[target] = joblib.load(_store.path(SCALER_FILENAMES[target]))
        try:
            bagging_models[target] = joblib.load(_store.path(BAGGING_FILENAMES[target]))
        except FileNotFoundError:
            bagging_models[target] = []
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("No se pudo cargar bagging para %s: %s", target, exc)
            bagging_models[target] = []

    logger.info("ml16 artifacts loaded — targets=%s", list(models))
    return {
        "models": models,
        "scalers": scalers,
        "bagging_models": bagging_models,
        "train_config": train_config,
    }
