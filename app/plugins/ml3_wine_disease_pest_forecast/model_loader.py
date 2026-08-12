"""Model loading for the ml3 wine disease/pest Deep Ensemble plugin.

Resolves the three Keras checkpoints, the StandardScaler and the LabelEncoder
via :class:`ArtifactStore` — the fixed S3 artifacts are never mutated.
"""
from __future__ import annotations

import logging
from typing import Any

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml3_wine_disease_pest_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    LABEL_ENCODER_FILENAME,
    MODEL_FILENAMES,
    SCALER_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _load_models(paths: list) -> list[Any]:
    """Load a list of Keras ``.keras`` checkpoints (self-contained architecture)."""
    # TensorFlow is imported lazily so the plugin modules stay light for unit tests.
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from tensorflow.keras.models import load_model

    return [load_model(_store.path(name)) for name in paths]


def load_artifacts() -> tuple[list[Any], Any, Any, list[str]]:
    """Load the fixed ensemble bundle.

    Returns (models, scaler, label_encoder, class_names).
    """
    models = _load_models(MODEL_FILENAMES)
    scaler = joblib.load(_store.path(SCALER_FILENAME))
    le = joblib.load(_store.path(LABEL_ENCODER_FILENAME))
    class_names = list(le.classes_)
    logger.info(
        "ml3 artifacts loaded — %d models, %d classes",
        len(models), len(class_names),
    )
    return models, scaler, le, class_names


def load_user_artifacts(local_path: str) -> tuple[list[Any], Any, Any, list[str]]:
    """Load a user-retrained bundle from an MLflow download directory.

    The class set is read dynamically from the user's own label_encoder, so the
    ensemble output dimension matches the retrained models.
    """
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from tensorflow.keras.models import load_model

    models = [load_model(f"{local_path}/{name}") for name in MODEL_FILENAMES]
    scaler = joblib.load(f"{local_path}/{SCALER_FILENAME}")
    le = joblib.load(f"{local_path}/{LABEL_ENCODER_FILENAME}")
    return models, scaler, le, list(le.classes_)
