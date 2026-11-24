"""Artifact loading for ml38 — the four delivered pickles, served bit for bit.

The same bundle layout is used for the base artifacts (ArtifactStore, local copy or S3) and for a
user model retrained into MLflow, so both go through :func:`load_bundle_from`.

Keras is never imported at module level: ml1 sets ``KERAS_BACKEND=torch`` when registry.py
imports it, and importing keras earlier would freeze another backend for the whole process.
Unpickling ``severity_models.pkl`` imports keras lazily, after every plugin module is loaded.
The severity MLPs were pickled under the TensorFlow backend and load under either backend
(manifest known_issues KI-04).
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml38_wine_machinery_fault_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    FAMILY_MODEL_FILENAME,
    FAMILY_SCALER_FILENAME,
    FEATURE_NAMES,
    SEVERITY_MODELS_FILENAME,
    SEVERITY_SCALERS_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


@dataclass(frozen=True)
class ModelBundle:
    """Family RandomForest + its scaler, and the per-family severity MLPs + their scalers."""

    family_model: Any
    family_scaler: Any
    severity_models: dict[str, Any]
    severity_scalers: dict[str, Any]

    @property
    def families(self) -> list[str]:
        """Classes of the family classifier, in its own order."""
        return [str(c) for c in self.family_model.classes_]


def _joblib_load(path: str) -> Any:
    """joblib.load that logs (once) the sklearn version mismatch instead of spamming warnings.

    The delivered pickles come from scikit-learn 1.7.2 and the repo pins 1.5.2 (KI-03); the
    predictions were checked to be identical (877/877), so the warning is logged, not raised.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = joblib.load(path)
    versions = {str(w.message).rsplit("from version ", maxsplit=1)[-1].split(" when")[0]
                for w in caught if w.category.__name__ == "InconsistentVersionWarning"}
    if versions:
        logger.warning("ml38: %s se serializó con scikit-learn %s (versión distinta de la "
                       "instalada); predicción verificada en el informe de verificación.",
                       os.path.basename(str(path)), ", ".join(sorted(versions)))
    return obj


def load_bundle_from(resolve: Callable[[str], Any]) -> ModelBundle:
    """Load the four artifacts, ``resolve(filename)`` giving the path of each one."""
    family_model = _joblib_load(resolve(FAMILY_MODEL_FILENAME))
    family_scaler = _joblib_load(resolve(FAMILY_SCALER_FILENAME))
    severity_models = _joblib_load(resolve(SEVERITY_MODELS_FILENAME))
    severity_scalers = _joblib_load(resolve(SEVERITY_SCALERS_FILENAME))
    n_features = getattr(family_model, "n_features_in_", None)
    if n_features != len(FEATURE_NAMES):
        raise ValueError(
            f"El clasificador de familia espera {n_features} características; el contrato del "
            f"modelo son {len(FEATURE_NAMES)}."
        )
    if not isinstance(severity_models, dict) or set(severity_models) != set(severity_scalers):
        raise ValueError(
            "severity_models.pkl y severity_scalers.pkl no cubren las mismas familias."
        )
    return ModelBundle(family_model, family_scaler, dict(severity_models), dict(severity_scalers))


def load_artifacts() -> ModelBundle:
    """Load the base (delivered) bundle through the ArtifactStore (local first, then S3)."""
    bundle = load_bundle_from(lambda name: str(_store.path(name)))
    logger.info("ml38 artifacts loaded — RF %d clases, severidad para %s",
                len(bundle.families), sorted(bundle.severity_models))
    return bundle
