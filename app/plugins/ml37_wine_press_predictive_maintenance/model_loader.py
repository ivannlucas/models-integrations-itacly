"""Model loading for ml37 — LSTM multitask classifier (4 heads), served variant
no_virtual_v1__rand_scool.

CRITICAL (faithful to src/training/train.py, inbox/a37/codigo/): KERAS_BACKEND=torch MUST be set
before keras is imported for the first time anywhere in the process — same pattern verified
working for ml1 (Keras 3 with backend=torch needs no TensorFlow at all, which resolves the real
tensorflow==2.19.0 wheel gap for this sandbox's Python version, see manifest known_issues).
"""
from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402  pylint: disable=wrong-import-position

from app.infrastructure.artifact_store import ArtifactStore  # noqa: E402  pylint: disable=wrong-import-position
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (  # noqa: E402  pylint: disable=wrong-import-position
    ARTIFACT_FOLDER_NAME,
    CLASS_MAPPINGS_FILENAME,
    CLASS_ORDERS,
    F_SCHEMA,
    MODEL_FILENAME,
    PHYSICAL_SENSORS,
    SCALER_FILENAME,
    TARGET_NAMES,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _load_class_mappings(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        mappings = json.load(fh)

    feature_order = mappings.get("feature_order_final")
    if not isinstance(feature_order, list) or len(feature_order) != F_SCHEMA:
        raise ValueError(
            f"class_mappings.json::feature_order_final debe tener {F_SCHEMA} slots, "
            f"encontrado: {feature_order}"
        )
    if tuple(feature_order[: len(PHYSICAL_SENSORS)]) != PHYSICAL_SENSORS:
        raise ValueError(
            "class_mappings.json::feature_order_final no coincide con el orden vinculante de "
            f"PHYSICAL_SENSORS en constants.py. Esperado={PHYSICAL_SENSORS} | "
            f"encontrado={feature_order[:len(PHYSICAL_SENSORS)]}"
        )

    class_order = mappings.get("class_order_by_target", {})
    for target in TARGET_NAMES:
        expected = CLASS_ORDERS[target]
        found = tuple(class_order.get(target, ()))
        if found != expected:
            raise ValueError(
                f"class_mappings.json::class_order_by_target[{target}] no coincide con "
                f"CLASS_ORDERS en constants.py. Esperado={expected} | encontrado={found}"
            )
    return mappings


def _load_scaler(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        scaler = json.load(fh)
    mean, scale = scaler.get("mean"), scaler.get("scale")
    valid = (
        isinstance(mean, list) and isinstance(scale, list)
        and len(mean) == F_SCHEMA and len(scale) == F_SCHEMA
    )
    if not valid:
        raise ValueError(
            f"scaler.json inválido: 'mean'/'scale' deben ser listas de longitud {F_SCHEMA}"
        )
    return scaler


def _load_bundle(model_path: str, scaler_path: str, mappings_path: str) -> dict:
    class_mappings = _load_class_mappings(mappings_path)
    scaler = _load_scaler(scaler_path)
    model = keras.models.load_model(model_path, compile=False)

    output_names = list(getattr(model, "output_names", TARGET_NAMES))
    missing = [t for t in TARGET_NAMES if t not in output_names]
    if missing:
        raise ValueError(f"El modelo cargado no contiene los heads esperados: {missing}")

    return {
        "model": model,
        "scaler": scaler,
        "class_mappings": class_mappings,
    }


def load_model_bundle() -> dict:
    """Load the real LSTM multitask model + scaler + class mappings for no_virtual_v1__rand_scool
    (the served base artifact, via ArtifactStore)."""
    bundle = _load_bundle(
        str(_store.path(MODEL_FILENAME)),
        str(_store.path(SCALER_FILENAME)),
        str(_store.path(CLASS_MAPPINGS_FILENAME)),
    )
    logger.info("ml37 base bundle loaded — targets=%s", TARGET_NAMES)
    return bundle


def load_model_bundle_from_dir(local_path: str) -> dict:
    """Load a user-retrained bundle from a downloaded MLflow run folder (mlflow_utils.py)."""
    return _load_bundle(
        os.path.join(local_path, MODEL_FILENAME),
        os.path.join(local_path, SCALER_FILENAME),
        os.path.join(local_path, CLASS_MAPPINGS_FILENAME),
    )
