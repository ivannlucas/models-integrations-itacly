"""Model loading for ml39 — the XAI companion to ml37. Loads a39's own copy of ml37's exact
LSTM multitask model (same run_id no_virtual_v1__rand_scool) plus the SHAP background sample.

Unlike ml37 (which sidesteps a missing tensorflow==2.19.0 wheel via Keras 3 standalone +
backend=torch), a39 needs REAL TensorFlow: Integrated Gradients uses ``tf.GradientTape``, which
the standalone Keras 3 package does not expose. ``tensorflow>=2.15.0`` is already a repo-wide
dependency (requirements.txt, used directly by ml3/ml18) — see inbox/a39/manifest.yaml
known_issues for the exact-pin-vs-range trade-off this plugin accepts.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import tensorflow as tf

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml39_wine_predictive_maintenance_xai.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASS_MAPPINGS_FILENAME,
    CLASS_ORDERS,
    F_SCHEMA,
    MODEL_FILENAME,
    PHYSICAL_SENSORS,
    SCALER_FILENAME,
    SHAP_BACKGROUND_FILENAME,
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


def load_model_bundle() -> dict:
    """Load the model/scaler/class_mappings (+ SHAP background, if already built) via
    ArtifactStore. The SHAP background is optional: without it, ``include_shap`` requests fail
    with a clear error instead of silently degrading to a degenerate background."""
    model_path = str(_store.path(MODEL_FILENAME))
    scaler_path = str(_store.path(SCALER_FILENAME))
    mappings_path = str(_store.path(CLASS_MAPPINGS_FILENAME))

    class_mappings = _load_class_mappings(mappings_path)
    scaler = _load_scaler(scaler_path)
    # no-member: tf.keras is a lazy loader pylint cannot see.
    model = tf.keras.models.load_model(model_path, compile=False)  # pylint: disable=no-member

    output_names = list(getattr(model, "output_names", TARGET_NAMES))
    missing = [t for t in TARGET_NAMES if t not in output_names]
    if missing:
        raise ValueError(f"El modelo cargado no contiene los heads esperados: {missing}")

    shap_background = _load_shap_background()
    logger.info(
        "ml39 bundle loaded — targets=%s shap_background=%s", TARGET_NAMES,
        "yes" if shap_background is not None else "no",
    )
    return {
        "model": model, "scaler": scaler, "class_mappings": class_mappings,
        "shap_background": shap_background,
    }


def _load_shap_background() -> np.ndarray | None:
    try:
        path = _store.path(SHAP_BACKGROUND_FILENAME)
    except FileNotFoundError:
        return None
    return np.load(str(path))


# ── How shap_background.npy was built (reproducible, not re-run on every load) ──────────────
# 80 cycles (SHAP_N_BACKGROUND, "full" profile) sampled with np.random.default_rng(42) over
# ml37's real train_idx (data/splits/no_virtual_v1__rand_scool/splits.json, 1543 cycles),
# resampled to the schema_v1 (600, 40) tensor and scaled with this plugin's own scaler.json —
# same recipe as _explain_cycle, just applied to the background cycles once, offline. This is
# an independently, cleanly seeded sample (not a replay of the entregable's own internal rng
# call sequence, which draws from a single shared rng across several unrelated steps before
# reaching its background sample — see inbox/a39/manifest.yaml known_issues on SHAP
# reproducibility). Regeneration recipe:
#
#   rng = np.random.default_rng(42)
#   bg_idx = rng.choice(len(train_idx), size=80, replace=False)
#   bg_cycle_idx = train_idx[bg_idx]
#   # for each cidx in bg_cycle_idx: build_schema_v1_tensor(X[cidx][:, :14]), then apply_scaler
