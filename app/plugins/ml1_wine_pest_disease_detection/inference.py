"""Inference core for ml1 — faithful port of src/predict/predictor.py::VidPredictor.run_inference
/ run_inference_tabular_only (decoupled + sin-gases production config) from the delivered code.
"""
from __future__ import annotations

import numpy as np

from app.domain.services.exceptions import DataContractError
from app.plugins.ml1_wine_pest_disease_detection.constants import (
    HEALTHY_CLASSES,
    HEALTHY_CONFIDENCE_THRESHOLD,
    HEALTHY_RISK_DAMPING,
    NO_IMAGE_LABEL,
)


def run_inference(bundle: dict, img_rgb: np.ndarray, sensors_aligned: np.ndarray) -> dict:
    """Deep Ensemble soft-voting over the 3 backbones (image -> class, tabular -> risk).

    img_rgb: HxWx3 float32 in [0, 1] (already YOLO-cropped and resized — see preprocessing.py).
    sensors_aligned: (1, n_sensors) float32, already validated/ordered.
    """
    img_batch = np.expand_dims(img_rgb, axis=0)
    class_mapping = bundle["class_mapping"]

    sum_probs_clase = None
    sum_preds_riesgo = None
    ensemble = bundle["ensemble"]

    for model in ensemble.values():
        preds = model.predict({"input_image": img_batch, "input_tabular": sensors_aligned}, verbose=0)
        if sum_probs_clase is None:
            sum_probs_clase = preds[0]
            sum_preds_riesgo = preds[1]
        else:
            sum_probs_clase = sum_probs_clase + preds[0]
            sum_preds_riesgo = sum_preds_riesgo + preds[1]

    num_models = len(ensemble)
    avg_probs = sum_probs_clase / num_models
    avg_riesgo = sum_preds_riesgo / num_models

    idx_predicho = int(np.argmax(avg_probs[0]))
    confianza = float(avg_probs[0][idx_predicho])
    enfermedad = class_mapping[idx_predicho]
    riesgo_final = float(np.clip(float(avg_riesgo[0][0]), 0.0, 1.0))

    # predictor.py::run_inference business rule: a confidently-healthy call dampens risk.
    if enfermedad in HEALTHY_CLASSES and confianza > HEALTHY_CONFIDENCE_THRESHOLD:
        riesgo_final *= HEALTHY_RISK_DAMPING

    return {
        "enfermedad": enfermedad,
        "confianza": confianza,
        "riesgo": riesgo_final,
        "metodo": "HIBRIDO_ENSAMBLE",
    }


def run_inference_tabular_only(bundle: dict, sensors_aligned: np.ndarray) -> dict:
    """Decoupled-mode contingency path (no image): class is NOT inferred from the RF classifier
    — predictor.py only does that when use_fusion_layer=True, which is not the production
    config (see inbox/a01/manifest.yaml known_issues: the memoria says otherwise in 3 places,
    the code disagrees, and the RF classifier's own audited accuracy — 11.6% over 15 classes —
    explains why). Risk still comes from the RF regressor, same as the original in both modes.
    """
    if bundle.get("rf_riesgo") is None:
        raise DataContractError(
            "Sin imagen no se puede estimar el riesgo: el experto RF de riesgo no está disponible "
            "en este despliegue. Envía una imagen."
        )
    riesgo_estimado = float(bundle["rf_riesgo"].predict(sensors_aligned)[0])
    return {
        "enfermedad": NO_IMAGE_LABEL,
        "confianza": 0.0,
        "riesgo": float(np.clip(riesgo_estimado, 0.0, 1.0)),
        "metodo": "RIESGO_TABULAR_EXPERT",
    }
