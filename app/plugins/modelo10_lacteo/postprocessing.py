"""Funciones de postprocesamiento para el plugin Modelo10Lacteo."""
from __future__ import annotations

from typing import Any

import torch


def classify_crop(
    classifier,
    tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    """
    Clasifica un crop con MobileNetV3.
    Devuelve (nombre_clase, confianza).
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return class_names[pred_idx], float(probs[pred_idx].item())


def build_inline_result(
    model_id: str,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Construye el dict de respuesta inline a partir de la lista de detecciones.
    La predicción dominante es la especie con mayor cls_conf.
    """
    if not detections:
        return {
            "model_id": model_id,
            "prediction": "no_vectors",
            "confidence": 0.0,
            "vectors_count": 0,
            "detections": [],
            "species_summary": {},
        }

    dominant = max(detections, key=lambda d: d["cls_conf"])
    species_summary: dict[str, int] = {}
    for d in detections:
        species_summary[d["species"]] = species_summary.get(d["species"], 0) + 1

    return {
        "model_id": model_id,
        "prediction": dominant["species"],
        "confidence": dominant["cls_conf"],
        "vectors_count": len(detections),
        "detections": detections,
        "species_summary": species_summary,
    }
