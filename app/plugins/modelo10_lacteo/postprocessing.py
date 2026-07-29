"""Funciones de postprocesamiento para el plugin Modelo10Lacteo."""
from __future__ import annotations

import base64
import io
from typing import Any

import torch
from PIL import Image, ImageDraw

SPECIES_COLORS: dict[str, tuple[int, int, int]] = {
    "fly": (220, 80, 80),
    "mos": (80, 160, 220),
    "tick": (80, 200, 80),
}
MAX_ANNOTATED_SIZE = 640
JPEG_QUALITY = 70


def render_annotated_image(image_pil, detections: list[dict[str, Any]]) -> str:
    """Draw detection bounding boxes on *image_pil* and return it as base64 JPEG."""
    img = image_pil.convert("RGB")

    w, h = img.size
    if max(w, h) > MAX_ANNOTATED_SIZE:
        scale = MAX_ANNOTATED_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        scale_w, scale_h = int(w * scale) / w, int(h * scale) / h
    else:
        scale_w = scale_h = 1.0

    draw = ImageDraw.Draw(img)
    for d in detections:
        bbox = d["bbox"]
        x1 = bbox["x1"] * scale_w
        y1 = bbox["y1"] * scale_h
        x2 = bbox["x2"] * scale_w
        y2 = bbox["y2"] * scale_h
        color = SPECIES_COLORS.get(d["species"], (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(y1 - 14, 0)), f"{d['species']} {d['cls_conf']:.2f}", fill=color)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


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
