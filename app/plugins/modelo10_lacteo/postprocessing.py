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


def _annotated_scale(w: int, h: int) -> tuple[float, float]:
    """Scale factors render_annotated_image applies when downscaling to MAX_ANNOTATED_SIZE.

    Shared with build_heatmap_crops so its bbox coordinates land in the exact same pixel
    space as the annotated_image the platform composites Grad-CAM heatmaps onto — a drift
    between the two would misplace every composited heatmap.
    """
    if max(w, h) > MAX_ANNOTATED_SIZE:
        scale = MAX_ANNOTATED_SIZE / max(w, h)
        return int(w * scale) / w, int(h * scale) / h
    return 1.0, 1.0


def render_annotated_image(image_pil, detections: list[dict[str, Any]]) -> str:
    """Draw detection bounding boxes on *image_pil* and return it as base64 JPEG."""
    img = image_pil.convert("RGB")

    w, h = img.size
    scale_w, scale_h = _annotated_scale(w, h)
    if (scale_w, scale_h) != (1.0, 1.0):
        img = img.resize((int(w * scale_w), int(h * scale_h)), Image.LANCZOS)

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


def build_heatmap_crops(
    image_pil, detections: list[dict[str, Any]], max_crops: int = 5,
) -> list[dict[str, Any]]:
    """Up to *max_crops* detections (by cls_conf), each with its own base64 JPEG crop.

    Each crop is the same plain rectangular region (no padding) preprocessing.crop_to_tensor
    takes before classifying — the exact input the classifier itself was fed for that
    detection. The platform runs Grad-CAM against each crop and composites the resulting
    per-detection heatmaps onto annotated_image (bbox already rescaled to match its
    coordinate space via _annotated_scale), instead of running a single Grad-CAM against the
    full uncropped scene: that classifier only ever sees crops like these, never a full
    scene, in training or at inference, so Grad-CAM against the full scene produced an
    out-of-distribution activation with no real correspondence to any detection — and picking
    only one detection to explain hid every other vector present in the same image.
    """
    if not detections:
        return []
    img = image_pil.convert("RGB")
    w, h = img.size
    scale_w, scale_h = _annotated_scale(w, h)

    top = sorted(detections, key=lambda d: d["cls_conf"], reverse=True)[:max_crops]
    result = []
    for d in top:
        bbox = d["bbox"]
        crop = img.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=JPEG_QUALITY)
        result.append({
            "species": d["species"],
            "cls_conf": d["cls_conf"],
            "crop_base64": base64.b64encode(buf.getvalue()).decode(),
            "bbox": {
                "x1": round(bbox["x1"] * scale_w, 1),
                "y1": round(bbox["y1"] * scale_h, 1),
                "x2": round(bbox["x2"] * scale_w, 1),
                "y2": round(bbox["y2"] * scale_h, 1),
            },
        })
    return result


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
