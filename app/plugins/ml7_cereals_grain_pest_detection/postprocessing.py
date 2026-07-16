"""Postprocessing: convert YOLO results to the platform output schema."""
import base64
import io
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES: Dict[str, str] = {
    "cf": "Cryptolestes ferrugineus",
    "sz": "Sitophilus spp.",
    "rd": "Rhyzopertha dominica",
    "tc": "Tribolium castaneum",
    "os": "Oryzaephilus surinamensis",
}

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cf": (220, 80, 80),
    "sz": (80, 160, 220),
    "rd": (80, 200, 80),
    "tc": (220, 160, 40),
    "os": (160, 80, 220),
}

MAX_ANNOTATED_SIZE = 640
JPEG_QUALITY = 70


def yolo_results_to_dict(
    results: Any, original_image: np.ndarray, conf_threshold: float = 0.28
) -> dict:
    """Convert an ultralytics Results object to the platform output dict.

    Returns: prediction, confidence, total_detections, species_counts,
    detections (list), annotated_image (base64 JPEG).
    """
    detections: List[Dict[str, Any]] = []

    if results.boxes is not None and len(results.boxes) > 0:
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        xyxy = results.boxes.xyxy.cpu().numpy()
        names = results.names  # {0: 'cf', 1: 'sz', ...}

        for i in range(len(class_ids)):
            conf = float(confidences[i])
            if conf < conf_threshold:
                continue
            cls_code = names.get(int(class_ids[i]), str(class_ids[i]))
            detections.append({
                "class": cls_code,
                "class_name": CLASS_NAMES.get(cls_code, cls_code),
                "confidence": round(conf, 4),
                "bbox": [round(float(v), 1) for v in xyxy[i]],
            })

    species_counts: Dict[str, int] = {}
    for d in detections:
        species_counts[d["class"]] = species_counts.get(d["class"], 0) + 1

    prediction = "none"
    confidence = 0.0
    if detections:
        conf_sums: Dict[str, float] = defaultdict(float)
        for d in detections:
            conf_sums[d["class"]] += d["confidence"]
        prediction = max(species_counts, key=lambda k: (species_counts[k], conf_sums[k]))
        # "confidence" = highest single-detection confidence among boxes of the predicted
        # class only (not the max/mean/min across all detected classes in the image).
        confidence = round(max(d["confidence"] for d in detections if d["class"] == prediction), 4)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "total_detections": len(detections),
        "species_counts": species_counts,
        "detections": detections,
        "annotated_image": _render_annotated_image(original_image, detections),
    }


def _render_annotated_image(image: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """Draw bounding boxes on the image and return it as base64 JPEG."""
    img = Image.fromarray(image).convert("RGB")

    w, h = img.size
    if max(w, h) > MAX_ANNOTATED_SIZE:
        scale = MAX_ANNOTATED_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        scale_w = int(w * scale) / w
        scale_h = int(h * scale) / h
    else:
        scale_w = scale_h = 1.0

    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        x1, y1, x2, y2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
        color = CLASS_COLORS.get(d["class"], (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(y1 - 14, 0)), f"{d['class']} {d['confidence']:.2f}", fill=color)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()
