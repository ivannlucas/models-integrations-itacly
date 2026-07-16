"""Postprocessing for the fungal leaf-disease CNN plugin.

Turns raw model logits into the response dicts consumed by the predict DTOs.
"""
import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MAX_HEATMAP_SIZE = 640
JPEG_QUALITY = 70
HEATMAP_ALPHA = 0.4


def compute_and_encode_cam(
    feature_map: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_idx: int,
    original_image: Image.Image,
) -> str:
    """Overlay a Class Activation Map for ``class_idx`` on the original image and return
    it as a base64 JPEG data URI.

    LeafCNN's classifier is a single Linear layer applied right after global-average-pooling
    the last conv block, so CAM (Zhou et al., 2016) is exact here and needs no backward
    pass: just a weighted sum of that block's feature maps, weighted by the predicted
    class's own classifier row. For this architecture Grad-CAM reduces to the same result.
    """
    weights = classifier_weight[class_idx].detach().cpu().numpy()  # (C,)
    feat = feature_map[0].detach().cpu().numpy()  # (C, H, W)
    cam = np.tensordot(weights, feat, axes=([0], [0]))  # (H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    img = original_image.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_HEATMAP_SIZE:
        scale = MAX_HEATMAP_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_arr = np.array(img).astype(np.float32)
    overlay = (heatmap.astype(np.float32) * HEATMAP_ALPHA + img_arr * (1 - HEATMAP_ALPHA)).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_inline_response(
    logits: torch.Tensor,
    classes: list[str],
    model_id: str,
) -> dict:
    """Convert raw logits into the inline response dict (class + per-class probabilities)."""
    probs = F.softmax(logits.detach(), dim=-1).squeeze(0)
    idx = int(probs.argmax().item())

    return {
        "model_id": model_id,
        "prediction": classes[idx],
        "confidence": round(float(probs[idx].item()), 6),
        "probabilities": {
            classes[i]: round(float(probs[i].item()), 6)
            for i in range(len(probs))
        },
    }


def build_batch_response(predictions: list[dict], model_id: str) -> dict:
    """Wrap a list of per-image prediction dicts into the batch response dict."""
    return {
        "model_id": model_id,
        "predictions": predictions,
        "output_path": None,
    }
