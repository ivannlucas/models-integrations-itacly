"""Postprocessing: convert thermal-CNN logits into a prediction dict."""
import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image

from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.constants import CLASS_NAMES

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

    BaselineModel's classifier is Dropout (no-op in eval mode) + a single Linear layer
    applied right after global-average-pooling the backbone's last feature map, so CAM
    (Zhou et al., 2016) is exact here and needs no backward pass — same reasoning as
    ml2_fungal_cnn_disease_detection's compute_and_encode_cam.
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


def decode_logits(logits: torch.Tensor) -> dict:
    """Softmax the logits and return prediction, class index and per-class probabilities."""
    probs = torch.softmax(logits[0], dim=0)
    confidence, idx = torch.max(probs, dim=0)
    pred_idx = int(idx.item())
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "predicted_class_index": pred_idx,
        "confidence": float(confidence.item()),
        "probability_healthy": float(probs[0].item()),
        "probability_scm": float(probs[1].item()),
    }
