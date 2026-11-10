"""Class Activation Map overlay for the CNN image classifiers (ml2, ml4)."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image

MAX_HEATMAP_SIZE = 640
JPEG_QUALITY = 70
HEATMAP_ALPHA = 0.4


def class_activation_map(
    feature_map: torch.Tensor, classifier_weight: torch.Tensor, class_idx: int
) -> np.ndarray:
    """CAM of ``class_idx``: ReLU of the classifier-weighted sum of the feature maps, in [0, 1]."""
    weights = classifier_weight[class_idx].detach().cpu().numpy()  # (C,)
    feat = feature_map[0].detach().cpu().numpy()  # (C, H, W)
    cam = np.tensordot(weights, feat, axes=([0], [0]))  # (H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def compute_and_encode_cam(
    feature_map: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_idx: int,
    original_image: Image.Image,
) -> str:
    """Overlay the CAM of ``class_idx`` on the original image; return a base64 JPEG data URI.

    Valid for a classifier that is a single Linear layer (Dropout aside) right after a global
    average pooling of the last feature map: then CAM (Zhou et al., 2016) is exact and needs no
    backward pass — a weighted sum of those feature maps by the predicted class's classifier
    row. For such an architecture Grad-CAM reduces to the same result.
    """
    cam = class_activation_map(feature_map, classifier_weight, class_idx)

    img = original_image.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_HEATMAP_SIZE:
        scale = MAX_HEATMAP_SIZE / max(w, h)
        # pylint: disable-next=no-member  # Image.LANCZOS exists (Pillow alias); inference FP
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size

    # pylint: disable=no-member  # cv2 is a C extension; pylint cannot see its members
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # pylint: enable=no-member

    img_arr = np.array(img).astype(np.float32)
    overlay = (
        heatmap.astype(np.float32) * HEATMAP_ALPHA + img_arr * (1 - HEATMAP_ALPHA)
    ).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
