import base64
import logging
from typing import Any

import cv2
import numpy as np
import torch

from app.domain.services.exceptions import InvalidImageError

logger = logging.getLogger(__name__)

CROP_SIZE = 224
ALPHA = 4


def decode_frames_base64(frames_b64: list[str]) -> np.ndarray:
    frames: list[np.ndarray] = []
    for i, b64 in enumerate(frames_b64):
        try:
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError(f"Frame {i} could not be decoded.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
            frames.append(img)
        except InvalidImageError:
            raise
        except Exception as exc:
            raise InvalidImageError(f"Failed to decode frame {i}: {exc}") from exc
    return np.array(frames, dtype=np.uint8)


def prepare_slowfast_tensor(frames_np: np.ndarray, device: Any, alpha: int = ALPHA) -> list[torch.Tensor]:
    T = frames_np.shape[0]
    frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0
    slow_indices = torch.arange(0, T, alpha)
    slow = frames_t[:, slow_indices, :, :].unsqueeze(0).to(device)
    fast = frames_t.unsqueeze(0).to(device)
    return [slow, fast]


def extract_cow_roi(frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    roi = cv2.resize(frame_bgr[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
