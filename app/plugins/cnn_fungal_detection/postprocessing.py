from __future__ import annotations

from typing import Any, Dict

import torch

from app.plugins.cnn_fungal_detection.model_loader import CLASSES


def logits_to_prediction(logits: torch.Tensor) -> Dict[str, Any]:
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return {
        "prediction": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {cls: float(probs[i].item()) for i, cls in enumerate(CLASSES)},
    }
