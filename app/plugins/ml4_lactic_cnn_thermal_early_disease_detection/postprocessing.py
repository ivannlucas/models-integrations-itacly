"""Postprocessing: convert thermal-CNN logits into a prediction dict."""
import torch

from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.constants import CLASS_NAMES


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
