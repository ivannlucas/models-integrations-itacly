"""Postprocessing for the fungal leaf-disease CNN plugin.

Turns raw model logits into the response dicts consumed by the predict DTOs.
"""
import torch
import torch.nn.functional as F


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
