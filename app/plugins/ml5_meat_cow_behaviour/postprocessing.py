"""Postprocessing: convert SlowFast logits to behaviour labels."""
import torch


def decode_logits(
    logits: torch.Tensor,
    idx_to_behavior: dict[int, str],
    anomaly_threshold: float = 0.5,
) -> dict:
    """Convert raw SlowFast logits ``(1, num_classes)`` into a structured prediction dict.

    Returns keys: ``prediction``, ``confidence``, ``is_anomaly``, ``behavior_idx``, ``all_probs``.
    """
    probs = torch.softmax(logits[0], dim=0)
    confidence, pred_idx = torch.max(probs, dim=0)

    idx = int(pred_idx.item())
    conf = float(confidence.item())
    behavior = idx_to_behavior.get(idx, f"unknown_{idx}")

    all_probs = {
        idx_to_behavior.get(i, f"unknown_{i}"): round(float(p), 4)
        for i, p in enumerate(probs.tolist())
    }

    return {
        "prediction": behavior,
        "confidence": conf,
        "is_anomaly": conf < anomaly_threshold,
        "behavior_idx": idx,
        "all_probs": all_probs,
    }
