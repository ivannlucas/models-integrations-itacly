import torch


def decode_logits(logits: torch.Tensor, idx_to_behavior: dict[int, str], anomaly_threshold: float = 0.5) -> dict:
    probs = torch.softmax(logits[0], dim=0)
    confidence, pred_idx = torch.max(probs, dim=0)
    idx = int(pred_idx.item())
    conf = float(confidence.item())
    behavior = idx_to_behavior.get(idx, f"unknown_{idx}")
    return {
        "prediction": behavior,
        "confidence": conf,
        "is_anomaly": conf < anomaly_threshold,
        "behavior_idx": idx,
    }
