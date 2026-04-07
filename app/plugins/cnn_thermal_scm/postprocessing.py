import torch

from app.plugins.cnn_thermal_scm.predict_dto import CLASS_NAMES, PredictResponse


def decode_logits(logits: torch.Tensor) -> PredictResponse:
    probabilities = torch.softmax(logits[0], dim=0)
    confidence, predicted_index = torch.max(probabilities, dim=0)
    pred_idx = predicted_index.item()
    return PredictResponse(
        prediction=CLASS_NAMES[pred_idx],
        predicted_class_index=pred_idx,
        confidence=float(confidence.item()),
        probability_healthy=float(probabilities[0].item()),
        probability_scm=float(probabilities[1].item()),
    )
