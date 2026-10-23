"""Ensemble inference and metrics — mirrors src/main.py::predict and src/predict/predictor.py."""
from collections import Counter

import torch
from sklearn.metrics import accuracy_score, classification_report

from app.plugins.ml6_meat_poultry_disease_classification.constants import CLASS_NAMES


def majority_vote(votes: list[int]) -> int:
    """src/main.py: ``Counter(votes).most_common(1)[0][0]``.

    On a three-way tie the vote of the first model (config.yaml order) wins — kept on purpose.
    """
    return int(Counter(votes).most_common(1)[0][0])


def run_ensemble(models: dict, tensor: torch.Tensor, task: str) -> dict:
    """Run every base model on one image and aggregate by majority vote."""
    class_names = CLASS_NAMES[task]
    votes: list[int] = []
    model_predictions: dict[str, str] = {}
    model_probabilities: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, model in models.items():
            output = model(tensor)
            # predictor.py::run_inference_img — argmax of exp(logits)
            vote = int(torch.max(torch.exp(output), 1)[1].item())
            probs = torch.softmax(output[0], dim=0).cpu().tolist()
            votes.append(vote)
            model_predictions[name] = class_names[vote]
            model_probabilities[name] = {c: round(float(p), 6) for c, p in zip(class_names, probs)}
    winner = majority_vote(votes)
    return {
        "prediction": class_names[winner],
        "predicted_class_index": winner,
        "confidence": votes.count(winner) / len(votes),
        "model_predictions": model_predictions,
        "model_probabilities": model_probabilities,
    }


def classification_metrics(y_true: list[int], y_pred: list[int], task: str) -> dict:
    """Same report as predictor.py::save_metrics (4 decimals, zero_division=0)."""
    class_names = list(CLASS_NAMES[task])
    report = classification_report(
        y_true, y_pred, output_dict=True, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_weighted": round(report["weighted avg"]["precision"], 4),
        "recall_weighted": round(report["weighted avg"]["recall"], 4),
        "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
        "precision_macro": round(report["macro avg"]["precision"], 4),
        "recall_macro": round(report["macro avg"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in class_names
        },
        "support": len(y_true),
    }
