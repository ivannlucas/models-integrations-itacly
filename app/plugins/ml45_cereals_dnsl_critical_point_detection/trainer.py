"""Fine-tuning for m45 — clones the loaded checkpoint and continues training on user data.

The original src/training/trainer.py::run_train_pipeline (268 lines) is a from-scratch training
run with disk checkpointing, early stopping, LR scheduling and post-hoc threshold re-search —
appropriate for training the delivered model on the full synthetic dataset, not for fine-tuning
an already-trained checkpoint on a user-submitted CSV. Following this repo's established
fine-tuning pattern (ml34/ml35: same loss/optimizer as the original, fixed epoch count, no
early stopping), this module reuses the same DNFLoss + Adam + hyperparameters from the
checkpoint's own model_cfg, without the scheduler/checkpointing/threshold-search machinery.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.loss import DNFLoss
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.model import (
    ParallelDeepNeuroFuzzyModel,
)

FINE_TUNE_EPOCHS = 30


def _compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Same formula as src/training/metrics.py::compute_class_weights."""
    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    return torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)


def fine_tune(
    model: ParallelDeepNeuroFuzzyModel,
    model_cfg: dict,
    sequences_scaled: np.ndarray,
    stats_scaled: np.ndarray,
    y_labels: np.ndarray,
    threshold: float,
    epochs: int = FINE_TUNE_EPOCHS,
) -> tuple[ParallelDeepNeuroFuzzyModel, dict]:
    """Fine-tune a clone of *model* on (sequences_scaled, stats_scaled, y_labels).

    y_labels is the per-window binary anomaly label (0/1) already produced by
    preprocess.create_sequences()'s window-level _anomaly_mask.
    """
    fine_model = ParallelDeepNeuroFuzzyModel(copy.deepcopy(model_cfg))
    fine_model.load_state_dict(model.state_dict())
    fine_model.train()

    train_cfg = model_cfg["training_kwargs"]
    y_anom = (np.asarray(y_labels) > 0).astype(np.int64)

    pos_weight = _compute_class_weights(y_anom)
    dnf_loss_kwargs = copy.deepcopy(train_cfg.get("dnf_loss", {}))
    criterion = DNFLoss(anomaly_pos_weight=pos_weight, **dnf_loss_kwargs)

    optimizer = torch.optim.Adam(
        fine_model.parameters(),
        lr=float(train_cfg["lr_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    X = torch.tensor(sequences_scaled, dtype=torch.float32)
    S = torch.tensor(stats_scaled, dtype=torch.float32)
    y = torch.tensor(y_anom, dtype=torch.float32)

    grad_clip = train_cfg.get("grad_clip")
    for _ in range(epochs):
        optimizer.zero_grad()
        out = fine_model(X, S)
        loss, _ = criterion(
            anomaly_score=out["anomaly_score"],
            logit_anomaly_dl=out["logit_anomaly_dl"],
            logit_anomaly_fuzzy=out["logit_anomaly_fuzzy"],
            rule_activations=out["rule_activations"],
            alpha_probs=out["alpha_probs"],
            y_anomaly=y,
        )
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(fine_model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

    fine_model.eval()
    with torch.no_grad():
        out = fine_model(X, S)
        probs = torch.sigmoid(out["anomaly_score"].view(-1)).numpy()

    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_anom, preds)),
        "f1": float(f1_score(y_anom, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_anom, probs)) if len(np.unique(y_anom)) > 1 else float("nan"),
        "n_windows": int(len(y_anom)),
        "n_epochs": int(epochs),
    }
    return fine_model, metrics
