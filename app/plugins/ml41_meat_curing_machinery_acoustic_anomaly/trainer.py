"""Fine-tuning logic, ported from inbox/a41/codigo/src/training/trainer.py.

Same optimizer (AdamW with selective weight decay), same CosineWarmupScheduler, same
per-machine hyperparameters (never invented — see constants.TRAINING_HYPERPARAMS_BY_MACHINE,
sourced from config_vit_tiny_{machine}.yaml). Adapted to consume in-memory tensors built
from a ZIP of WAV files (train()'s data_path) instead of pre-split .npy files under
data/processed/{train,val}/, since the plugin receives raw audio, not the original
pipeline's intermediate artifacts.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.audio_mae import AudioMAE
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import TRAINING_COMMON

logger = logging.getLogger(__name__)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing — standard for ViT/MAE."""

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


def _train_one_epoch(model: AudioMAE, loader: DataLoader, optimizer, device, mask_ratio: float) -> float:
    model.train()
    total_loss = 0.0
    for (imgs,) in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(imgs, mask_ratio=mask_ratio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def _validate(model: AudioMAE, loader: DataLoader, device, mask_ratio: float) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (imgs,) in loader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def fine_tune(
    model: AudioMAE,
    train_data: np.ndarray,
    val_data: np.ndarray,
    device: torch.device,
    hyperparams: dict,
) -> tuple[float, dict]:
    """Fine-tune (or train from scratch) `model` in place on already-normalized data.

    train_data/val_data: (N, 1, n_mels, target_frames) float32 arrays, already normalized
    with the combination's norm_mean/norm_std.

    Returns (best_val_loss, history).
    """
    seed = TRAINING_COMMON["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(torch.from_numpy(train_data).float())
    val_ds = TensorDataset(torch.from_numpy(val_data).float())

    batch_size = hyperparams["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(k in name for k in ("bias", "norm", "cls_token", "mask_token")):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": hyperparams["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=hyperparams["lr"], betas=TRAINING_COMMON["betas"],
    )

    epochs = TRAINING_COMMON["epochs"]
    patience = TRAINING_COMMON["patience"]
    scheduler = CosineWarmupScheduler(optimizer, hyperparams["warmup_epochs"], epochs, min_lr=1e-6)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history: dict = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device, hyperparams["mask_ratio"])
        val_loss = _validate(model, val_loader, device, hyperparams["mask_ratio"])
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch, val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return best_val_loss, history


def compute_threshold_audit_fnr(
    normal_scores: np.ndarray, abnormal_scores: np.ndarray, max_fnr: float = 0.10,
) -> float:
    """Ported from inbox/a41/codigo/src/predict/postprocess.py:compute_threshold(method='audit_fnr').

    Highest threshold that still guarantees TPR >= 1 - max_fnr (FNR <= max_fnr) — the
    project's success criterion (manifest.yaml metrics_reported.fnr_objetivo_negocio).
    """
    from sklearn.metrics import roc_curve

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(abnormal_scores))])
    scores = np.concatenate([normal_scores, abnormal_scores])
    _fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(tpr >= 1.0 - max_fnr)[0]
    if len(valid) == 0:
        return float(thresholds[int(np.argmax(tpr))])
    return float(thresholds[valid[0]])


def evaluate_against_abnormal(
    normal_maha: np.ndarray, abnormal_maha: np.ndarray, normal_mse: np.ndarray, abnormal_mse: np.ndarray,
) -> dict:
    """Compute auc_mse, auc_maha, threshold, fnr, fpr, recall — only called when abnormal
    audio was provided for a combination (see train_dto.py docstring)."""
    from sklearn.metrics import confusion_matrix, roc_auc_score

    y_true = np.concatenate([np.zeros(len(normal_maha)), np.ones(len(abnormal_maha))])
    maha_scores = np.concatenate([normal_maha, abnormal_maha])
    mse_scores = np.concatenate([normal_mse, abnormal_mse])

    auc_maha = float(roc_auc_score(y_true, maha_scores))
    auc_mse = float(roc_auc_score(y_true, mse_scores))
    threshold = compute_threshold_audit_fnr(normal_maha, abnormal_maha)

    y_pred = (maha_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0

    return {
        "auc_mse": auc_mse, "auc_maha": auc_maha, "threshold": threshold,
        "fnr": fnr, "fpr": fpr, "recall": recall,
    }
