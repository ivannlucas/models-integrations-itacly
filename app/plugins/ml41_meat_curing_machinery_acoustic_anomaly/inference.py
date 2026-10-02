"""Inference on a loaded AudioMAE checkpoint: MSE reconstruction score + Mahalanobis
distance in the (PCA-projected) CLS-embedding latent space.

Ported from inbox/a41/codigo/src/predict/predictor.py. run_inference() intentionally
does NOT fix a random seed for the masking used in the MSE score — this matches the
original code exactly and is documented in manifest.yaml as a known, expected source of
run-to-run variance (~5-6%) in mse_score. maha_score is deterministic and is the actual
production decision signal (see manifest.yaml outputs.predict_inline).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.audio_mae import AudioMAE
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import MASK_RATIO, NUM_MASKS


def run_inference_single(
    model: AudioMAE,
    spec: torch.Tensor,
    device: torch.device,
    mask_ratio: float = MASK_RATIO,
    num_masks: int = NUM_MASKS,
) -> float:
    """MSE reconstruction score for a single (1, 1, n_mels, target_frames) spectrogram."""
    model.eval()
    single = spec.to(device)
    with torch.no_grad():
        if num_masks > 1:
            losses = []
            for _ in range(num_masks):
                loss, _, _ = model(single, mask_ratio=mask_ratio)
                losses.append(loss.item())
            return float(np.mean(losses))
        loss, _, _ = model(single, mask_ratio=mask_ratio)
        return float(loss.item())


def get_cls_embedding_single(model: AudioMAE, spec: torch.Tensor, device: torch.device) -> np.ndarray:
    """CLS token of the encoder (no masking) for a single sample. Returns shape (embed_dim,)."""
    model.eval()
    with torch.no_grad():
        latent, _, _ = model.forward_encoder(spec.to(device), mask_ratio=0.0)
        cls = latent[:, 0, :]  # (1, D)
    return cls.cpu().numpy()[0]


def mahalanobis_score(
    embedding: np.ndarray,
    mean: np.ndarray,
    inv_cov: np.ndarray,
    pca_components: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    """Mahalanobis distance of a single CLS embedding from the training normal distribution."""
    emb = embedding
    if pca_components is not None:
        pca_mean, v_k = pca_components
        emb = (emb - pca_mean) @ v_k.T
    diff = emb - mean
    dist = np.sqrt(diff @ inv_cov @ diff.T)
    return float(dist)


def load_mahalanobis_stats(path: str):
    """Load (mean, inv_cov, pca_components) from a maha_stats.npz file."""
    d = np.load(path)
    pca_components = (d["pca_mean"], d["pca_v"]) if bool(d["has_pca"]) else None
    return d["mean"], d["inv_cov"], pca_components


def fit_mahalanobis(
    embeddings: np.ndarray,
    n_pca: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Fit (mean, inv_cov[, pca]) on a batch of CLS embeddings — used by train()/recalibration.

    Ported from inbox/a41/codigo/src/predict/predictor.py:fit_mahalanobis (accepts
    precomputed embeddings here rather than a DataLoader+model, since the caller already
    has to iterate samples for MSE loss during fine-tuning).
    """
    _n_samples, embed_dim = embeddings.shape
    pca_components = None
    emb = embeddings

    if n_pca and 0 < n_pca < embed_dim:
        pca_mean = np.mean(emb, axis=0)
        emb_c = emb - pca_mean
        cov_for_pca = np.cov(emb_c, rowvar=False)
        _eigvals, eigvecs = np.linalg.eigh(cov_for_pca)
        v_k = eigvecs[:, -n_pca:][:, ::-1].T
        emb = emb_c @ v_k.T
        pca_components = (pca_mean, v_k)

    mean = np.mean(emb, axis=0)
    cov = np.cov(emb, rowvar=False) + np.eye(emb.shape[1]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    return mean, inv_cov, pca_components
