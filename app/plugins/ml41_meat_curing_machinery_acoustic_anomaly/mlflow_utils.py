"""MLflow helper for ml41 — download user fine-tuned checkpoint(s) from MLflow.

A single MLflow run may hold one or more fine-tuned (machine, machine_id, snr)
checkpoints (train() can fine-tune several combinations from one ZIP). Artifacts are
uploaded under model/<machine>/<machine_id>/<snr>/{best.pth,maha_stats.npz} — mirrors the
local on-disk layout so the same three-level walk works for both.
"""
from __future__ import annotations

import logging
import os

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import (
    CHECKPOINT_FILENAME,
    MAHA_STATS_FILENAME,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.inference import load_mahalanobis_stats
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.model_loader import build_model, _safe_device

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download every fine-tuned combination logged under this MLflow run.

    Returns (combinations, temp_dir) where combinations is a dict keyed by
    (machine, machine_id, snr) -> LoadedCombination-like dict {model, norm_mean,
    norm_std, maha_mean, maha_inv_cov, maha_pca, device}, or None if nothing could be
    downloaded. Caller MUST shutil.rmtree(temp_dir) after inference (try/finally).
    """
    import tempfile

    import torch

    tmp = tempfile.mkdtemp(prefix="mlflow_ml41_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    device = _safe_device()
    combinations: dict[tuple[str, str, str], dict] = {}

    for machine in os.listdir(local_path):
        machine_dir = os.path.join(local_path, machine)
        if not os.path.isdir(machine_dir):
            continue
        for machine_id in os.listdir(machine_dir):
            machine_id_dir = os.path.join(machine_dir, machine_id)
            if not os.path.isdir(machine_id_dir):
                continue
            for snr in os.listdir(machine_id_dir):
                combo_dir = os.path.join(machine_id_dir, snr)
                ckpt_path = os.path.join(combo_dir, CHECKPOINT_FILENAME)
                maha_path = os.path.join(combo_dir, MAHA_STATS_FILENAME)
                if not (os.path.isfile(ckpt_path) and os.path.isfile(maha_path)):
                    continue

                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = build_model(machine, device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                maha_mean, maha_inv_cov, maha_pca = load_mahalanobis_stats(maha_path)

                combinations[(machine, machine_id, snr)] = {
                    "model": model,
                    "norm_mean": float(checkpoint.get("norm_mean", 0.0)),
                    "norm_std": float(checkpoint.get("norm_std", 1.0)),
                    "maha_mean": maha_mean,
                    "maha_inv_cov": maha_inv_cov,
                    "maha_pca": maha_pca,
                    "device": device,
                }

    if not combinations:
        return None

    logger.info("Downloaded %d user-trained combination(s) from MLflow run_id=%s",
                len(combinations), run_id)
    return combinations, tmp
