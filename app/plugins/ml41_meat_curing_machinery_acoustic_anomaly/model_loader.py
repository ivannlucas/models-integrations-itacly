"""Lazy, per-combination checkpoint loading with a small in-memory cache.

The 48 (machine, machine_id, snr) combinations total ~2GB of weights — loading all of
them eagerly in load() would be wasteful for a deployment that typically monitors a
handful of physical machines. Instead, load() only verifies/downloads the artifact
directory, and each combination's checkpoint is loaded on first use and cached (LRU,
bounded by CHECKPOINT_CACHE_SIZE).
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from app.domain.services.exceptions import UnsupportedMachineConfigurationError
from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.audio_mae import AudioMAE
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import (
    ARCHITECTURE_BY_MACHINE,
    ARTIFACT_FOLDER_NAME,
    CHECKPOINT_CACHE_SIZE,
    CHECKPOINT_FILENAME,
    MACHINE_IDS,
    MACHINES,
    MAHA_STATS_FILENAME,
    SNRS,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.inference import load_mahalanobis_stats

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _safe_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())
        return torch.device("cuda")
    except Exception:
        logger.warning("CUDA detected but not functional — falling back to CPU.")
        return torch.device("cpu")


def is_supported_combination(machine: str, machine_id: str, snr: str) -> bool:
    return machine in MACHINES and machine_id in MACHINE_IDS and snr in SNRS


def _require_supported(machine: str, machine_id: str, snr: str) -> None:
    if not is_supported_combination(machine, machine_id, snr):
        raise UnsupportedMachineConfigurationError(
            f"(machine={machine!r}, machine_id={machine_id!r}, snr={snr!r}) is not one "
            f"of the 48 trained combinations. Valid values: machine in {MACHINES}, "
            f"machine_id in {MACHINE_IDS}, snr in {SNRS}."
        )


def combination_dir(machine: str, machine_id: str, snr: str) -> str:
    """Relative path (inside ARTIFACT_FOLDER_NAME) to a combination's checkpoint dir."""
    return f"vit_tiny_{machine}/{machine}/{machine_id}/{snr}"


def build_model(machine: str, device: torch.device) -> AudioMAE:
    """Instantiate AudioMAE with the per-machine architecture — do this BEFORE
    load_state_dict, the checkpoint itself does not self-describe depth/num_heads."""
    arch = ARCHITECTURE_BY_MACHINE[machine]
    model = AudioMAE(
        img_size=arch["img_size"],
        patch_size=arch["patch_size"],
        in_chans=arch["in_chans"],
        embed_dim=arch["embed_dim"],
        depth=arch["depth"],
        num_heads=arch["num_heads"],
        decoder_embed_dim=arch["decoder_embed_dim"],
        decoder_depth=arch["decoder_depth"],
        decoder_num_heads=arch["decoder_num_heads"],
        norm_pix_loss=arch["norm_pix_loss"],
    ).to(device)
    return model


@dataclass
class LoadedCombination:
    model: AudioMAE
    norm_mean: float
    norm_std: float
    maha_mean: np.ndarray
    maha_inv_cov: np.ndarray
    maha_pca: Optional[Tuple[np.ndarray, np.ndarray]]
    device: torch.device


def load_checkpoint(machine: str, machine_id: str, snr: str, device: torch.device) -> LoadedCombination:
    """Load one (machine, machine_id, snr) checkpoint + Mahalanobis stats from disk."""
    _require_supported(machine, machine_id, snr)
    combo_dir = combination_dir(machine, machine_id, snr)

    ckpt_path = _store.path(f"{combo_dir}/{CHECKPOINT_FILENAME}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(machine, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_mean = float(checkpoint.get("norm_mean", 0.0))
    norm_std = float(checkpoint.get("norm_std", 1.0))

    maha_path = _store.path(f"{combo_dir}/{MAHA_STATS_FILENAME}")
    maha_mean, maha_inv_cov, maha_pca = load_mahalanobis_stats(str(maha_path))

    return LoadedCombination(
        model=model, norm_mean=norm_mean, norm_std=norm_std,
        maha_mean=maha_mean, maha_inv_cov=maha_inv_cov, maha_pca=maha_pca,
        device=device,
    )


class CheckpointCache:
    """Bounded LRU cache of LoadedCombination, keyed by (machine, machine_id, snr)."""

    def __init__(self, maxsize: int = CHECKPOINT_CACHE_SIZE) -> None:
        self._maxsize = maxsize
        self._cache: "OrderedDict[tuple, LoadedCombination]" = OrderedDict()

    def get(self, machine: str, machine_id: str, snr: str, device: torch.device) -> LoadedCombination:
        key = (machine, machine_id, snr)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        loaded = load_checkpoint(machine, machine_id, snr, device)
        self._cache[key] = loaded
        if len(self._cache) > self._maxsize:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.info("Evicting cached checkpoint: %s", evicted_key)
        return loaded

    def clear(self) -> None:
        self._cache.clear()


def artifacts_available() -> bool:
    """True if the artifact directory exists locally with at least one valid combination."""
    if not _store.local_dir.exists():
        return False
    for machine in MACHINES:
        for machine_id in MACHINE_IDS:
            for snr in SNRS:
                combo_dir = combination_dir(machine, machine_id, snr)
                ckpt = _store.local_dir / combo_dir / CHECKPOINT_FILENAME
                if ckpt.exists():
                    return True
    return False


def ensure_artifacts_downloaded() -> None:
    """download_all_if_needed() if STORAGE_BUCKET is set — mirrors m47/ml35 pattern."""
    if os.environ.get("STORAGE_BUCKET"):
        _store.download_all_if_needed()
