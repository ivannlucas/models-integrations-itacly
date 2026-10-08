"""Enological reference rules used to build the 72 h targets (vendored from src/utils/enology.py).

Only needed by train(): the risk target is derived from the future free SO2 with these rules.
"""

from __future__ import annotations

import math

import numpy as np

BASE_FREE_SO2_TARGETS = {
    "must": 10.0,
    "fermentation": 14.0,
    "stabilization": 20.0,
    "storage": 24.0,
    "pre_bottling": 28.0,
}

WINE_TYPE_TARGET_OFFSETS = {
    "red": 0.0,
    "rose": 1.0,
    "white": 2.0,
}


def target_free_sulfite(wine_type: str, stage: str, ph: float) -> float:
    """Return the reference free SO2 level (mg/L) for a wine type, stage and pH."""
    base = BASE_FREE_SO2_TARGETS.get(stage, BASE_FREE_SO2_TARGETS["storage"])
    wine_adjustment = WINE_TYPE_TARGET_OFFSETS.get(wine_type, 0.0)
    ph_adjustment = max(0.0, ph - 3.30) * 8.0
    return float(base + wine_adjustment + ph_adjustment)


def estimate_underprotection_risk(  # pylint: disable=too-many-arguments  # vendored, keyword-only
    *,
    wine_type: str,
    stage: str,
    ph_lab: float,
    future_free_sulfite: float,
    dissolved_oxygen_mg_l: float,
    temperature_c: float,
    hours_since_last_lab: float,
) -> float:
    """Return the synthetic underprotection score in [0, 1] for a future lot state."""
    target = target_free_sulfite(wine_type, stage, ph_lab)
    gap = max(0.0, target - future_free_sulfite)
    score = (
        0.40 * gap
        + 0.14 * max(0.0, dissolved_oxygen_mg_l - 1.6)
        + 0.035 * max(0.0, temperature_c - 20.0)
        + 0.015 * max(0.0, hours_since_last_lab - 12.0)
    )
    return float(np.clip(1.0 / (1.0 + math.exp(-((score - 2.0) / 1.6))), 0.0, 1.0))
