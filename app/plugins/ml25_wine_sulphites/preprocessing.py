"""Preprocessing functions for the wine sulphite plugin."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature sets — order is mandatory (matches training column order)
FEATURES_PHYS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
FEATURES_QUAL = FEATURES_PHYS + ["free sulfur dioxide", "total sulfur dioxide"]
FEATURES_BOUND = FEATURES_PHYS + ["free sulfur dioxide"]

# Dissociation constant for SO2 molecular calculation
PKA_SO2 = 1.81


def map_request_to_wine_dict(request: Any) -> dict:
    """Map snake_case API fields to the space-separated column names used at training time.

    Accepts any object with the expected attributes (Pydantic model, SimpleNamespace, etc.).
    """
    return {
        "fixed acidity": request.fixed_acidity,
        "volatile acidity": request.volatile_acidity,
        "citric acid": request.citric_acid,
        "residual sugar": request.residual_sugar,
        "chlorides": request.chlorides,
        "density": request.density,
        "pH": request.pH,
        "sulphates": request.sulphates,
        "alcohol": request.alcohol,
        "free sulfur dioxide": request.free_sulfur_dioxide,
        "total sulfur dioxide": request.total_sulfur_dioxide,
    }


def build_simulation_grid(
    base_wine: dict,
    delta_max: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build simulation grid from current free SO2 up to current + delta_max."""
    try:
        current_free = float(base_wine["free sulfur dioxide"])
    except KeyError as exc:
        logger.error("Missing key 'free sulfur dioxide' in base_wine: %s", exc)
        raise ValueError("base_wine must contain 'free sulfur dioxide'") from exc
    except (TypeError, ValueError) as exc:
        logger.error("Invalid 'free sulfur dioxide' value in base_wine: %s", exc)
        raise ValueError("'free sulfur dioxide' must be a numeric value") from exc

    if delta_max < 0:
        logger.error("delta_max cannot be negative: %s", delta_max)
        raise ValueError("delta_max must be non-negative")

    try:
        free_targets = np.arange(current_free, current_free + delta_max + 1e-9, 1.0)
        n = len(free_targets)

        phys_base = {k: base_wine[k] for k in FEATURES_PHYS}
    except KeyError as exc:
        logger.error("Missing key in base_wine: %s", exc)
        raise ValueError(f"base_wine missing required key: {exc}") from exc

    phys_df = pd.DataFrame([phys_base] * n)

    bound_rows = phys_df.copy()
    bound_rows["free sulfur dioxide"] = free_targets

    # total SO2 placeholder — will be replaced after bound prediction
    qual_rows = phys_df.copy()
    qual_rows["free sulfur dioxide"] = free_targets
    try:
        qual_rows["total sulfur dioxide"] = float(base_wine["total sulfur dioxide"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Invalid or missing 'total sulfur dioxide': %s", exc)
        raise ValueError("base_wine must contain a numeric 'total sulfur dioxide'") from exc

    return free_targets, qual_rows[FEATURES_QUAL], bound_rows[FEATURES_BOUND]
