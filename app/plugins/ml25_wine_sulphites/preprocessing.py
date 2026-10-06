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

# Columna objetivo del entrenamiento.
TARGET_QUAL = "quality"

# El esquema de CSV que la plataforma le documenta al usuario usa guiones bajos
# (fixed_acidity, free_sulfur_dioxide...), pero los artefactos se entrenaron con
# los nombres separados por espacios de FEATURES_QUAL y las rutas de predicción
# construyen los dicts con esos mismos nombres (map_request_to_wine_dict). La
# convención con espacios es la canónica: un modelo reentrenado con otra tiene
# feature_names_in_ distintos y deja de servir a predict.
TRAIN_COLUMN_ALIASES = {c.replace(" ", "_"): c for c in FEATURES_QUAL if " " in c}

# Dissociation constant for SO2 molecular calculation
PKA_SO2 = 1.81


def normalize_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombrar las columnas del CSV de entrenamiento a la convención canónica.

    Acepta tanto los nombres con espacios de los artefactos como los que la
    plataforma documenta con guiones bajos. Solo renombra el alias cuando la
    columna canónica no viene ya en el fichero, así que un CSV que traiga las
    dos formas conserva la canónica.
    """
    renames = {
        alias: canonical
        for alias, canonical in TRAIN_COLUMN_ALIASES.items()
        if alias in df.columns and canonical not in df.columns
    }
    return df.rename(columns=renames) if renames else df


def require_training_columns(df: pd.DataFrame) -> None:
    """Fallar con un ValueError que nombre lo que falta, no con un KeyError."""
    missing = [c for c in FEATURES_QUAL if c not in df.columns]
    if TARGET_QUAL not in df.columns:
        missing.append(TARGET_QUAL)
    if missing:
        raise ValueError(
            "El CSV de entrenamiento no tiene las columnas: "
            + ", ".join(missing)
            + ". Se aceptan los nombres con espacios o con guiones bajos."
        )


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
    free_so2_p99: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build simulation grid from current free SO2 up to current + delta_max.

    ``free_so2_p99`` caps the upper bound at the 99th percentile of free SO2 in the
    training dataset — matches ``wine_quality.common.build_free_grid`` /
    ``SimulationConfig(sim_free_p_high=99.0)`` in the real delivered inference
    pipeline, which never explores doses far outside the training distribution
    regardless of how large ``delta_max`` is. ``None`` disables the cap (legacy
    behaviour) — callers should always pass it when available.
    """
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
        hi = current_free + delta_max
        if free_so2_p99 is not None:
            hi = min(hi, max(float(free_so2_p99), current_free))
        free_targets = np.arange(current_free, hi + 1e-9, 1.0)
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
