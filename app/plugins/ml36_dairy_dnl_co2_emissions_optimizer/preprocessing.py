"""Feature engineering for the ml36 MLP — port of ga_model_pipeline._build_feature_matrix_for_individual."""
# pylint: disable=invalid-name
from __future__ import annotations

import numpy as np
import pandas as pd

from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.constants import (
    FEATURE_ORDER,
    GA_BOUNDS,
    T_OUT_REFERENCE,
    THERMAL_INTENSITY_DIVISOR,
)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with Hydraulic_Resistance, T_serv_margin and Thermal_Intensity computed.

    Requires the 8 base columns. Formulas match the delivered simulator/pipeline and
    were verified against data/splits/test_raw.csv.
    """
    out = df.copy()
    f_milk = out["F_milk"].astype(float)
    denom = (f_milk ** 2).replace(0.0, np.nan)
    out["Hydraulic_Resistance"] = (out["Delta_P"].astype(float) / denom).fillna(0.0)
    out["T_serv_margin"] = out["T_serv"].astype(float) - T_OUT_REFERENCE
    out["Thermal_Intensity"] = (
        (T_OUT_REFERENCE - out["T_in"].astype(float))
        * (1.0 - out["Regeneration_perc"].astype(float) / 100.0)
    ) / THERMAL_INTENSITY_DIVISOR
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features + the 11 columns in scaler_X order."""
    return add_derived_features(df)[FEATURE_ORDER]


def clip_individual(individual) -> list[float]:
    """Clip (T_serv, Delta_P, Regeneration_perc) into the GA operating bounds."""
    t_serv, delta_p, regen = (float(g) for g in individual)
    return [
        float(np.clip(t_serv, *GA_BOUNDS["T_serv"])),
        float(np.clip(delta_p, *GA_BOUNDS["Delta_P"])),
        float(np.clip(regen, *GA_BOUNDS["Regeneration_perc"])),
    ]
