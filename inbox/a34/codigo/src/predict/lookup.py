"""
Optimization Lookup Table query.

Implements query_setpoints() which, given sensor readings,
returns the closest optimal setpoints from the pre-computed table.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.paths import GA_V4_RESULTS_PATH


def query_setpoints(
    df_lookup: pd.DataFrame,
    T_in: float,
    Delta_P: float,
    t_ciclo: float,
) -> Optional[Dict]:
    """
    Given a sensor state, return the optimal setpoints from the Lookup Table.

    Snaps values to the mesh resolution and finds the closest scenario.
    In production, this function would run every PLC control cycle.

    Parameters
    ----------
    df_lookup : pd.DataFrame
        Full Lookup Table (ga_v3_optimization_results.csv).
    T_in : float
        Milk inlet temperature (C).
    Delta_P : float
        Pressure drop (bar).
    t_ciclo : float
        Time since last CIP cleaning (min).

    Returns
    -------
    dict or None
        Dictionary with: query, snap, setpoints, prediction, feasible.
        None if no match is found.
    """
    # Snap to mesh resolution
    T_in_snap = round(round(T_in / 0.2) * 0.2, 1)
    dp_snap = round(round(Delta_P / 0.1) * 0.1, 1)
    t_ciclo_snap = round(round(t_ciclo / 50) * 50)

    # Clamp to mesh limits
    T_in_snap = max(0.0, min(8.0, T_in_snap))
    dp_snap = max(0.4, min(1.0, dp_snap))
    t_ciclo_snap = max(0, min(800, t_ciclo_snap))

    # Search in table
    match = df_lookup[
        (np.isclose(df_lookup["T_in_leche"], T_in_snap))
        & (np.isclose(df_lookup["Delta_P"], dp_snap))
        & (df_lookup["t_ciclo"] == t_ciclo_snap)
    ]

    if len(match) == 0:
        return None

    row = match.iloc[0]
    return {
        "query": {"T_in": T_in, "Delta_P": Delta_P, "t_ciclo": t_ciclo},
        "snap": {"T_in": T_in_snap, "Delta_P": dp_snap, "t_ciclo": t_ciclo_snap},
        "setpoints": {
            "F_flow_optimal": row["F_flow_optimo"],
            "T_servicio_optimal": row["T_servicio_optimo"],
        },
        "prediction": {
            "E_consumo": row["E_consumo_pred"],
            "T_out": row["T_out_pred"],
            "specific_consumption": row["consumo_especifico"],
        },
        "feasible": row["factible"],
    }


def load_lookup_table(path: str = None) -> pd.DataFrame:
    """
    Load the optimization Lookup Table.

    Parameters
    ----------
    path : str, optional
        Path to CSV. If None, uses the standard project path.

    Returns
    -------
    pd.DataFrame
    """
    p = path or str(GA_V4_RESULTS_PATH)
    return pd.read_csv(p)
