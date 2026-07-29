"""
Baseline KPI computation for the pasteurization process (PID control).

Encapsulates the reference metrics calculated in baseline.ipynb.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.utils.constants import T_SAFETY, SAMPLING_FREQ


def compute_baseline_kpis(df_prod: pd.DataFrame) -> Dict:
    """
    Compute reference metrics for the current process (PID).

    Parameters
    ----------
    df_prod : pd.DataFrame
        Production dataset (Is_Cleaning == 0), with columns:
        E_consumo, F_flow, T_out_leche, T_servicio, t_ciclo, Delta_P.

    Returns
    -------
    dict
        Dictionary with all baseline metrics aligned with
        GA optimization reports.
    """
    dt_h = SAMPLING_FREQ / 60  # hours per record

    # Consumption
    E_mean = float(df_prod["E_consumo"].mean())
    E_std = float(df_prod["E_consumo"].std())
    E_total_kwh = float((df_prod["E_consumo"] * dt_h).sum())

    # Production
    F_mean = float(df_prod["F_flow"].mean())
    F_std = float(df_prod["F_flow"].std())
    vol_total = float((df_prod["F_flow"] * dt_h).sum())

    # Specific consumption E/F
    df_prod = df_prod.copy()
    df_prod["E_over_F"] = df_prod["E_consumo"] / df_prod["F_flow"]
    EF_mean = float(df_prod["E_over_F"].mean())
    EF_std = float(df_prod["E_over_F"].std())

    # Service temperature
    Ts_mean = float(df_prod["T_servicio"].mean())
    Ts_std = float(df_prod["T_servicio"].std())

    # Thermal safety
    T_out_mean = float(df_prod["T_out_leche"].mean())
    T_out_min = float(df_prod["T_out_leche"].min())
    mean_margin = T_out_mean - T_SAFETY
    compliance_rate = float((df_prod["T_out_leche"] >= T_SAFETY).mean() * 100)

    # Fouling
    delta_T_serv, delta_DP_pct = _compute_fouling_impact(df_prod)

    return {
        "E_consumo_mean_kW": round(E_mean, 2),
        "E_consumo_std_kW": round(E_std, 2),
        "E_total_kWh": round(E_total_kwh, 0),
        "F_flow_mean_Lh": round(F_mean, 1),
        "F_flow_std_Lh": round(F_std, 1),
        "vol_total_L": round(vol_total, 0),
        "specific_consumption_kW_per_Lh": round(EF_mean, 5),
        "T_servicio_mean_C": round(Ts_mean, 2),
        "T_servicio_std_C": round(Ts_std, 2),
        "T_out_mean_C": round(T_out_mean, 2),
        "T_out_min_C": round(T_out_min, 2),
        "mean_margin_C": round(mean_margin, 2),
        "compliance_rate_pct": round(compliance_rate, 1),
        "delta_T_serv_fouling_C": round(delta_T_serv, 2),
        "delta_DP_fouling_pct": round(delta_DP_pct, 0),
    }


def _compute_fouling_impact(df_prod: pd.DataFrame):
    """
    Compute fouling impact on T_servicio and Delta_P between cycle start and end.

    Returns
    -------
    tuple[float, float]
        (average delta_T_servicio in C, average delta_DP percentage in %)
    """
    df_sorted = df_prod.sort_values("Time_min").reset_index(drop=True)
    df_sorted["cycle_id"] = (df_sorted["t_ciclo"].diff() < -10).cumsum()

    cycle_stats = df_sorted.groupby("cycle_id").agg(
        T_serv_start=("T_servicio", "first"),
        T_serv_end=("T_servicio", "last"),
        DP_start=("Delta_P", "first"),
        DP_end=("Delta_P", "last"),
        duration_min=("t_ciclo", "max"),
    ).reset_index()

    cycle_stats = cycle_stats[cycle_stats["duration_min"] > 50]

    if len(cycle_stats) == 0:
        return 0.0, 0.0

    delta_T = float((cycle_stats["T_serv_end"] - cycle_stats["T_serv_start"]).mean())
    delta_DP = float(((cycle_stats["DP_end"] / cycle_stats["DP_start"]).mean() - 1) * 100)

    return delta_T, delta_DP
