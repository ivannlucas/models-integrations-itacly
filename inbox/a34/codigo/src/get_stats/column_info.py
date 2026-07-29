"""
Dataset column inventory and description.

Provides get_column_info() which returns a structured description of every
column in the dataset, useful for documentation and auditing purposes.
"""

from typing import Dict, List


def get_column_info() -> List[Dict[str, str]]:
    """
    Return descriptive metadata for each column in the pasteurization dataset.

    Returns
    -------
    list[dict]
        Each dict has keys: name, symbol, description, role, unit.

    Example
    -------
    >>> from src.get_stats.column_info import get_column_info
    >>> for col in get_column_info():
    ...     print(f"{col['name']:15s} ({col['unit']:>5s}) — {col['description']}")
    """
    return [
        {
            "name": "Time_min",
            "symbol": "t",
            "description": "Simulation timestamp (cumulative minutes from start)",
            "role": "index",
            "unit": "min",
        },
        {
            "name": "T_in_leche",
            "symbol": "T_c,in",
            "description": "Raw milk inlet temperature (seasonal + noise)",
            "role": "feature — disturbance",
            "unit": "°C",
        },
        {
            "name": "F_flow",
            "symbol": "V_dot",
            "description": "Milk volumetric flow rate (controllable via pump)",
            "role": "feature — controllable / GA decision variable",
            "unit": "L/h",
        },
        {
            "name": "T_servicio",
            "symbol": "T_h,in",
            "description": "Heating fluid temperature (manipulated by PID / GA decision variable)",
            "role": "feature — controllable / GA decision variable",
            "unit": "°C",
        },
        {
            "name": "t_ciclo",
            "symbol": "t_cip",
            "description": "Elapsed time since last CIP cleaning (state indicator of fouling age)",
            "role": "feature — state",
            "unit": "min",
        },
        {
            "name": "Delta_P",
            "symbol": "ΔP",
            "description": "Pressure drop across the plate heat exchanger (fouling proxy)",
            "role": "feature — state",
            "unit": "bar",
        },
        {
            "name": "E_consumo",
            "symbol": "P_total",
            "description": "Instantaneous total energy consumption (thermal + pumping)",
            "role": "target (KPI to minimize)",
            "unit": "kW",
        },
        {
            "name": "T_out_leche",
            "symbol": "T_c,out",
            "description": "Pasteurized milk outlet temperature (safety constraint >= 72.3 °C)",
            "role": "target / constraint",
            "unit": "°C",
        },
        {
            "name": "Is_Cleaning",
            "symbol": "—",
            "description": "CIP cleaning flag (1 = cleaning in progress, 0 = production)",
            "role": "auxiliary (used for filtering, not a model feature)",
            "unit": "0/1",
        },
    ]


def print_column_info() -> None:
    """Print a formatted table of column metadata to stdout."""
    cols = get_column_info()
    header = f"{'Column':<15s} {'Symbol':<10s} {'Role':<45s} {'Unit':<6s} Description"
    print(header)
    print("-" * len(header))
    for c in cols:
        print(
            f"{c['name']:<15s} {c['symbol']:<10s} {c['role']:<45s} "
            f"{c['unit']:<6s} {c['description']}"
        )
