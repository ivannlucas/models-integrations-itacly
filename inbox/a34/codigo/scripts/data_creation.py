"""
Script: Data Creation

Generates only the raw synthetic pasteurization dataset from the V3.4
PID/supervisor simulator.

Usage:
    python scripts/data_creation.py
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.simulator import generar_dataset_pasteurizacion


if __name__ == "__main__":
    df_simulado = generar_dataset_pasteurizacion(save=True)

    print("Dataset V3.4 generado. Comprobando temperatura de pasteurizacion:")
    print(df_simulado["T_out_leche"].describe().round(2).to_string())
    registros_bajo_objetivo = (df_simulado["T_out_leche"] < 72.3).sum()
    print(f"Registros con T_out_leche < 72.3 C: {registros_bajo_objetivo}")
