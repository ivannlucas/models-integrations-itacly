#!/usr/bin/env python3
"""Descarga el tipo de cambio EUR/USD semanal desde Yahoo Finance.

Descarga la serie histórica del par EUR/USD, la remuestrea a frecuencia semanal
(inicio en lunes, tomando el último cierre disponible) y rellena huecos hacia adelante.

Entradas:
  - Yahoo Finance API (ticker EURUSD=X vía yfinance)

Salidas:
  - data/utils/eur_usd_weekly.csv
    (columnas: week_start, EUR_USD; una fila por semana ISO)

Uso:
  python src/data_processing/ingestion/ingest_eurusd_weekly.py
  python src/data_processing/ingestion/ingest_eurusd_weekly.py --start 2010-01-01
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "auto" / "utils" / "eur_usd_weekly.csv"


def main(start: str = "2000-01-01", output: Path | str = DEFAULT_OUTPUT) -> int:
    try:
        import yfinance as yf
    except Exception as e:
        logging.error("yfinance no está instalado. Instálalo con: pip install yfinance")
        return 2

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("=== OBTENIENDO DATOS MACRO (EUR/USD) ===")
    logging.info("Rango desde %s", start)

    # Descargar histórico
    try:
        eur_usd = yf.download('EURUSD=X', start=start, progress=False)
    except Exception as e:
        logging.exception("Error descargando datos desde yfinance: %s", e)
        return 3

    if eur_usd.empty:
        logging.warning("La descarga no devolvió datos. Archivo no creado.")
        return 4

    # Re-muestreo semanal (inicio lunes), tomar el último valor disponible (cierre)
    df_forex = eur_usd['Close'].resample('W-MON').last()

    # Normalizar a DataFrame con columna `EUR_USD`
    if isinstance(df_forex, pd.Series):
        df_forex = df_forex.to_frame(name='EUR_USD')
    else:
        df_forex.columns = ['EUR_USD']

    # Rellenar huecos hacia adelante
    df_forex = df_forex.ffill()

    # Guardar CSV con índice como week_start
    df_forex.index.name = 'week_start'
    df_forex.to_csv(out_path, index=True)
    logging.info("Guardado CSV en %s | filas=%d", out_path, len(df_forex))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga EUR/USD desde yfinance y guarda un CSV semanal.")
    parser.add_argument("--start", type=str, default="2000-01-01", help="Fecha de inicio (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Ruta de salida del CSV")
    args = parser.parse_args()

    sys.exit(main(start=args.start, output=args.output))
