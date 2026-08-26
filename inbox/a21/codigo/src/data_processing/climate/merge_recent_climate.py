#!/usr/bin/env python3
"""Agrega a mensual el CSV diario producido por ingest_weather_gee.py y lo
fusiona (upsert por año/mes/provincia) en el panel climático mensual.

Lee `data/external/clima_provincias_GEE_daily.csv` (columnas date/provincia/
temperature/precipitation, ya en Celsius/mm) y agrega por mes de calendario
real. Deliberadamente NO se usa el CSV semanal (`clima_provincias_GEE.csv`):
agregar por semana ISO asignaría toda una semana al mes de su lunes de
inicio, lo que en los límites de mes/año mezcla días del mes siguiente con
el anterior.

Uso:
  python -m src.data_processing.climate.merge_recent_climate
  python -m src.data_processing.climate.merge_recent_climate --src "data/external/clima_provincias_GEE_daily.csv" --out "data/processed/auto/utils/climate_monthly_provinces.csv"
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SRC = "data/external/clima_provincias_GEE_daily.csv"
DEFAULT_OUT = "data/processed/auto/utils/climate_monthly_provinces.csv"


def aggregate_daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month

    agg = (
        daily.groupby(["year", "month", "provincia"], sort=True)
        .agg(
            temp_mean_C=("temperature", "mean"),
            temp_std_C=("temperature", "std"),
            precip_total_mm=("precipitation", "sum"),
            precip_std_mm=("precipitation", "std"),
        )
        .reset_index()
        .rename(columns={"provincia": "province_name"})
    )
    agg["temp_std_C"] = agg["temp_std_C"].fillna(0.0)
    agg["precip_std_mm"] = agg["precip_std_mm"].fillna(0.0)
    return agg


def main(src: str, out: str) -> None:
    src_path = Path(src)
    out_path = Path(out)
    if not src_path.exists():
        raise SystemExit(f"No existe {src_path}. Ejecuta antes ingest_weather_gee.py.")

    daily = pd.read_csv(src_path)
    monthly_new = aggregate_daily_to_monthly(daily)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        key = ["year", "month", "province_name"]
        existing = existing[~existing.set_index(key).index.isin(monthly_new.set_index(key).index)]
        merged = pd.concat([existing, monthly_new], ignore_index=True)
    else:
        merged = monthly_new

    merged = merged.sort_values(["year", "month", "province_name"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    meses = sorted(monthly_new[["year", "month"]].drop_duplicates().apply(tuple, axis=1))
    print(
        f"Fusionado: {out_path}\n"
        f"  meses nuevos/actualizados: {meses}\n"
        f"  shape final={merged.shape}  provincias={merged['province_name'].nunique()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.src, args.out)
