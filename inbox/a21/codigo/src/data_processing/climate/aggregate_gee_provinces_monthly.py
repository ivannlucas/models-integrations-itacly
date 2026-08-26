#!/usr/bin/env python3
"""Agrega los CSVs de clima GEE por provincia a un único CSV mensual.

Lee todos los CSVs de climate_provinces_GEE_by_year/ y produce un CSV mensual
con temperatura en Celsius y precipitación en mm.

Formato de entrada (generado por ingest_weather_gee.py):
  Columnas: date_ms (epoch ms), province_name, temp_k, temp_std_k,
            precip_m (m/día), precip_std_m

Transformaciones:
  - date_ms  → year, month
  - temp_k   → temp_mean_C   = mean(temp_k  - 273.15) por mes
  - temp_std_k → temp_std_C  = mean(temp_std_k)
  - precip_m  → precip_total_mm = sum(precip_m  * 1000) por mes
  - precip_std_m → precip_std_mm = mean(precip_std_m * 1000)

Salida:
  Columnas: year, month, province_name, temp_mean_C, temp_std_C,
            precip_total_mm, precip_std_mm

Uso:
  python src/data_processing/climate/aggregate_gee_provinces_monthly.py
  python src/data_processing/climate/aggregate_gee_provinces_monthly.py --src "data/..." --out "data/..."
"""
import argparse
import glob
import os

import pandas as pd

DEFAULT_SRC = "data/processed/manual/climate_provinces_GEE_by_year/*.csv"
DEFAULT_OUT = "data/processed/auto/utils/climate_monthly_provinces.csv"

# Si date_ms supera este umbral, asumimos que está en milisegundos y lo dividimos.
EPOCH_MS_THRESHOLD = 1e10


def load_file(path: str) -> pd.DataFrame:
    """Carga un CSV GEE y devuelve DataFrame con columnas estandarizadas."""
    df = pd.read_csv(path)

    # ── Fecha ──────────────────────────────────────────────────────────────────
    if "date_ms" not in df.columns:
        raise ValueError(f"Columna 'date_ms' no encontrada en {path}")

    date_ms = df["date_ms"].astype(float)
    # Si los valores están en ms, convertir a segundos
    if date_ms.max() > EPOCH_MS_THRESHOLD:
        date_ms = date_ms / 1000.0
    dates = pd.to_datetime(date_ms, unit="s", utc=True).dt.tz_localize(None)
    df["year"]  = dates.dt.year
    df["month"] = dates.dt.month

    # ── Provincia ──────────────────────────────────────────────────────────────
    if "province_name" not in df.columns:
        raise ValueError(f"Columna 'province_name' no encontrada en {path}")

    # ── Temperatura (Kelvin → Celsius) ─────────────────────────────────────────
    if "temp_k" not in df.columns:
        raise ValueError(f"Columna 'temp_k' no encontrada en {path}")
    df["_temp_c"]     = df["temp_k"] - 273.15
    df["_temp_std_c"] = df["temp_std_k"] if "temp_std_k" in df.columns else 0.0

    # ── Precipitación (m → mm) ─────────────────────────────────────────────────
    if "precip_m" not in df.columns:
        raise ValueError(f"Columna 'precip_m' no encontrada en {path}")
    df["_precip_mm"]     = df["precip_m"] * 1000.0
    df["_precip_std_mm"] = df["precip_std_m"] * 1000.0 if "precip_std_m" in df.columns else 0.0

    return df[["year", "month", "province_name",
               "_temp_c", "_temp_std_c", "_precip_mm", "_precip_std_mm"]]


def main(src_glob: str, out_path: str) -> None:
    files = sorted(glob.glob(src_glob))
    if not files:
        raise SystemExit(f"No se encontraron ficheros con el patrón: {src_glob}")

    print(f"Procesando {len(files)} ficheros...")
    parts = []
    for f in files:
        try:
            parts.append(load_file(f))
        except Exception as exc:
            print(f"  SKIP {os.path.basename(f)}: {exc}")

    if not parts:
        raise SystemExit("No se cargaron datos. Revisa los ficheros de entrada.")

    df = pd.concat(parts, ignore_index=True)

    # Agregar por provincia × año × mes
    agg = (
        df.groupby(["year", "month", "province_name"], sort=True)
        .agg(
            temp_mean_C     = ("_temp_c",       "mean"),
            temp_std_C      = ("_temp_std_c",   "mean"),
            precip_total_mm = ("_precip_mm",    "sum"),
            precip_std_mm   = ("_precip_std_mm","mean"),
        )
        .reset_index()
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print(
        f"Escrito: {out_path}\n"
        f"  shape={agg.shape}  "
        f"años={agg['year'].min()}–{agg['year'].max()}  "
        f"provincias={agg['province_name'].nunique()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agrega CSVs diarios GEE a CSV mensual por provincia"
    )
    parser.add_argument(
        "--src", default=DEFAULT_SRC,
        help="Patrón glob de los CSVs de entrada (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Ruta del CSV de salida (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.src, args.out)
