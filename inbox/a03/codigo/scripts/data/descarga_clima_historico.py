"""
==============================================================================
descarga_clima_historico.py
==============================================================================
Descarga datos meteorológicos horarios históricos de la API de Open-Meteo
para 5 estaciones asociadas a las principales Denominaciones de Origen
vitivinícolas de Castilla y León.

Fuente: Open-Meteo Historical Weather API (gratuita, sin API key)
Rango:  2015-01-01 a 2025-12-31
Salida: Un fichero Parquet por estación en data/clima_real/

Uso:
    python descarga_clima_historico.py
==============================================================================
"""

import requests
import pandas as pd
import time
import sys
from pathlib import Path
import yaml

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================

API_URL = "https://archive-api.open-meteo.com/v1/archive"

ANIO_INICIO = 2015
ANIO_FIN = 2025

# Estaciones meteorológicas representativas de las D.O. de Castilla y León
ESTACIONES = {
    "ribera_del_duero": {
        "lat": 41.69, "lon": -3.93, "elev": 750,
        "nombre": "Roa de Duero (D.O. Ribera del Duero)"
    },
    "rueda": {
        "lat": 41.41, "lon": -4.96, "elev": 720,
        "nombre": "Rueda (D.O. Rueda)"
    },
    "toro": {
        "lat": 41.52, "lon": -5.39, "elev": 620,
        "nombre": "Toro (D.O. Toro)"
    },
    "bierzo": {
        "lat": 42.60, "lon": -6.73, "elev": 480,
        "nombre": "Cacabelos (D.O. Bierzo)"
    },
    "cigales": {
        "lat": 41.76, "lon": -4.70, "elev": 725,
        "nombre": "Cigales (D.O. Cigales)"
    },
}

# Variables horarias a descargar de Open-Meteo
VARIABLES_API = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Mapeo de nombres API → nombres del proyecto
RENOMBRAR_COLUMNAS = {
    "temperature_2m":       "Temp_Amb_C",
    "relative_humidity_2m": "Hum_Rel_Pct",
    "precipitation":        "Lluvia_mm",
    "wind_speed_10m":       "Viento_kmh",
    "shortwave_radiation":  "Radiacion_Wm2",
}

# Rutas de salida
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

OUTPUT_DIR = PROJECT_ROOT / config_yaml['paths']['clima_real_dir']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. FUNCIONES DE DESCARGA
# ==============================================================================

def descargar_anio(lat: float, lon: float, year: int, max_reintentos: int = 3) -> pd.DataFrame:
    """
    Descarga datos horarios de un año completo desde Open-Meteo.
    Incluye reintentos automáticos ante fallos de red.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(VARIABLES_API),
        "timezone": "Europe/Madrid",
    }

    for intento in range(1, max_reintentos + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                print(f"    ⚠️ Respuesta sin datos horarios para {year}. Reintentando...")
                time.sleep(5)
                continue

            df = pd.DataFrame(data["hourly"])
            df["Fecha"] = pd.to_datetime(df["time"])
            df.drop(columns=["time"], inplace=True)
            df.rename(columns=RENOMBRAR_COLUMNAS, inplace=True)

            return df

        except requests.exceptions.RequestException as e:
            print(f"    ❌ Error de red (intento {intento}/{max_reintentos}): {e}")
            if intento < max_reintentos:
                time.sleep(10 * intento)  # Backoff exponencial
            else:
                print(f"    🚫 No se pudo descargar el año {year} tras {max_reintentos} intentos.")
                return pd.DataFrame()


def descargar_estacion(id_estacion: str, config: dict) -> None:
    """
    Descarga todos los años (2015-2025) para una estación y guarda en Parquet.
    """
    print(f"\n📡 Descargando: {config['nombre']}")
    print(f"   Coordenadas: ({config['lat']}, {config['lon']}), Elevación: {config['elev']}m")

    frames = []

    for year in range(ANIO_INICIO, ANIO_FIN + 1):
        print(f"   📅 Año {year}...", end=" ")
        df_year = descargar_anio(config["lat"], config["lon"], year)

        if df_year.empty:
            print("⚠️ VACÍO")
            continue

        frames.append(df_year)
        print(f"✅ {len(df_year)} registros")

        # Pausa entre peticiones para no saturar la API
        time.sleep(1.0)

    if not frames:
        print(f"   🚫 No se obtuvieron datos para {id_estacion}. Saltando.")
        return

    # Concatenar todos los años
    df_completo = pd.concat(frames, ignore_index=True)
    df_completo.sort_values("Fecha", inplace=True)
    df_completo.reset_index(drop=True, inplace=True)

    # Añadir metadatos de la parcela
    df_completo["Parcela_ID"] = id_estacion
    df_completo["Latitud"] = config["lat"]
    df_completo["Longitud"] = config["lon"]
    df_completo["Elevacion_m"] = config["elev"]

    # Guardar
    out_path = OUTPUT_DIR / f"clima_{id_estacion}.parquet"
    df_completo.to_parquet(out_path, index=False)

    # Resumen
    n_nulos = df_completo[list(RENOMBRAR_COLUMNAS.values())].isnull().sum().sum()
    print(f"   💾 Guardado: {out_path.name}")
    print(f"   📊 Total: {len(df_completo)} registros | Rango: {df_completo['Fecha'].min()} → {df_completo['Fecha'].max()}")
    print(f"   🔍 Valores nulos en variables climáticas: {n_nulos}")


# ==============================================================================
# 3. EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("=" * 70)
    print("  DESCARGA DE DATOS METEOROLÓGICOS HISTÓRICOS")
    print(f"  Fuente: Open-Meteo Historical Weather API")
    print(f"  Rango: {ANIO_INICIO} - {ANIO_FIN}")
    print(f"  Estaciones: {len(ESTACIONES)}")
    print(f"  Salida: {OUTPUT_DIR}")
    print("=" * 70)

    inicio = time.time()

    for id_estacion, config in ESTACIONES.items():
        descargar_estacion(id_estacion, config)

    elapsed = (time.time() - inicio) / 60
    print(f"\n✅ Descarga completada en {elapsed:.1f} minutos.")
    print(f"   Archivos generados en: {OUTPUT_DIR}")

    # Listar archivos generados
    archivos = list(OUTPUT_DIR.glob("clima_*.parquet"))
    for f in archivos:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   📄 {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
