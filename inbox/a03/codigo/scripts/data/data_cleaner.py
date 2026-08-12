"""
==============================================================================
data_cleaner.py
==============================================================================
Limpia y preprocesa los datos meteorológicos descargados por 
descarga_clima_historico.py. Realiza imputación de huecos y calcula la 
Evapotranspiración de Referencia (ETo) horaria a partir de la radiación 
solar y la temperatura reales.

Entrada: data/clima_real/clima_*.parquet
Salida:  data/clima_real/clean/clima_*_clean.parquet

Uso:
    python data_cleaner.py
==============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

INPUT_DIR  = PROJECT_ROOT / config_yaml['paths']['clima_real_dir']
OUTPUT_DIR = PROJECT_ROOT / config_yaml['paths']['clima_clean_dir']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variables climáticas que deben existir y estar limpias
VARIABLES_CLIMATICAS = [
    "Temp_Amb_C",
    "Hum_Rel_Pct",
    "Lluvia_mm",
    "Viento_kmh",
    "Radiacion_Wm2",
]

# Rangos físicos válidos (para detectar valores absurdos)
RANGOS_VALIDOS = {
    "Temp_Amb_C":     (-30.0, 55.0),
    "Hum_Rel_Pct":    (0.0, 100.0),
    "Lluvia_mm":      (0.0, 200.0),   # mm/hora
    "Viento_kmh":     (0.0, 200.0),
    "Radiacion_Wm2":  (0.0, 1500.0),
}


# ==============================================================================
# 2. FUNCIONES DE LIMPIEZA
# ==============================================================================

def reemplazar_fuera_de_rango(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sustituye por NaN los valores que caen fuera de los rangos físicos válidos.
    Esto permite que la imputación posterior los rellene correctamente.
    """
    for col, (vmin, vmax) in RANGOS_VALIDOS.items():
        if col in df.columns:
            mask = (df[col] < vmin) | (df[col] > vmax)
            n_invalidos = mask.sum()
            if n_invalidos > 0:
                print(f"    ⚠️ {col}: {n_invalidos} valores fuera de rango [{vmin}, {vmax}] → NaN")
                df.loc[mask, col] = np.nan
    return df


def imputar_huecos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena valores NaN con una estrategia escalonada:
      1. Huecos ≤ 3 horas: Interpolación lineal
      2. Huecos restantes: Media de la misma hora del día (estacional)
      3. Residual: Forward fill + Backward fill
    """
    for col in VARIABLES_CLIMATICAS:
        if col not in df.columns:
            continue
            
        n_antes = df[col].isnull().sum()
        if n_antes == 0:
            continue

        # Paso 1: Interpolación lineal (máximo 3 huecos consecutivos)
        df[col] = df[col].interpolate(method="linear", limit=3, limit_direction="both")

        # Paso 2: Rellenar con media horaria (misma hora del día, misma estación)
        n_aun_nulo = df[col].isnull().sum()
        if n_aun_nulo > 0 and "Fecha" in df.columns:
            df["_hora"] = df["Fecha"].dt.hour
            df["_mes"] = df["Fecha"].dt.month
            media_horaria = df.groupby(["_mes", "_hora"])[col].transform("mean")
            df[col] = df[col].fillna(media_horaria)
            df.drop(columns=["_hora", "_mes"], inplace=True, errors="ignore")

        # Paso 3: Forward/Backward fill para residuos
        df[col] = df[col].ffill().bfill()

        n_despues = df[col].isnull().sum()
        reparados = n_antes - n_despues
        if reparados > 0:
            print(f"    🔧 {col}: {reparados} valores imputados (quedan {n_despues} NaN)")

    return df


def calcular_eto_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la Evapotranspiración de Referencia (ETo) horaria usando una
    aproximación de Hargreaves simplificada basada en radiación real y
    temperatura.
    
    ETo (mm/h) ≈ 0.0023 × Rs(MJ/m²/h) × (T + 17.8)
    
    Donde Rs se convierte de W/m² a MJ/m²/h multiplicando por 0.0036.
    """
    # Conversión: W/m² → MJ/m²/hora  (1 W/m² = 0.0036 MJ/m²/h)
    Rs_MJ = df["Radiacion_Wm2"] * 0.0036

    # Factor de poder secante del aire (simplificado con viento)
    factor_viento = 1.0 + 0.3 * (df["Viento_kmh"] / 10.0)  # Normalizado

    # Hargreaves simplificada
    df["ETo"] = 0.0023 * Rs_MJ * (df["Temp_Amb_C"] + 17.8) * factor_viento

    # ETo no puede ser negativa ni existir de noche (radiación = 0 ya lo maneja)
    df["ETo"] = df["ETo"].clip(lower=0.0)

    return df


def verificar_continuidad_temporal(df: pd.DataFrame, id_estacion: str) -> None:
    """
    Verifica que las fechas son continuas (sin saltos) y reporta huecos.
    """
    if "Fecha" not in df.columns or len(df) < 2:
        return

    diff = df["Fecha"].diff()
    saltos = diff[diff > pd.Timedelta(hours=1)]

    if len(saltos) > 0:
        print(f"    ⚠️ {id_estacion}: {len(saltos)} saltos temporales detectados")
        for idx, delta in saltos.head(5).items():
            fecha = df.loc[idx, "Fecha"]
            print(f"       Salto de {delta} en {fecha}")
        if len(saltos) > 5:
            print(f"       ... y {len(saltos) - 5} más")
    else:
        print(f"    ✅ {id_estacion}: Serie temporal continua (sin saltos)")


# ==============================================================================
# 3. PIPELINE PRINCIPAL
# ==============================================================================

def procesar_estacion(filepath: Path) -> None:
    """
    Ejecuta el pipeline completo de limpieza para un archivo de estación.
    """
    id_estacion = filepath.stem.replace("clima_", "")
    print(f"\n🧹 Procesando: {id_estacion} ({filepath.name})")

    df = pd.read_parquet(filepath)
    print(f"   📊 Registros: {len(df)} | Rango: {df['Fecha'].min()} → {df['Fecha'].max()}")

    # Paso 1: Verificar continuidad temporal
    verificar_continuidad_temporal(df, id_estacion)

    # Paso 2: Reemplazar valores fuera de rango
    df = reemplazar_fuera_de_rango(df)

    # Paso 3: Imputar huecos
    nulos_antes = df[VARIABLES_CLIMATICAS].isnull().sum().sum()
    df = imputar_huecos(df)
    nulos_despues = df[VARIABLES_CLIMATICAS].isnull().sum().sum()
    print(f"   📈 Nulos climáticos: {nulos_antes} → {nulos_despues}")

    # Paso 4: Calcular ETo
    df = calcular_eto_horaria(df)
    print(f"   🌡️ ETo calculada: media={df['ETo'].mean():.4f} mm/h, max={df['ETo'].max():.4f} mm/h")

    # Paso 5: Guardar
    out_path = OUTPUT_DIR / f"clima_{id_estacion}_clean.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"   💾 Guardado: {out_path.name} ({size_mb:.1f} MB)")


def main():
    print("=" * 70)
    print("  LIMPIEZA Y PREPROCESAMIENTO DE DATOS METEOROLÓGICOS")
    print(f"  Entrada: {INPUT_DIR}")
    print(f"  Salida:  {OUTPUT_DIR}")
    print("=" * 70)

    archivos = sorted(INPUT_DIR.glob("clima_*.parquet"))

    if not archivos:
        print(f"\n❌ No se encontraron archivos clima_*.parquet en {INPUT_DIR}")
        print("   Ejecuta primero: python descarga_clima_historico.py")
        return

    print(f"\n📂 Archivos encontrados: {len(archivos)}")
    for f in archivos:
        print(f"   📄 {f.name}")

    for filepath in archivos:
        procesar_estacion(filepath)

    print(f"\n✅ Limpieza completada. Archivos limpios en: {OUTPUT_DIR}")

    # Resumen final
    archivos_clean = sorted(OUTPUT_DIR.glob("clima_*_clean.parquet"))
    for f in archivos_clean:
        df = pd.read_parquet(f)
        nulos = df[VARIABLES_CLIMATICAS + ["ETo"]].isnull().sum().sum()
        print(f"   📄 {f.name}: {len(df)} registros, {nulos} NaN restantes")


if __name__ == "__main__":
    main()
