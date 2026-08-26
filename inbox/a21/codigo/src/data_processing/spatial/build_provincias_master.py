"""Construye un maestro provincial con distancia al puerto mas cercano."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "auto" / "spatial"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "auto" / "utils" / "provincias_master_final.csv"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia Haversine en kilometros.

    Args:
        lat1: Latitud punto 1.
        lon1: Longitud punto 1.
        lat2: Latitud punto 2.
        lon2: Longitud punto 2.

    Returns:
        Distancia en km.
    """
    import math

    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def build_provincias_master(
    provincias_csv: Path = SPATIAL_DIR / "provincias_geometria.csv",
    puertos_json: Path = SPATIAL_DIR / "puertos_coords.json",
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Crea un maestro con distancia al puerto mas cercano.

    Args:
        provincias_csv: Ruta al CSV de provincias.
        puertos_json: Ruta al JSON de puertos.
        output_path: Ruta del CSV de salida.

    Returns:
        DataFrame con columnas agregadas.
    """
    provincias = pd.read_csv(provincias_csv)
    with puertos_json.open(encoding="utf-8") as fh:
        puertos = json.load(fh)

    def _closest_port(row: pd.Series) -> pd.Series:
        lat = float(row["lat_centroide"])
        lon = float(row["lon_centroide"])
        best_name = None
        best_dist = None
        for name, coords in puertos.items():
            dist = _haversine_km(lat, lon, float(coords["lat"]), float(coords["lon"]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_name = name
        return pd.Series({"dist_puerto_min_km": best_dist, "puerto_referencia": best_name})

    closest = provincias.apply(_closest_port, axis=1)
    out = pd.concat([provincias, closest], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    build_provincias_master()
