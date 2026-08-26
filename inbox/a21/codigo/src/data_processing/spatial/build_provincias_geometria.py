"""Genera un CSV maestro de geometria y vecinos de provincias."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape


OUTPUT_PATH = Path("data/processed/auto/utils/provincias_geometria.csv")

GEOJSON_URLS = [
    "https://services1.arcgis.com/OLiydejKCZTGhvWg/ArcGIS/rest/services/Provincias_IGN/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson",
    "https://raw.githubusercontent.com/codeforgermany/click_that_hood/master/public/data/spain-provinces.geojson",
]


def _download_geojson(urls: Iterable[str], timeout: int = 30) -> dict:
    """Descarga el GeoJSON desde una lista de URLs.

    Args:
        urls: Lista de URLs candidatas.
        timeout: Timeout en segundos.

    Returns:
        GeoJSON como dict.

    Raises:
        RuntimeError: Si ninguna descarga funciona.
    """
    last_error = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"No se pudo descargar GeoJSON: {last_error}")


def _infer_code(props: dict) -> str | None:
    """Infere el codigo INE (2 digitos) desde propiedades.

    Args:
        props: Propiedades del feature.

    Returns:
        Codigo INE con 2 digitos o None.
    """
    candidates = []
    for key, value in props.items():
        if value is None:
            continue
        val = str(value).strip()
        if len(val) == 2 and val.isdigit():
            candidates.append(val)
        if "prov" in key.lower() or "ine" in key.lower() or "cod" in key.lower():
            if val.isdigit() and len(val) <= 3:
                candidates.append(val.zfill(2))
    if candidates:
        return candidates[0]
    return None


def _infer_name(props: dict) -> str | None:
    """Infere el nombre de provincia desde propiedades.

    Args:
        props: Propiedades del feature.

    Returns:
        Nombre de provincia o None.
    """
    for key in ["name", "nombre", "provincia", "prov_name", "provincia_nombre"]:
        if key in props and props[key]:
            return str(props[key]).strip()
    for key, value in props.items():
        if "prov" in key.lower() and value:
            return str(value).strip()
    return None


def _build_geodataframe(geojson: dict) -> gpd.GeoDataFrame:
    """Convierte un GeoJSON a GeoDataFrame con columnas normalizadas.

    Args:
        geojson: GeoJSON como dict.

    Returns:
        GeoDataFrame con columnas provincia_id y nombre.
    """
    features = geojson.get("features", [])
    rows = []
    for feature in features:
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry")
        if geom is None:
            continue
        rows.append(
            {
                "provincia_id": _infer_code(props),
                "nombre": _infer_name(props),
                "geometry": shape(geom),
                "_props": json.dumps(props, ensure_ascii=False),
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.dropna(subset=["nombre"]).copy()
    return gdf


def _compute_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calcula centroides en CRS proyectado para mejor precision.

    Args:
        gdf: GeoDataFrame en EPSG:4326.

    Returns:
        GeoDataFrame con lat/lon de centroides.
    """
    gdf_proj = gdf.to_crs("EPSG:3857")
    centroids = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    gdf["lat_centroide"] = centroids.y
    gdf["lon_centroide"] = centroids.x
    return gdf


def build_provincias_geometria(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Genera el CSV maestro con geometria y vecinos de provincias.

    Args:
        output_path: Ruta de salida.

    Returns:
        DataFrame final.
    """
    geojson = _download_geojson(GEOJSON_URLS)
    gdf = _build_geodataframe(geojson)

    if gdf.empty:
        raise RuntimeError("GeoJSON sin provincias validas.")

    gdf = _compute_centroids(gdf)
    gdf = gdf.sort_values(["provincia_id", "nombre"], na_position="last").reset_index(drop=True)

    df_out = gdf[["provincia_id", "nombre", "lat_centroide", "lon_centroide"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    return df_out


if __name__ == "__main__":
    build_provincias_geometria()
