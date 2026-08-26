"""Geocodifica muelles de graneles en puertos de Espana con Nominatim."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from geopy.geocoders import Nominatim


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "auto" / "spatial" / "puertos_coords.json"


@dataclass(frozen=True)
class PortQuery:
    """Representa una consulta de geocodificacion con fallback."""

    key: str
    query: str
    fallback_city: str


QUERIES: list[PortQuery] = [
    PortQuery(
        key="tarragona_graneles",
        query="Muelle de graneles solidos, Puerto de Tarragona, Spain",
        fallback_city="Tarragona, Spain",
    ),
    PortQuery(
        key="santander_graneles",
        query="Terminal de graneles, Puerto de Santander, Spain",
        fallback_city="Santander, Spain",
    ),
    PortQuery(
        key="bilbao_graneles",
        query="Puerto de Bilbao (muelle de graneles), Spain",
        fallback_city="Bilbao, Spain",
    ),
    PortQuery(
        key="musel_gijon",
        query="Puerto del Musel, Gijon, Spain",
        fallback_city="Gijon, Spain",
    ),
    PortQuery(
        key="huelva_muelle_sur",
        query="Puerto de Huelva (muelle sur), Spain",
        fallback_city="Huelva, Spain",
    ),
    PortQuery(
        key="sevilla_eurovia",
        query="Puerto de Sevilla (eurovia Guadalquivir), Spain",
        fallback_city="Sevilla, Spain",
    ),
    PortQuery(
        key="valencia_graneles",
        query="Terminal de graneles, Puerto de Valencia, Spain",
        fallback_city="Valencia, Spain",
    ),
    PortQuery(
        key="cartagena_escombreras",
        query="Puerto de Cartagena (muelle de Escombreras), Spain",
        fallback_city="Cartagena, Spain",
    ),
]


def _geocode(geocoder: Nominatim, query: str):
    """Lanza una consulta de geocodificacion.

    Args:
        geocoder: Instancia de Nominatim.
        query: Texto de busqueda.

    Returns:
        Objeto de geopy o None.
    """
    return geocoder.geocode(query)


def build_ports_coords(
    queries: Iterable[PortQuery] = QUERIES,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, dict[str, float]]:
    """Geocodifica puertos y devuelve diccionario con lat/lon.

    Args:
        queries: Lista de consultas con fallback.
        output_path: Ruta para guardar JSON.

    Returns:
        Diccionario {key: {"lat": .., "lon": ..}}.
    """
    geocoder = Nominatim(user_agent="datagia-ports-geocoder")

    results: dict[str, dict[str, float]] = {}
    for item in queries:
        location = _geocode(geocoder, item.query)
        if location is None:
            location = _geocode(geocoder, item.fallback_city)
        if location is None:
            results[item.key] = {"lat": float("nan"), "lon": float("nan")}
        else:
            results[item.key] = {
                "lat": float(location.latitude),
                "lon": float(location.longitude),
            }
        time.sleep(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    build_ports_coords()
