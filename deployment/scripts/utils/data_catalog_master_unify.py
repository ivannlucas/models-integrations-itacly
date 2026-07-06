#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Herramienta de unificación del catálogo de datos.

Este script procesa un directorio de archivos JSON autogenerados que describen
procesos ETL y los enriquece con esquemas de validación manual.
El resultado es un catálogo de datos maestro unificado en formatos JSON y Markdown.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

# --- Constantes ---

# Nombres de los archivos de salida
MASTER_JSON_FILENAME = "master_catalog.json"
MASTER_MD_FILENAME = "data_catalog.md"

# Marcadores de texto que se consideran "por definir"
TBD_MARKERS = ('[TBD]', 'TBD', '')


# --- Carga y Procesamiento de Esquemas ---

def load_validation_schemas(validation_directory: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Carga los esquemas de validación manual desde un directorio.

    Busca recursivamente archivos '*.json'.
    Devuelve un diccionario donde la clave es el nombre base de la tabla
    (ej. 'od_areas') y el valor es un diccionario de sus columnas
    (clave = nombre de columna en minúsculas).

    Args:
        validation_directory: Ruta al directorio con los JSON de validación.

    Returns:
        Un diccionario que mapea nombres de tablas a sus esquemas de columnas.
    """
    validation_schemas: Dict[str, Dict[str, Dict[str, str]]] = {}
    search_pattern = os.path.join(validation_directory, "**", "*.json")
    validation_files = glob.glob(search_pattern, recursive=True)

    if not validation_files:
        print(
            f"Advertencia: No se encontraron archivos '*.json' en el directorio de validación: {validation_directory}")
        return validation_schemas

    print(f"\nCargando {len(validation_files)} archivos de validación desde '{validation_directory}'...")

    for val_file_path in validation_files:
        # Extrae el nombre base, p.ej. 'od_areas' de 'ruta/od_areas.json'
        base_name = os.path.basename(val_file_path).replace(".json", "")

        try:
            with open(val_file_path, 'r', encoding='utf-8') as f:
                schema_list: List[Dict[str, Any]] = json.load(f)

            # Convierte la lista de columnas en un diccionario para búsqueda rápida
            column_map: Dict[str, Dict[str, str]] = {}
            for col in schema_list:
                col_name = col.get("name")
                if col_name:
                    # Normalizamos la clave a minúsculas para facilitar la coincidencia
                    column_map[col_name.lower()] = {
                        "type": col.get("type", "[TBD]"),
                        "description": col.get("description", "[TBD]")
                    }

            validation_schemas[base_name] = column_map
            print(f"  -> Esquema de validación '{base_name}' cargado con {len(column_map)} columnas.")

        except json.JSONDecodeError as e:
            print(f"    Error al decodificar JSON en {val_file_path}: {e}")
            continue
        except IOError as e:
            print(f"    Error de E/S al leer {val_file_path}: {e}")
            continue

    return validation_schemas


def _enrich_column(col_auto: Dict[str, Any], val_info: Dict[str, str]) -> Dict[str, Any]:
    """
    Enriquece una única columna autogenerada con la información de validación.

    Replica la lógica de merge original:
    - Tipo: Se actualiza si es [TBD]. Si no, comprueba conflictos.
    - Descripción: Se actualiza si es [TBD]. Si no, anexa la nueva descripción.

    Args:
        col_auto: El diccionario de la columna del archivo autogenerado.
        val_info: El diccionario de la columna del archivo de validación.

    Returns:
        El diccionario de la columna autogenerada, modificado.
    """
    # --- Lógica de Tipo ---
    current_type = col_auto.get('inferred_type')
    validation_type_lower = val_info.get('type', '[TBD]').lower()

    if current_type in TBD_MARKERS:
        col_auto['inferred_type'] = validation_type_lower
    elif validation_type_lower.startswith(str(current_type).lower()):
        # Lógica original: si el tipo de validación "comienza con" el tipo actual
        # (p.ej. 'string' vs 'string(10)'), no se hace nada.
        pass
    else:
        # Conflicto detectado
        col_auto['inferred_type'] = f"{validation_type_lower}"

    # --- Lógica de Descripción ---
    current_desc = col_auto.get('description')
    validation_desc = val_info.get('description', '[TBD]')

    if current_desc in TBD_MARKERS:
        col_auto['description'] = validation_desc
    else:
        # Lógica original: anexa la descripción de validación a la existente
        col_auto['description'] = f"{current_desc}\n --- \n{validation_desc}"

    return col_auto


def _process_etl_sources(etl_name: str, etl_data: Dict[str, Any], master_catalog: Dict[str, Dict[str, Any]]) -> None:
    """
    Procesa la sección 'data_sources' de un archivo ETL.

    Actualiza el catálogo maestro para registrar qué ETLs consumen cada fuente.

    Args:
        etl_name: Nombre del archivo ETL que se está procesando.
        etl_data: Contenido JSON del archivo ETL.
        master_catalog: El catálogo maestro (se modifica in-place).
    """
    source_identifiers = [
        source.get("source_identifier")
        for source in etl_data.get("data_sources", [])
        if source.get("source_identifier")
    ]

    for source_id in source_identifiers:
        master_catalog[source_id]["entity_name"] = source_id

        if etl_name not in master_catalog[source_id]["consumed_by_etls"]:
            master_catalog[source_id]["consumed_by_etls"].append(etl_name)


def _process_etl_targets(etl_name: str, etl_data: Dict[str, Any], master_catalog: Dict[str, Dict[str, Any]],
                         validation_schemas: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    """
    Procesa la sección 'data_targets' de un archivo ETL.

    Actualiza el catálogo maestro con el linaje, propietario (ETL) y enriquece
    los metadatos de las columnas usando los esquemas de validación.

    Args:
        etl_name: Nombre del archivo ETL que se está procesando.
        etl_data: Contenido JSON del archivo ETL.
        master_catalog: El catálogo maestro (se modifica in-place).
        validation_schemas: El mapa de esquemas de validación cargado.
    """
    # Identificar las fuentes de este ETL para vincularlas a los destinos
    upstream_sources = [
        source.get("source_identifier")
        for source in etl_data.get("data_sources", [])
        if source.get("source_identifier")
    ]

    for target in etl_data.get("data_targets", []):
        target_id = target.get("target_identifier")
        if not target_id:
            continue

        # Registrar la entidad y su ETL de escritura
        master_catalog[target_id]["entity_name"] = target_id
        if master_catalog[target_id]["written_by_etl"] is not None:
            print(f"    ¡ADVERTENCIA DE CONFLICTO! La entidad '{target_id}' ya está definida por "
                  f"'{master_catalog[target_id]['written_by_etl']}'.")
            print(f"    Será SOBRESCRITA por la definición de '{etl_name}'.")

        master_catalog[target_id]["written_by_etl"] = etl_name
        master_catalog[target_id]["upstream_sources"] = upstream_sources

        # --- Lógica de Enriquecimiento de Linaje ---
        original_lineage = target.get("data_catalog_lineage", [])
        enriched_lineage = []

        # Obtener el nombre de la tabla (ej. 'od_areas' de 'dev_master.od_areas')
        table_name_only = target_id.split('.')[-1]
        validation_map = validation_schemas.get(table_name_only)

        if validation_map:
            print(f"    -> Enriqueciendo '{target_id}' usando el esquema de validación '{table_name_only}'.")
            for col_auto in original_lineage:
                col_name_auto = col_auto.get('column')
                validation_info = None

                if col_name_auto:
                    # Buscar usando la clave normalizada en minúsculas
                    validation_info = validation_map.get(col_name_auto.lower())

                if validation_info:
                    # Enriquecer la columna
                    enriched_col = _enrich_column(col_auto, validation_info)
                    enriched_lineage.append(enriched_col)
                else:
                    # Mantener la columna autogenerada como está si no hay info de validación
                    enriched_lineage.append(col_auto)
            master_catalog[target_id]["schema_lineage"] = enriched_lineage
        else:
            # Si no hay esquema de validación, usar el linaje original
            master_catalog[target_id]["schema_lineage"] = original_lineage


# --- Generación de Salidas ---

def _filter_catalog(master_catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Filtra el catálogo maestro para incluir solo entidades que tengan linaje.

    Esto excluye entidades que solo son 'fuentes' (consumidas) pero
    no 'destinos' (escritas por un ETL con linaje).

    Args:
        master_catalog: El catálogo maestro completo.

    Returns:
        Un nuevo diccionario de catálogo filtrado.
    """
    final_catalog: Dict[str, Dict[str, Any]] = {}
    for entity_name, entity_data in master_catalog.items():
        # Incluir si 'schema_lineage' existe y no está vacío
        if entity_data.get('schema_lineage'):
            final_catalog[entity_name] = entity_data
    return final_catalog


def _write_json_catalog(catalog_data: Dict[str, Dict[str, Any]], filename: str) -> None:
    """
    Escribe el diccionario del catálogo maestro en un archivo JSON.

    Args:
        catalog_data: El diccionario del catálogo.
        filename: El nombre del archivo de salida.
    """
    print(f"\nEscribiendo catálogo JSON en '{filename}'...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Ordenar las claves del JSON final para un archivo consistente
            json.dump(catalog_data, f, indent=4, sort_keys=True)

        print("¡Éxito! Catálogo JSON maestro creado.")
        print(f"Total de entidades generadas en el catálogo: {len(catalog_data)}")

    except IOError as e:
        print(f"Error al escribir el archivo de salida '{filename}': {e}")
        sys.exit(1)


def generate_markdown_catalog(catalog_data: Dict[str, Dict[str, Any]], output_filename: str) -> None:
    """
    Genera un archivo Markdown a partir del catálogo maestro.

    Args:
        catalog_data: El diccionario del catálogo.
        output_filename: El nombre del archivo de salida.
    """
    print(f"\nGenerando catálogo Markdown en '{output_filename}'...")
    md_content = ["# Catálogo de Datos Maestro\n\n"]

    # Ordenar entidades por nombre para un catálogo consistente
    sorted_entities = sorted(catalog_data.keys())

    for entity_name in sorted_entities:
        entity_data = catalog_data[entity_name]
        md_content.append(f"## Entidad: `{entity_name}`\n")

        # --- Metadatos Generales ---
        written_by = entity_data.get('written_by_etl') or "N/A (Fuente Externa)"
        md_content.append(f"**ETL Propietario (Escritura):** `{written_by}`\n")

        consumed_by = entity_data.get('consumed_by_etls', [])
        if consumed_by:
            md_content.append("**Consumido por (ETLs Lectura):**\n")
            for consumer in sorted(consumed_by):
                md_content.append(f"- `{consumer}`\n")
        else:
            md_content.append("**Consumido por (ETLs Lectura):** Ninguno\n")

        upstreams = entity_data.get('upstream_sources', [])
        if upstreams:
            md_content.append("**Fuentes Directas (Upstream):**\n")
            for source in sorted(upstreams):
                md_content.append(f"- `{source}`\n")
        else:
            md_content.append("**Fuentes Directas (Upstream):** Ninguna (Es una entidad raíz)\n")

        # --- Tabla de Linaje de Columnas ---
        schema_lineage = entity_data.get('schema_lineage', [])
        md_content.append("\n### Linaje de Columnas (Esquema)\n")

        if schema_lineage:
            md_content.append(
                "| Columna Destino | Tipo | Descripción | Lógica de Transformación | Columnas Origen |")
            md_content.append("| :--- | :--- | :--- | :--- | :--- |")

            for col in schema_lineage:
                col_name = col.get('column', 'N/A')
                col_type = col.get('inferred_type', 'N/A')
                # Escapar el carácter '|' si aparece en la lógica
                col_description = str(col.get('description', 'N/A')).replace("|", "\\|")
                col_logic = str(col.get('transformation_logic', 'N/A')).replace("|", "\\|")
                col_sources = col.get('source_columns', [])
                col_source_str = f"`{', '.join(col_sources)}`" if col_sources else "N/A"

                md_content.append(f"| `{col_name}` | {col_type} | {col_description} | {col_logic} | {col_source_str} |")
        else:
            md_content.append("_No hay linaje de columnas definido._")

        md_content.append("\n\n---\n\n")  # Separador entre entidades

    # --- Escritura del archivo ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))
        print("¡Éxito! Catálogo Markdown creado.")
    except IOError as e:
        print(f"Error al escribir el archivo Markdown '{output_filename}': {e}")
        sys.exit(1)


# --- Función Principal de Orquestación ---

def build_master_catalog(autogen_directory: str, validation_directory: str) -> None:
    """
    Orquesta el proceso completo de creación del catálogo.

    1. Carga los esquemas de validación.
    2. Procesa los archivos ETL autogenerados para construir un catálogo en memoria.
    3. Filtra el catálogo para mantener solo entidades con linaje.
    4. Escribe los catálogos finales en JSON y Markdown.

    Args:
        autogen_directory: Ruta al directorio con los JSON de ETLs.
        validation_directory: Ruta al directorio con los JSON de validación.
    """

    # 1. Cargar los esquemas de validación primero
    validation_schemas = load_validation_schemas(validation_directory)

    # 2. Preparar el catálogo maestro
    master_catalog = defaultdict(lambda: {
        "entity_name": "",
        "written_by_etl": None,
        "consumed_by_etls": [],
        "schema_lineage": [],
        "upstream_sources": []
    })

    # 3. Procesar los archivos autogenerados
    search_pattern = os.path.join(autogen_directory, "**", "*.json")
    etl_files = glob.glob(search_pattern, recursive=True)

    if not etl_files:
        print(f"Error: No se encontraron archivos '*.json' en el directorio: {autogen_directory}")
        return

    print(f"\nProcesando {len(etl_files)} archivos ETL autogenerados en '{autogen_directory}'...")

    for etl_file_path in etl_files:
        etl_name = os.path.basename(etl_file_path).replace(".json", ".py")
        print(f"  -> Procesando: {etl_name}")

        try:
            with open(etl_file_path, 'r', encoding='utf-8') as f:
                etl_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"    Error al leer {etl_file_path}: {e}")
            continue

        # Procesar fuentes (construye 'consumed_by_etls')
        _process_etl_sources(etl_name, etl_data, master_catalog)

        # Procesar destinos (construye 'written_by_etl', 'schema_lineage', etc.)
        _process_etl_targets(etl_name, etl_data, master_catalog, validation_schemas)

    # 4. Filtrar el catálogo para incluir solo entidades generadas (con linaje)
    #    Esta es la lógica que estaba comentada en tu script original.
    final_catalog = _filter_catalog(dict(master_catalog))

    if not final_catalog:
        print("\nNo se generó ningún catálogo. El catálogo final está vacío después de filtrar.")
        return

    # 5. Escribir los archivos de salida
    _write_json_catalog(final_catalog, MASTER_JSON_FILENAME)
    generate_markdown_catalog(final_catalog, MASTER_MD_FILENAME)


def main() -> None:
    """
    Punto de entrada principal.
    Parsea argumentos de línea de comandos y lanza el proceso.
    """
    parser = argparse.ArgumentParser(
        description="Agrega archivos JSON de ETL en un catálogo maestro (JSON y Markdown).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-a", "--autogen-dir",
        required=True,
        help="Ruta al directorio que contiene los archivos *.json autogenerados."
    )

    parser.add_argument(
        "-v", "--validation-dir",
        required=True,
        help="Ruta al directorio que contiene los archivos *.json de validación."
    )

    args = parser.parse_args()

    # Validar que ambos directorios existan
    if not os.path.isdir(args.autogen_dir):
        print(f"Error: El directorio de autogenerados '{args.autogen_dir}' no es un directorio válido.")
        sys.exit(1)

    if not os.path.isdir(args.validation_dir):
        print(f"Error: El directorio de validación '{args.validation_dir}' no es un directorio válido.")
        sys.exit(1)

    build_master_catalog(args.autogen_dir, args.validation_dir)


if __name__ == "__main__":
    main()
