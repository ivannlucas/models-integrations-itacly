# Script: build_master_catalog.py

## Resumen y Objetivo

Este script tiene como objetivo consolidar información de múltiples archivos JSON de configuración de ETL (Extract,
Transform, Load) distribuidos en un directorio específico. Su propósito es generar un catálogo maestro unificado, tanto
en formato JSON como en Markdown, que sirva como una fuente centralizada para entender el linaje de datos, las
dependencias entre procesos ETL y los esquemas de las entidades de datos.

## Arquitectura y Lógica Principal

El script opera de la siguiente manera:

1. **Parseo de Argumentos:** Utiliza `argparse` para recibir la ruta a un directorio que contiene los archivos JSON de
   configuración de ETL.
2. **Búsqueda de Archivos ETL:** Emplea `glob` para encontrar todos los archivos que terminan en `.json` dentro del
   directorio especificado.
3. **Procesamiento de Archivos:** Itera sobre cada archivo JSON encontrado:
    * Lee el contenido del archivo JSON.
    * Identifica las `data_sources` (fuentes de datos de entrada) y sus `source_identifier`.
    * Identifica las `data_targets` (destinos de datos de salida) y sus `target_identifier`.
    * Para cada `source_id`, actualiza el catálogo maestro registrando el ETL actual como un consumidor (
      `consumed_by_etls`) y las fuentes directas (`upstream_sources`).
    * Para cada `target_id`, actualiza el catálogo maestro registrando el ETL actual como el escritor (
      `written_by_etl`), y almacena el linaje del esquema (`schema_lineage`) y las fuentes directas.
4. **Consolidación del Catálogo Maestro:** Utiliza un `defaultdict` para construir un diccionario que representa el
   catálogo maestro, agregando o actualizando la información de cada entidad de datos encontrada.
5. **Generación de Salida:**
    * Guarda el catálogo maestro consolidado en un archivo `master_catalog.json`.
    * Llama a la función `generate_markdown_catalog` para crear un archivo `data_catalog.md` legible por humanos, que
      detalla cada entidad, su origen, consumo y linaje de columnas.

## Configuración Requerida

El script requiere un único argumento de línea de comandos:

* `--directory` o `-d`: La ruta al directorio que contiene los archivos `etl_*.json` que se van a procesar.

**Ejemplo de uso:**
`python build_master_catalog.py -d ./etl_configs`

## Entradas y Salidas

* **Entradas:**
    * Un directorio (`--directory`) que contiene uno o más archivos JSON con la configuración de procesos ETL.
    * Cada archivo JSON debe tener una estructura que incluya `data_sources` (con `source_identifier`) y
      `data_targets` (con `target_identifier` y opcionalmente `data_catalog_lineage`).

* **Salidas:**
    * `master_catalog.json`: Un archivo JSON que contiene el catálogo maestro consolidado de todas las entidades de
      datos y sus relaciones.
    * `data_catalog.md`: Un archivo Markdown que presenta el catálogo maestro de forma legible, detallando el linaje y
      las dependencias.

## Ejemplos de Uso

Para ejecutar el script y generar los catálogos:

```bash
python build_master_catalog.py --directory /ruta/a/tu/directorio/de/etls
```

Esto procesará todos los archivos `.json` en `/ruta/a/tu/directorio/de/etls` y creará `master_catalog.json` y
`data_catalog.md` en el directorio actual desde donde se ejecutó el script.

## Mantenimiento y Puntos Clave

* **Dependencia de Formato JSON:** El script depende en gran medida de que los archivos JSON de configuración sigan una
  estructura esperada (específicamente, la presencia de `data_sources`, `source_identifier`, `data_targets`,
  `target_identifier`, y `data_catalog_lineage`). Cualquier desviación puede causar errores o resultados incompletos.
* **Manejo de Errores:** El script incluye manejo básico de errores para la lectura de archivos JSON y la escritura de
  archivos de salida, pero podría mejorarse para manejar casos más específicos de datos malformados dentro de los JSON.
* **Consistencia de Identificadores:** La efectividad del catálogo maestro depende de la consistencia en el uso de
  `source_identifier` y `target_identifier` a través de los diferentes archivos ETL.
* **Escalabilidad:** Para un número muy grande de archivos ETL o entidades de datos, el tiempo de procesamiento y el
  tamaño de los archivos de salida podrían aumentar significativamente. Se podría considerar optimizaciones si se
  manejan volúmenes masivos.
