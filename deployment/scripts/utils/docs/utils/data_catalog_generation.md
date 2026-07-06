# Script: data_catalog_generation.py

## Resumen y Objetivo

Este script automatiza la creación de entradas para un catálogo de datos. Su objetivo es analizar archivos de código
fuente (Python, en este caso) y generar metadatos estructurados en formato JSON que describan los datos o la lógica de
procesamiento de datos contenida en ellos. Esto facilita la comprensión, el descubrimiento y la gestión de los activos
de datos dentro de un proyecto.

## Arquitectura y Lógica Principal

1. **Parseo de Argumentos:** Utiliza `argparse` para definir y procesar argumentos de línea de comandos, permitiendo
   especificar un archivo/directorio de entrada y un directorio de salida.
2. **Inicialización:** Configura el modelo de IA (`setup_generative_model`) y carga las instrucciones para el agente de
   IA (`load_agent_instructions`).
3. **Determinación de Archivos:** Identifica los archivos a procesar. Si no se proporciona una ruta, analiza los
   archivos modificados en Git (implícito por la función `get_files_to_process`).
4. **Iteración y Procesamiento:**
    * Para cada archivo identificado:
        * Verifica si se ha alcanzado el límite de tokens.
        * Lee el contenido del archivo.
        * Determina el lenguaje del archivo.
        * Llama a la función `process_file_with_ia` para que el modelo de IA analice el contenido del código y genere
          una salida estructurada (JSON).
        * Calcula la ruta relativa del archivo procesado respecto a la raíz del proyecto.
        * Construye la ruta de salida para el archivo JSON generado, manteniendo la estructura de directorios original.
        * Serializa el resultado del análisis de IA a formato JSON.
        * Guarda el archivo JSON resultante en el directorio de salida especificado.
5. **Registro:** Utiliza la función `log_message` para proporcionar feedback sobre el progreso y los errores.

## Configuración Requerida

* **Variables de Entorno:** No se especifican explícitamente variables de entorno en el script, pero las funciones
  importadas (`files`, `ia`, `ui`) podrían depender de ellas (ej. para credenciales de API de IA, configuración de
  logging).
* **Argumentos de Línea de Comandos:**
    * `path` (opcional): Ruta al archivo o directorio a procesar. Si se omite, se procesan los archivos modificados en
      Git.
    * `--output-dir` (opcional, por defecto: `docs/catalog`): Directorio donde se guardarán los archivos JSON generados.

## Entradas y Salidas

* **Entradas:**
    * Archivos de código fuente (ej. `.py`, `.sql`, etc., según `ALLOWED_EXTENSIONS`).
    * Instrucciones para el agente de IA (definidas en `PROMPT_FILENAME`).
    * Archivos modificados en Git (si no se proporciona `path`).
* **Salidas:**
    * Archivos JSON (`.json`) que contienen metadatos del catálogo de datos, generados en el directorio especificado por
      `--output-dir`.
    * Mensajes de log en la consola indicando el progreso y los errores.

## Ejemplos de Uso

1. **Procesar un archivo específico y guardar en el directorio por defecto:**
   ```bash
   python scripts/data_catalog_generation.py mi_script_de_datos.py
   ```
   Esto generará `docs/catalog/mi_script_de_datos.json`.

2. **Procesar un directorio y especificar un directorio de salida:**
   ```bash
   python scripts/data_catalog_generation.py src/data_processing/ --output-dir generated_catalog
   ```
   Esto procesará todos los archivos soportados dentro de `src/data_processing/` y guardará los JSON resultantes en
   `generated_catalog/`, manteniendo la estructura de subdirectorios.

3. **Procesar archivos modificados en Git (sin argumentos de path):**
   ```bash
   python scripts/data_catalog_generation.py
   ```
   Analizará los archivos que han sido modificados en el repositorio Git y generará sus correspondientes JSON en
   `docs/catalog/`.

## Mantenimiento y Puntos Clave

* **Dependencia de IA:** La calidad y utilidad de los metadatos generados dependen en gran medida de la efectividad del
  modelo de IA y de las instrucciones proporcionadas (`PROMPT_FILENAME`). Es crucial mantener y refinar estas
  instrucciones.
* **Límite de Tokens:** El script implementa un límite de tokens (`MAX_TOTAL_TOKENS`) para evitar costes excesivos o
  tiempos de procesamiento muy largos. Si se procesan muchos archivos o archivos muy grandes, el script puede detenerse
  prematuramente.
* **Estructura de Archivos:** El script asume una cierta estructura de proyecto para determinar la ruta raíz y generar
  las rutas de salida relativas. Cambios significativos en la estructura del proyecto podrían requerir ajustes.
* **Manejo de Errores:** El script incluye manejo básico de errores (ej. fallo al inicializar el modelo, no poder leer
  archivos, resultados de IA no válidos), pero se podría mejorar para casos más específicos.
* **Formatos Soportados:** La lista de extensiones de archivo soportadas (`ALLOWED_EXTENSIONS`) debe mantenerse
  actualizada si se añaden nuevos tipos de archivos de código o configuración al proyecto.
