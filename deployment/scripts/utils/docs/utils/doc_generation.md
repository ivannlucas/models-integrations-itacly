# Script: doc_generation.py

## Resumen y Objetivo

Este script automatiza la generación de documentación para archivos de código, preservando la estructura de directorios.
Su objetivo es facilitar el onboarding de nuevos desarrolladores y el mantenimiento a largo plazo de la base de código,
generando archivos `README.md` claros y concisos.

## Arquitectura y Lógica Principal

1. **Análisis de Argumentos:** Utiliza `argparse` para definir y procesar argumentos de línea de comandos, incluyendo la
   ruta del archivo/directorio a procesar, el directorio de salida, y opciones para la integración con Confluence y la
   generación de archivos Swagger/Data Catalog.
2. **Inicialización:** Configura el modelo de IA y carga las instrucciones del agente (prompts).
3. **Obtención de Archivos:** Determina los archivos a procesar, ya sea por una ruta especificada o por archivos
   modificados en Git.
4. **Procesamiento por Archivo:**
    * Lee el contenido del archivo.
    * Llama a la IA (`process_file_with_ia`) para generar la documentación en formato Markdown, clasificando el archivo
      y extrayendo dependencias clave.
    * Maneja el límite de tokens para evitar sobrecostos o tiempos de procesamiento excesivos.
5. **Generación de Salida:**
    * **Guardado Local:** Si no se especifica la opción de Confluence, guarda el archivo Markdown generado en la
      estructura de directorios especificada.
    * **Generación de Artefactos Adicionales:** Para scripts de tipo `ModelInference_API` o `WebService_General`, genera
      un archivo `swagger.yaml`. Para scripts `DataPipeline_ETL`, genera un catálogo de datos.
    * **Carga en Confluence:** Si se especifica la opción de Confluence, utiliza el cliente de Confluence para crear o
      actualizar páginas, asegurando la jerarquía de páginas según la estructura de directorios.
6. **Registro de Mensajes:** Utiliza la función `log_message` para proporcionar feedback sobre el progreso y los
   errores.

## Configuración Requerida

* **Variables de Entorno (para Confluence):**
    * `CONFLUENCE_URL`: URL de la instancia de Confluence.
    * `CONFLUENCE_USERNAME`: Nombre de usuario para la autenticación.
    * `CONFLUENCE_PASSWORD`: Contraseña o token de API para la autenticación.
* **Argumentos de Línea de Comandos:**
    * `path`: Ruta al archivo o directorio a documentar (opcional, por defecto analiza archivos modificados en Git).
    * `--output-dir`: Directorio donde se guardarán los archivos Markdown generados (por defecto: `docs`).
  * `--extra-docs`: Habilita la creación de documentación extra específica para ciertos tipos de ficheros (p.e.: swagger
    para servicios o catálogo de datos para ETLs).
    * `--swagger-output`: Ruta para el archivo `swagger.yaml` (por defecto: `swagger.yaml`).
    * `--confluence`: Bandera para habilitar la carga automática a Confluence.
    * `--space`: Clave del espacio de Confluence (requerido si `--confluence` está habilitado).
    * `--parent-id`: ID de la página padre en Confluence (opcional).

## Entradas y Salidas

* **Entradas:**
    * Archivos de código fuente (Python, en este caso, según `ALLOWED_EXTENSIONS`).
    * Instrucciones del agente (`doc_generation.txt`).
    * Configuración de Confluence (si se usa la opción `--confluence`).
* **Salidas:**
    * Archivos Markdown (`.md`) en el directorio especificado (si no se usa Confluence).
    * Archivo `swagger.yaml` (para APIs).
    * Archivos de catálogo de datos (para pipelines ETL).
    * Páginas de documentación en Confluence (si se usa la opción `--confluence`).

## Ejemplos de Uso

1. **Documentar un directorio y guardar localmente:**
   ```bash
   python scripts/doc_generation.py ./src --output-dir ./generated_docs
   ```

2. **Documentar un archivo específico y subir a Confluence:**
   ```bash
   python scripts/doc_generation.py ./src/my_module.py --confluence --space "PROJ" --parent-id "12345678"
   ```

3. **Documentar archivos y generar Swagger (u otros):**
   ```bash
   python scripts/doc_generation.py ./src --extra-docs --swagger-output ./api/spec.yaml
   ```

4. **Documentar archivos modificados en una PR en Git y generar Swagger:**
   ```bash
   python scripts/doc_generation.py
   ```

## Mantenimiento y Puntos Clave

* **Dependencia de IA:** La calidad de la documentación generada depende en gran medida del modelo de IA y de la
  claridad de las instrucciones proporcionadas (`PROMPT_FILENAME`).
* **Gestión de Tokens:** El script implementa un límite de tokens (`MAX_TOTAL_TOKENS`) para controlar el uso y el costo.
  Si se alcanza este límite, el procesamiento se detiene.
* **Integración con Confluence:** Requiere configuración correcta de credenciales y permisos para Confluence. La gestión
  de la jerarquía de páginas es crucial para mantener el orden.
* **Subprocesos:** La ejecución de otros scripts (Swagger, Data Catalog) mediante `subprocess.run` puede ser frágil si
  las rutas o los argumentos cambian. Se debe asegurar que estos scripts auxiliares estén disponibles y funcionen
  correctamente.
* **Extensión de Archivos:** Actualmente, solo soporta extensiones definidas en `ALLOWED_EXTENSIONS`. Para soportar
  otros lenguajes, se debe actualizar esta configuración.
