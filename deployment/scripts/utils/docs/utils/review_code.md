# Título: Script: review_code.py

## Resumen y Objetivo

Este script, `review_code.py`, está diseñado para analizar la calidad del código fuente de diferentes lenguajes de
programación (Python, SQL, Java) utilizando un modelo de inteligencia artificial generativa. Su objetivo principal es
proporcionar un informe detallado sobre la calidad del código, identificar áreas de mejora y ayudar a los
desarrolladores a mantener estándares de alta calidad en el codebase.

## Arquitectura y Lógica Principal

1. **Inicialización:**
    * Configura el analizador de argumentos (`argparse`) para aceptar una ruta de archivo o directorio, o para analizar
      archivos modificados en Git por defecto.
    * Inicializa el modelo de IA generativa (`setup_generative_model`).
2. **Obtención de Archivos:**
    * Determina los archivos a procesar basándose en la ruta proporcionada o en los cambios de Git (
      `get_files_to_process`).
    * Identifica el lenguaje de programación de cada archivo y carga las instrucciones específicas del agente para ese
      lenguaje (`AGENT_INSTRUCTIONS`).
3. **Procesamiento de Archivos:**
    * Itera sobre cada archivo a procesar.
    * Lee el contenido del archivo (`read_file_content`).
    * Envía el contenido del código y las instrucciones del agente al modelo de IA para su análisis (
      `process_file_with_ia`).
    * Registra la respuesta del modelo y el uso de tokens.
    * Maneja el límite de tokens para evitar exceder la capacidad del modelo.
4. **Generación de Informe:**
    * Una vez procesados todos los archivos (o hasta el límite de tokens), genera un informe de calidad (
      `quality_report`).
    * El informe incluye una puntuación de calidad por archivo, un análisis detallado por categoría (estado y
      observación) y sugerencias de mejora.
    * Calcula una puntuación promedio final y determina el estado de salida del script (éxito, advertencia o error)
      basado en la puntuación y si se alcanzó el límite de tokens.

## Configuración Requerida

* **Argumentos de Línea de Comandos:**
    * `path` (opcional): Ruta a un archivo o directorio específico para analizar. Si se omite, se analizarán los
      archivos modificados en el repositorio Git actual.

* **Variables de Entorno:**
    * [TBD] Se asume que la inicialización del modelo de IA (`setup_generative_model`) puede requerir configuraciones
      específicas (ej. claves API, configuración del modelo) que no se detallan en este script.

## Entradas y Salidas

* **Entradas:**
    * Archivos de código fuente en formatos soportados (Python, SQL, Java).
    * Instrucciones de agente predefinidas para cada lenguaje.
    * [TBD] Posibles configuraciones o credenciales para el modelo de IA.

* **Salidas:**
    * Mensajes de log en la consola indicando el progreso del análisis.
    * Un informe de calidad de código en la consola, que incluye:
        * Puntuación de calidad por archivo.
        * Análisis detallado de cada archivo.
        * Sugerencias de mejora.
        * Una puntuación promedio final.
    * Códigos de salida del script para indicar el estado de la ejecución (0 para éxito, 1 para error/advertencia).

## Ejemplos de Uso

* **Analizar un archivo específico:**
  ```bash
  python review_code.py /ruta/a/tu/codigo.py
  ```

* **Analizar todos los archivos modificados en Git:**
  ```bash
    python review_code.py
  ```

* **Analizar un directorio completo:**
  ```bash
  python review_code.py /ruta/a/tu/directorio/
  ```

## Mantenimiento y Puntos Clave

* **Dependencia de IA:** El rendimiento y la precisión del análisis dependen en gran medida de la calidad del modelo de
  IA subyacente y de las instrucciones del agente (`AGENT_INSTRUCTIONS`). Cualquier cambio en estos componentes puede
  afectar los resultados.
* **Límite de Tokens:** El script tiene un límite de tokens (`MAX_TOTAL_TOKENS`) para evitar el uso excesivo de
  recursos. Si este límite se alcanza, no todos los archivos podrán ser analizados, lo cual se indicará en el informe
  final.
* **Soporte de Lenguajes:** Actualmente, el script soporta Python, SQL y Java. La adición de soporte para otros
  lenguajes requerirá la definición de nuevas instrucciones de agente y la actualización de la lógica de detección de
  extensiones.
* **Gestión de Errores:** El script incluye manejo básico de errores para la inicialización del modelo y la lectura de
  archivos, pero podría beneficiarse de un manejo de errores más robusto para casos excepcionales durante la
  comunicación con la IA o el procesamiento de archivos.
* **Archivos `files.py` y `ia.py`:** La funcionalidad principal del script depende de módulos externos (`files`, `ia`,
  `ui`). Asegúrate de que estos módulos estén disponibles y funcionen correctamente.
