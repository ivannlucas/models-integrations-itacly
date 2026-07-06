# Título

Script: swagger_generation.py

## Resumen y Objetivo

Este script tiene como objetivo automatizar la generación de un archivo `swagger.yaml` (especificación OpenAPI 3.1.0) a
partir del código fuente de una aplicación Flask. Resuelve la necesidad de documentar manualmente las APIs, ahorrando
tiempo y reduciendo errores.

## Arquitectura y Lógica Principal

1. **Parseo de Argumentos:** Utiliza `argparse` para aceptar la ruta al archivo principal de la aplicación Flask y una
   ruta opcional para el archivo de salida `swagger.yaml`.
2. **Inicialización:** Configura un modelo de IA (`setup_generative_model`) y carga las instrucciones específicas para
   la generación de Swagger (`load_agent_instructions`).
3. **Validación de Entrada:** Verifica que el archivo de entrada especificado exista.
4. **Lectura de Código:** Lee el contenido del archivo Python de la aplicación Flask.
5. **Procesamiento con IA:** Envía el contenido del código y las instrucciones a un modelo de IA (
   `process_file_with_ia`) para que genere el contenido del archivo `swagger.yaml`.
6. **Guardado de Salida:** Guarda el contenido generado en el archivo `swagger.yaml` especificado.

## Configuración Requerida

* **Argumentos de Línea de Comandos:**
    * `path` (obligatorio): Ruta al archivo principal `.py` de la aplicación Flask.
    * `--output` (opcional): Ruta de salida para el archivo `swagger.yaml`. Por defecto es `swagger.yaml`.

* **Variables de Entorno:** [TBD] - No se especifican variables de entorno en el script, pero el modelo de IA subyacente
  podría requerirlas.

## Entradas y Salidas

* **Entradas:**
    * Código fuente de una aplicación Flask (archivo `.py`).
    * Instrucciones de prompt para la IA (archivo `swagger_generation.txt`).
* **Salidas:**
    * Un archivo `swagger.yaml` que contiene la especificación OpenAPI 3.1.0 de la API Flask.

## Ejemplos de Uso

Para generar un `swagger.yaml` a partir de un archivo `app.py` y guardarlo como `api_spec.yaml`:

```bash
python scripts/swagger_generation.py app.py --output api_spec.yaml
```

Para usar el nombre de archivo de salida por defecto (`swagger.yaml`):

```bash
python scripts/swagger_generation.py main_flask_app.py
```

## Mantenimiento y Puntos Clave

* **Dependencia de IA:** La calidad del `swagger.yaml` generado depende en gran medida de la capacidad del modelo de IA
  y de la calidad de las instrucciones (`PROMPT_FILENAME`).
* **Estructura del Código Flask:** El script asume que el código Flask está estructurado de una manera que la IA puede
  interpretar para extraer las rutas, métodos HTTP y esquemas.
* **Manejo de Errores:** El script incluye validaciones básicas para la existencia del archivo de entrada y la
  inicialización del modelo/instrucciones, pero errores más complejos en la IA o en la estructura del código Flask
  podrían no ser manejados explícitamente.
* **Actualización de Prompts:** Si la estructura de las aplicaciones Flask cambia o se requieren formatos de Swagger más
  específicos, el archivo `swagger_generation.txt` deberá ser actualizado.
