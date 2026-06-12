# Script: utils/ia.py

## Resumen y Objetivo

Este script proporciona un módulo de utilidad para interactuar con la API de Google Gemini. Su objetivo principal es
facilitar la integración de modelos de lenguaje grandes (LLMs) en otras aplicaciones, ofreciendo funciones para
configurar el modelo, invocarlo con prompts y procesar sus respuestas, ya sea en formato JSON o texto plano.

## Arquitectura y Lógica Principal

El módulo se organiza en las siguientes funciones clave:

1. **`setup_generative_model()`**: Configura y devuelve una instancia del modelo `GenerativeModel` de Google Gemini.
   Requiere la variable de entorno `GEMINI_API_KEY` y permite especificar la versión del modelo a través de
   `GEMINI_MODEL`. Maneja errores de configuración y claves faltantes.
2. **`parse_json_from_response()`**: Extrae y parsea un objeto JSON de una respuesta de texto, buscando bloques de
   código JSON (` ```json ... ``` `) o la primera ocurrencia de `{...}`. Es robusto ante respuestas que contienen texto
   adicional o markdown.
3. **`invoke_model()`**: Envía un prompt al modelo Gemini configurado. Permite especificar si se espera una respuesta en
   formato JSON (`application/json`) o texto plano, ajustando la configuración de generación (temperatura, tokens
   máximos). Maneja errores durante la llamada a la API.
4. **`process_file_with_ia()`**: La función central que orquesta el proceso. Construye un prompt completo incluyendo
   instrucciones y el contenido del código a procesar. Invoca el modelo usando `invoke_model()` y luego procesa la
   respuesta (parseando JSON o limpiando texto) según el parámetro `is_json_output`. Añade metadatos como `filepath` y
   `code_content` a las respuestas JSON.

## Configuración Requerida

* **Variables de Entorno:**
    * `GEMINI_API_KEY`: Clave de API necesaria para autenticarse con Google Gemini.
    * `GEMINI_MODEL` (Opcional): Especifica la versión del modelo Gemini a utilizar (ej. `gemini-2.5-flash-lite`). Si no
      se establece, se usará `gemini-2.5-flash-lite` por defecto.

## Entradas y Salidas

* **Entradas:**
    * Instancia de `genai.GenerativeModel`.
    * `agent_instructions`: Un string que contiene las instrucciones para la IA.
    * `code_content`: Un string con el contenido del código a procesar.
    * `filepath`: El nombre del archivo que se está procesando.
    * `language`: El lenguaje del código (por defecto `python`).
    * `is_json_output`: Booleano indicando si se espera una respuesta JSON.

* **Salidas:**
    * Si `is_json_output` es `True`: Un diccionario (`Dict[str, Any]`) que contiene los datos JSON parseados, junto con
      `filepath` y `code_content`, o `None` si falla el parseo o la invocación.
    * Si `is_json_output` es `False`: Un string (`str`) con el texto plano procesado (ej. YAML), o `None` si falla la
      invocación.
    * En caso de error, se retorna `None` y el número de tokens usados (que puede ser 0).

## Ejemplos de Uso

```python
import os
from utils.ia import setup_generative_model, process_file_with_ia

# Asegúrate de que GEMINI_API_KEY esté configurada en tu entorno
# os.environ["GEMINI_API_KEY"] = "TU_API_KEY"

model = setup_generative_model()

if model:
    agent_prompt = "Extract the main function name and its description from the following Python code."
    code_to_analyze = """\ndef calculate_sum(a, b):
    \"\"\"Calculates the sum of two numbers.\n    \"\"\"
    return a + b
"""
    filepath = "example.py"

    # Ejemplo para obtener salida JSON
    result_json, tokens = process_file_with_ia(model, agent_prompt, code_to_analyze, filepath, is_json_output=True)
    if result_json:
        print("JSON Output:", result_json)
        print(f"Tokens used: {tokens}")

    # Ejemplo para obtener salida de texto plano (ej. para generar documentación)
    text_prompt = "Generate a short documentation for the following Python function.\n**Format:**
```markdown
## Function: [function_name]

**Description:** [description]
```"
    result_text, tokens = process_file_with_ia(model, text_prompt, code_to_analyze, filepath, is_json_output=False)
    if result_text:
        print("Text Output:", result_text)
        print(f"Tokens used: {tokens}")
else:
    print("Failed to initialize the generative model.")

```

## Mantenimiento y Puntos Clave

* **Dependencia de API Externa:** El correcto funcionamiento depende de la disponibilidad y el rendimiento de la API de
  Google Gemini. Los errores de red o de la API pueden causar fallos.
* **Gestión de Tokens:** Se implementan límites de tokens (`MAX_TOTAL_TOKENS`, `MAX_REQUEST_TOKENS`) para controlar el
  consumo y evitar costos excesivos o respuestas truncadas. Es importante monitorizar el uso de tokens.
* **Robustez del Parseo JSON:** La función `parse_json_from_response` intenta ser robusta, pero respuestas malformadas
  de la IA pueden requerir ajustes adicionales en la lógica de limpieza o parseo.
* **Configuración de Modelo:** La elección del modelo (`GEMINI_MODEL`) y la temperatura (`MODEL_TEMPERATURE`) pueden
  afectar significativamente la calidad y el formato de las respuestas. Ajustar estos parámetros puede ser necesario
  para casos de uso específicos.
* **Logging:** El script utiliza una función `log_message` (importada de `ui`) para registrar información y errores.
  Asegúrate de que este módulo esté disponible y configurado correctamente.
