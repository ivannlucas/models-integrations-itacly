# utils/ia.py
# Description: Shared module for interacting with the Google Gemini API.

import json
import os
import re
from typing import Dict, Any, Optional

import google.generativeai as genai

from ui import log_message


# --- CONFIGURATION ---
def get_env_int(key: str, default: int) -> int:
    """
    Gets an environment variable and safely converts it to an integer.

    Returns:
        The variable's integer value, or the default if missing or invalid.
    """
    try:
        value = os.getenv(key)
        return int(value)
    except (TypeError, ValueError):
        return default


MAX_TOTAL_TOKENS = get_env_int("MAX_TOTAL_TOKENS",
                               75000)  # Safety limit to avoid excessive consumption on large executions
MAX_REQUEST_TOKENS = get_env_int("MAX_REQUEST_TOKENS",
                                 8192)  # Safety limit to avoid excessive consumption on single executions
MODEL_TEMPERATURE = 0.2  # Controls the randomness of an AI model's output from deterministic (0) to creative (1)
MODEL_VERSION = os.getenv("GEMINI_MODEL", 'gemini-2.5-flash-lite')

# Precios en USD por 1,000,000 (1 millón) de tokens
# (A fecha de Octubre 2025, estos son precios de ejemplo basados en los datos públicos)
MODEL_PRICING_PER_MILLION_TOKENS = {
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.0,
        "cache": 0.125,
    },
    "gemini-2.5-flash": {
        "input": 0.3,
        "output": 2.5,
        "cache": 0.03,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.1,
        "output": 0.4,
        "cache": 0.01,
    }
}


def setup_generative_model() -> Optional[genai.GenerativeModel]:
    """
    Configures and returns a Gemini Generative Model instance.

    Returns:
        A configured GenerativeModel instance or None if the API key is missing.
    """
    # log_message("Initializing Generative Model...", "INFO")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log_message("The GEMINI_API_KEY environment variable is not set.", "ERROR")
        return None
    try:
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash as it is fast and capable for these tasks
        model = genai.GenerativeModel(MODEL_VERSION)
        # log_message(f"Model {model_version} initialized successfully.", "SUCCESS")
        return model
    except Exception as e:
        log_message(f"Failed to configure the generative model: {e}", "ERROR")
        return None


def calculate_consumption_price(consumes, model):
    model_prices = MODEL_PRICING_PER_MILLION_TOKENS.get(model, {"input": 2.5, "output": 20.0, "cache": 0.5})
    input_price = consumes.get("prompt") * model_prices["input"] / 1000000
    output_price = consumes.get("candidates") * model_prices["output"] / 1000000
    cache_price = consumes.get("cached_content") * model_prices["cache"] / 1000000

    price = input_price + output_price + cache_price

    return price


def parse_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses a JSON object from a string, which might contain markdown code blocks.
    It robustly finds a JSON block and decodes it.

    Args:
        response_text: The text response from the AI model.

    Returns:
        A dictionary if a valid JSON object is found and parsed, otherwise None.
    """
    # First, try to find a JSON block inside markdown ```json ... ```
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If no markdown block is found, try to find the first '{' and the last '}'
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end + 1]
        else:
            log_message("No valid JSON structure found in the model's response.", "ERROR")
            log_message(f"Received response:\n{response_text}", "INFO")
            return None

    try:
        # Clean up potential artifacts before parsing
        json_str = json_str.strip()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        log_message(f"Failed to decode JSON from the response: {e}", "ERROR")
        log_message(f"Problematic JSON string:\n{json_str}", "INFO")
        return None


def invoke_model(model: genai.GenerativeModel, prompt: str, is_json_output: bool = True):
    """
    Invokes the generative model with a given prompt and handles the response.

    Args:
        model: The generative model instance.
        prompt: The full prompt to send to the model.
        is_json_output: If True, expects a JSON response and configures the model accordingly.

    Returns:
        The full generation response object, or None if an error occurs.
    """
    try:
        generation_config = {"temperature": MODEL_TEMPERATURE, "max_output_tokens": MAX_REQUEST_TOKENS}
        if is_json_output:
            # This tells the model to ensure its output is valid JSON.
            generation_config["response_mime_type"] = "application/json"

        response = model.generate_content(prompt, generation_config=generation_config)
        return response
    except Exception as e:
        log_message(f"Error during API call to Gemini: {e}", "ERROR")
        return None


def process_file_with_ia(model: Any, agent_instructions: str, code_content: str,
                         filepath: str, language: str = "python",
                         is_json_output: bool = True) -> tuple[Optional[Dict[str, Any] | str], int, float]:
    """
    Función central para procesar un archivo con la IA.

    1. Construye el prompt completo.
    2. Invoca el modelo.
    3. Procesa la respuesta (parsea JSON o limpia texto).

    Args:
        model: El cliente del modelo generativo.
        agent_instructions: El prompt base para la IA.
        code_content: Code to eval
        filepath: Ruta al fichero que se va a evaluar
        language: Extensión del fichero que se va a revisar
        is_json_output: True si se espera un JSON, False si se espera texto plano (YAML).

    Returns:
        Un diccionario (si es JSON) o un string (si es texto), o None si falla.
    """
    # Nomenclatura unificada para el prompt
    full_prompt = f"{agent_instructions}\n\n**Code to process:**\n```{language}\n{code_content}\n```"

    response = invoke_model(model, full_prompt, is_json_output=is_json_output)
    if not response or not response.text:
        log_message("Failed to get a response from IA.", "ERROR")
        empty_result = {'filepath': str(filepath), 'code_content': code_content}
        return empty_result, 0, 0

    tokens_used = {
        "total": response.usage_metadata.total_token_count,
        "prompt": response.usage_metadata.prompt_token_count,
        "candidates": response.usage_metadata.candidates_token_count,
        "cached_content": response.usage_metadata.cached_content_token_count,
    } if response.usage_metadata else {"total": 0, "prompt": 0, "candidates": 0, "cached_content": 0}

    consumption = calculate_consumption_price(tokens_used, response.model_version)

    if is_json_output:
        try:
            # Limpieza básica de la respuesta JSON
            cleaned_str = response.text.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:]
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3]
            cleaned_str = cleaned_str.strip()

            parsed_data = json.loads(cleaned_str)

            # Añadir metadata al dict
            if isinstance(parsed_data, dict):
                parsed_data['filepath'] = str(filepath)
                parsed_data['code_content'] = code_content

            return parsed_data, tokens_used.get("total"), consumption

        except json.JSONDecodeError as e:
            log_message(f"Failed to decode JSON from response for {filepath}: {e}", "ERROR")
            log_message(f"Problematic JSON string after cleaning: {cleaned_str}", "INFO")
            empty_result = {'filepath': str(filepath), 'code_content': code_content}
            return empty_result, tokens_used.get("total"), consumption
    else:
        # Es texto plano (YAML para swagger)
        # Limpieza básica de ```yaml o ```
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```yaml"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        return cleaned_text.strip(), tokens_used.get("total"), consumption
