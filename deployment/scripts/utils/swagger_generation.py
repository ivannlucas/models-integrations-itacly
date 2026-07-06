# scripts/swagger_generation.py
# Purpose: Generate a swagger.yaml file from a Flask application's source code.

import argparse
import sys
from pathlib import Path

from files import load_agent_instructions, save_output_file, read_file_content
from ia import setup_generative_model, process_file_with_ia
from ui import log_message

PROMPT_FILENAME = "swagger_generation.txt"


def main():
    """
    Main function to run the script from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Generates a swagger.yaml from a Flask source code file."
    )
    # Argumentos
    parser.add_argument("path", type=str, help="Path to the main .py file of the Flask app.")
    parser.add_argument("--output", type=str, default="swagger.yaml", help="Output path for the swagger.yaml file.")
    args = parser.parse_args()

    log_message("INITIALIZING SWAGGER GENERATOR", "HEADER")

    # Inicialización de modelo
    model = setup_generative_model()
    if not model:
        log_message("Failed to initialize model. Exiting.", "ERROR")
        sys.exit(1)

    # Carga de instrucciones
    agent_instructions = load_agent_instructions(PROMPT_FILENAME)
    if not agent_instructions:
        log_message("Failed to initialize agent instructions. Exiting.", "ERROR")
        sys.exit(1)

    filepath = Path(args.path)

    # Validar que el archivo de entrada existe antes de procesar
    if not filepath.is_file():
        log_message(f"Input file not found: {filepath}", "ERROR")
        sys.exit(1)

    log_message(f"Processing file: {filepath}", "INFO")
    code_content = read_file_content(filepath)
    if not code_content:
        log_message(f"Skipping {filepath}: Could not read file content.", "ERROR")
        sys.exit(1)

    # Llamada a la función de procesamiento unificada
    # Le indicamos que NO esperamos un JSON, sino texto plano (YAML)
    swagger_content, tokens_call, consumption_call = process_file_with_ia(model, agent_instructions, code_content,
                                                                          str(filepath), is_json_output=False)
    # log_message(f"Tokens consumed: {tokens_call}  --> {consumption_call} $", "INFO")

    if swagger_content and isinstance(swagger_content, str):
        output_path = Path(args.output)

        # Función de guardado unificada
        save_output_file(output_path, swagger_content)
        # El log de éxito ahora está en save_output_file, pero podemos añadir uno específico si queremos.
        log_message(f"OpenAPI 3.1.0 specification generated at '{output_path}'", "SUCCESS")
    else:
        log_message("Swagger generation process failed. No valid content received from IA.", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
