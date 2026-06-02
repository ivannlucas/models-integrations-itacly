# scripts/data_catalog_generation.py
# Purpose: Automate data catalog creation from code files.

import argparse
import json
import sys
from pathlib import Path

from files import load_agent_instructions, get_files_to_process, determine_root_path, save_output_file, \
    read_file_content, ALLOWED_EXTENSIONS
from ia import setup_generative_model, process_file_with_ia, MAX_TOTAL_TOKENS
from ui import log_message

PROMPT_FILENAME = "data_catalog_generation.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Generates data catalog entries for a file or directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Argumentos
    parser.add_argument("path", nargs='?', default=None, type=str,
                        help="Path to the file/directory. If omitted, analyzes modified files in Git.")
    parser.add_argument("--output-dir", type=str, default="docs/catalog",
                        help="Directory to save the generated JSON files.")

    args = parser.parse_args()

    log_message("STARTING DATA CATALOG GENERATOR", "HEADER")

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

    # Obtención de archivos
    files_to_process = get_files_to_process(args.path)
    if not files_to_process:
        sys.exit(0)  # Salimos si no hay archivos

    # Determinación de ruta raíz
    root_path = determine_root_path(files_to_process)

    log_message(f"{len(files_to_process)} file(s) will be processed.", "INFO")

    total_tokens_used = 0
    consumption_total = 0
    file_number_processed = 1
    for filepath in files_to_process:
        if total_tokens_used >= MAX_TOTAL_TOKENS:
            log_message(f"Token limit reached ({MAX_TOTAL_TOKENS}). Stopping.", "WARNING")
            break

        log_message(f"Processing file: {filepath}. File {file_number_processed} from {len(files_to_process)}", "INFO")
        file_number_processed += 1
        language = ALLOWED_EXTENSIONS.get(filepath.suffix)
        if not language:
            log_message(f"Unsupported file extension: {filepath.suffix}", "WARNING")
            continue

        code_content = read_file_content(filepath)
        if not code_content:
            log_message(f"Skipping {filepath}: Could not read file content.", "WARNING")
            continue

        # Llamada a la función de procesamiento unificada
        # Esperamos una respuesta JSON (is_json_output=True por defecto)
        result, tokens_call, consumption_call = process_file_with_ia(model, agent_instructions, code_content,
                                                                     str(filepath))
        total_tokens_used += tokens_call
        consumption_total += consumption_call
        log_message(f"Tokens consumed: {tokens_call} | Total: {total_tokens_used} --> {round(consumption_total, 4)} $",
                    "INFO")

        if not result or not isinstance(result, dict):
            log_message(f"Skipping file {filepath} due to processing error or invalid format.", "WARNING")
            continue

        # Usamos resolve() para asegurar que relative_to funcione
        relative_path = filepath.resolve().relative_to(root_path)

        # Preparar la ruta de salida
        output_filepath = Path(args.output_dir).resolve() / relative_path
        output_filepath = output_filepath.with_suffix('.json')

        # Convertimos el dict resultante a un string JSON
        try:
            output_content = json.dumps(result, indent=2, ensure_ascii=False)
        except TypeError as e:
            log_message(f"Could not serialize result to JSON for {filepath}: {e}", "ERROR")
            continue

        # Función de guardado
        save_output_file(output_filepath, output_content)

    log_message("DATA CATALOG PROCESS FINISHED", "HEADER")


if __name__ == "__main__":
    main()
