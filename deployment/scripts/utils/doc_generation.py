# scripts/doc_generation.py
# Purpose: Automate documentation creation for code files, preserving directory structure.

import argparse
import subprocess
import sys
from pathlib import Path

from confluence import get_confluence_client, create_or_update_confluence_page, ensure_page_hierarchy_exists
from files import load_agent_instructions, get_files_to_process, determine_root_path, save_output_file, \
    read_file_content, ALLOWED_EXTENSIONS
from ia import setup_generative_model, process_file_with_ia, MAX_TOTAL_TOKENS
from ui import log_message

# Definimos la ruta del prompt
PROMPT_FILENAME = "doc_generation.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Generates documentation for a file or directory, preserving the folder structure.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Argumentos
    parser.add_argument("path", nargs='?', default=None, type=str,
                        help="Path to the file/directory. If omitted, analyzes modified files in Git.")
    parser.add_argument("--output-dir", type=str, default="docs",
                        help="Directory to save the generated Markdown files.")

    # Argumentos específicos de este script
    parser.add_argument("--extra-docs", action="store_true",
                        help="Enables generate specific documentation for certain file types.")
    parser.add_argument("--swagger-output", type=str, default="swagger.yaml",
                        help="Output path for the swagger.yaml file if generated.")
    parser.add_argument("--confluence", action="store_true",
                        help="Enables automatic upload to Confluence, saving files locally instead.")
    parser.add_argument("--space", type=str,
                        help="Confluence space key to upload the documentation.")
    parser.add_argument("--parent-id", type=str, default=None,
                        help="Confluence parent page ID.")

    args = parser.parse_args()

    log_message("STARTING DOCUMENTATION GENERATOR", "HEADER")

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

    # Lógica de Confluence
    confluence_client = None
    if args.confluence:
        if not args.space:
            log_message("Argument --space is required when uploading to Confluence.", "ERROR")
            sys.exit(1)
        confluence_client = get_confluence_client()
        if not confluence_client:
            log_message("Could not establish a connection with Confluence. Exiting.", "ERROR")
            sys.exit(1)

    # Obtención de archivos
    files_to_process = get_files_to_process(args.path)
    if not files_to_process:
        sys.exit(0)  # Salimos si no hay archivos

    # Determinación de ruta raíz
    root_path = determine_root_path(files_to_process)

    log_message(f"{len(files_to_process)} file(s) will be processed.", "INFO")

    hierarchy_cache = {}  # Caché para Confluence

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

        # Llamada a la función de procesamiento
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

        documentation = result.get("documentation")
        if not documentation:
            log_message(f"No documentation was generated for {filepath}.", "WARNING")
            continue

        # Usamos resolve() para asegurar que relative_to funcione
        relative_path = filepath.resolve().relative_to(root_path)

        if not args.confluence:
            # --- Guardado Local ---
            output_filepath = Path(args.output_dir).resolve() / relative_path
            output_filepath = output_filepath.with_suffix('.md')

            # Función de guardado
            save_output_file(output_filepath, documentation)

            # --- Lógica de Subprocesos para ciertos tipos de ficheros
            if args.extra_docs:
                file_type = result.get("classification")
                if file_type == "ModelInference_API" or file_type == "WebService_General":
                    log_message(f"{file_type} file detected, generating Swagger...", "INFO")
                    subprocess.run(
                        f"python ./deployment/scripts/utils/swagger_generation.py {filepath} --output {args.swagger_output}",
                        shell=True, text=True)

                elif file_type == "DataPipeline_ETL":
                    log_message(f"{file_type} file detected, generating Data Catalog...", "INFO")
                    subprocess.run(
                        f"python ./deployment/scripts/utils/data_catalog_generation.py {filepath} --output-dir {args.output_dir}/catalog",
                        shell=True, text=True)
                else:
                    log_message(f"No extra documentation needed for {file_type} files", "INFO")

        else:
            # --- Lógica de escritura en Confluence ---
            hierarchy_parts = tuple(relative_path.parts[:-1])
            page_title = relative_path.stem
            document_parent_id = args.parent_id

            if hierarchy_parts:
                if hierarchy_parts in hierarchy_cache:
                    document_parent_id = hierarchy_cache[hierarchy_parts]
                    log_message(f"Using cached parent ID for hierarchy {hierarchy_parts}.", "INFO")
                else:
                    final_parent_id = ensure_page_hierarchy_exists(
                        client=confluence_client,
                        space_key=args.space,
                        hierarchy=list(hierarchy_parts),
                        base_parent_id=args.parent_id
                    )

                    if final_parent_id:
                        hierarchy_cache[hierarchy_parts] = final_parent_id
                        document_parent_id = final_parent_id
                    else:
                        log_message(f"Could not ensure parent hierarchy for {filepath}. Skipping upload.", "ERROR")
                        continue

            try:
                create_or_update_confluence_page(
                    client=confluence_client,
                    space_key=args.space,
                    title=page_title,
                    body=documentation,
                    parent_id=document_parent_id
                )
            except Exception as e:
                log_message(f"Skipping upload for {page_title} due to an error: {e}", "WARNING")

    log_message("DOCUMENTATION PROCESS FINISHED", "HEADER")


if __name__ == "__main__":
    main()
