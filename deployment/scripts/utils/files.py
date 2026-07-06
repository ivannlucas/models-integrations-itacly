# utils/files.py
# Description: Shared module for file system operations.

import os
import sys
from pathlib import Path
from typing import List, Optional

from git import get_pr_context

from ui import log_message

# Allowed file extensions, mapped to their corresponding language.
ALLOWED_EXTENSIONS = {
    ".py": "python",
    ".sql": "sql",
    ".java": "java"
}

# List of filenames to be excluded from analysis.
EXCLUDED_FILES = [
    "__init__.py",
]

PROMPT_BASE_PATH = Path("./deployment/scripts/utils/llm_prompts")


def read_file_content(filepath: Path) -> Optional[str]:
    """
    Reads the content of a file safely.

    Args:
        filepath: The path to the file.

    Returns:
        The content of the file as a string, or None if an error occurs.
    """
    try:
        return filepath.read_text(encoding='utf-8')
    except Exception as e:
        log_message(f"Could not read file '{filepath}': {e}", "ERROR")
        return None


def find_files_to_process(path: Path) -> List[Path]:
    """
    Finds all valid files to process in a given path.
    If the path is a file, it returns it in a list. If it's a directory,
    it recursively searches for all files with allowed extensions.
    """
    files_to_process = []
    if not path.exists():
        log_message(f"Path does not exist: {path}", "ERROR")
        sys.exit(1)

    if path.is_file():
        if path.suffix in ALLOWED_EXTENSIONS:
            files_to_process.append(path)
    elif path.is_dir():
        for extension in ALLOWED_EXTENSIONS:
            files_to_process.extend(path.rglob(f"*{extension}"))

    return [f for f in files_to_process if f.name not in EXCLUDED_FILES]


def get_pr_files() -> List[Path]:
    """
    Gets the list of modified files in the PR against the target branch.
    Filters out non-existent, empty, or non-allowed files.
    """
    modified_files_str = get_pr_context().get('modified_files', [])

    files_to_analyze = []
    for file_path_str in modified_files_str:
        if not file_path_str:
            continue
        file_path = Path(file_path_str)
        if not file_path.exists():
            continue

        # Ignore files with not allowed extensions and in the excluded list
        if file_path.suffix not in ALLOWED_EXTENSIONS or file_path.name in EXCLUDED_FILES:
            continue

        try:
            # Ignore files that only contain whitespace
            if file_path.read_text(encoding='utf-8').strip() == "":
                continue
        except (FileNotFoundError, IOError, UnicodeDecodeError):
            continue

        files_to_analyze.append(file_path)

    return files_to_analyze


def get_files_to_process(path_arg: Optional[str]) -> List[Path]:
    """
    Obtiene la lista de archivos a procesar, ya sea desde un 'path'
    específico o desde los archivos modificados en un PR de Git.
    """
    if path_arg:
        log_message(f"Processing files from specified path: {path_arg}", "INFO")
        files = find_files_to_process(Path(path_arg))
    else:
        log_message("Processing modified files from Git PR.", "INFO")
        files = get_pr_files()

    if not files:
        log_message("No valid files found to process.", "WARNING")
        # No salimos, el script principal puede decidir qué hacer

    return files


def determine_root_path(files_to_process: List[Path]) -> Path:
    """
    Determina la ruta raíz común para una lista de archivos.
    Esto es útil para calcular las rutas relativas al guardar los documentos.
    """
    if not files_to_process:
        return Path.cwd().resolve()

    try:
        # Resolvemos las rutas para que commonpath funcione correctamente
        resolved_paths = [str(p.resolve()) for p in files_to_process]
        common_path_str = os.path.commonpath(resolved_paths)

        # Replicamos la lógica original: la raíz es el 'padre' del path común
        root_path = Path(common_path_str).parent

    except ValueError:
        root_path = Path.cwd().resolve()
        log_message(f"Could not determine a common path. Using CWD as root: {root_path}", "WARNING")

    log_message(f"Determined root path: {root_path}", "INFO")
    return root_path


def load_agent_instructions(prompt_filename: str) -> str:
    """
    Carga las instrucciones del agente (prompt) desde un archivo.
    Si falla, termina el script.

    Args:
        prompt_filename: El nombre del archivo .txt en el directorio PROMPT_BASE_PATH.

    Returns:
        El contenido del archivo de prompt como string.
    """
    prompt_path = PROMPT_BASE_PATH / prompt_filename
    # log_message(f"Loading agent instructions from: {prompt_path}", "INFO")
    instructions = read_file_content(prompt_path)
    if not instructions:
        log_message(f"Could not read prompt file: {prompt_path}. Exiting.", "ERROR")
        sys.exit(1)
    return instructions


def save_output_file(output_path: Path, content: str):
    """
    Guarda el contenido en un archivo de salida, creando los directorios
    padre si no existen.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
        log_message(f"Output saved successfully to: '{output_path}'", "SUCCESS")
    except IOError as e:
        log_message(f"Failed to write output file at {output_path}: {e}", "ERROR")
    except Exception as e:
        log_message(f"An unexpected error occurred while saving {output_path}: {e}", "ERROR")
