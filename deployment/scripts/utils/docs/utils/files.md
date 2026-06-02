# Script: utils/files.py

## Resumen y Objetivo

Este script proporciona un conjunto de funciones de utilidad para la gestión de archivos y directorios. Su objetivo
principal es facilitar la lectura, escritura, búsqueda y filtrado de archivos dentro de un proyecto, especialmente en el
contexto de operaciones de control de versiones como Pull Requests (PRs) de Git.

## Arquitectura y Lógica Principal

El módulo `utils/files.py` se organiza en varias funciones clave:

1. **`read_file_content(filepath)`**: Lee el contenido de un archivo dado de forma segura, manejando posibles errores de
   lectura y codificación.
2. **`find_files_to_process(path)`**: Busca archivos recursivamente en un directorio o devuelve un archivo específico si
   la ruta proporcionada es un archivo. Filtra los archivos basándose en `ALLOWED_EXTENSIONS` y `EXCLUDED_FILES`.
3. **`get_pr_context()` (dependencia externa)**: Se asume que esta función (importada de `git`) obtiene información
   sobre los archivos modificados en un PR.
4. **`get_pr_files()`**: Utiliza `get_pr_context()` para obtener la lista de archivos modificados en un PR, filtrando
   aquellos que no existen, están vacíos o no tienen extensiones permitidas.
5. **`get_files_to_process(path_arg)`**: Actúa como un punto de entrada principal para obtener la lista de archivos. Si
   se proporciona un argumento de ruta (`path_arg`), usa `find_files_to_process`. De lo contrario, utiliza
   `get_pr_files` para procesar archivos de un PR.
6. **`determine_root_path(files_to_process)`**: Calcula la ruta raíz común para una lista de archivos, lo cual es útil
   para determinar rutas relativas.
7. **`load_agent_instructions(prompt_filename)`**: Carga el contenido de un archivo de prompt desde una ubicación
   predefinida (`PROMPT_BASE_PATH`).
8. **`save_output_file(output_path, content)`**: Guarda contenido en un archivo, asegurándose de que los directorios
   padre existan y manejando errores de escritura.

El script también define constantes como `ALLOWED_EXTENSIONS` y `EXCLUDED_FILES` para configurar el comportamiento de
búsqueda y filtrado de archivos.

## Configuración Requerida

* **`PROMPT_BASE_PATH`**: Una variable de entorno o configuración implícita que define la ruta base para cargar
  prompts (`./deployment/scripts/utils/llm_prompts`).
* **Dependencia de `git`**: Para la funcionalidad `get_pr_files`, se requiere que el script se ejecute en un entorno
  donde la biblioteca `git` pueda acceder al contexto de un PR.
* **`ui.log_message`**: Se asume la existencia de una función `log_message` para el registro de eventos y errores.

## Entradas y Salidas

* **Entradas**:
    * Rutas de archivos o directorios (como argumento `path_arg`).
    * Nombres de archivos de prompt (para `load_agent_instructions`).
    * Contexto de un PR de Git (implícito para `get_pr_files`).
* **Salidas**:
    * Listas de objetos `Path` que representan los archivos a procesar.
    * Contenido de archivos como strings.
    * Archivos de salida guardados en el disco.
    * Mensajes de log (a través de `ui.log_message`).

## Ejemplos de Uso

**1. Procesar un directorio específico:**

```python
from pathlib import Path
from utils import files

# Asumiendo que el script se llama main.py y utils/files.py está en el mismo nivel
# o en una ruta importable.

# Ejemplo de cómo usar las funciones dentro de otro script:

# Para encontrar todos los archivos .py y .sql en un directorio 'src':
path_to_process = Path("./src")
files_to_analyze = files.find_files_to_process(path_to_process)

for file_path in files_to_analyze:
    content = files.read_file_content(file_path)
    if content:
        print(f"Contenido de {file_path}:\n{content[:100]}...") # Muestra los primeros 100 caracteres

# Para cargar instrucciones de un prompt:
agent_prompt = files.load_agent_instructions("system_prompt.txt")
print(f"\nInstrucciones del agente:\n{agent_prompt}")

# Para guardar un archivo de salida:
output_file_path = Path("./output/results.txt")
output_content = "Este es el contenido de prueba."
files.save_output_file(output_file_path, output_content)
```

**2. Procesar archivos modificados en un PR (ejecución desde la línea de comandos si `get_files_to_process` es el punto
de entrada principal):**

```bash
# Si este script se ejecuta en un entorno de PR y no se pasa un path:
python tu_script_principal.py
```

## Mantenimiento y Puntos Clave

* **Dependencia de `git`**: La función `get_pr_files` depende de la correcta configuración y acceso al contexto de un PR
  de Git. Si se ejecuta fuera de este contexto, no devolverá archivos.
* **Manejo de Errores**: Las funciones de lectura y escritura de archivos incluyen manejo básico de excepciones, pero se
  debe asegurar que los mensajes de log sean claros y que las excepciones no manejadas sean capturadas adecuadamente en
  los niveles superiores.
* **Ruta Raíz Común**: La lógica para determinar `root_path` puede ser sensible a la estructura de directorios. Si los
  archivos a procesar están en directorios muy dispares, `os.path.commonpath` podría no dar el resultado esperado, y se
  recurre a `Path.cwd()`.
* **Codificación de Archivos**: Se asume codificación `utf-8`. Si los archivos utilizan otras codificaciones,
  `read_file_content` y `save_output_file` podrían fallar o corromper datos.
* **Extensión de Archivos**: La lista `ALLOWED_EXTENSIONS` y `EXCLUDED_FILES` debe mantenerse actualizada según las
  necesidades del proyecto.