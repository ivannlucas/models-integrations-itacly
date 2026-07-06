# Script: utils/git.py

## Resumen y Objetivo

Este módulo proporciona funciones de utilidad para interactuar con el sistema de control de versiones Git,
específicamente diseñado para obtener información contextual de solicitudes de extracción (Pull Requests) en entornos de
integración continua (CI), con un enfoque inicial en Bitbucket.

El objetivo es facilitar la automatización de tareas relacionadas con la revisión de código y el despliegue,
proporcionando datos sobre archivos modificados, mensajes de commit y detalles del repositorio y la rama.

## Arquitectura y Lógica Principal

El módulo se organiza en las siguientes funciones:

1. **`_run_git_command(command)`**: Una función auxiliar interna que ejecuta comandos de Git en el shell, captura su
   salida, la decodifica como texto y maneja errores. Devuelve la salida como una lista de líneas.
2. **`get_modified_files(target_branch)`**: Utiliza `_run_git_command` para obtener una lista de nombres de archivos
   modificados entre la rama actual (HEAD) y una rama objetivo especificada (por defecto, `develop`).
3. **`get_commit_messages(target_branch)`**: Utiliza `_run_git_command` para obtener los mensajes de commit (solo la
   línea de asunto) entre la rama actual (HEAD) y una rama objetivo especificada.
4. **`get_bitbucket_pr_context()`**: Recopila información específica del entorno de CI de Bitbucket, como el nombre de
   la rama, los archivos modificados, los mensajes de commit y varias variables de entorno relacionadas con el proyecto
   y el PR. Utiliza `get_modified_files` y `get_commit_messages`.
5. **`get_pr_context(source_control)`**: Una función genérica que actúa como punto de entrada para obtener el contexto
   de PR. Actualmente, solo soporta `bitbucket`, delegando la llamada a `get_bitbucket_pr_context()`. Si se especifica
   otro sistema de control de origen, lanza un `ValueError`.

## Configuración Requerida

Este script depende de las variables de entorno que suelen estar disponibles en entornos de CI de Bitbucket. Las
variables clave incluyen:

* `BITBUCKET_BRANCH`: Nombre de la rama actual.
* `BITBUCKET_PR_DESTINATION_BRANCH`: Rama de destino del PR (por defecto `develop` si no está definida).
* `BITBUCKET_PROJECT_KEY`: Clave del proyecto en Bitbucket.
* `BITBUCKET_REPO_SLUG`: Nombre corto del repositorio.
* `BITBUCKET_REPO_FULL_NAME`: Nombre completo del repositorio.
* `BITBUCKET_COMMIT`: Hash del commit actual.
* `BITBUCKET_PR_DESTINATION_COMMIT`: Hash del commit de destino del PR.
* `BITBUCKET_REPO_IS_PRIVATE`: Indica si el repositorio es privado.
* `CI`: Indica si se está ejecutando en un entorno CI.
* `BITBUCKET_WORKSPACE`: Nombre del espacio de trabajo de Bitbucket.
* `BITBUCKET_REPO_OWNER`: Propietario del repositorio.
* `BITBUCKET_GIT_HTTP_ORIGIN`: Origen HTTP del repositorio.

## Entradas y Salidas

* **Entradas**: El script no toma argumentos directos de línea de comandos. Su entrada principal son las variables de
  entorno establecidas por el sistema de CI (principalmente Bitbucket) y el estado del repositorio Git local o remoto.
* **Salidas**: Las funciones devuelven diccionarios (`context`) que contienen información estructurada sobre el contexto
  del PR, listas de strings (archivos modificados, mensajes de commit) o lanzan excepciones (`ValueError`) si el sistema
  de control de origen no es compatible.

## Ejemplos de Uso

Para usar la función principal `get_pr_context` en un script Python:

```python
import utils.git

try:
    # Obtener contexto de PR para Bitbucket
    pr_context = utils.git.get_pr_context(source_control='bitbucket')
    print(f"Branch Name: {pr_context['branch_name']}")
    print(f"Modified Files: {pr_context['modified_files']}")
    print(f"Commit Messages: {pr_context['commit_messages']}")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

## Mantenimiento y Puntos Clave

* **Dependencia de Entorno CI**: La funcionalidad principal de `get_bitbucket_pr_context` depende en gran medida de la
  presencia y el formato correcto de las variables de entorno de Bitbucket. Si se ejecuta fuera de un entorno Bitbucket
  CI o si las variables de entorno cambian, el script podría no funcionar como se espera o devolver datos incompletos.
* **Soporte Limitado de Control de Origen**: Actualmente, solo se soporta Bitbucket. La función `get_pr_context` lanzará
  un error si se solicita otro sistema de control de origen.
* **Manejo de Errores de Git**: La función `_run_git_command` está diseñada para devolver una lista vacía en caso de
  `subprocess.CalledProcessError`, lo que puede ocultar fallos de comandos Git si no se maneja explícitamente en las
  funciones que la llaman.
* **Evolución Futura**: Para mantener el script, se podría considerar añadir soporte para otros sistemas de control de
  origen (como GitHub Actions o GitLab CI) y mejorar el manejo de errores para proporcionar mensajes más informativos en
  caso de fallos de comandos Git.