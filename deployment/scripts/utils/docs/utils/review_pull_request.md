# Script: review_pull_request.py

## Resumen y Objetivo

Este script automatiza la validación de las reglas de las Pull Requests (PRs) según las convenciones del proyecto. Su
objetivo es asegurar la calidad y consistencia del código y la documentación antes de que los cambios sean fusionados,
detectando incumplimientos en la nomenclatura de ramas, formato de commits, y actualizaciones de archivos clave como
`README.md` y `.VERSION`.

## Arquitectura y Lógica Principal

1. **Obtención de Contexto:** Recupera información relevante de la PR (ramas, archivos modificados, mensajes de commit)
   utilizando la función `get_pr_context` del módulo `git` y enriquece esta información buscando versiones en
   `README.md` y `.VERSION`.
2. **Definición de Reglas:** Se define una lista de diccionarios (`RULES`), cada uno representando una regla de
   validación con su nombre, categoría, criticidad y una función asociada para ejecutar la comprobación.
3. **Ejecución de Reglas:** Itera sobre cada regla definida, llamando a su función correspondiente con el contexto de la
   PR.
4. **Registro de Resultados:** Almacena el resultado de cada regla (PASS/FAIL) junto con un mensaje explicativo.
5. **Generación de Informe:** Imprime un informe detallado en la consola, agrupado por categorías, indicando el estado
   de cada regla y los mensajes de error si los hay.
6. **Determinación del Estado Final:** Calcula el número total de fallos y fallos críticos. El script sale con un código
   de error si hay fallos críticos, si el número total de fallos excede un umbral predefinido, o sale con éxito si todas
   las reglas pasan.

## Configuración Requerida

El script depende de la información proporcionada por el entorno de ejecución de la PR (probablemente a través de
variables de entorno o argumentos pasados a `get_pr_context`). Las siguientes configuraciones son internas al script:

* `REPO_ROOT`: Directorio raíz del repositorio (obtenido con `os.getcwd()`).
* `MAIN_BRANCHES`: Lista de ramas principales (`['master', 'develop', 'staging']`).
* `WORK_BRANCHES`: Lista de tipos de ramas de trabajo (`['feature', 'fix', 'bugfix', 'hotfix', 'release']`).
* `ALLOWED_BRANCH_TYPES`: Combinación de ramas principales y de trabajo.
* `CONVENTIONAL_COMMIT_TYPES`: Conjunto de tipos de commit permitidos según el estándar Conventional Commits.
* `CRITICAL_SEVERITY`, `MEDIUM_SEVERITY`, `LOW_SEVERITY`: Niveles de criticidad para las reglas.
* `MAX_FAILURES_ALLOWED`: Umbral máximo de fallos permitidos antes de que el script falle.

## Entradas y Salidas

* **Entradas:**
    * Contexto de la Pull Request (obtenido de `git.get_pr_context()`), que incluye:
        * Nombre de la rama fuente (`branch_name`).
        * Rama de destino (`bitbucket_pr_destination_branch`).
        * Archivos modificados (`modified_files`).
        * Mensajes de commit (`commit_messages`).
        * Nombre del repositorio (`bitbucket_repo_name`).
        * Estado de privacidad del repositorio (`bitbucket_repo_is_private`).
    * Contenido de `README.md` (para buscar la sección de Changelog).
    * Contenido de `.VERSION` (para obtener la versión actual).

* **Salidas:**
    * Un informe detallado en la consola (`stdout`) que resume el estado de cada regla de validación.
    * Un código de salida del script: `0` si pasa, `1` si falla (debido a fallos críticos, exceso de fallos, o errores
      irrecuperables).

## Ejemplos de Uso

Este script está diseñado para ejecutarse automáticamente en un pipeline de CI/CD. No se espera una ejecución manual
directa, pero si fuera necesario, se ejecutaría desde la raíz del repositorio:

```bash
python scripts/review_pull_request.py
```

## Mantenimiento y Puntos Clave

* **Dependencia Externa:** El script depende de un módulo `git` (probablemente una librería externa como `GitPython` o
  una implementación interna) para obtener el contexto de la PR. La disponibilidad y correcto funcionamiento de este
  módulo son cruciales.
* **Manejo de Errores:** El script incluye manejo básico de excepciones para la lectura de archivos y la ejecución de
  reglas, pero errores inesperados en `get_pr_context` podrían detener la ejecución.
* **Flexibilidad de Ramas:** Las listas `MAIN_BRANCHES` y `WORK_BRANCHES` deben mantenerse actualizadas si la estrategia
  de branching del proyecto cambia.
* **Convenciones de Commit:** La regla `check_conventional_commits` se basa en un patrón de expresiones regulares.
  Cualquier cambio en el estándar de Conventional Commits podría requerir la actualización de este patrón.
* **Actualización de Reglas:** Se pueden añadir, modificar o eliminar reglas fácilmente editando la lista `RULES`.
* **Informes Detallados:** El script proporciona información útil para depurar por qué una PR podría ser rechazada,
  facilitando la corrección por parte de los desarrolladores.