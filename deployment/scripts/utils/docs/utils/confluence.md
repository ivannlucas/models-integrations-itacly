# Script: confluence.py

## Resumen y Objetivo

Este script proporciona utilidades para interactuar con la API de Confluence. Su objetivo principal es facilitar la
creación y actualización de páginas de documentación en Confluence de forma programática, permitiendo la gestión
automatizada del contenido de conocimiento.

## Arquitectura y Lógica Principal

1. **Configuración:** Carga las credenciales de Confluence (URL, usuario, token) desde variables de entorno.
2. **Cliente Confluence:** Inicializa un cliente `atlassian.Confluence` si las credenciales son válidas.
3. **Gestión de Jerarquía:** La función `ensure_page_hierarchy_exists` verifica y crea páginas anidadas según una lista
   de títulos proporcionada, devolviendo el ID de la página final.
4. **Creación/Actualización de Páginas:** La función `create_or_update_confluence_page` busca una página existente por
   título y padre, y la crea o actualiza según sea necesario. Convierte el contenido Markdown a formato de
   almacenamiento de Confluence (XHTML) usando `markdown2`.
5. **Logging:** Utiliza una función `log_message` (asumida de `ui.py`) para reportar el estado de las operaciones y
   errores.

## Configuración Requerida

El script requiere las siguientes variables de entorno:

* `CONFLUENCE_URL`: La URL base de la instancia de Confluence.
* `CONFLUENCE_USERNAME`: El nombre de usuario para autenticarse en Confluence.
* `CONFLUENCE_API_TOKEN`: El token de API asociado a la cuenta de usuario.

## Entradas y Salidas

* **Entradas:**
    * Credenciales de Confluence (variables de entorno).
    * `space_key`: Clave del espacio de Confluence donde se crearán/actualizarán las páginas.
    * `title`: Título de la página a crear/actualizar.
    * `body`: Contenido de la página en formato Markdown.
    * `hierarchy`: Una lista de strings representando la jerarquía de páginas padre.
    * `parent_id`: Opcionalmente, el ID de la página padre para anidar la nueva página.
* **Salidas:**
    * Un cliente `Confluence` inicializado.
    * El ID de la página final en una jerarquía de páginas asegurada.
    * Páginas creadas o actualizadas en Confluence.
    * Mensajes de log indicando el éxito o fracaso de las operaciones.

## Ejemplos de Uso

```python
from confluence import get_confluence_client, ensure_page_hierarchy_exists, create_or_update_confluence_page

# Asumiendo que las variables de entorno están configuradas
client = get_confluence_client()

if client:
    space = "MYSPACE"
    page_title = "My New Documentation Page"
    page_content = "# Welcome!\n\nThis is the **main content** of the page.\n\n```python\nprint(\"Hello, Confluence!\")\n```"

    # Ejemplo 1: Crear una página simple
    create_or_update_confluence_page(client, space, page_title, page_content)

    # Ejemplo 2: Crear una página dentro de una jerarquía
    hierarchy = ["Project Docs", "API Reference", "v1.0"]
    parent_page_id = ensure_page_hierarchy_exists(client, space, hierarchy)

    if parent_page_id:
        sub_page_title = "Users Endpoint"
        sub_page_content = "## Users API\n\nDetails about the /users endpoint..."
        create_or_update_confluence_page(client, space, sub_page_title, sub_page_content, parent_id=parent_page_id)

```

## Mantenimiento y Puntos Clave

* **Dependencia de API Externa:** El script depende de la disponibilidad y el correcto funcionamiento de la API de
  Confluence. Los cambios en la API de Atlassian podrían requerir actualizaciones en este script.
* **Gestión de Credenciales:** Las credenciales se cargan desde variables de entorno. Es crucial asegurar que estas
  variables estén configuradas de forma segura en el entorno de ejecución.
* **Formato de Contenido:** El script convierte Markdown a formato de almacenamiento de Confluence. Asegúrate de que el
  Markdown generado sea compatible con `markdown2` y el formato esperado por Confluence.
* **Manejo de Errores:** El script incluye manejo básico de excepciones, pero se recomienda una estrategia de reintentos
  o un manejo de errores más robusto para operaciones críticas en entornos de producción.
* **ID de Página:** La función `ensure_page_hierarchy_exists` devuelve el ID de la última página creada/encontrada. Este
  ID es necesario para anidar páginas subsecuentes. La lógica de búsqueda y creación de páginas debe ser cuidadosa para
  evitar duplicados o la creación en la ubicación incorrecta.
* **`ui.log_message`:** El script depende de una función `log_message` externa. Asegúrate de que `ui.py` esté disponible
  y que la función `log_message` maneje adecuadamente los mensajes de log (incluyendo los niveles `ERROR`, `INFO`,
  `SUCCESS`).