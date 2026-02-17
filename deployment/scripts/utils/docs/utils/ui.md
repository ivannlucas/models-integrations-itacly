# Título

Script: utils/ui.py

# Resumen y Objetivo

Este módulo proporciona utilidades para la interfaz de usuario de consola, centrándose en la visualización de mensajes
con colores y prefijos para mejorar la legibilidad del registro.

# Arquitectura y Lógica Principal

El script define:

1. Una clase `Colors` que contiene códigos de escape ANSI para colorear el texto en la terminal.
2. Un diccionario `LOG_LEVEL_CONFIG` que mapea niveles de registro a prefijos y colores predefinidos.
3. Un diccionario `STATUS_COLORS` para mapear estados de mensajes a colores específicos.
4. Una función `log_message` que toma un mensaje, un nivel de registro y un flujo de salida, y lo imprime en la consola
   con el formato y color apropiados según el nivel.

# Configuración Requerida

No se requieren variables de entorno o argumentos de línea de comandos específicos para este script, ya que es una
biblioteca de utilidades.

# Entradas y Salidas

* **Entradas:**
    * `message` (str): El texto del mensaje a mostrar.
    * `level` (str): El nivel de registro (ej. "INFO", "ERROR").
    * `stream`: El flujo de salida (por defecto `sys.stderr`).
* **Salidas:**
    * Mensajes formateados y coloreados impresos en la consola.

# Ejemplos de Uso

```python
from utils.ui import log_message, Colors

log_message("Operación completada exitosamente.", level="INFO")
log_message("Algo salió mal.", level="ERROR")
log_message("Advertencia: El archivo no se encontró.", level="WARNING")

# Ejemplo usando colores directamente
print(f"{Colors.BOLD}Este es un mensaje en negrita.{Colors.ENDC}")
```

# Mantenimiento y Puntos Clave

* La funcionalidad de coloreado depende de que la terminal del usuario soporte códigos de escape ANSI.
* La lista de niveles de registro y sus correspondientes colores/prefijos se definen en `LOG_LEVEL_CONFIG` y pueden ser
  extendidos.
* La función `log_message` utiliza `sys.stderr` por defecto, lo cual es una práctica común para mensajes de registro,
  pero puede ser redirigido a `sys.stdout` u otros flujos si es necesario.