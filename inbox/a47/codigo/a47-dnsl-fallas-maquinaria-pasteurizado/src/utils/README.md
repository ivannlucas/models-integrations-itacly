# Utilidades (`src/utils`)

El directorio de utilidades contiene fragmentos de código auxiliares que proporcionan soporte a través de todo el entorno del proyecto. Típicamente, no contienen lógica de negocio exclusiva del problema de pasteurizado, sino funcionalidades de soporte transversal.

## Archivos Principales

- **`logging.py`**:
  Proporciona la configuración maestra y las funciones necesarias para registrar (loggear) todas las acciones, métricas y posibles errores (Warnings/Exceptions) de manera centralizada. Esto es esencial a la hora de rastrear la ejecución de `scripts/` o analizar cómo se comporta el pipeline en producción o durante largas jornadas de entrenamiento.
