# Obtencion de Estadisticas (`src/get_stats`)

Este directorio contiene herramientas para el analisis exploratorio y descriptivo básico de los datos. Sirve de apoyo para entender la naturaleza de las variables y generar descriptores utiles antes y despues de entrenar.

## Archivos Principales

- **`column_info.py`**:
  Este script extrae informacion clave de cada variable (columna) en el sistema. Analiza el comportamiento, tipos de datos, estadisticas basicas y cualquier otro descriptor necesario para respaldar o validar de que tratan los datos en cualquier punto del pipeline (por ejemplo, variables categoricas, numericas, rangos esperados para el pasteurizado, etc.).
