# Procesamiento de Datos (`src/data_processing`)

Este submódulo se encarga de la ingestión y transformación de los datos crudos a un formato óptimo para el entrenamiento y la predicción del modelo.

## Archivos Principales

- **`load_data.py`**:
  Responsable de cargar los conjuntos de datos desde su ubicación original (por ejemplo, archivos CSV o parquet en la carpeta `/data`). Maneja la lectura eficiente, garantizando que los datos ingresen correctamente al entorno de Python para que puedan ser pasados al siguiente paso del proceso.

- **`preprocess.py`**:
  Contiene la lógica pesada encargada de limpiar y tratar los datos de entrada. Esto incluye, entre posibles tareas:
  - Limpieza de datos nulos (valores faltantes).
  - Manejo de anomalías (outliers).
  - Transformación y escalado numérico aplicable a series temporales (Standardization, MinMaxScaler, etc.).
  - Codificación y estructuración para modelos temporales.
