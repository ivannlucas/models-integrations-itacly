# Código Fuente (Src Directory)

Este directorio (`src`) representa el corazón de la aplicación, conteniendo toda la lógica modular, dividida en pequeños componentes de acuerdo con las distintas fases del pipeline de Machine Learning de la maquinaria de pasteurizado.

Esta separación modular favorece el mantenimiento, las buenas prácticas de la ingeniería de software y la fácil depuración del proyecto.

## Estructura Principal

- **`main.py`**:
  Es el módulo orquestador central que une todos los componentes (procesamiento, entrenamiento, predicción, calibración, utilidades). Define funciones principales (como `run_data_processing()`, `run_train()`, `run_predict()`, `run_fine_tuning()`) que encapsulan los pipelines completos y que posteriormente son llamadas por los ejecutables en la carpeta `scripts/`.

## Subdirectorios Core

Cada subcarpeta se encarga de una responsabilidad única y detallada (cada una posee su propio README interno):

- **`data_processing/`**: Contiene los módulos para la carga de los datos (`load_data.py`) y toda la lógica de transformación y preprocesamiento (`preprocess.py`).
- **`get_stats/`**: Maneja la extracción y generación de estadísticas e información estructural de las columnas del dataset (`column_info.py`).
- **`predict/`**: Contiene la lógica necesaria para inferir predicciones con el modelo pre-entrenado (`predictor.py`) y postprocesar dichos resultados para su salida (`postprocess.py`).
- **`training/`**: Define la arquitectura del modelo predictivo (ej. en `model.py`) y encapsula el lazo o bucle de entrenamiento, la validación y el cálculo de métricas (`trainer.py`).
- **`fine_tuning/`**: Contiene la lógica de calibración por Transfer Learning (`fine_tuner.py`). Congela el backbone CNN preentrenado y re-entrena únicamente las 4 cabezas de clasificación con datos reales del fluido de producción (leche), adaptando los umbrales de presión, densidad y viscosidad al nuevo entorno de planta.
- **`utils/`**: Contiene módulos utilitarios que pueden ser compartidos entre todos los demás submódulos, como por ejemplo la configuración del sistema de logs (`logging.py`).
