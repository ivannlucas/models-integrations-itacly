import os
from pathlib import Path

base_path = Path(__file__).resolve().parent

folders = [
    "data/raw", "data/processed", "data/predictions", "data/splits",
    "models/artifacts", "models/metrics",
    "config",
    "src/training", "src/predict", "src/data_processing", "src/get_stats", "src/utils",
    "scripts"
]

init_files = [
    "src/__init__.py",
    "src/training/__init__.py",
    "src/predict/__init__.py",
    "src/data_processing/__init__.py",
    "src/get_stats/__init__.py",
    "src/utils/__init__.py"
]

readmes = {
    "data/README.md": "# Data\n\nContiene los datos del proyecto organizados por etapas (raw, processed, predictions, splits).",
    "models/README.md": "# Models\n\nAlmacena los artefactos finales (.keras, .pkl) en `artifacts/` y las métricas resultantes del entrenamiento en `metrics/`.",
    "config/README.md": "# Config\n\nArchivos de configuración, constantes, rutas y parámetros para controlar el modelo (`config.yaml`).",
    "src/README.md": "# Src\n\nCódigo fuente principal del proyecto de predicción de plagas de la vid. Módulo ejecutable e importable.",
    "src/training/README.md": "# Training\n\nMódulos y funciones específicos para el entrenamiento de los ensambles de Deep Learning.",
    "src/predict/README.md": "# Predict\n\nMódulos y funciones diseñados para realizar inferencias utilizando los modelos entrenados.",
    "src/data_processing/README.md": "# Data Processing\n\nRutinas de limpieza, preprocesamiento, escalado y empacado temporal de datos.",
    "src/get_stats/README.md": "# Get Stats\n\nFunciones auxiliares para obtener estadísticas, información de columnas y análisis exploratorio básico del dataset.",
    "src/utils/README.md": "# Utils\n\nUtilidades compartidas en todo el proyecto, como el sistema de logs general.",
    "scripts/README.md": "# Scripts\n\nPunto de entrada ejecutable `.py` para orquestar los flujos de train, predict, data_processing y get_stats conectando a `src/main.py`."
}

for f in folders:
    (base_path / f).mkdir(parents=True, exist_ok=True)

for init in init_files:
    (base_path / init).touch(exist_ok=True)

for path, content in readmes.items():
    with open(base_path / path, "w", encoding='utf-8') as f:
        f.write(content)

print("Estructura base creada con éxito.")
