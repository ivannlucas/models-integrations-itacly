# Entrenamiento (`src/training`)

Este es el modulo central del aprendizaje automatico en el sistema. Se encarga de definir, compilar y buscar los parametros optimos del modelo basandose en los datos historicos del proceso de pasteurizado.

## Archivos Principales

- **`model.py`**:
  Este script define internamente la arquitectura del modelo de Deep Learning a emplear para el proyecto (arquitecturas de capas). Contiene las especificaciones, funciones de activacion, y estructuras base del modelo.

- **`trainer.py`**:
  Contiene el bucle (loop) principal de entrenamiento. Gestiona el ciclo de vida del aprendizaje, incluyendo:
  - Generacion de conjuntos de Entrenamiento/Validacion/Test.
  - Orquestacion iterativa para optimizar los pesos de `model.py` minimizando la funcion de costo (Loss function).
  - Calculo exhaustivo de las metricas sobre los conjuntos validacionales (como matrices de confusion, accuracy, recall, etc.).
  - Guardado del modelo ajustado junto a sus artefactos.
