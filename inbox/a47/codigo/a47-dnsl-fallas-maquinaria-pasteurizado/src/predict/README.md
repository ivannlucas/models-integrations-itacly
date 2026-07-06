# Prediccion (`src/predict`)

Este submodulo es el encargado de cargar un modelo previamente entrenado para generar nuevas predicciones. También incluye pasos finales para formatear la salida del modelo en algo asimilable por el usuario final o un sistema externo.

## Archivos Principales

- **`predictor.py`**:
  Contiene la logica central para realizar las inferencias. Carga el modelo (pesos, arquitectura, hiperparametros guardados) y emplea observaciones nuevas (que ya fueron pre-procesadas) de la maquinaria de pasteurizado para estimar las anomalías o el estado resultante de la maquina en cada ciclo de trabajo.

- **`postprocess.py`**:
  Una vez `predictor.py` devuelve resultados numericos o probabilisticos en crudo, este archivo se encarga de transformarlos y estructurarlos. Puede incluir reglas de negocio o de presentacion (por ejemplo, determinar un nivel de severidad de fallo como 'SANO' o 'CRITICO', estructurar el DataFrame final, etc.) antes de guardar los resultados en la salida. Destaca tambien por su capacidad de emparejar las etiquetas reales (Ground Truth) con las predicciones, imprimiendo un registro visual claro (OK/ERR) en la terminal de si el modelo ha acertado en caso de realizar la inferencia con alguno de los ciclos del dataset.
