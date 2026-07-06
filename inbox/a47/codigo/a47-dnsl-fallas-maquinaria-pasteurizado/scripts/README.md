# Scripts Directory

Este directorio contiene los scripts de punto de entrada (entry points) diseniados para ejecutar las distintas fases del ciclo de vida del modelo de Machine Learning (procesamiento de datos, obtencion de estadisticas, entrenamiento y prediccion).

Estos scripts son simplemente envoltorios que importan las funciones principales desde el modulo central `src/main.py` y las ejecutan, permitiendo su uso directamente desde la linea de comandos, parseando posibles argumentos.

## Archivos Principales

- **`data_processing.py`**:
  Ejecuta la limpieza y preparacion de los datos en bruto. Llama a la funcion `run_data_processing` de la carpeta `src`. No requiere argumentos por defecto y procesa los datos configurados en el sistema.

- **`get_stats.py`**:
  Extrae la informacion estadistica y descriptiva de las variables utilizadas en el proyecto. Llama a la funcion `run_get_stats` de `src`.

- **`train.py`**:
  Inicia el proceso de entrenamiento del modelo. Llama a `run_train`.
  - **Argumentos**: Soporta el flag `--metrics`. Si se incluye en la ejecucion (ej: `python scripts/train.py --metrics`), el pipeline de entrenamiento también calculara y guardara las métricas de evaluacion sobre el conjunto de test (como la matriz de confusion).

- **`predict.py`**:
  Ejecuta inferencias o predicciones utilizando el modelo pre-entrenado. Llama a `run_predict`.
  - **Argumentos**:
    - `--input`: Ruta al archivo CSV con los datos de entrada (por defecto `data/raw/hydraulic_raw.csv`).
    - `--output`: Ruta al archivo CSV donde se guardaran los resultados predictivos (por defecto `data/predictions/prediction_output.csv`).
    - `--cycle`: (Opcional) Permite filtrar y realizar predicciones de un `Cycle_ID` en especifico. Ademas, detecta automaticamente la presencia de las etiquetas originales en el ciclo indicado, calculando y mostrando en pantalla una comparación amigable entre la PREDICCIÓN, la REALIDAD (Ground Truth) y si hubo ACIERTO para cada componente.

- **`fine_tune.py`**:
  Realiza el fine-tuning (calibración) del modelo preentrenado para adaptarlo a datos reales de planta. Congela el backbone CNN y re-entrena únicamente las cabezas de clasificación con los nuevos ciclos del fluido de producción real (p.ej. leche). Llama a `run_fine_tuning` del orquestador central.
  - **Argumentos**:
    - `--train_input` (**requerido**): Ruta al CSV con ciclos reales de planta para entrenamiento del fine-tuning. Debe incluir las columnas de sensores y las etiquetas (`Target_Fouling`, `Target_Valvula`, `Target_Bomba`, `Target_Acumulador`).
    - `--val_input` (**requerido**): Ruta al CSV con ciclos reales de planta para validación/early stopping.
    - `--fluid_density`: Densidad del fluido real en kg/L (por defecto `1.03` para leche entera).
    - `--fluid_cp`: Calor específico del fluido real en kJ/(kg·K) (por defecto `3.93` para leche entera).
    - `--epochs`: Número máximo de épocas de calibración (por defecto `50`).
    - `--patience`: Paciencia del Early Stopping (por defecto `7`).

  Ejemplo de uso:
  ```bash
  python scripts/fine_tune.py \
      --train_input data/planta/ciclos_train.csv \
      --val_input   data/planta/ciclos_val.csv
  ```
  El modelo calibrado se guarda en `models/artifacts/neurosymbolic_cnn_finetuned.pth` sin sobreescribir el modelo base.

## Nota

Estos scripts son simplemente puntos de entrada (entry points) que delegan toda la logica al modulo central `src/main.py`. No contienen logica de negocio propia.
