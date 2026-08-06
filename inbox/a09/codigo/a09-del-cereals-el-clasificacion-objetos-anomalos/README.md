# Modelo de Clasificacion de Infestacion de Cereales con Series Temporales

## Resumen

Este proyecto implementa un pipeline secuencial para clasificar ventanas temporales de una serie sintetica en tres clases:
- `sano`
- `insectos`
- `moho_critico`

La arquitectura actual se apoya en modelos secuenciales reales:
- `LSTM`
- `GRU`

Ambos modelos se entrenan sobre ventanas deslizantes construidas a partir del dataset sintetico procesado. El pipeline compara sus resultados y guarda automaticamente el mejor modelo segun la metrica de seleccion configurada.

## Objetivo

El objetivo del sistema es aprender dependencias temporales reales de la serie para detectar el estado de la infestacion a partir de la evolucion reciente de las señales.

La salida principal es la clase predicha por ventana temporal. Ademas, el pipeline genera probabilidades, informes de clasificacion, matrices de confusion y un resumen comparativo entre LSTM y GRU.

## Estructura del proyecto

```text
Modelo9_INF_CER/
|- config/
|  |- config.yaml
|
|- data/
|  |- raw/
|  |- processed/
|  |- splits/
|  |- predictions/
|
|- models/
|  |- artifacts/
|  |- metrics/
|     |- figures/
|
|- notebooks/
|  |- EDA/
|  |- evaluacion/
|  |- modelo/
|
|- scripts/
|  |- data_processing.py
|  |- train.py
|  |- evaluate.py
|  |- predict.py
|  |- get_stats.py
|
|- src/
|  |- main.py
|  |- data_processing/
|  |- evaluation/
|  |- predict/
|  |- training/
|  |- utils/
|
|- requirements.txt
|- README.md
```

## Entorno

Se recomienda usar un entorno virtual.

El proyecto ha sido validado con Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Si quieres fijar la version en Windows:

```powershell
py -3.12 -m venv .venv
```

## Ejecucion en Windows

### Punto de partida

Los comandos de este README estan pensados para ejecutarse dentro de `Modelo9_INF_CER/`. Si estas en la raiz del repo, entra primero en esa carpeta:

```powershell
cd .\Modelo9_INF_CER\
```

### Flujo A: entrenamiento desde cero

Usa este camino cuando no exista el raw o quieras regenerar todo el caso sintetico desde la configuracion del proyecto.

1. Generar el raw sintetico y procesarlo
```powershell
python .\scripts\data_processing.py --synthetic
```

2. Entrenar modelos secuenciales
```powershell
python .\scripts\train.py
```

3. Evaluar el modelo guardado
```powershell
python .\scripts\evaluate.py
```

4. Inferencia
```powershell
python .\scripts\predict.py --input-csv data\processed\dataset_infestacion_cereales_sintetico_processed.csv
```

5. Estadisticas de columnas
```powershell
python .\scripts\get_stats.py
```

Notas:
- `--synthetic` regenera `data/raw/dataset_infestacion_cereales_sintetico.csv` y tambien `dataset_infestacion_cereales_sintetico_summary.csv` y `dataset_infestacion_cereales_sintetico_metadata.json`.
- `train.py` entrena `LSTM` y `GRU`, compara ambos y deja persistido el mejor modelo.
- `evaluate.py` no reentrena; solo recalcula las evidencias de hold-out.

### Flujo B: raw ya disponible, con entrenamiento

Usa este camino cuando ya tengas el fichero raw en `data/raw/dataset_infestacion_cereales_sintetico.csv` y solo quieras preparar el dataset y entrenar.

1. Procesar el raw existente
```powershell
python .\scripts\data_processing.py
```

2. Entrenar modelos secuenciales
```powershell
python .\scripts\train.py
```

3. Evaluar el modelo guardado
```powershell
python .\scripts\evaluate.py
```

4. Inferencia
```powershell
python .\scripts\predict.py --input-csv data\processed\dataset_infestacion_cereales_sintetico_processed.csv
```

5. Estadisticas de columnas
```powershell
python .\scripts\get_stats.py
```

Notas:
- Sin `--synthetic`, el script lee el raw existente y genera `dataset_infestacion_cereales_sintetico_processed.csv`.
- En este modo no se regeneran `summary.csv` ni `metadata.json`.

### Flujo C: modelo ya entrenado, sin volver a entrenar

Usa este camino cuando ya existan:
- `models/artifacts/lstm_best.pt`
- `models/artifacts/gru_best.pt`
- `models/artifacts/final_winner.pt`
- `models/artifacts/model_bundle_metadata.json`
- `models/artifacts/sequence_scaler.pkl`

1. Procesar el raw existente o regenerar el sintetico
```powershell
python .\scripts\data_processing.py
```

Si quieres regenerar todo desde cero:
```powershell
python .\scripts\data_processing.py --synthetic
```

2. Evaluar el modelo guardado sobre el dataset procesado actual
```powershell
python .\scripts\evaluate.py
```

3. Inferencia
```powershell
python .\scripts\predict.py --input-csv data\processed\dataset_infestacion_cereales_sintetico_processed.csv
```

4. Estadisticas de columnas
```powershell
python .\scripts\get_stats.py
```

Notas:
- `evaluate.py` carga el bundle ya guardado y genera metricas y figuras de hold-out, incluida una traza temporal de una serie de ejemplo para revisar el comportamiento ventana a ventana.
- Si solo quieres inferencia y no necesitas regenerar el paquete de evaluacion, puedes saltarte `evaluate.py`.

## Datos de entrada

### Raw

- `data/raw/dataset_infestacion_cereales_sintetico.csv`
  - Dataset sintetico base.
  - Es el punto de partida para el resto del pipeline.

### Procesado

- `data/processed/dataset_infestacion_cereales_sintetico_processed.csv`
  - Dataset que consume el pipeline.
  - Conserva el historico necesario para construir las ventanas secuenciales.

### Auxiliares de trazabilidad

- `data/processed/dataset_infestacion_cereales_sintetico_summary.csv`
- `data/processed/dataset_infestacion_cereales_sintetico_metadata.json`

El `config_hash` y la version del dataset se mantienen estables para una misma configuracion.

## Que hace cada paso

### `scripts/data_processing.py`

- valida columnas obligatorias, timestamps y tipos de entrada
- genera el dataset sintetico si se usa `--synthetic`
- prepara el dataset procesado para el pipeline secuencial

### `scripts/train.py`

- construye features secuenciales
- crea ventanas deslizantes
- separa train, validation y test por `sample_id`
- entrena `LSTM` y `GRU`
- compara ambos modelos
- guarda el mejor modelo final

### `scripts/evaluate.py`

- carga el bundle ya entrenado
- evalua `LSTM` y `GRU` sobre hold-out
- genera reportes, metricas y figuras
- produce una comparativa entre modelos

### `scripts/predict.py`

- carga el modelo ganador
- reconstruye las ventanas de entrada
- genera predicciones y probabilidades

### `scripts/get_stats.py`

- genera estadisticas descriptivas de columnas

## Artefactos principales

### Modelos

- `models/artifacts/lstm_best.pt` - mejor checkpoint obtenido para `LSTM`.
- `models/artifacts/gru_best.pt` - mejor checkpoint obtenido para `GRU`.
- `models/artifacts/final_winner.pt` - modelo final seleccionado para produccion o inferencia.
- `models/artifacts/model_bundle_metadata.json` - bundle con la configuracion, los splits y la ruta del ganador.
- `models/artifacts/sequence_scaler.pkl` - escalador ajustado sobre las ventanas de entrenamiento.

### Metricas

- `models/metrics/lstm_class_report.csv` - precision, recall y F1 por clase para `LSTM`.
- `models/metrics/gru_class_report.csv` - precision, recall y F1 por clase para `GRU`.
- `models/metrics/lstm_search_summary.csv` - resumen de la busqueda de hiperparametros de `LSTM`.
- `models/metrics/gru_search_summary.csv` - resumen de la busqueda de hiperparametros de `GRU`.
- `models/metrics/lstm_training_history.csv` - evolucion de entrenamiento de `LSTM` por epoca.
- `models/metrics/gru_training_history.csv` - evolucion de entrenamiento de `GRU` por epoca.
- `models/metrics/model_comparison.csv` - comparativa final entre ambos modelos.
- `models/metrics/metrics_summary.json` - resumen estructurado con la metrica principal y el ganador.
- `models/metrics/reporte_modelo.md` - resumen legible de metricas y lectura tecnica.

### Graficas

- `models/metrics/figures/lstm_confusion.png` - matriz de confusion de `LSTM`.
- `models/metrics/figures/gru_confusion.png` - matriz de confusion de `GRU`.
- `models/metrics/figures/winner_confusion.png` - matriz de confusion del modelo ganador.
- `models/metrics/figures/winner_temporal_trace.png` - evolucion temporal de una serie de hold-out para revisar clase real, prediccion y probabilidades por ventana.
- `models/metrics/figures/winner_initial_context.png` - foco sobre las primeras ventanas de una serie de hold-out para revisar el arranque de la clasificacion.
- `models/metrics/figures/winner_feature_importance.png` - visualizacion auxiliar de importancia de variables del modelo ganador.

### Predicciones

- `data/predictions/holdout_predictions_lstm.csv` - predicciones sobre hold-out de `LSTM`.
- `data/predictions/holdout_predictions_gru.csv` - predicciones sobre hold-out de `GRU`.
- `data/predictions/holdout_predictions_winner.csv` - predicciones sobre hold-out del ganador.
- `data/predictions/predictions_sequence.csv` - salida de inferencia final sobre ventanas temporales.

### Splits

- `data/splits/model_train.csv` - dataset de entrenamiento exportado con `sample_id`, `timestamp`, `target` y features secuenciales.
- `data/splits/model_validation.csv` - dataset de validacion exportado con el mismo formato.
- `data/splits/model_test.csv` - dataset de prueba final exportado con el mismo formato.

## Como interpretar el modelo

El modelo trabaja con secuencias, no con filas independientes. Cada muestra del modelo corresponde a una ventana temporal construida a partir del historico reciente de cada `sample_id`.

Forma conceptual de los datos:
- `X_seq`: `[n_ventanas, window_size, n_features]`
- `y_seq`: clase asociada a cada ventana
- `group_seq`: identificador de la serie original para evitar fuga de datos

La comparacion entre `LSTM` y `GRU` se hace sobre la metrica de seleccion definida en configuracion. En la version actual la eleccion final se guarda en `models/artifacts/final_winner.pt`.

## Metricas

Las metricas principales que se revisan son:
- `accuracy` - porcentaje de aciertos global.
- `balanced accuracy` - promedio equilibrado entre clases, util si hay desbalance.
- `precision macro` - precision media tratando todas las clases por igual.
- `recall macro` - cobertura media por clase.
- `f1 macro` - equilibrio entre precision y recall por clase.
- `log loss` - penaliza predicciones inseguras o mal calibradas.

Estas metricas describen comportamiento del hold-out secuencial; no deben leerse como una garantia de robustez fuera del escenario sintetico actual.

Ademas, `models/metrics/reporte_modelo.md` resume la lectura tecnica de las metricas y la comparacion entre `LSTM` y `GRU`.

## Notebooks

- `notebooks/EDA/notebook_preparacion_dataset_sintetico.ipynb`
  - preparacion y analisis exploratorio del dataset sintetico y de sus transiciones temporales
- `notebooks/evaluacion/validacion_modelo.ipynb`
  - lectura y validacion del flujo de evaluacion, incluyendo la traza temporal del hold-out
- `notebooks/modelo/notebook_modelo_completo.ipynb`
  - vision global del pipeline del modelo

## Notas tecnicas

- El pipeline evita fuga de informacion separando por `sample_id`.
- El dataset sintetico base sigue siendo la fuente de entrada.
- El entrenamiento puede usar GPU si esta disponible y, si no, cae automaticamente a CPU.
