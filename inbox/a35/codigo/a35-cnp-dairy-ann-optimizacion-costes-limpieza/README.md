# Datagia: Optimizacion de Consumo de Agua en Pasteurizacion

Este proyecto implementa un gemelo digital basado en una red neuronal ANN y un optimizador prescriptivo basado en algoritmo genetico. El objetivo es estimar y optimizar el consumo de agua en un proceso de pasteurizacion de leche manteniendo la restriccion de seguridad alimentaria definida por PU >= 13.

El repositorio funciona con Python 3.13.

## Instalacion

1. Clona el repositorio.
2. Crea un entorno virtual:
   `python -m venv venv`
3. Activa el entorno:
   `venv\Scripts\activate`
4. Instala dependencias:
   `pip install -r requirements.txt`

> **Nota:** Todos los comandos deben ejecutarse desde la raiz del proyecto (`a35-cnp-dairy-ann-optimizacion-costes-limpieza/`). Los scripts usan rutas relativas y fallaran si se invocan desde otro directorio.

## Ejecucion E2E

1. Generar datos sinteticos:
   `python scripts/data_generating.py`
2. Procesar datos y generar splits definitivos:
   `python scripts/data_processing.py`
   > **Importante:** Este paso es obligatorio. Genera los splits de entrenamiento con las variables de proceso normalizadas y sobreescribe los splits del paso anterior. El modelo se entrena sobre los splits de este paso.
3. Entrenar el modelo:
   `python scripts/train.py`
4. Evaluar el modelo en test:
   `python scripts/predict.py`
5. Optimizar el contexto unico definido en `config/config.yaml` bajo `contexto_actual`:
   `python scripts/optimize.py`
6. Reproducir la validacion masiva oficial sobre los escenarios definidos en `optimization_scenarios`:
   `python scripts/optimize.py --mode massive_mode`

## Inferencia sobre datos externos

Para predecir consumo sobre nuevos datos de entrada:

`python scripts/predict.py --input_path data/processed/nuevos_datos.csv`

Opcionalmente se puede indicar una salida concreta:

`python scripts/predict.py --input_path data/processed/nuevos_datos.csv --output_path data/predictions/external_predictions.csv`

El CSV de inferencia debe contener estas variables:

- `temp_entrada_leche`
- `temp_ambiente`
- `temp_setpoint_leche`
- `flujo_leche_lh`
- `horas_desde_limpieza`
- `presion_diferencial_bar`

Si no se incluyen `temp_proceso_leche` y `temp_agua_servicio`, se calculan automaticamente como:

- `temp_proceso_leche = temp_setpoint_leche`
- `temp_agua_servicio = temp_proceso_leche + 10.0`

## Optimizacion con CSV externo

Para validar escenarios externos en modo CSV:

`python scripts/optimize.py --mode csv_mode --input_path data/processed/archivo.csv`

El CSV debe contener:

- `temp_entrada_leche`
- `temp_ambiente`
- `horas_desde_limpieza`
- `presion_diferencial_bar`

## Fine-tuning con datos del cliente

Para calibrar el modelo preentrenado con datos reales etiquetados:

`python scripts/fine_tune.py --input_path data/processed/datos_cliente.csv`

El CSV debe contener las variables de entrada del modelo y la columna objetivo `consumo_agua_l`.

Por defecto se generan:

- `models/artifacts/model_ann_finetuned.pt`
- `models/metrics/fine_tune_metrics.txt`

Tambien pueden definirse rutas especificas:

`python scripts/fine_tune.py --input_path data/processed/datos_cliente.csv --output_model_path models/artifacts/model_cliente.pt --metrics_path models/metrics/fine_tune_cliente.txt`

## Salidas principales

- Metricas: `models/metrics/`
  - `test_metrics.txt` es la metrica de generalizacion honesta (datos no vistos durante entrenamiento).
  - `train_metrics.txt` y `val_metrics.txt` se incluyen para analisis de overfitting, no como evidencia de generalizacion.
- Artefactos del modelo: `models/artifacts/`
- Predicciones y resultados de optimizacion: `data/predictions/`

## Limitaciones y contexto de los resultados

El modelo y el optimizador han sido entrenados y evaluados sobre **datos sinteticos** generados mediante un balance termico y modelo de fouling simplificado. Los resultados de optimizacion (por ejemplo, ahorro hidrico proyectado) son estimaciones del modelo como proxy, no mediciones reales en planta. Requieren validacion con datos operacionales reales antes de cualquier despliegue productivo.
