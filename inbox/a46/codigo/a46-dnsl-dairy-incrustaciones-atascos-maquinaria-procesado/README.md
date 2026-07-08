# CU07 DNSL para la detección de incrustación y obstrucción

## Estructura

```text
requirements.txt
README.md

config/
  config.yaml

data/
  raw/
  processed/
  predictions/
  splits/

models/
  artifacts/
  metrics/

src/
  __init__.py
  main.py
  data_processing/
    __init__.py
    synthetic_generator.py
    targets.py
    pipeline.py
  training/
    __init__.py
    common.py
    data.py
    datasets.py
    evaluation.py
    features.py
    model.py
    persistence.py
    pipeline.py
  predict/
    __init__.py
    predictor.py
  get_stats/
    __init__.py
    eda.py
    pipeline.py
    column_info.py
  utils/
    __init__.py
    common.py
    config.py
    logging.py
    paths.py

scripts/
  data_processing.py
  train.py
  predict.py
  get_stats.py

notebooks/
  EDA/
  tuning_hyperparameters/
  opcional/
  evaluacion/
```

## Módulos

- `src/data_processing/`: generación sintética y anotación de targets futuros.
- `src/training/`: carga/alineación de datos, feature engineering, construcción de ventanas, arquitectura TCN, entrenamiento, evaluación, calibración de policy y persistencia.
- `src/predict/`: carga del modelo final, preparación de features, inferencia real sobre CSV procesado y guardado de predicciones/alertas.
- `src/get_stats/`: EDA completo, catálogo de columnas y resumen del modelo/artefactos.
- `src/utils/`: utilidades compartidas de logging, configuración, rutas y funciones comunes de series temporales.

## Requisitos e instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No se han fijado versiones exactas salvo mínimos razonables, porque los scripts fuente no incluían un entorno bloqueado. Se documenta esta limitación para auditoría.

## Comandos de uso

### 1) Generación / procesamiento de datos

Genera los CSV procesados y la metadata en la estructura del repositorio:

```bash
python scripts/data_processing.py
```

Ejemplo con override rápido:

```bash
python scripts/data_processing.py --assets 4 --cycles-per-asset 6 --dt 300
```

Salida por defecto:

- `data/processed/telemetry_processed.csv`
- `data/processed/maintenance_processed.csv`
- `data/processed/generation_metadata.json`

### 2) Entrenamiento

Entrena el modelo TCN, exporta splits, métricas, checkpoints y predicciones de validación/test:

```bash
python scripts/train.py
```

Ejemplo con overrides:

```bash
python scripts/train.py --epochs 4 --batch-size 32 --seq-len 60 --stride 5 --device cpu
```

### 3) Inferencia

Ejecuta inferencia real sobre un CSV procesado y usa los artefactos guardados en `models/artifacts/`:

```bash
python scripts/predict.py --input data/processed/telemetry_processed.csv
```

También admite elegir escenario (`auto`, `full`, `no_clock`):

```bash
python scripts/predict.py --input data/processed/telemetry_processed.csv --scenario auto
```

### 4) Estadísticas / EDA

Genera un paquete EDA útil y un catálogo de columnas:

```bash
python scripts/get_stats.py
```

Ejemplo con salida personalizada:

```bash
python scripts/get_stats.py --outdir models/metrics/stats
```

## Flujo end-to-end recomendado

```bash
python scripts/data_processing.py
python scripts/train.py
python scripts/predict.py --input data/processed/telemetry_processed.csv
python scripts/get_stats.py
```

## Artefactos generados

### Después de `data_processing`

- `data/processed/telemetry_processed.csv`
- `data/processed/maintenance_processed.csv`
- `data/processed/generation_metadata.json`

### Después de `train`

- `models/artifacts/selected_model.pt`
- `models/artifacts/model_manifest.json`
- `models/artifacts/feature_artifacts.json`
- `models/artifacts/training_config.json`
- `models/artifacts/scenarios/*/best_model.pt`
- `models/metrics/*.json`, `*.csv` (métricas, thresholds, feature reports, training history)
- `data/predictions/*` (predicciones de validación/test y alertas)
- `data/splits/*` (reparto de activos y export de splits)

### Después de `predict`

- `data/predictions/<input>_predictions.csv`
- `data/predictions/<input>_alerts.csv`
- `data/predictions/<input>_inference_manifest.json`

### Después de `get_stats`

- `models/metrics/stats/report.html`
- `models/metrics/stats/report.md`
- `models/metrics/stats/column_catalog.csv`
- `models/metrics/stats/summary.json`
- tablas/plots/QC asociados dentro de `models/metrics/stats/`


## Limitaciones conocidas

- La fuente original no contenía un ETL de datos reales; por ello `data_processing()` implementa de forma honesta la generación sintética y la persistencia en `data/processed/`.
- La inferencia espera un CSV procesado con columnas compatibles con el pipeline entrenado. Si faltan targets reales, el pipeline sigue siendo ejecutable porque usa defaults y artefactos guardados.
- Las métricas dependen de la distribución del dataset sintético generado y del régimen de entrenamiento configurado en `config/config.yaml`.

## Reproducibilidad

- Las semillas se controlan desde `config/config.yaml` y los wrappers CLI.
- Todas las rutas se resuelven relativas a la raíz del repo.
- El manifiesto del modelo guarda el escenario seleccionado, el checkpoint usado y la configuración de entrenamiento asociada.
