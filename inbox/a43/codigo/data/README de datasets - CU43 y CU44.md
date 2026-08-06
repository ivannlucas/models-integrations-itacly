# Estructura del Dataset — CU43 y CU44

## Contenido de `data/`

La carpeta `data/` contiene todos los datos necesarios tanto para **inferencia** como para **entrenamiento**:

```
data/
├── input/         # CSV del cliente → Entradas para INFERENCIA
├── raw/           # Dataset fuente (sintético o externo) + datasets XAI auxiliares → ENTRENAMIENTO y VALIDACIÓN XAI
├── splits/        # Splits tabulares: train.csv, val.csv, test.csv → ENTRENAMIENTO
├── processed/     # Arrays procesados: X_*.npy, y_*.npy, stats_*.csv → ENTRENAMIENTO
└── predictions/   # Salidas de inferencia y reportes XAI → Salidas de INFERENCIA
    └── xai/       # Archivos específicos de explicabilidad (CSV + JSON de reportes XAI)
```

---

## Para Inferencia (CU43 y CU44)

El modelo DNF se entrega ya entrenado y listo para inferencia por parte del cliente.

### Pasos

1. **Preparar un CSV** con las columnas esperadas (13 sensores según `data_generation.sensors` de `config/config.yaml`; `timestamp` obligatorio, se usa para ordenar cronológicamente; `cycle_id` opcional).

2. **Guardar el archivo** en `data/input/` (por ejemplo, `data/input/datos_prueba.csv`).

   > **Nota:** En `data/input/` se incluye un archivo `datos_prueba.csv` como ejemplo para probar la inferencia.
   >
   > **Para generar `datos_prueba.csv`**, se ejecutó `generate_dataset.py` con una configuración modificada respecto a la de entrenamiento. Solo se usó el split de *test*, renombrándolo y copiándolo a `data/input/`. Cambios en `config.yaml`:
   >
   > | Parámetro | Entrenamiento | `datos_prueba.csv` |
   > |---|---|---|
   > | `project.seed` | 42 | 43 |
   > | `data_generation.n_cycles_test` | 600 | 52 |
   > | `data_generation.split_ratios.test` | 80 / 20 | 82 / 18 |
   > | `data_generation.fault_start` | [10 %, 80 %] | [20 %, 75 %] |
   > | `data_generation.fault_duration` | [40 %, 75 %] | [45 %, 80 %] |
   > | `data_generation.severity` | [0.60, 0.90] | [0.55, 0.85] |

3. **Ejecutar la predicción** indicando la ruta del archivo con `--input`. Dos opciones:

   **Sin explicabilidad** (solo predicción de fallos en csv):
   ```bash
   python scripts/predict.py --input data/input/datos_prueba.csv
   ```
   → Guarda el CSV con las predicciones en `data/predictions/`.

   **Con explicabilidad** (predicción en csv + explicaciones y acciones correctivas en JSON):
   ```bash
   python scripts/predict.py --input data/input/datos_prueba.csv --xai
   ```
   → Guarda el CSV con las predicciones y el JSON con las explicaciones en `data/predictions/xai/`.

---

## Para Entrenamiento (CU43)

El entrenamiento se realiza a partir de los siguientes datos:

| Ruta | Descripción |
|---|---|
| `data/raw/oven_full_dataset.csv` | Versión compactada de los datos generados (sin split). |
| `data/splits/*` | Splits tabulares: `train.csv`, `val.csv`, `test.csv`. Creados a partir de los datasets en `data/raw/`. |
| `data/processed/*` | Arrays procesados: `X_*.npy`, `y_*.npy`, `stats_*.csv`. Creados a partir de los splits en `data/splits/`. Son los que verdaderamente se usan para el entrenamiento. |

> **Importante:** Estos datos suben a un contenedor externo, ya que son demasiado pesados para el repositorio.

### Ejecución

Teniendo los datos en sus carpetas correspondientes, ejecutar:

```bash
python scripts/train.py
```

Para obtener las métricas sobre el conjunto **TEST**, ejecutar posteriormente:

```bash
python scripts/get_stats.py
```

### Alternativa: Generar los datos de entrenamiento

En lugar de mover los datos de entrenamiento a `data/raw/`, `data/splits/` y `data/processed/`, es posible **generarlos de forma sintética** ejecutando los scripts a continuación:

```bash
# Generar datos en data/raw/ y data/splits/ (ambas opciones son equivalentes)
python scripts/generate_dataset.py
# o
python scripts/generate_dataset.py --xai

# Procesar datos (crear arrays en data/processed/)
python scripts/data_processing.py
```

La ejecución de ambos scripts y la consecuente generación y procesamiento de datos tarda **aproximadamente 1 minuto**.

---

## Validación de la Capa XAI y Acciones Correctivas (CU44)

Esta validación se lleva a cabo en el notebook situado en **`notebooks/CU44.ipynb`**.

### Datos requeridos

| Ruta | Descripción |
|---|---|
| `data/raw/interpretability_val.csv` | Permite verificar consistencia interpretativa en un escenario balanceado. |
| `data/raw/actions_system_eval.csv` | Permite probar la lógica de acciones correctivas en una distribución más realista. |
| `data/raw/xai_background.csv` | Sirve como base para el background de SHAP usado por la capa XAI. |

> **Importante:** Estos datos se suben a un contenedor externo, ya que son demasiado pesados para el repositorio.

### Ejecución

Teniendo los datos en sus carpetas correspondientes, abrir y ejecutar el notebook:

```
notebooks/CU44.ipynb
```

### Alternativa: Generar los datos XAI

En lugar de mover los datos de explicabilidad a `data/raw/`, es posible **generarlos de forma sintética** ejecutando:

```bash
python scripts/generate_dataset.py --xai
```

La ejecución de este script y la consecuente generación de datos tarda **aproximadamente 1 minuto**.

---

> **Advertencia:** Sin estos datasets específicos de explicabilidad, el entrenamiento del modelo base puede completarse (CU43), pero la validación completa de la capa XAI (`notebooks/CU44.ipynb`) y de las acciones correctivas de CU44 **no se puede realizar**.
