# Estructura del Dataset — CU45

## Contenido de `data/`

La carpeta `data/` contiene todos los datos necesarios para **inferencia**, **entrenamiento**, **explicabilidad (XAI)** y **control de puntos críticos de control (PCC)**:

```
data/
├── input/         # CSV del cliente → Entradas para INFERENCIA (incluye datos_cliente.csv como ejemplo)
├── raw/           # Dataset fuente (sintético o externo) + datasets XAI auxiliares → ENTRENAMIENTO y VALIDACIÓN XAI
├── splits/        # Splits tabulares: train.csv, val.csv, test.csv → ENTRENAMIENTO
├── processed/     # Arrays procesados: X_*.npy, y_*.npy, stats_*.csv → ENTRENAMIENTO
└── predictions/   # Salidas de inferencia, reportes XAI y monitor PCC → Salidas de INFERENCIA
```

---

## Para Inferencia (CU45)

El modelo Deep Neuro-Fuzzy (DNF) se entrega ya entrenado y listo para inferencia por parte del cliente.

### Pasos

1. **Preparar un CSV** con las columnas esperadas (11 sensores según `data_generation.sensors` de `config/config.yaml`; `timestamp` obligatorio, sin valores nulos o vacíos en esta columna y con formato homogéneo; `cycle_id` opcional).

2. **Guardar el archivo** en `data/input/` (por ejemplo, `data/input/mis_datos.csv`).

   > **Nota:** En `data/input/` se incluye ya un archivo **`datos_cliente.csv`** como ejemplo, listo para probar la inferencia sin necesidad de cargar datos propios.

3. **Ejecutar la predicción** indicando la ruta del archivo con `--input`:

   ```bash
   python scripts/predict.py --input data/input/datos_cliente.csv
   ```

   El argumento `--input` indica la ruta explícita al dataset. Si no se proporciona, se usa la ruta por defecto en `paths.input_data` de `config/config.yaml`. Cuando esta ruta es un directorio, selecciona automáticamente el primer CSV en orden alfabético, por lo que en producción es preferible indicar el archivo explícitamente.

   La inferencia genera automáticamente las predicciones de anomalía junto con las explicaciones XAI, el estado de monitorización PCC y las recomendaciones para el operador.

### Salidas generadas en `data/predictions/`

| Archivo | Descripción |
|---|---|
| `predictions_<timestamp>.csv` | Predicciones por ventana con `window_index`, `cycle_id` si existe, `timestamp_init`, `timestamp_end`, `predicted_anomaly_class`, `predicted_anomaly_label`, `anomaly_probability` y `decision_threshold`. |
| `PCC_monitor_<timestamp>.json` | Array JSON con un objeto por ventana. Cada ventana procesada con éxito incluye `Índice de ventana`, `ID de ciclo` si existe, `Timestamp cubierto por ventana`, `Estado interpretativo`, `Evidencia`, `Probabilidad de anomalia`, `Umbral de detección de anomalias`, `Margen respecto al umbral` y `Recomendacion`. Si el procesamiento XAI de una ventana falla (por ejemplo, probabilidad no válida, error en SHAP, error en el módulo PCC, etc.), el objeto incluye solo `Índice de ventana` y `error` con el mensaje de la excepción. El pipeline continúa procesando las ventanas restantes; si todas fallan, se lanza un error global. |

Cada objeto dentro del array JSON tiene el siguiente formato:

```json
  {
    "Índice de ventana": 1,
    "ID de ciclo": 0,
    "Timestamp cubierto por ventana": "2026-01-01 00:00:00 / 2026-01-01 03:59:00",
    "Estado interpretativo": "Vigilancia",
    "Evidencia": "No se identifica un perfil catalogado de criticidad, pero hay indicios de anomalia.",
    "Probabilidad de anomalia": 0.43,
    "Umbral de detección de anomalias": 0.73,
    "Margen respecto al umbral": 0.3,
    "Recomendacion": "Se recomienda vigilancia reforzada y seguimiento."
  }
```

**Interpretación de campos:**
- `Índice de ventana`: índice de la ventana procesada en el JSON.
- `ID de ciclo`: ciclo de origen de la ventana, si el CSV incluye `id_column`.
- `Timestamp cubierto por ventana`: rango `timestamp_init / timestamp_end` cubierto por la ventana.
- En el CSV de predicciones, los campos equivalentes de ventana son `window_index`, `cycle_id`, `timestamp_init` y `timestamp_end`.
- `predicted_anomaly_class`: clase binaria predicha (`0` = no fallo, `1` = fallo).
- `predicted_anomaly_label`: etiqueta legible asociada a la clase (`No Fallo` o `Fallo`).
- `anomaly_probability`: probabilidad de anomalia estimada por el modelo.
- `decision_threshold`: umbral aplicado para convertir la probabilidad en clase.
- `Estado interpretativo`: estado operativo de la ventana (`Normal`, `Vigilancia`, `Criticidad detectada`).
- `Evidencia`: resumen textual de la evidencia detectada o coincidencia con un perfil crítico catalogado.
- `Probabilidad de anomalia`: probabilidad de anomalía estimada por el modelo (0–1).
- `Umbral de detección de anomalias`: umbral de decisión usado para clasificar la ventana.
- `Margen respecto al umbral`: distancia entre la probabilidad y el umbral (mayor margen = mayor confianza).
- `Recomendacion`: acción sugerida para el operador según el estado y la evidencia.

---

## Para Entrenamiento (CU45)

El entrenamiento del modelo requiere datos ubicados en las carpetas `data/raw/`, `data/splits/` y `data/processed/`. Existen dos opciones para disponer de estos datos: utilizar los datos sintéticos del repositorio (Opción A) o emplear un dataset externo del cliente (Opción B).

### Opción A: Datos sintéticos (por defecto)

El entrenamiento se realiza a partir de los siguientes datos:

| Ruta | Descripción |
|---|---|
| `data/raw/dryer_full_dataset.csv` | Versión compactada de los datos generados (sin split). Se analiza desde `notebooks/EDA_control_dryer.ipynb`. |
| `data/splits/*` | Splits tabulares: `train.csv`, `val.csv`, `test.csv`. Creados a partir de los datasets en `data/raw/`. |
| `data/processed/*` | Arrays procesados: `X_*.npy`, `y_*.npy`, `stats_*.csv`. Creados a partir de los splits en `data/splits/`. Son los que verdaderamente se usan para el entrenamiento. |

> **Importante:** Estos datos están subidos a un contenedor externo, ya que son demasiado pesados para el repositorio.

#### Alternativa: Generar los datos de entrenamiento

En lugar de mover los datos de entrenamiento a `data/raw/`, `data/splits/` y `data/processed/`, es posible **generarlos de forma sintética** ejecutando:

```bash
# Generar datos en data/raw/ y data/splits/
python scripts/generate_dataset.py
```

La ejecución de este script de generación de datos tarda **aproximadamente 1 minuto**.

### Opción B: Dataset externo del cliente

Se utiliza cuando el cliente dispone de datos reales para reentrenar el modelo.

#### Formato del dataset

El CSV debe colocarse en `data/raw/` y contener:
- Las 11 variables de sensor definidas en `data_generation.sensors` de `config/config.yaml`.
- La columna objetivo `fault_name` (definida en `config/config.yaml`en `data_processing.target_column`). Para ciclos normales, usar cualquiera de las etiquetas en `data_processing.normal_tokens`; cualquier otra etiqueta se interpretará como anomalía.
- La columna `timestamp`, obligatoria para orden temporal y construcción de ventanas. No puede contener vacíos; puede ser fecha/hora parseable u ordinal numérico, pero debe mantener un único formato en todo el CSV.
- Opcionalmente, `cycle_id` para respetar la independencia entre ciclos.

> **Nota:** Se recomienda evitar columnas adicionales no reconocidas, ya que el preprocesamiento podría incorporarlas como variables predictoras.

#### División en splits

Configurar en `config/config.yaml` bajo `data_processing.external_data_split` los porcentajes `train_pct`, `val_pct`, `test_pct` (deben sumar 100%).

```bash
python scripts/split_external_data.py --input data/raw/mi_dataset.csv
```

Este script valida los sensores y la variable objetivo, y genera `data/splits/train.csv`, `val.csv`, `test.csv`. La validación completa de `timestamp`, duplicados, filas vacías, tipos de sensores y nulos parciales se aplica después en `scripts/data_processing.py`, usando `src/data_processing/input_validation.py`.

> ⚠️ **Importante:** `split_external_data.py` sobrescribe los archivos en `data/splits/`. No ejecutar `generate_dataset.py` después, ya que sustituiría los splits externos por sintéticos.

---

### Entrenamiento del modelo

Una vez disponibles los splits en `data/splits/`, existen dos opciones de entrenamiento:

#### Opción 1 — Pipeline convencional mediante scripts

Teniendo los datos en sus carpetas correspondientes, ejecutar:

```bash
# 1. Procesar datos (crear arrays en data/processed/)
python scripts/data_processing.py

# 2. Entrenar modelo Deep Neuro-Fuzzy
python scripts/train.py
```

Para obtener las métricas sobre el conjunto **TEST**, ejecutar posteriormente:

```bash
python scripts/get_stats.py
```

#### Opción 2 — Optimización con Optuna

Como alternativa al pipeline convencional, se puede ejecutar **`notebooks/tuning_optuna.ipynb`**, que optimiza los hiperparámetros entrenando 80 trials sobre los splits de `data/splits/`.

**Salidas adicionales:** `models/metrics/best_optuna_trial.json` con los hiperparámetros del mejor trial.

> ⚠️ **Importante:** El notebook no modifica `config/config.yaml`. Los hiperparámetros del mejor trial deben copiarse manualmente a `config/config.yaml` para que la inferencia posterior sea coherente. Una vez sincronizado, NO es necesario volver a ejecutar `train.py`.
---

## Validación de la Capa XAI y PCC tras el Entrenamiento

La validación completa de la capa XAI, la detección de Perfiles Críticos de Control (PCC) y el monitor online se lleva a cabo en el notebook situado en **`notebooks/XAI.ipynb`**.

### Datos requeridos

| Ruta | Descripción |
|---|---|
| `data/raw/interpretability_val.csv` | Dataset balanceado (fault_ratio=0.5, 1500 ciclos) para verificar consistencia interpretativa del pipeline XAI por grupos TP/TN/FP/FN. |
| `data/raw/pcc_system_eval.csv` | Dataset con distribución más realista (fault_ratio=0.2, 500 ciclos) para probar la lógica de acciones correctivas y el monitor online PCC. |
| `data/raw/xai_background.csv` | Base para el background de SHAP usado por la capa XAI. **Incluido en el repositorio.** |

> **Importante:** `interpretability_val.csv` y `pcc_system_eval.csv` están subidos a un contenedor externo, ya que son demasiado pesados para el repositorio. `xai_background.csv` es necesario para la inferencia y se incluye directamente en el repositorio.

> Si el modelo se ha reentrenado con datos externos del cliente, estos conjuntos deberían construirse también a partir de datos representativos del mismo dominio. Mantener los datasets sintéticos permite comprobar el funcionamiento técnico de la capa XAI, pero no demuestra que los perfiles PCC identificados sean válidos para el entorno real del cliente.

### Ejecución

Teniendo los datos en sus carpetas correspondientes, abrir y ejecutar en orden las celdas de:

```
notebooks/XAI.ipynb
```

El notebook está organizado en las siguientes secciones:

1. **Librerías** — Importación de dependencias y configuración del pipeline.
2. **Interpretabilidad local para detección de PCCs** — Constructor del contexto de evaluación XAI y validación por grupos de confusión (TP/TN/FP/FN).
3. **Detección de PCC** — Perfilado de criticidad operativa por pares de subsistemas dominantes y tramos temporales SHAP.
4. **Evaluación monitor** — Evaluación del monitor online PCC con política de estados (`Normal`, `Vigilancia`, `Criticidad detectada`).
5. **Aplicación** — Ejecución completa del pipeline con los datos de validación, incluyendo la configuración de subsistemas PCC, catálogo de perfiles críticos y política de monitorización.

### Advertencia tras reentrenamiento

> ⚠️ Un nuevo entrenamiento del modelo puede modificar las variables dominantes, las activaciones de las reglas fuzzy y los perfiles interpretativos emergentes. Estos cambios pueden invalidar la configuración actual del catálogo y la política de monitorización. Tras cualquier reentrenamiento, se recomienda abrir **`notebooks/XAI.ipynb`** y ajustar manualmente las tres variables configurables de la sección **Aplicación** — `SUBSYSTEMS_PCC`, `PCC_CATALOG` y `MONITOR_POLICY` — hasta encontrar la nueva combinación óptima. Los nuevos valores deben copiarse manualmente en `config/config.yaml` a sus equivalentes (`xai.pcc.subsystems`, `xai.pcc.catalog`, `xai.pcc.monitor_policy`), ya que `config.yaml` es la fuente utilizada por el flujo de inferencia en producción.

### Alternativa: Generar los datos XAI de forma sintética

En lugar de mover los datos de explicabilidad a `data/raw/`, es posible **generarlos de forma sintética** ejecutando:

```bash
python scripts/generate_dataset.py
```

La ejecución de este script tarda **aproximadamente 1 minuto** y genera en `data/raw/`:

| Archivo | Parámetros de generación (`xai.dataset_generation` en `config/config.yaml`) |
|---|---|
| `interpretability_val.csv` | seed=32, n_cycles=1500, fault_ratio=0.5 |
| `pcc_system_eval.csv` | seed=40, n_cycles=500, fault_ratio=0.2 |
| `xai_background.csv` | seed=52, n_cycles=25, fault_ratio=0.5 |

---

> ⚠️ **Advertencia:** Sin estos datasets específicos de explicabilidad, el entrenamiento del modelo base puede completarse, pero la validación completa de la capa XAI , la detección de PCC y la evaluación del monitor online en `notebooks/XAI.ipynb` **no se pueden realizar**.
