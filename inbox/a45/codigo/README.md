# CU45 - Implementación del modelo Deep Neuro-Fuzzy para la detección de puntos críticos de control en los procesos y la maquinaria

## Visión general

El sistema implementa un modelo de mantenimiento predictivo para secadoras de grano de flujo mixto. Su objetivo es detectar comportamientos anómalos del proceso y complementar la predicción con explicaciones interpretables y recomendaciones operativas para el operador.

El modelo híbrido **Deep Neuro-Fuzzy** combina:

- **Rama LSTM con atención temporal**: captura patrones secuenciales en ventanas temporales de sensores.

- **Rama neuro-difusa basada en reglas TSK**: procesa estadísticas agregadas de cada ventana para complementar la rama secuencial.

Además, se incorpora una **capa XAI** que genera explicaciones locales y un estado de monitorización PCC basado en perfiles interpretativos catalogados.

## Estructura del repositorio

```text
a45-dnsl-cereals-deteccion-puntos-criticos/
├── .gitignore                         # Reglas de exclusión de archivos para Git
├── README.md                          # Documentación principal de instalación, uso, entrenamiento e inferencia
├── requirements.txt                   # Dependencias Python necesarias para ejecutar el proyecto
├── config/
│   └── config.yaml                    # Configuración central: rutas, datos, modelo, entrenamiento, XAI y monitor PCC
├── data/
│   ├── README de datasets - CU45.md   # Descripción complementaria de los datasets del proyecto
│   ├── input/                         # CSV de entrada para inferencia en modo producción
│   │   └── datos_cliente.csv          # Dataset de ejemplo para probar scripts/predict.py
│   ├── raw/                           # Datasets fuente (sintético o externo) y auxiliares XAI
│   ├── splits/                        # Splits tabulares de entrenamiento, validación y test
│   ├── processed/                     # Arrays y estadísticos procesados para entrenamiento/evaluación
│   └── predictions/                   # Salidas de inferencia y reportes XAI/PCC
├── models/
│   ├── artifacts/                     # Artefactos del modelo entrenado
│   │   ├── best_dnf_model.pt          # Checkpoint principal del modelo DNF entrenado
│   │   ├── scaler.pkl                 # Scalers congelados: scaler_x para secuencias y scaler_num para estadísticos
│   │   ├── xai_background.npy         # Background SHAP preprocesado para explicabilidad temporal
│   │   └── baseline_models/           # Artefactos de modelos baseline
│   └── metrics/                       # Métricas, gráficos e historial del modelo DNF
│       ├── best_optuna_trial.json     # Parámetros, metadatos y puntuación del mejor trial de Optuna
│       ├── evaluation_metrics.png     # Gráfico de evaluación del modelo sobre test
│       ├── results.json               # Métricas finales y umbral seleccionado del modelo DNF
│       ├── threshold_diagnostics.png  # Diagnóstico del umbral de decisión
│       ├── training_history.json      # Historial de entrenamiento por época
│       └── training_history.png       # Gráfico del historial de entrenamiento
├── notebooks/
│   ├── baseline.ipynb                 # Entrenamiento y evaluación de modelos baseline
│   ├── EDA_control_dryer.ipynb        # Análisis exploratorio del generador y del proceso simulado
│   ├── tuning_optuna.ipynb            # Búsqueda de hiperparámetros y entrenamiento optimizado con Optuna
│   └── XAI.ipynb                      # Validación interpretativa local, perfiles PCC y evaluación del monitor
├── scripts/
│   ├── data_processing.py             # Ejecuta el preprocesamiento: ventanas, estadísticos, escalado y artefactos
│   ├── generate_dataset.py            # Genera dataset sintético y datasets auxiliares XAI
│   ├── get_stats.py                   # Calcula métricas y gráficos finales sobre test
│   ├── predict.py                     # Ejecuta inferencia con predicción, XAI y monitor PCC
│   ├── split_external_data.py         # Divide un dataset externo etiquetado en train/val/test
│   └── train.py                       # Entrena el modelo Deep Neuro-Fuzzy con la configuración actual
└── src/
    ├── main.py                        # Orquestador lógico: generación, procesamiento, entrenamiento, inferencia y métricas
    ├── data_processing/               # Módulos de carga, generación y preprocesamiento de datos
    │   ├── generate_dataset.py        # Simulador sintético del proceso de secado y generación de datasets XAI
    │   ├── input_validation.py        # Validación común de entradas: timestamp, sensores, nulos, duplicados e imputación
    │   ├── load_data.py               # Carga de CSV y normalización inicial de columnas
    │   └── preprocess.py              # Splits, validación de datos, creación de ventanas temporales con etiquetado, estadísticos y escalado
    ├── get_stats/                     # Módulos para evaluación final del modelo
    │   └── stats.py                   # Cálculo de métricas, curvas, diagnósticos de umbral y gráficos
    ├── predict/                       # Módulos de inferencia y postprocesado
    │   ├── postprocess.py             # Decodificación de predicciones y serialización segura a JSON
    │   └── xai_predictor.py           # Pipeline de inferencia con XAI y monitorización PCC
    ├── training/                      # Módulos de entrenamiento del modelo DNF
    │   ├── loss.py                    # Funciones de pérdida y regularización
    │   ├── metrics.py                 # Métricas de entrenamiento y validación
    │   ├── model.py                   # Arquitectura Deep Neuro-Fuzzy: rama LSTM, rama fuzzy y fusión
    │   └── trainer.py                 # Bucle de entrenamiento, validación, early stopping y checkpointing
    ├── utils/                         # Utilidades comunes del proyecto
    │   ├── common.py                  # Carga de configuración, creación de carpetas y control de semillas
    │   └── logging.py                 # Configuración del sistema de logging
    └── xai/                           # Capa de explicabilidad y monitorización PCC
        ├── explainer.py               # Orquestador XAI: fuzzy, temporal, fusión y PCC
        ├── fusion.py                  # Fusión de explicaciones y probabilidades de ramas fuzzy/LSTM
        ├── fuzzy_explainer.py         # Explicabilidad de reglas neuro-difusas y support ratio
        ├── pcc.py                     # Ranking de subsistemas, catálogo PCC y política del monitor
        ├── temporal_explainer.py      # Explicabilidad temporal con SHAP sobre la rama LSTM
        └── utils.py                   # Utilidades auxiliares para interpretabilidad
```

## Datos

El modelo trabaja con series temporales en ventanas deslizantes sobre sensores de una secadora de grano de flujo mixto.

En esta fase, el repositorio incluye un generador de datos sintéticos controlados y también permite preparar splits desde un dataset externo etiquetado.

### Sensores de entrada

| Sensor | Unidad | Descripción |
|---|---:|---|
| `plenum_temp` | C | Temperatura del plenum |
| `exhaust_air_temp` | C | Temperatura del aire de escape |
| `exhaust_air_humidity` | % | Humedad del aire de escape |
| `static_pressure` | mbar | Presión estática del sistema |
| `burner_power` | % | Potencia del quemador |
| `fan_speed` | RPM | Velocidad del ventilador |
| `discharge_frequency` | Hz | Frecuencia de descarga |
| `grain_moisture_in` | % | Humedad de entrada del grano |
| `ambient_temp` | C | Temperatura ambiente |
| `ambient_humidity` | % | Humedad ambiente |
| `setpoint_temp` | C | Setpoint de temperatura |

### Metadatos reconocidos

El pipeline reconoce como metadatos, y no como variables predictoras principales, las siguientes columnas:

| Columna | Descripción |
|---|---|
| `timestamp` | Marca temporal obligatoria de cada lectura. Puede venir como fecha/hora parseable o como ordinal numérico (`1, 2, 3, ...`), pero debe mantener un único formato en todo el CSV. |
| `cycle_id` | Identificador único de cada ciclo de secado. Utilizado para splits por entidad y para la construcción de ventanas temporales. |
| `phase` | Fase operativa del secador en cada registro: `heating` (calefacción inicial), `drying` (secado en régimen) o `cooling` (enfriamiento y parada). |
| `grain_type` | Tipo de grano procesado en el ciclo: `CORN`, `WHEAT`, `BARLEY` o `SUNFLOWER`. Cada tipo define un perfil propio de setpoint, humedad y resistencia. |
| `fault_name` | Nombre del escenario de fallo inyectado en el ciclo (p. ej. `NORMAL`, `FILTER_CLOGGED`, `BURNER_DEGRADED`). Sirve como columna objetivo para el etiquetado. |
| `fault_label` | Etiqueta numérica del escenario de fallo: `0` para operación normal y `1`–`5` para cada tipo de fallo definido en `fault_types`. |

### Modos de operación sintéticos

| Nombre | Descripción |
|---|---|
| `NORMAL` | Operación normal sin fallas |
| `FILTER_CLOGGED` | Filtro obstruido, con mayor presión y menor flujo |
| `BURNER_DEGRADED` | Pérdida de potencia térmica del quemador |
| `DISCHARGE_JAM` | Bloqueo en el sistema de descarga |
| `HUMIDITY_SENSOR_DRIFT` | Deriva en la lectura de humedad de escape |
| `PLENUM_THERMAL_LEAK` | Fuga térmica o aislamiento deficiente |

### Ventanas temporales

La configuración actual utiliza:

- `sequence_length = 240`
- `solapamiento_beta = 0.5`
- Desplazamiento entre ventanas: 120 registros
- Frecuencia sintética de muestreo: 60 segundos

Por tanto, cada ventana contiene 240 minutos de histórico, equivalentes a cuatro horas, con un solapamiento del 50 % entre ventanas consecutivas.

### Ubicación de datos

Resumen práctico de las carpetas incluidas dentro de `data/`:

- `data/input/`: CSV del cliente o archivo de ejemplo utilizado como entrada de inferencia. Por defecto, el repositorio incluye `data/input/datos_cliente.csv`.

- `data/raw/`: datasets fuente y auxiliares. Cuando se ejecuta `scripts/generate_dataset.py`, aquí se generan el dataset sintético completo (`dryer_full_dataset.csv`) y los datasets auxiliares de XAI (`interpretability_val.csv`, `pcc_system_eval.csv` y `xai_background.csv`).

- `data/splits/`: particiones tabulares `train.csv`, `val.csv` y `test.csv`. Pueden proceder del generador sintético o de `scripts/split_external_data.py`.

- `data/processed/`: arrays y estadísticos listos para entrenamiento y evaluación, generados por `scripts/data_processing.py` o por `notebooks/tuning_optuna.ipynb`. Incluye `X_*.npy`, `y_*.npy` y `stats_*.csv`.

- `data/predictions/`: salidas generadas por `scripts/predict.py`, incluyendo predicciones en CSV y reportes XAI/PCC en JSON.

- `data/README de datasets - CU45.md`: documento auxiliar con información sobre los datasets del proyecto.

## Qué recibe y qué entrega

**Entrada de inferencia:**

- CSV con series temporales de sensores de la secadora de grano.
- Debe contener las 11 columnas de sensores definidas en `config/config.yaml`, dentro de `data_generation.sensors`.
- Debe contener `timestamp`, definido por `data_processing.timestamp_column`; esta columna se usa para ordenar los registros y delimitar las ventanas de salida.
- Opcionalmente, puede incluir `cycle_id`.
- El orden de columnas no es estricto: el sistema normaliza nombres y reordena las variables conforme a la configuración.

**Validación de entrada:**

La validación de datos de entrada, compartida por la inferencia y el entrenamiento, vive en `src/data_processing/input_validation.py`.

- Se normalizan nombres de columnas y se detectan columnas duplicadas tras normalizar.
- Se ignoran filas completamente vacías, dejando un warning en logs.
- `timestamp` es obligatorio y no puede estar vacío; se acepta formato fecha/hora o formato ordinal numérico, siempre que no se mezclen ambos formatos.
- Si existe `cycle_id`, los duplicados se comprueban por pareja `cycle_id`/`timestamp`; si no existe, `timestamp` debe ser único a nivel global.
- Se valida que haya registros suficientes para conformar una ventana, según lo dispuesto en `sequence_length`: se valida por ciclo si existe `cycle_id` o globalmente en caso contrario.
- Se validan sensores requeridos, tipos numéricos, columnas completamente vacías y nulos parciales.
- Los nulos parciales permitidos dentro del umbral `partial_null_max_ratio` se imputan con criterio temporal (solo en columnas de sensores; nunca en `timestamp`) antes de inferencia y antes de crear las secuencias de entrenamiento.
- Al preparar el DataFrame se filtran columnas no utilizadas por el modelo, se conservan las columnas de sensores, se conserva `timestamp`, se conserva `cycle_id` cuando existe y se ordena por `[cycle_id, timestamp]` o por `timestamp`.

**Salida de inferencia:**

- `data/predictions/predictions_YYYYMMDD_HHMMSS.csv`: predicciones por ventana con `window_index`, `cycle_id` si existe, `timestamp_init`, `timestamp_end`, `predicted_anomaly_class`, `predicted_anomaly_label`, `anomaly_probability` y `decision_threshold`.
- `data/predictions/PCC_monitor_YYYYMMDD_HHMMSS.json`: reporte XAI/PCC por ventana con `Índice de ventana`, `ID de ciclo` cuando existe, `Timestamp cubierto por ventana`, `Estado interpretativo`, `Evidencia`, `Probabilidad de anomalía`, `Umbral de detección de anomalías`, `Margen respecto al umbral` y `Recomendación`. Si una ventana falla durante la explicación, el objeto incluye `Índice de ventana` y `error`.



## Configuración

La configuración principal se gestiona desde:

```text
config/config.yaml
```

Bloques principales:

- `project`: nombre del proyecto y semilla global.
- `paths`: rutas de datos de entrada/salida, artefactos y métricas.
- `data_generation`: parámetros del generador sintético, sensores y tipos de fallo.
- `data_processing`: columna objetivo, identificador de ciclo, ventana temporal, solapamiento, escalado, reglas de etiquetado y estadísticos de la rama neuro-difusa.
- `model`: hiperparámetros de la rama LSTM y de la rama neuro-difusa.
- `training`: parámetros de entrenamiento, scheduler, early stopping, métrica de monitorización y función de pérdida.
- `xai`: parámetros de explicabilidad, background SHAP, perfiles PCC y política del monitor.

## Pasos comunes para inferencia y entrenamiento

Estos pasos preparan el entorno local. Deben realizarse antes de ejecutar inferencia o entrenamiento.

### Paso 1 — Instalar Python 3.11

El repositorio está preparado para Python 3.11. En Windows, puede instalarse con:

```bash
winget install -e --id Python.Python.3.11
```

### Paso 2 — Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd <nombre-de-la-carpeta>
```

### Paso 3 — Crear entorno virtual e instalar dependencias

```bash
py -3.11 -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

**Notas importantes:**

- El sistema selecciona automáticamente el dispositivo disponible para PyTorch cuando corresponde.
- No se requiere GPU para ejecutar inferencia; la CPU es suficiente, aunque más lenta.
- El entrenamiento y el tuning con Optuna pueden beneficiarse de GPU.

Para verificar la instalación de PyTorch y la disponibilidad de CUDA:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda)"
```

## Flujo de inferencia para uso en producción

Este flujo se utiliza cuando se dispone de un CSV de entrada y se desea obtener tanto las predicciones de normalidad/fallo como las explicaciones XAI y el estado del monitor PCC.

### Paso 1 — Preparación del dataset de inferencia

El archivo debe contener obligatoriamente las 11 variables definidas en `config/config.yaml`, dentro de `data_generation.sensors`:

| Sensor |
|---|
| `plenum_temp` |
| `exhaust_air_temp` |
| `exhaust_air_humidity` |
| `static_pressure` |
| `burner_power` |
| `fan_speed` |
| `discharge_frequency` |
| `grain_moisture_in` |
| `ambient_temp` |
| `ambient_humidity` |
| `setpoint_temp` |

Además, debe incluir la columna temporal configurada en `data_processing.timestamp_column`, por defecto `timestamp`. `cycle_id` es opcional:

- `timestamp` es obligatorio, permite ordenar temporalmente los registros y se propaga a las salidas como rango de ventana. Puede ser fecha/hora o un ordinal numérico, pero no una mezcla de ambos formatos.
- `cycle_id` garantiza que las ventanas se construyan de forma independiente para cada ciclo, sin mezclar registros pertenecientes a ciclos distintos.

A modo de ejemplo, para probar la inferencia, el repositorio incluye el archivo:

```text
data/input/datos_cliente.csv
```

### Paso 2 — Ejecutar inferencia

```bash
python scripts/predict.py --input data/input/datos_cliente.csv
```

Se recomienda utilizar `--input` para indicar explícitamente la ruta al dataset de inferencia. Si no se proporciona este argumento, se emplea por defecto la ruta configurada en `config/config.yaml`, dentro de `paths.input_data`. Cuando esta ruta es un directorio, el sistema selecciona automáticamente el primer CSV en orden alfabético, por lo que en producción es preferible indicar el archivo de entrada de forma explícita.

### Salidas generadas

Por defecto, las salidas se guardan en `data/predictions/`:

- `predictions_YYYYMMDD_HHMMSS.csv`: contiene `window_index`, `cycle_id` si existe, `timestamp_init`, `timestamp_end`, `predicted_anomaly_class`, `predicted_anomaly_label`, `anomaly_probability` y `decision_threshold`.

- `PCC_monitor_YYYYMMDD_HHMMSS.json`: contiene, para cada ventana explicada correctamente, `Índice de ventana`, `ID de ciclo` cuando existe, `Timestamp cubierto por ventana`, `Estado interpretativo`, `Evidencia`, `Probabilidad de anomalía`, `Umbral de detección de anomalías`, `Margen respecto al umbral` y `Recomendación`. Si una ventana no puede explicarse, el objeto registra `Índice de ventana` y `error`.

Interpretación rápida de las columnas del CSV:

- `window_index`: índice ordinal de la ventana procesada.
- `cycle_id`: ciclo de origen de la ventana, si el CSV incluye `id_column`.
- `timestamp_init` y `timestamp_end`: registro temporal inicial y final de la ventana procesada.
- `predicted_anomaly_class`: clase binaria predicha (`0` = no fallo, `1` = fallo).
- `predicted_anomaly_label`: etiqueta legible asociada a la clase (`No Fallo` o `Fallo`).
- `anomaly_probability`: probabilidad de anomalía usada por el clasificador.
- `decision_threshold`: umbral aplicado para convertir probabilidad en clase.

Interpretación rápida de los campos del JSON:

- `Índice de ventana`: índice ordinal de la ventana procesada.
- `ID de ciclo`: ciclo de origen de la ventana, si el CSV incluye `id_column`.
- `Timestamp cubierto por ventana`: rango temporal de registros cubierto por la ventana.
- `Estado interpretativo`: estado asignado por el monitor (`Normal`, `Vigilancia` o `Criticidad detectada`).
- `Evidencia`: resumen textual de la coincidencia, o ausencia de coincidencia, con un perfil catalogado.
- `Probabilidad de anomalía`: probabilidad de anomalía estimada por el modelo.
- `Umbral de detección de anomalías`: umbral utilizado para clasificar la ventana.
- `Margen respecto al umbral`: distancia absoluta entre la probabilidad y el umbral.
- `Recomendación`: acción sugerida para el operador según el estado y la evidencia.

Ejemplo de salida del JSON:

```json
{
  "Índice de ventana": 17,
  "ID de ciclo": 3,
  "Timestamp cubierto por ventana": "2026-05-18 08:00:00 / 2026-05-18 11:59:00",
  "Estado interpretativo": "Normal",
  "Evidencia": "No se identifica un perfil catalogado de criticidad y no hay indicios fuertes de anomalía.",
  "Probabilidad de anomalía": 0.07,
  "Umbral de detección de anomalías": 0.73,
  "Margen respecto al umbral": 0.66,
  "Recomendación": "Mantener monitorización ordinaria."
}
```

Si una ventana produce un error durante la explicación, el error se registra en el JSON y el procesamiento continúa con las ventanas restantes. El CSV de predicciones solo incluye las ventanas procesadas correctamente.

## Flujo de entrenamiento

El modelo entregado en `models/artifacts/best_dnf_model.pt` ya se encuentra entrenado.

Opcionalmente, este modelo se puede reentrenar a partir de dos fuentes de datos:

- Datos sintéticos generados por el repositorio.
- Dataset externo etiquetado del cliente.

Asimismo, existen dos rutas posibles de entrenamiento:

- Pipeline convencional mediante scripts y los hiperparámetros en `config/config.yaml`. 
- Entrenamiento con optimización de hiperparámetros mediante Optuna en `notebooks/tuning_optuna.ipynb`.

### Paso 1 — Elección de la fuente de datos

#### Opción A — Datos sintéticos

Esta es la opción por defecto. Permite generar los datos desde el repositorio a partir de la configuración definida en `config/config.yaml`, dentro de `data_generation`.

Genera el dataset completo de entrenamiento, sus particiones `train`, `val` y `test`, y varios conjuntos auxiliares para explicabilidad.

Ejecutar:

```bash
python scripts/generate_dataset.py
```

Este comando genera:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/raw/dryer_full_dataset.csv`
- `data/raw/interpretability_val.csv`
- `data/raw/pcc_system_eval.csv`
- `data/raw/xai_background.csv`

#### Opción B — Datos externos del cliente

Esta opción se utiliza cuando se dispone de un dataset real o externo para reentrenar el modelo.

El dataset externo debe estar en formato CSV y se recomienda colocarlo en `data/raw/`. Debe contener:

- Las 11 variables definidas en `config/config.yaml`, dentro de `data_generation.sensors`.
- La columna objetivo `fault_name`, definida en `data_processing.target_column`.
- La columna `timestamp`, obligatoria para ordenar los registros durante la construcción de secuencias. No puede contener vacíos; puede ser fecha/hora parseable u ordinal numérico, pero debe mantener un único formato en todo el CSV.
- Opcionalmente, `cycle_id`, para construir ventanas respetando los ciclos de origen.

En la columna `fault_name`, los ciclos normales pueden identificarse con cualquiera de las etiquetas incluidas en `data_processing.normal_tokens`. Cualquier otro valor se interpreta como anomalía.

Se recomienda evitar columnas adicionales distintas de los sensores y metadatos reconocidos, ya que algunos flujos de preprocesamiento de entrenamiento podrían incorporarlas como variables predictoras si no se controlan adecuadamente.

##### Configurar proporciones de split

En `config/config.yaml`, dentro de `data_processing.external_data_split`, ajustar:

- `train_pct`
- `val_pct`
- `test_pct`

Los tres porcentajes deben sumar 100 %. En caso contrario, el script mostrará un error.

##### Dividir el dataset externo en splits

```bash
python scripts/split_external_data.py --input data/raw/mi_dataset.csv
```

Este script valida la presencia de los 11 sensores y de la variable objetivo, y genera los splits. La validación completa de los datos (`timestamp`, duplicados, filas vacías, tipos de sensores y nulos parciales) se aplica posteriormente en `scripts/data_processing.py`, siguiendo el mismo criterio descrito en la sección [Validación de entrada](#validacion-de-entrada).

Genera:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Comportamiento de la división:

- Si existe `cycle_id` y se detectan al menos tres ciclos, estos se distribuyen entre entrenamiento, validación y test siguiendo el orden de primera aparición de sus identificadores. Un mismo ciclo nunca aparece en más de una partición. La división no aplica aleatorización ni estratificación por clase, y las proporciones finales pueden ser aproximadas si los ciclos tienen duraciones distintas.

- Si `cycle_id` no existe, hay menos de tres identificadores de ciclo o la división por ciclos produce una partición vacía, se utiliza una división secuencial por filas.

Importante:

- La ejecución de `split_external_data.py` sobrescribe los archivos existentes en `data/splits/`.

- Si se utiliza la vía de datos externos, no debe ejecutarse posteriormente `scripts/generate_dataset.py`, ya que esto sustituiría los splits creados por splits sintéticos.

- `split_external_data.py` únicamente genera las particiones de entrenamiento, validación y test. No crea automáticamente conjuntos externos equivalentes a `interpretability_val.csv`, `pcc_system_eval.csv` o `xai_background.csv`. Si el modelo se entrena con datos del cliente, estos conjuntos deberían construirse también a partir de datos representativos e independientes del mismo dominio. Mantener los datasets sintéticos permite comprobar el funcionamiento técnico de la capa XAI, pero no demuestra que los perfiles PCC identificados sean válidos para el entorno real del cliente.

### Paso 2 — Selección del modo de entrenamiento

Una vez obtenidos los splits mediante cualquiera de las dos opciones anteriores, se debe elegir el modo de entrenamiento.

#### Ruta A — Pipeline convencional mediante scripts

Esta ruta utiliza la configuración definida en `config/config.yaml`.

##### Paso 2.1A — Procesamiento de datos

Antes de ejecutar este paso, revisar los bloques `data_processing`, `model`, `training` y `xai` de `config/config.yaml`. Se recomienda no modificarlos si se quiere reproducir el modelo actual.

Posteriormente, ejecutar:

```bash
python scripts/data_processing.py
```

Este script carga los splits de `data/splits/`, aplica la construcción de secuencias temporales y escalado, y guarda los artefactos procesados.

Genera:

- `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy`
- `data/processed/y_train.npy`, `y_val.npy`, `y_test.npy`
- `data/processed/stats_train.csv`, `stats_val.csv`, `stats_test.csv`
- `models/artifacts/scaler.pkl`
- `models/artifacts/xai_background.npy`, si existe `data/raw/xai_background.csv`

##### Paso 2.2A — Entrenamiento del modelo Deep Neuro-Fuzzy

```bash
python scripts/train.py
```

Este comando genera o actualiza:

- `models/artifacts/best_dnf_model.pt`
- `models/metrics/results.json`
- `models/metrics/training_history.json`

#### Ruta B — Optimización y entrenamiento con Optuna

Esta ruta sustituye conjuntamente a `python scripts/data_processing.py` y `python scripts/train.py`.

##### Paso 2.1B — Ejecución del estudio de Optuna

Habiendo generado los splits en el paso 1, abrir y ejecutar en orden las celdas de:

```text
notebooks/tuning_optuna.ipynb
```

Este notebook parte de los conjuntos de entrenamiento, validación y prueba almacenados en `data/splits/`. Para cada combinación de hiperparámetros propuesta por Optuna, vuelve a preparar los datos cuando la configuración afecta a la forma de construir las entradas del modelo, por ejemplo, la longitud de las ventanas temporales, su grado de solapamiento o las estadísticas calculadas para la rama neuro-difusa.

De este modo, cada configuración se entrena y evalúa con unos datos procesados de forma coherente con sus propios parámetros, en lugar de reutilizar unas ventanas preparadas para una configuración diferente. Para reducir cálculos repetidos, las configuraciones de preprocesamiento ya evaluadas se almacenan temporalmente y se reutilizan cuando vuelven a aparecer.

El notebook entrena un total de 80 trials y publica directamente:

- `models/artifacts/best_dnf_model.pt`
- `models/artifacts/scaler.pkl`
- Los arrays y estadísticos de `data/processed/`
- `models/metrics/results.json`
- `models/metrics/training_history.json`
- `models/metrics/best_optuna_trial.json`

Importante:

- Una vez termina el entrenamiento con tuning de Optuna, los hiperparámetros del mejor trial se guardan en `models/metrics/best_optuna_trial.json`.

- `notebooks/tuning_optuna.ipynb` no modifica automáticamente `config/config.yaml` con dichos hiperparámetros.

- Para garantizar que la predicción y la explicabilidad en inferencia se correspondan con el mejor modelo entrenado, los hiperparámetros del mejor trial deben trasladarse manualmente a `config/config.yaml`, dentro de los bloques correspondientes de `data_processing`, `model` y `training`, antes de utilizar el nuevo modelo en inferencia.

- En el repositorio actual, los parámetros de `config/config.yaml` ya se corresponden con los del mejor trial obtenido en el tuning con Optuna.

Una vez copiados los hiperparámetros a `config/config.yaml`, no es necesario volver a ejecutar `scripts/train.py`, ya que el notebook de Optuna guarda automáticamente el mejor checkpoint del modelo entrenado en `models/artifacts/best_dnf_model.pt`.

Nota: este paso puede requerir varias horas de ejecución, aproximadamente dos horas dependiendo del hardware.

### Paso 3 — Verificación de métricas sobre test

Con independencia de la ruta de entrenamiento empleada, ejecutar:

```bash
python scripts/get_stats.py
```

Este paso evalúa sobre el split de test el modelo almacenado en `models/artifacts/best_dnf_model.pt`. Sobrescribe `models/metrics/results.json` con el informe final de métricas.

### Paso 4 — Auditoría del monitor PCC tras reentrenamiento

Para reproducir el análisis de interpretabilidad correspondiente al modelo actual, ejecutar las celdas de:

```text
notebooks/XAI.ipynb
```

La configuración de variables que aparece en la sección de aplicación del notebook recoge la combinación de `SUBSYSTEMS_PCC`, `PCC_CATALOG` y `MONITOR_POLICY` seleccionada para el modelo entrenado según los criterios de validación interpretativa del proyecto.

No obstante, un nuevo entrenamiento puede modificar las variables dominantes, las activaciones de las reglas neuro-difusas y los perfiles interpretativos emergentes. Estos cambios podrían invalidar la configuración actual del catálogo y de la política de monitorización.

Por ello, tras cualquier reentrenamiento se recomienda abrir `notebooks/XAI.ipynb` y ajustar manualmente las tres variables configurables:

- `SUBSYSTEMS_PCC`
- `PCC_CATALOG`
- `MONITOR_POLICY`

Los nuevos valores deben copiarse manualmente en `config/config.yaml`, en sus equivalentes:

- `xai.pcc.subsystems`
- `xai.pcc.catalog`
- `xai.pcc.monitor_policy`

Es importante que `config/config.yaml` esté sincronizado con `notebooks/XAI.ipynb`, ya que `config.yaml` es la fuente utilizada por el flujo de inferencia.

## Notebooks y funcionalidad

Los notebooks del repositorio se utilizan como espacios de análisis, comparación experimental, tuning e interpretabilidad. No sustituyen al flujo de inferencia por scripts, pero documentan y permiten reproducir partes importantes del desarrollo del modelo.

- `notebooks/EDA_control_dryer.ipynb`: notebook de análisis exploratorio del proceso simulado. Bebe directamente de `data/raw/dryer_full_dataset.csv`. Permite inspeccionar las variables del secador, las distribuciones de sensores, la evolución temporal de los ciclos, las fases operativas y el comportamiento de los fallos sintéticos generados.

- `notebooks/baseline.ipynb`: notebook de comparación con modelos baseline. Carga los splits tabulares desde `data/splits/` (`train.csv`, `val.csv`, `test.csv`) y, en el caso de los modelos LSTM y ANFIS, utiliza los arrays preprocesados de `data/processed/` (`X_*.npy`, `y_*.npy`, `stats_*.csv`). Entrena y evalúa alternativas más simples o convencionales, como Random Forest, XGBoost, LSTM simple y ANFIS simple, con el objetivo de disponer de referencias frente al modelo Deep Neuro-Fuzzy.

- `notebooks/tuning_optuna.ipynb`: notebook de optimización de hiperparámetros con Optuna. Carga los splits de `data/splits/`, repite el preprocesamiento cuando la configuración evaluada lo requiere, entrena múltiples configuraciones del modelo DNF y publica el mejor checkpoint, scalers, métricas y metadatos del mejor trial.

- `notebooks/XAI.ipynb`: notebook de validación interpretativa de la capa XAI. Utiliza `data/raw/interpretability_val.csv` para la validación por grupo de confusión y el perfilado de criticidad operativa, y `data/raw/pcc_system_eval.csv` para la evaluación offline del monitor PCC. Ejecuta explicaciones locales por grupo de confusión, construye perfiles interpretativos recurrentes mediante el análisis de subsistemas y tramos temporales, permite ajustar `SUBSYSTEMS_PCC`, `PCC_CATALOG` y `MONITOR_POLICY`, y evalúa de forma offline el monitor PCC sobre un conjunto independiente.

## Reproducibilidad

El proyecto fija semillas desde `config/config.yaml`, tanto a nivel global como por dataset. Semillas utilizadas:

| Dataset | Clave en configuración | Semilla |
|---|---|---|
| Dataset principal (`dryer_full_dataset.csv`) | `project.seed` | `42` |
| Dataset de validación XAI (`interpretability_val.csv`) | `xai.dataset_generation.interpretability_val.seed` | `32` |
| Dataset de evaluación PCC (`pcc_system_eval.csv`) | `xai.dataset_generation.pcc_system_eval.seed` | `40` |
| Dataset background SHAP (`xai_background.csv`) | `xai.dataset_generation.xai_background.seed` | `52` |

Además, `project.seed` (`42`) se aplica como semilla global antes de cada pipeline de entrenamiento o inferencia, fijando los generadores aleatorios de Python, NumPy y PyTorch.

Aún así, cambios en versiones de PyTorch, CUDA o cuDNN, diferencias en GPU, drivers, sistema operativo o configuración determinista pueden introducir pequeñas variaciones en los resultados de entrenamiento.

La reproducibilidad estricta solo puede garantizarse cuando el entrenamiento se ejecuta bajo el mismo entorno software y hardware.
