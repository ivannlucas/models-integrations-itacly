# CU43 - Diseño y entrenamiento de un modelo basado en Deep Neuro-Fuzzy Learning para detectar anomalías y predecir fallas en hornos y CU44 - Desarrollo de un modelo predictivo basado en XAI para generar acciones correctivas

## Visión general

Este proyecto implementa un ecosistema de inteligencia híbrida que engloba:

- **CU43**: Detección de anomalías mediante Deep Neuro-Fuzzy Learning.
- **CU44**: Capa XAI para generar acciones correctivas prescriptivas.

El sistema es un modelo de mantenimiento predictivo para hornos industriales orientado a transitar desde la detección de fallos de caja negra hacia una estrategia de mantenimiento prescriptivo con recomendaciones comprensibles, útiles y justificables para operadores.

El modelo híbrido **Deep Neuro-Fuzzy** (DNF) combina:

- **LSTM unidireccional con atención temporal**: captura patrones secuenciales en las ventanas temporales de los sensores.
- **Bloque estadístico fuzzy**: procesa estadísticas agregadas de cada ventana para complementar la rama secuencial.

Para el **CU44**, se añade una **capa XAI** que genera explicaciones y acciones correctivas recomendadas basadas en los resultados de predicción.

## Estructura del repositorio

```text
a43-dnsl-cereals-neurofuzzy-anomalias-fallas/
├── config/
│   └── config.yaml                 # Configuración central (paths, datos, modelo, entrenamiento, XAI)
├── data/
│   ├── input/                      # CSV del cliente para inferencia (ej: datos_prueba.csv)
│   ├── predictions/                # Salidas de inferencia y reportes XAI
│   │   └── xai/                    # Archivos específicos de explicabilidad (CSV + JSON)
│   ├── processed/                  # Arrays procesados: X_*.npy, y_*.npy, stats_*.csv
│   ├── raw/                        # Dataset fuente (sintético o externo) + datasets XAI auxiliares
│   └── splits/                     # Splits tabulares: train.csv, val.csv, test.csv
├── models/
│   ├── artifacts/                  # Artefactos del modelo entrenado
│   │   ├── baseline_models/        # Modelos baseline (RF, LSTM, GB, ANFIS)
│   │   ├── args.yaml               # Hiperparámetros del mejor modelo DNF
│   │   ├── best_dnf_model.pt       # Checkpoint del modelo DNF
│   │   ├── best_optuna_model.pt    # Mejor modelo encontrado con Optuna
│   │   ├── scaler.pkl              # Scalers (scaler_x y scaler_num)
│   │   └── xai_background.npy      # Background de SHAP para XAI
│   └── metrics/                    # Métricas y gráficos de entrenamiento
│       ├── best_optuna_results.json   # Resultados del tuning con Optuna
│       ├── evaluation_metrics.png  # Gráfico de evaluación sobre test
│       ├── results.json            # Métricas finales del modelo DNF
│       ├── threshold_diagnostics.png  # Diagnóstico de umbral de decisión
│       ├── training_history.csv    # Historial de entrenamiento (formato CSV)
│       ├── training_history.json   # Historial de entrenamiento (por epoch)
│       └── training_history.png    # Gráfico de métricas durante entrenamiento
├── notebooks/                      # Notebooks de experimentación y análisis
│   ├── baseline.ipynb              # Baselines: RF, LSTM, GB, ANFIS
│   ├── CU44.ipynb                  # Capa XAI: interpretabilidad y acciones correctivas
│   ├── EDA_anomaly_detection.ipynb # Análisis exploratorio del dataset
│   └── tuning_optuna.ipynb         # Búsqueda de hiperparámetros con Optuna
├── scripts/                        # Scripts CLI para ejecutar cada fase del pipeline
│   ├── data_processing.py          # Preprocesamiento: secuencias, escalado, arrays
│   ├── generate_dataset.py         # Genera dataset sintético (+ --xai para datasets XAI)
│   ├── get_stats.py                # Métricas y gráficos sobre test
│   ├── predict.py                  # Inferencia (+ --xai para explicabilidad)
│   ├── split_external_data.py      # Divide dataset externo en train/val/test
│   └── train.py                    # Entrenamiento del modelo DNF
├── src/                            # Código fuente modular
│   ├── main.py                     # Orquestador: generate_dataset, data_processing, split_external_data, train, predict, get_stats
│   ├── data_processing/            # Carga, generación y preprocesamiento de datos
│   ├── get_stats/                  # Generación de métricas y gráficos
│   ├── predict/                    # Predictor estándar, postprocesado y predictor XAI
│   ├── training/                   # Modelo DNF, loss, métricas y trainer
│   ├── utils/                      # Utilidades comunes (logging, config, seed)
│   └── xai/                        # Capa XAI: explainers, fuzzy, temporal, fusion, acciones correctivas
├── .gitignore
├── README.md
└── requirements.txt
```

## Datos

El modelo trabaja con series temporales procesadas en **ventanas deslizantes** de sensores. Cada ventana contiene lecturas de **13 sensores** del horno industrial.

El flujo de entrenamiento admite dos vías: datos sintéticos generados por el repositorio o datos externos proporcionados por el cliente. Ambas vías convergen en el mismo pipeline de procesamiento y entrenamiento. Los detalles se describen en la sección [Entrenamiento y reentrenamiento](#entrenamiento-y-reentrenamiento).

| Sensor                 | Unidad | Descripción                        |
|------------------------|--------|------------------------------------|
| `temp_zona1`           | C      | Temperatura zona 1                 |
| `temp_zona2`           | C      | Temperatura zona 2                 |
| `temp_zona3`           | C      | Temperatura zona 3                 |
| `temp_salida_gases`    | C      | Temperatura de salida de gases     |
| `presion_camara`       | mbar   | Presión interior de la cámara      |
| `presion_ventilacion`  | mbar   | Presión del sistema de ventilación |
| `potencia_kw`          | kW     | Potencia eléctrica consumida       |
| `flujo_gas`            | m3/h   | Flujo de gas de combustión         |
| `humedad_relativa`     | %      | Humedad relativa interior          |
| `temp_ambiente`        | C      | Temperatura ambiente exterior      |
| `setpoint_temp`        | C      | Temperatura objetivo configurada   |
| `posicion_valvula`     | %      | Apertura de la válvula de gas      |
| `velocidad_ventilador` | RPM    | Velocidad del ventilador extractor |

En el flujo sintético por defecto se generan **7 modos de operación**: 1 estado normal y 6 tipos de fallo, distribuidos como:

- **Train**: 2000 ciclos
- **Validación**: 400 ciclos
- **Test**: 600 ciclos

| Nombre                 | Descripción                                      |
|------------------------|--------------------------------------------------|
| NORMAL                 | Operación normal sin fallas                      |
| AISLAMIENTO_DEGRADADO  | Deterioro del aislamiento térmico                |
| RESISTENCIA_DEGRADADA  | Pérdida de eficiencia en resistencias eléctricas |
| REFRACTARIO_EROSIONADO | Erosión de elementos refractarios                |
| SENSOR_DESCALIBRADO    | Sensor con lecturas erróneas                     |
| VALVULA_OBSTRUIDA      | Válvula con flujo reducido                       |
| VENTILADOR_DEFECTUOSO  | Ventilador con reducción de extracción           |

**Nota**: El preprocesamiento aplica escalado normalizado y genera secuencias temporales según la configuración en `config/config.yaml`.

## Instalación en Windows

El repositorio está diseñado para trabajar con Python 3.11, a fin de evitar problemas de incompatibilidad entre librerías.

Si no se tiene instalado Python 3.11 en el dispositivo, se puede ejecutar el siguiente comando (instala automáticamente la versión 3.11.9):

```bash
winget install -e --id Python.Python.3.11
```

Una vez instalado Python 3.11 (en caso de no tenerlo previamente), clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd <nombre-de-la-carpeta>
```

A continuación, crear el entorno virtual y activarlo:

```bash
py -3.11 -m venv env
env\Scripts\activate
```

Por último, instalar dependencias:

```bash
pip install -r requirements.txt
```

**Notas importantes:**

- `requirements.txt` incluye PyTorch preconfigurado para GPU (CUDA 12.1).
- **Sin GPU**: el sistema cae automáticamente a CPU (más lento en entrenamiento, suficiente para inferencia).
- **Para inferencia en cliente**: no se requiere GPU.

**Verificar GPU tras instalación:**

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda)"
```

## Configuración

Todo se gestiona desde `config/config.yaml`.

Bloques principales:

- `project`: nombre del proyecto y semilla global.
- `paths`: rutas de datos de entrada/salida, artefactos y métricas.
- `data_generation`: parámetros para crear dataset sintético (ciclos, proporciones y severidad de fallos).
- `data_processing`: columna objetivo, configuración temporal (ventana/solape), reglas de etiquetado y creación de estadísticas para la rama fuzzy.
- `model`: hiperparámetros de la rama LSTM y del bloque fuzzy.
- `training`: batch size, learning rate, epochs, scheduler, métrica de monitorización y pesos de pérdida.
- `inference`: parámetros de inferencia (por ejemplo, batch size de predicción).
- `xai`: parámetros de explicabilidad (fondo, reglas/variables top y configuración de acciones correctivas).

## Qué recibe y qué entrega

**Entrada:**

- CSV con series temporales de sensores del horno (13 columnas de sensores + `timestamp` obligatorio + `cycle_id` opcional). La columna `timestamp` se utiliza para ordenar cronológicamente los datos antes de construir las ventanas temporales.
- Las 13 columnas de sensores deben corresponder a las definidas en `data_generation.sensors` de `config/config.yaml`. El orden de las columnas en el CSV no tiene por qué coincidir con el de la configuración, ya que el pipeline reordena automáticamente las columnas de sensores al orden esperado por el modelo y el scaler antes de la inferencia.

**Salida CU43** (detección de anomalías):

- En inferencia estándar (sin `--xai`), se genera un CSV `predictions_YYYYMMDD_HHMMSS.csv` en `data/predictions/` con clase y probabilidad de anomalía.

**Salida CU44** (explicabilidad + acciones, opcional):

- En inferencia con `--xai`, se generan en `data/predictions/xai/`:
  - `xai_predictions_YYYYMMDD_HHMMSS.json`.
  - `xai_predictions_YYYYMMDD_HHMMSS.csv`.

**Ubicación de datos (resumen práctico):**

- `data/input/`: CSV reales del cliente para inferencia/predicción en producción. Se incluye `datos_prueba.csv` como ejemplo de uso para inferencia.
- `data/raw/`: se incluye tanto el dataset sintético para re-entrenamiento (`oven_full_dataset.csv`), como los datasets auxiliares de la capa XAI generados con `python scripts/generate_dataset.py --xai` (estos son: `interpretability_val.csv`, `actions_system_eval.csv` y `xai_background.csv`). En caso de querer re-entrenar con un dataset externo, se debe colocar en esta carpeta.
- `data/splits/`: splits tabulares `train.csv`, `val.csv`, `test.csv` generados por `generate_dataset.py` o `split_external_data.py`, consumidos por el pipeline de procesamiento.
- `data/processed/`: arrays procesados (secuencias y estadísticas) generados por `data_processing.py`, consumidos directamente por el modelo durante el entrenamiento.


## Uso en inferencia (predicciones)

El modelo DNF se entrega ya entrenado y listo para inferencia por parte del cliente. Este es el flujo de uso más habitual en producción. Pasos:

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

### Formato de salida en modo explicabilidad (XAI)

Si se quiere obtener la predicción más un reporte de explicabilidad (capa XAI) en el mismo flujo, ejecutar:
Esto genera en `data/predictions/xai/`:
- `xai_predictions_YYYYMMDD_HHMMSS.csv`: predicciones tabulares (clase y probabilidad de anomalía).
- `xai_predictions_YYYYMMDD_HHMMSS.json`: reporte detallado con estado interpretativo, variables clave y acciones correctivas sugeridas. Ejemplo típico de un caso con alerta/anomalía en el JSON:

```json
  {
    "Índice_de_ventana": 6,
    "Timestamp_cubierto_por_ventana": "2029-04-15 12:00:00 / 2029-04-15 14:59:00",
    "Estado_del_sistema": {
      "Estado_interpretativo": "Normal con señales",
      "Probabilidad_anomalia": 0.399,
      "Umbral_decision": 0.55,
      "Margen_umbral": -0.151
    },
    "Bloques_principales": [
      "combustion_control_flujo",
      "ventilacion_presion"
    ],
    "Variables_clave": [
      "flujo_gas",
      "posicion_valvula",
      "potencia_kw"
    ],
    "Acciones_sugeridas": [
      "Verificar alimentación de gas y respuesta de la válvula de control."
    ]
  },
```

Interpretación rápida:

**Campos siempre presentes:**
- `Índice_de_ventana`: identificador numérico de la ventana temporal.
- `Timestamp_cubierto_por_ventana`: rango temporal que abarca la ventana (formato `"inicio / fin"`).
- `Estado_del_sistema`: agrupa la interpretación del sistema. Contiene siempre tres campos:
  - `Estado_interpretativo`: clasificación del estado del horno. Los cuatro valores posibles son:
    - **Normal**: operación sin señales de fallo. Ambas ramas del modelo predicen normal con margen suficiente y acuerdo entre ellas.
    - **Normal con señales**: predicción general normal, pero con margen bajo o desacuerdo leve entre ramas. Indica que conviene mantener la vigilancia.
    - **Alerta no confirmada**: se detecta posible anomalía pero sin consenso total o margen insuficiente para confirmarla. Requiere atención del operador.
    - **Anomalía confirmada**: ambas ramas predicen fallo con margen suficiente y soporte adecuado. Indica presencia de anomalía consolidada.
  - `Probabilidad_anomalia`: probabilidad estimada de anomalía (valor entre 0 y 1). Valores más altos indican mayor riesgo.
  - `Umbral_decision`: umbral de decisión usado para clasificar la muestra. Si `Probabilidad_anomalia` supera este valor, la muestra se considera anomalía.

**Campos condicionales dentro de `Estado_del_sistema`:**
- `Margen_umbral`: diferencia entre la probabilidad de anomalía y el umbral de decisión. Aparece cuando el estado no es `Normal`.
- `Evidencia_dominante`: indica qué rama del modelo aporta mayor evidencia (`Rama temporal` o `Rama neuro-difusa`). Solo aparece en `Alerta no confirmada`.

**Campos condicionales según estado:**
- `Bloques_principales`: dos grupos de sensores mejor rankeados según la importancia XAI (ej: `combustion_control_flujo`, `ventilacion_presion`). Cada bloque agrupa variables físicas relacionadas del horno. Solo aparece cuando el estado no es `Normal`.
- `Variables_clave`: sensores con mayor contribución a la predicción, extraídos de los bloques principales. Solo aparece cuando el estado no es `Normal`.
- `Acciones_sugeridas`: recomendaciones operativas asociadas a los bloques principales, definidas en `action_config` de `config/config.yaml`. Solo aparece cuando el estado no es `Normal`.
- `Mensaje_operativo`: lista con mensajes operativos indicando funcionamiento correcto. Solo aparece cuando el estado es `Normal`. Ejemplo:
  ```json
  "Mensaje_operativo": [
      "No se requieren acciones correctivas.",
      "Mantener monitorización ordinaria."
  ]
  ```


## Flujo de entrenamiento (dos vías posibles)

Ejecutar solo si se desea reentrenar el modelo.

Prerrequisitos:

- Entorno virtual activado con `env\Scripts\activate`.
- Dependencias instaladas con `pip install -r requirements.txt`.
- Configuración revisada en `config/config.yaml`.

### Vía 1: Datos sintéticos (por defecto)

El primer paso es generar el dataset que se empleará para el re-entrenamiento, así como los datasets auxiliares para explicabilidad (CU44). Hay dos opciones:

**1a. Generar SOLO el dataset sintético para el re-entrenamiento (train=2000 ciclos, val=400 ciclos, test=600 ciclos; 13 sensores; 7 modos):**

```bash
python scripts/generate_dataset.py
```
Esto genera el dataset de entrenamiento `data/raw/oven_full_dataset.csv` y los splits `test.csv`, `train.csv` y `val.csv` en `data/splits`.

**Nota**: el generador sintético se implementa en `src/data_processing/generate_dataset.py` y se configura desde `config/config.yaml`. Si se desea cambiar cantidad de ciclos, proporciones o severidad de fallos, debe hacerse desde ese archivo de configuración.

**1.b Generar tanto el dataset de re-entrenamiento como los datasets específicos para explicabilidad.**

Esto es CRÍTICO para el flujo CU44 (XAI). 
Además del dataset de entrenamiento y los splits, crea:
- `data/raw/interpretability_val.csv`: permite verificar consistencia interpretativa en un escenario balanceado.
- `data/raw/actions_system_eval.csv`: permite probar la lógica de acciones correctivas en una distribución más realista.
- `data/raw/xai_background.csv`: sirve como base para background de SHAP usado por la capa XAI.

```bash
python scripts/generate_dataset.py --xai
```
**Nota**: En un reentrenamiento para CU44, el argumento `--xai` prepara datasets específicos para validar y estabilizar la explicabilidad. Sin estos datasets, el entrenamiento del modelo base puede completarse, pero la validación completa de la capa XAI y de las acciones correctivas de CU44 queda incompleta.

### Vía 2: Datos externos del cliente

Para el reentrenamiento del modelo (sin XAI), en lugar de generar un dataset sintético, se puede colocar un CSV externo en data/raw/ (por ejemplo, data/raw/mi_dataset.csv).
Este dataset debe incluir:
- Las 13 columnas de sensores (ver data_generation.sensors en config.yaml).
- Una columna de etiquetas de fallo (fault_name por defecto).
- `timestamp` (obligatorio, para ordenar cronológicamente).
- Opcionalmente: `cycle_id` (para agrupar ventanas por ciclo y evitar mezclar datos de distintos ciclos).

Después se debe dividir en splits train/val/test. Las proporciones se configuran directamente en `config/config.yaml` bajo `data_processing.external_data_split`:

```yaml
external_data_split:
  train_pct: 66.7
  val_pct: 13.3
  test_pct: 20.0
```

Los tres porcentajes deben sumar 100%. Una vez configurados, ejecutar:

```bash
python scripts/split_external_data.py --input data/raw/mi_dataset.csv
```

### Pipeline común (ambas vías)

**2. Procesar datos: secuencias temporales, escalado, splits**
```bash
python scripts/data_processing.py
```

**3. (Opcional) Encontrar con Optuna la mejor combinación de hiperparámetros para usar después en el entrenamiento del modelo (train.py)**. 

Para ello, ejecutar el notebook `notebooks/tuning_optuna.ipynb`.

**IMPORTANTE**: este paso puede llevar bastante tiempo (varias horas, según configuración y hardware). Una vez terminado, se deben actualizar manualmente los mejores hiperparámetros en el `config/config.yaml`.

**4. Entrenar el modelo Deep Neuro-Fuzzy:**
```bash
python scripts/train.py
```

**5. Visualizar métricas de entrenamiento sobre test:**
```bash
python scripts/get_stats.py
```

> **IMPORTANTE**: cambios en versiones de PyTorch, CUDA o cuDNN, así como diferencias en GPU, drivers, sistema operativo o configuración de ejecución determinista, pueden introducir pequeñas variaciones en los resultados del entrenamiento. La reproducibilidad estricta solo puede garantizarse bajo el mismo entorno software y hardware.

## Notebooks y su funcionalidad

La carpeta `notebooks/` concentra experimentación, análisis y soporte al reentrenamiento:

- `notebooks/EDA_anomaly_detection.ipynb`: análisis exploratorio del dataset (distribución de clases, separabilidad, ANOVA y visualizaciones) para validar calidad de datos y señales antes de entrenar.
- `notebooks/baseline.ipynb`: entrenamiento y comparación de baselines (por ejemplo, Random Forest, LSTM simple y ANFIS) para tener una referencia de desempeño frente al modelo Deep Neuro-Fuzzy escogido.
- `notebooks/tuning_optuna.ipynb`: búsqueda de hiperparámetros del modelo DNF con Optuna. Es clave dentro del reentrenamiento cuando se desea optimizar desempeño antes de ejecutar `scripts/train.py`.
- `notebooks/CU44.ipynb`: notebook principal de la capa XAI (CU44). Permite analizar interpretabilidad, revisar reglas/variables relevantes y validar la generación de acciones correctivas. En el flujo CU44, este notebook es el punto central de validación funcional de explicaciones y recomendaciones/acciones correctivas. Una vez se evalua satisfactoriamente el sistema de generación de acciones correctivas, es preciso escribir los parámetros finales en los campos correspondientes del config/config.yaml (bloque xai) para incluirlo efectivamente dentro del mecanismo de inferencia.

En términos prácticos:

- Si solo se reentrena CU43 (detección), `tuning_optuna.ipynb` es el notebook más importante para mejorar el modelo.
- Si se incluye CU44 (explicabilidad y acciones), `CU44.ipynb` es crítico para validar que la salida XAI sea coherente y operativa.
