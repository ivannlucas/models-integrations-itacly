# DATAGIA — Optimización Energética de Pasteurización mediante IA Neuroevolutiva

---

## Índice

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Instalación y Requisitos](#instalación-y-requisitos)
3. [Ejecución del Pipeline — Guía para el Auditor](#ejecución-del-pipeline--guía-para-el-auditor)
4. [Interpretación de Resultados y Criterios de Éxito](#interpretación-de-resultados-y-criterios-de-éxito)
5. [Arquitectura de la Solución](#arquitectura-de-la-solución)
6. [Estructura del Repositorio](#estructura-del-repositorio)
7. [Datos — Generación y Reproducción](#datos--generación-y-reproducción)
8. [Modelo Predictivo (MLP)](#modelo-predictivo-mlp)
9. [Optimización con Algoritmo Genético (GA mono-objetivo v4)](#optimización-con-algoritmo-genético-ga-mono-objetivo-v4)
10. [Limitaciones, Riesgos y Consideraciones](#limitaciones-riesgos-y-consideraciones)
11. [Estado Actual y Trabajo Pendiente](#estado-actual-y-trabajo-pendiente)
12. [Notebooks — Documentación del Proceso Experimental](#notebooks--documentación-del-proceso-experimental)

---

## Descripción del Proyecto

**DATAGIA** aborda la optimización energética de un proceso industrial de pasteurización de leche mediante un enfoque híbrido **Red Neuronal + Algoritmo Genético**.

El objetivo principal es **reducir el consumo energético específico** ($E_{consumo}/F_{flow}$, kW por L/h procesado) del intercambiador de calor manteniendo la seguridad alimentaria ($T_{out} \geq 72.3°C$ — límite legal 72.0°C + margen de seguridad de +0.3°C para incertidumbre PT100/PMO-FDA).

El sistema funciona en dos fases:
1. **Gemelo Digital (MLP):** Una red neuronal multicapa (PyTorch) aprende la relación entre las variables del proceso y predice el consumo energético y la temperatura de salida.
2. **Motor de Búsqueda (GA mono-objetivo):** Un algoritmo genético (DEAP) utiliza el MLP como función surrogate para encontrar, por cada escenario de planta, el par de setpoints (`F_flow`, `T_servicio`) que minimiza el consumo específico `E_consumo/F_flow`, penalizando las soluciones que incumplen la restricción de seguridad `T_out ≥ 72.3°C`. Se ejecuta en tiempo real por escenario (HallOfFame de tamaño 1, sin frente de Pareto).

### Referencias Científicas Base
- **Tarapata et al. (2025)** — Modelado y predicción de fouling en intercambiadores.
- **Yang et al. (2023)** — Ejemplo de metodología híbrida GA-ANN para optimización de procesos en alimentación.
- **Deka y Datta (2017)** — Minimización de costes operativos en redes de intercambiadores con incrustaciones de leche.
- **Manika et al. (2004)** — Modelado y optimización de pasteurización de leche.
- **Abakarov et al. (2009)** — Optimización térmica en procesamiento de alimentos.

---

## Instalación y Requisitos

### Prerrequisitos
- Python 3.10+
- GPU NVIDIA (opcional) con driver que soporte CUDA ≥ 12.1 (`nvidia-smi` debe reportar `CUDA Version: 12.1` o superior). Sin GPU, la instalación funciona igual y cae automáticamente a CPU.

### Instalación

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd DATAGIA-local

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota sobre CUDA/PyTorch:** `requirements.txt` fija `torch==2.5.1+cu121` / `torchvision==0.20.1+cu121` y declara `--extra-index-url https://download.pytorch.org/whl/cu121`. Esto es intencional: el índice público de PyPI para Windows solo publica builds **CPU-only** de `torch`/`torchvision` bajo el mismo número de versión, así que sin ese índice extra `pip install -r requirements.txt` instalaría en silencio una versión sin soporte CUDA (incluso con GPU disponible), rompiendo la reproducibilidad de `torch.cuda.is_available()` entre máquinas. No elimines esa línea. Si necesitas instalar en una máquina sin GPU o con CUDA distinto, sustituye el sufijo `+cu121` y la URL por la variante deseada (p. ej. `+cpu` con `https://download.pytorch.org/whl/cpu`, o `+cu124` con `https://download.pytorch.org/whl/cu124`).

### Dependencias Principales

| Paquete | Uso |
|---------|-----|
| `torch` (2.5.1) | Red neuronal MLP |
| `deap` | Algoritmo genético mono-objetivo (v4) |
| `pandas`, `numpy` | Manipulación de datos |
| `scikit-learn` | Preprocesado (MinMaxScaler), métricas |
| `matplotlib`, `seaborn` | Visualización |
| `scipy` | Análisis estadístico |
| `tqdm` | Barras de progreso |
| `pyyaml` | Lectura de configuración YAML |

> **Nota:** El entorno virtual `.venv` ya está creado en el repositorio. Activarlo con `.venv\Scripts\activate` antes de ejecutar cualquier comando.

---

## Ejecución del Pipeline

El código del proyecto se expone a través de **cinco funciones del pipeline**, cada una invocable con un único comando. Cada función es independiente y puede ejecutarse por separado, siempre que sus entradas existan. El orden natural de ejecución es:

### Paso 1 — Generación de Datos

```bash
python scripts/data_processing.py
# equivalente: python -m src.main data_processing
```

**Qué hace:** Ejecuta el simulador físico para generar el dataset sintético, filtra los registros de limpieza CIP y crea los splits temporales de entrenamiento/validación/test.

**Entradas:** Ninguna (genera desde cero con semilla fija `seed=1`).

**Artefactos generados:**

| Archivo | Descripción |
|---------|-------------|
| `data/raw/pasteurizacion_dataset_simulado.csv` | Dataset completo del simulador (51 840 registros) |
| `data/processed/final_data_sim.csv` | Solo registros de producción (`Is_Cleaning=0`, referencia actual: 51 369 registros) |
| `data/splits/train.csv` | 70% del dataset de producción (referencia actual: 35 957 registros) |
| `data/splits/val.csv` | 15% del dataset de producción (referencia actual: 7 704 registros) |
| `data/splits/test.csv` | 15% del dataset de producción (referencia actual: 7 708 registros) |

> ⚠️ Este paso regenera los datos desde cero. Si los artefactos de `data/` ya existen, se sobreescriben. La semilla `seed=1` garantiza reproducibilidad.

---

### Paso 2 — Estadísticas y Baseline

```bash
python scripts/get_stats.py
# equivalente: python -m src.main get_stats
```

**Qué hace:** Imprime el inventario de columnas del dataset y calcula los KPIs de referencia de la operación histórica con PID.

**Entradas:** `data/processed/final_data_sim.csv`

**Artefactos generados:**

| Archivo | Descripción |
|---------|-------------|
| `models/metrics/baseline_metrics.json` | KPIs de la operación histórica (consumo, caudal, temperaturas) |

**Salida esperada en consola** (resumen del JSON generado):

```
E_consumo_mean_kW        : 435.14
F_flow_mean_Lh           : 4917.5
specific_consumption     : 0.08845 kW/(L/h)
T_out_min_C              : 72.18 °C
compliance_rate_pct      : 99.6 %
```

Estos son los valores de referencia contra los que se mide la mejora del sistema IA.

---

### Paso 3 — Entrenamiento del Modelo MLP

```bash
python scripts/train.py
# equivalente: python -m src.main train
```

**Qué hace:** Carga los datos procesados, normaliza con MinMaxScaler (ajuste solo en train), entrena la `DynamicMLP` con early stopping y evalúa en el set de test.

**Entradas:** `data/processed/final_data_sim.csv`, `models/artifacts/model_config.json` (hiperparámetros; usa fallback si no existe).

**Artefactos generados:**

| Archivo | Descripción |
|---------|-------------|
| `models/artifacts/mlp_predictor.pt` | Pesos del modelo entrenado |
| `models/artifacts/model_config.json` | Arquitectura y parámetros del modelo |
| `models/artifacts/scaler_X.pkl` | Scaler de features (MinMaxScaler) |
| `models/artifacts/scaler_y.pkl` | Scaler de targets (MinMaxScaler) |
| `models/metrics/train_metrics.json` | Hiperparámetros usados y RMSE, MAE, R² por variable objetivo, desglosados en `train`, `val` y `test` |


**Valores de referencia ya obtenidos (`train_metrics.json`, hiperparámetros tuneados: 3 capas, 64 neuronas, ReLU, lr=0.0005):**

| Split | `E_consumo` (RMSE / MAE / R²) | `T_out_leche` (RMSE / MAE / R²) |
|-------|-------------------------------|----------------------------------|
| train | 5.3543 kW / 4.2453 kW / 0.9783 | 0.0635 °C / 0.0474 °C / 0.3779 |
| val   | 5.3504 kW / 4.2300 kW / 0.9783 | 0.0634 °C / 0.0473 °C / 0.3725 |
| test  | 5.3838 kW / 4.2571 kW / **0.9779** | 0.0643 °C / 0.0473 °C / **0.3759** |

- Las tres particiones son consistentes entre sí (sin señales de overfitting).
- `E_consumo` se predice con muy alta precisión (R²≈0.98). `T_out_leche` tiene un R² bajo porque el PID/supervisor la mantiene casi constante en un rango muy estrecho (72.18–73.91 °C, σ≈0.08 °C) — con tan poca varianza que explicar, el R² es sensible incluso a errores absolutos pequeños (MAE≈0.047 °C).
- Para comparación final y reporting de generalización se usa habitualmente el bloque `test`.

---

### Paso 4 — Inferencia (Predicciones)

```bash
python scripts/predict.py
# equivalente:
python -m src.main predict --input data/splits/test.csv --output data/predictions/predictions.csv
```

**Qué hace:** Carga los artefactos del modelo y ejecuta inferencia batch sobre el CSV de entrada. Por defecto usa `data/splits/test.csv`.

**Entradas:** `models/artifacts/` (modelo + scalers), CSV con columnas `T_in_leche`, `F_flow`, `T_servicio`, `t_ciclo`, `Delta_P`.

**Artefactos generados:**

| Archivo | Descripción |
|---------|-------------|
| `data/predictions/predictions.csv` | CSV original con columnas `E_consumo_pred` y `T_out_pred` añadidas |

---

### Paso 5 — Optimización GA (Backtesting en tiempo real)

```bash
python scripts/optimize.py
# equivalente:
python -m src.main optimize --input data/splits/test.csv --output data/predictions/evaluation_rt_hist_vs_ia.csv
```

**Qué hace:** Para cada fila del CSV de entrada, ejecuta un algoritmo genético mono-objetivo (DEAP) con el MLP como función surrogate: minimiza el consumo específico `E_consumo/F_flow`, penalizando las soluciones con `T_out < 72.3°C`, y selecciona el mejor individuo con `HallOfFame(1)` (no hay frente de Pareto ni objetivo de producción independiente). Si el CSV contiene columnas históricas (`F_flow`, `T_servicio`, `E_consumo`, `T_out_leche`), añade columnas de comparación directa IA vs histórico.

**Entradas:** `models/artifacts/` (modelo + scalers), CSV con columnas `T_in_leche`, `Delta_P`, `t_ciclo`.

**Artefactos generados:**

| Archivo | Descripción |
|---------|-------------|
| `data/predictions/evaluation_rt_hist_vs_ia.csv` | Setpoints IA, predicciones y comparativa vs histórico por fila |
| `models/metrics/evaluation_rt_backtesting_report.json` | KPIs agregados de la comparativa (solo si el CSV de entrada trae columnas históricas) |

> ⚠️ Este paso es computacionalmente costoso. Para el set de test completo (7 708 instancias) tarda del orden de 80-90 minutos (~650-680 ms/instancia, depende de la máquina). Para una verificación rápida, pasar un subconjunto con `--input`.

**KPIs esperados** (valores de referencia en `evaluation_rt_backtesting_report.json`, 7 708 instancias):

| KPI | Histórico (PID) | IA (GA mono-objetivo) | Δ |
|-----|----------------|--------------|---|
| `E_consumo` medio | 435.4 kW | 426.27 kW | **−1.48%** |
| `F_flow` medio | 4 920.6 L/h | 5 398.8 L/h | **+9.72%** |
| Consumo específico `E/F` | 0.088448 | 0.078960 | **−10.73% ✅** |
| `T_servicio` medio | 81.55 °C | 81.35 °C | −0.19 °C |
| Cumplimiento `T_out ≥ 72.3 °C` | 99.42% | **100% ✅** | — |

---

## Interpretación de Resultados y Criterios de Éxito

### Criterios de aceptación del proyecto

El proyecto define dos criterios de éxito en `config/config.yaml`:

| Criterio | Umbral | Estado |
|----------|--------|--------|
| Tasa de cumplimiento `T_out ≥ 72.3 °C` | 100% | ✅ Cumplido (backtesting IA sobre test) |
| Mejora en consumo específico `E/F` vs baseline | ≥ 3% | ✅ Superado (+10.73%) |

### Cómo leer `train_metrics.json`

```json
{
    "train": {
        "E_consumo":   { "RMSE": 5.3543, "MAE": 4.2453, "R2": 0.9783 },
        "T_out_leche": { "RMSE": 0.0635, "MAE": 0.0474, "R2": 0.3779 }
    },
    "val": {
        "E_consumo":   { "RMSE": 5.3504, "MAE": 4.2300, "R2": 0.9783 },
        "T_out_leche": { "RMSE": 0.0634, "MAE": 0.0473, "R2": 0.3725 }
    },
    "test": {
        "E_consumo":   { "RMSE": 5.3838, "MAE": 4.2571, "R2": 0.9779 },
        "T_out_leche": { "RMSE": 0.0643, "MAE": 0.0473, "R2": 0.3759 }
    }
}
```

- **RMSE y MAE** están en las unidades originales (kW para energía, °C para temperatura). El archivo incluye los tres bloques `train`, `val` y `test`; para comparación final y reporting de generalización se usa habitualmente el bloque `test`.
- Los hiperparámetros de la arquitectura ganadora (3 capas, 64 neuronas, ReLU, lr=0.0005), encontrados por la búsqueda aleatoria en `predict_model_t_h.ipynb` (no volver a ejecutar esa búsqueda), se guardan aparte en `models/artifacts/model_config.json`, que `scripts/train.py` lee y reutiliza directamente.
- `T_out_leche` tiene un R² bajo (~0.37-0.38) porque el PID/supervisor la mantiene casi constante (σ≈0.08 °C) en las tres particiones; el MAE absoluto (~0.047 °C) es el indicador más informativo de la calidad de esa predicción.

### Cómo leer `predictions.csv`

La salida `data/predictions/predictions.csv` conserva las columnas originales del CSV de entrada y añade las predicciones del modelo.

- **Filas esperadas:** exactamente el mismo número de filas que el archivo de entrada.
- **Caso por defecto:** si se ejecuta `python scripts/predict.py` (entrada por defecto `data/splits/test.csv`), el CSV de salida tendrá el mismo número de filas que `test.csv` (actualmente 7 708).

**Columnas e interpretación (caso por defecto con `test.csv`):**

| Columna | Interpretación |
|---------|----------------|
| `T_in_leche` | Temperatura de entrada de la leche (°C) en el instante evaluado |
| `F_flow` | Caudal histórico de planta (L/h) |
| `T_servicio` | Temperatura de servicio histórica (°C) |
| `t_ciclo` | Tiempo desde último CIP (min) |
| `Delta_P` | Caída de presión (bar), proxy de ensuciamiento |
| `E_consumo` | Consumo energético histórico/real de referencia (kW) |
| `T_out_leche` | Temperatura de salida histórica/real (°C) |
| `E_consumo_pred` | Predicción del MLP para consumo energético (kW) |
| `T_out_pred` | Predicción del MLP para temperatura de salida (°C) |

> Si se usa un CSV de entrada personalizado con columnas adicionales, esas columnas también se conservan en la salida.

### Cómo leer `evaluation_rt_backtesting_report.json`

La métrica clave sigue siendo `mejora_pct` en `kpi_eficiencia_especifica`:

```json
"kpi_eficiencia_especifica": {
    "E_F_hist_medio": 0.088448,   ← consumo por unidad producida, operación PID
    "E_F_ia_medio":   0.078960,   ← consumo por unidad producida, sistema IA
    "mejora_pct":     10.73       ← % de mejora (criterio de éxito: ≥ 3%)
}
```

**Interpretación:** En este run (7 708 instancias de test), la IA **reduce tanto el consumo absoluto como el específico**, y además procesa más caudal. Esto se refleja en:
- `kpi_energia.ahorro_medio_pct` = +1.48% de ahorro en consumo absoluto.
- `kpi_produccion.delta_F_flow_pct` = +9.72% más caudal procesado.
- `kpi_T_servicio.delta_T_servicio` = −0.19 °C en temperatura de servicio (prácticamente sin cambios).
- `kpi_seguridad.cumplimiento_ia_pct` = 100% (vs 99.42% histórico) — la restricción de seguridad alimentaria nunca se viola con los setpoints IA.

### Cómo leer `evaluation_rt_hist_vs_ia.csv`

El archivo `data/predictions/evaluation_rt_hist_vs_ia.csv` contiene la recomendación IA y la comparación contra histórico para cada escenario evaluado.

- **Filas esperadas:** exactamente el mismo número de filas que el CSV de entrada (`1 fila de entrada -> 1 fila de salida`).
- **Caso por defecto:** con `python scripts/optimize.py` (entrada `data/splits/test.csv`), la salida tiene 7 708 filas.

**Columnas e interpretación (salida por defecto, cuando la entrada incluye histórico):**

| Columna | Descripción |
|---------|-------------|
| `T_in_leche` | Condición de planta no controlable: temperatura de entrada (°C) |
| `Delta_P` | Condición de planta no controlable: caída de presión (bar) |
| `t_ciclo` | Condición de planta no controlable: tiempo desde CIP (min) |
| `IA_F_flow` | Setpoint óptimo recomendado por IA para caudal (L/h) |
| `IA_T_servicio` | Setpoint óptimo recomendado por IA para temperatura de servicio (°C) |
| `IA_E_consumo` | Consumo energético predicho con los setpoints IA (kW) |
| `IA_T_out` | Temperatura de salida predicha con setpoints IA (°C) |
| `IA_consumo_especifico` | Consumo específico IA: `IA_E_consumo / IA_F_flow` |
| `IA_factible` | `True` si `T_out ≥ 72.3 °C`, `False` en caso contrario |
| `fitness_final` | Valor de fitness del mejor individuo (`HallOfFame(1)`) al final del GA |
| `HIST_F_flow` | Caudal histórico de referencia (L/h) |
| `HIST_T_servicio` | Temperatura de servicio histórica (°C) |
| `HIST_E_consumo` | Consumo energético histórico (kW) |
| `HIST_T_out` | Temperatura de salida histórica (°C) |
| `Ahorro_kW` | Diferencia absoluta de consumo: `HIST_E_consumo - IA_E_consumo` |
| `Ahorro_pct` | % de ahorro energético respecto al histórico |
| `HIST_Eficiencia` | Consumo específico histórico: `HIST_E_consumo / HIST_F_flow` |
| `IA_Eficiencia` | Consumo específico IA: `IA_E_consumo / IA_F_flow` |
| `Mejora_Eficiencia` | `HIST_Eficiencia - IA_Eficiencia` (positivo = mejora IA) |
| `Delta_F_flow` | Cambio de caudal propuesto: `IA_F_flow - HIST_F_flow` |
| `Delta_T_servicio` | Cambio de temperatura de servicio: `IA_T_servicio - HIST_T_servicio` |

> Si el CSV de entrada no contiene columnas históricas (`F_flow`, `T_servicio`, `E_consumo`, `T_out_leche`), la salida mantiene las columnas IA y no incluye los campos comparativos `HIST_*`, `Ahorro_*`, `Eficiencia` y `Delta_*`.

---

## Arquitectura de la Solución

```
┌─────────────────────────────────────────────────────────┐
│              Condiciones de Planta (reales)             │
│    T_in_leche · t_ciclo · Delta_P (no controlables)     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  GA mono-objetivo (DEAP)     │
        │  Variables de decisión:      │
        │   • F_flow  [3500–5500 L/h]  │
        │   • T_servicio [76–95 °C]    │
        │                              │
        │  Objetivo:                   │
        │   • min E_consumo / F_flow   │
        │     (consumo específico)     │
        │                              │
        │  Restricción (penalización): │
        │   • T_out ≥ 72.3 °C          │
        └──────────┬───────────────────┘
                   │ evalúa candidatos
                   ▼
        ┌──────────────────────────────┐
        │     MLP Surrogate (PyTorch)  │
        │  5 inputs → ANN (MLP) → 2    │
        │  Predice: E_consumo, T_out   │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │   HallOfFame(1)  (GA)        │
        │   Mejor individuo por        │
        │   fitness (min E/F, penal.)  │
        │   → Setpoints óptimos        │
        └──────────────────────────────┘
```

---

## Estructura del Repositorio

```
DATAGIA-local/
│
├── README.md                    ← Este archivo
├── requirements.txt             ← Dependencias del proyecto
│
├── config/
│   └── config.yaml              ← Configuración centralizada (hiperparámetros, rutas, GA, criterios)
│
├── scripts/                     ← Scripts ejecutables del pipeline
│   ├── data_processing.py       ← Genera datos, filtra y crea splits
│   ├── train.py                 ← Entrena el modelo MLP
│   ├── predict.py               ← Ejecuta inferencia sobre un dataset
│   ├── optimize.py              ← Ejecuta el GA mono-objetivo por escenario y guarda recomendaciones
│   └── get_stats.py             ← Calcula estadísticas y baseline KPIs
│
├── data/                        ← Datos del proyecto (presentes en esta copia local)
│   ├── raw/
│   │   └── pasteurizacion_dataset_simulado.csv   ← Dataset sintético (51 840 registros)
│   ├── processed/
│   │   └── final_data_sim.csv                    ← Dataset procesado (sin CIP, limpio)
│   ├── splits/
│   │   ├── train.csv                             ← Conjunto de entrenamiento
│   │   ├── val.csv                               ← Conjunto de validación
│   │   └── test.csv                              ← Conjunto de test
│   ├── predictions/
│   │   ├── predictions.csv                       ← Salida por defecto de scripts/predict.py
│   │   ├── evaluation_rt_hist_vs_ia.csv          ← Salida por defecto de scripts/optimize.py
│   │   └── ga_v4_optimization_results.csv        ← Resultados GA v4 sobre malla de escenarios (histórico, no operativo)
│   └── images/                                   ← Gráficos generados
│
├── models/
│   ├── artifacts/
│   │   ├── mlp_predictor.pt                      ← Pesos del modelo MLP entrenado
│   │   ├── model_config.json                     ← Configuración arquitectura del MLP
│   │   ├── scaler_X.pkl                          ← Scaler de features (MinMaxScaler)
│   │   └── scaler_y.pkl                          ← Scaler de targets (MinMaxScaler)
│   ├── artifacts_debug/                          ← Artefactos y configs en fase de desarrollo/debug
│   └── metrics/
│       ├── train_metrics.json                    ← Métricas de entrenamiento del MLP (test set)
│       ├── baseline_metrics.json                 ← KPIs del proceso histórico (PID)
│       ├── evaluation_rt_backtesting_report.json ← Evaluación en tiempo real (GA mono-objetivo vs histórico)
│       └── ga_v3_optimization_report.json, ga_v4_optimization_report.json  ← Reportes de versiones anteriores del GA (histórico)
│
├── notebooks/
│   ├── EDA/                       ← Exploración y generación de datos
│   ├── training/                  ← Entrenamiento MLP + Optimización GA
│   │   └── old_versions/          ← Trabajos de iteraciones previas del Algoritmo Genético
│   ├── tuning_hyperparameters/    ← Ajuste de hiperparámetros
│   ├── evaluacion/                ← Baseline y evaluación final
│   └── opcional/                  ← Material didáctico complementario
│
├── src/                           ← Código fuente modular
│   ├── __init__.py
│   ├── main.py                    ← Punto de entrada: train(), predict(), optimize(), data_processing(), get_stats()
│   ├── utils/
│   │   ├── constants.py           ← Constantes físicas, GA y listas de features/targets
│   │   ├── paths.py               ← Rutas centralizadas del proyecto
│   │   ├── reproducibility.py     ← Semillas y detección CPU/GPU
│   │   └── logging.py             ← Logger centralizado (get_logger)
│   ├── data_processing/
│   │   ├── simulator.py           ← Gemelo Digital de la planta pasteurizadora
│   │   └── preprocessing.py       ← Carga, filtrado, split temporal y normalización
│   ├── training/
│   │   ├── model.py               ← DynamicMLP (red neuronal configurable)
│   │   ├── trainer.py             ← Bucle de entrenamiento con early stopping
│   │   ├── hyperparameter_search.py ← Random Search
│   │   └── artifacts.py           ← Guardar/cargar modelo, scalers y config
│   ├── predict/
│   │   ├── inference.py           ← Inferencia individual, batch y save_predictions
│   │   ├── optimization.py        ← GA mono-objetivo v4, min E/F con penalización (DEAP)
│   │   └── lookup.py              ← Consulta a la Lookup Table precomputada
│   └── get_stats/
│       ├── metrics.py             ← RMSE, MAE, R² por variable objetivo
│       ├── baseline.py            ← KPIs de referencia del proceso PID
│       └── column_info.py         ← Inventario de columnas del dataset
│
└── documentation/
    └── datagia_lit_cient/         ← Literatura científica recopilada
```

---

## Datos — Generación y Reproducción

> **Nota de auditoría:** En esta copia de trabajo la carpeta `data/` está disponible. Si en una publicación se excluyera por tamaño o política de datos, puede regenerarse ejecutando los notebooks/scripts en el orden descrito abajo.

### ¿Por qué no hay datos públicos?

No existen datasets públicos que combinen la **bioquímica de la leche** (ensuciamiento por β-lactoglobulina) con la **termodinámica de intercambiadores de calor** y el **consumo energético** detallado. Los datos SCADA reales de plantas lecheras son propiedad industrial protegida. Por ello se optó por un enfoque **Sim2Real**: generar datos sintéticos mediante un simulador basado en ecuaciones físicas validadas por la literatura científica.

### Pipeline de Datos (cómo reproducir `data/`)

La siguiente tabla describe el flujo completo de generación de datos. **Ejecutar los notebooks en este orden recrea toda la carpeta `data/`:**

| Paso | Notebook | Entrada | Salida | Descripción |
|------|----------|---------|--------|-------------|
| **1** | `notebooks/EDA/data_creation.ipynb` | — (generación desde cero) | `data/raw/pasteurizacion_dataset_simulado.csv` | Simulador de planta de pasteurización basado en física (V3.4, PID/supervisor de lazo abierto). Genera 51 840 registros (180 días, muestreo cada 5 min). Semilla: `np.random.seed(1)` |
| **2** | `notebooks/EDA/eda.ipynb` | `data/raw/pasteurizacion_dataset_simulado.csv` | `data/processed/final_data_sim.csv` | Análisis Exploratorio. Filtra registros de limpieza CIP (`Is_Cleaning == 0`, quedan 51 369 registros) y guarda el dataset limpio |
| **3** | `notebooks/tuning_hyperparameters/predict_model_t_h.ipynb` | `data/processed/final_data_sim.csv` | `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`, `models/artifacts/model_config.json` | Split temporal por cuartiles (70/15/15 → 35 957 / 7 704 / 7 708 registros) y búsqueda aleatoria de hiperparámetros del MLP. **No volver a ejecutar la búsqueda**: no es reproducible byte a byte y su resultado (3 capas, 64 neuronas, ReLU, lr=0.0005) ya está guardado en `model_config.json`; `scripts/train.py` lo reutiliza directamente |
| **4** | `notebooks/evaluacion/ga_evaluation.ipynb` | `data/splits/test.csv` + modelo MLP | `data/predictions/evaluation_rt_hist_vs_ia.csv`, `models/metrics/evaluation_rt_backtesting_report.json` | Backtesting en tiempo real: GA mono-objetivo (min `E/F`, `HallOfFame(1)`) ejecutado por escenario, comparativa punto a punto histórico vs IA |

> **Nota:** Los notebooks de versiones anteriores del GA (`training/old_versions/optimization_ga.ipynb`, `optimization_ga_v2.ipynb`, `training/optimization_ga_v3.ipynb`) y la malla de escenarios precomputada (`ga_v3_optimization_results.csv`, `ga_v4_optimization_results.csv`) son material de referencia histórico de iteraciones anteriores (NSGA-II bi-objetivo, lookup table por malla). La arquitectura actual y operativa es el GA mono-objetivo v4 ejecutado en tiempo real por escenario, documentado en `ga_evaluation.ipynb` y encapsulado en `src/predict/optimization.py`.

### Simulador Físico (Paso 1 en detalle)

En el notebook `data_creation.ipynb` se desarrolló un **Gemelo Digital (Digital Twin)** basado en física que simula un intercambiador de calor de placas (PHE) en configuración contracorriente bajo régimen transitorio. No genera datos aleatorios: resuelve iterativamente las ecuaciones diferenciales de transferencia de calor y deposición de proteínas descritas en la literatura científica del estado del arte, garantizando que las correlaciones aprendidas por el modelo de ML posterior respeten las leyes de la termodinámica. No hace falta ejecutarlo para la prueba del codigo del repositorio, puesto que la simulación de datos está incluida en el archivo `src/data_processing/simulator.py` (función `generar_dataset_pasteurizacion()`), pero se recomienda su visualización para entender el proceso de simulación de la planta.

**Referencias Científicas Base:**
- **Modelado de Fouling:** *Tarapata et al. (2025)* - "Approaches for Measuring and Predicting Fouling During Thermal Processing of Dairy Solutions".
- **Dinámica del Proceso:** *Manika et al. (2004)* - "Modelling and optimisation of milk pasteurisation processes".
- **Optimización de Costes:** *Deka & Datta (2017)* - "Operational cost minimization in heat exchanger network under milk fouling".

#### A. Modelo de Transferencia de Calor (ε-NTU)

En lugar de utilizar la media logarítmica de temperatura (LMTD) que asume estado estacionario, se utiliza el método **Efectividad-NTU**, más adecuado para simulaciones donde el coeficiente global de transferencia ($U$) cambia con el tiempo debido a la suciedad.

La resistencia térmica total ($R_{total}$) aumenta progresivamente:

$$\frac{1}{U_{sucio}(t)} = \frac{1}{U_{limpio}} + R_f(t)$$

Donde $R_f(t)$ es el **Factor de Ensuciamiento** (Fouling Factor) en el tiempo $t$.

La transferencia de calor real ($Q$) se calcula como:

$$Q = \epsilon \cdot C_{min} \cdot (T_{servicio} - T_{in\_leche})$$

Donde $\epsilon$ (efectividad) depende del número de unidades de transferencia ($NTU = \frac{U \cdot A}{C_{min}}$). Para intercambiador de placas contracorriente (asumiendo $C_r \approx 1$):

$$\epsilon = \frac{NTU}{1 + NTU}$$

#### B. Modelo Dinámico de Ensuciamiento (Fouling)

Basado en *Tarapata et al. (2025)* y *Manika (2004)*, el ensuciamiento se modela como una competencia entre la **deposición química** (desnaturalización de β-lactoglobulina) y la **remoción mecánica** (esfuerzo cortante del flujo):

$$\frac{dR_f}{dt} = \underbrace{k_d \cdot \exp\left(\frac{-E_a}{R \cdot T_{film}}\right)}_{\text{Deposición Térmica}} - \underbrace{k_r \cdot \tau_{wall}}_{\text{Arrastre por Flujo}}$$

Simplificado en el simulador como:

$$\frac{dR_f}{dt} = \frac{k_{dep} \cdot \exp(\alpha \cdot T_{servicio})}{F_{flow} / F_{ref}}$$

- **Implicación:** Si aumentamos la temperatura del servicio ($T_{servicio}$), la deposición crece exponencialmente. Si aumentamos el caudal ($F_{flow}$), la limpieza por arrastre aumenta.

#### C. Modelo Hidráulico (Caída de Presión)

La caída de presión ($\Delta P$) es el indicador físico principal del bloqueo de las placas. Se modela siguiendo la ecuación de Darcy-Weisbach adaptada a canales rectangulares con obstrucción variable:

$$\Delta P(t) = k_{geo} \cdot \dot{m}^2 \cdot (1 + \alpha \cdot R_f(t))$$

- La presión aumenta con el **cuadrado del flujo** y linealmente con el espesor de la suciedad ($R_f$).

#### D. Modelo Energético Total

El consumo energético total ($E_{consumo}$) se calcula como la suma de dos componentes principales:

$$E_{consumo} = \frac{Q_{térmico}}{\eta_{caldera}} + \frac{\dot{V} \cdot \Delta P}{\eta_{bomba}}$$

Donde:
- **Consumo térmico:** Energía necesaria para calentar la leche, dividida por la eficiencia de la caldera ($\eta_{caldera} = 0.90$).
- **Consumo de bombeo:** Potencia hidráulica requerida para vencer la caída de presión, dividida por la eficiencia de la bomba ($\eta_{bomba} = 0.75$).

> **Implicación del fouling:** A medida que $R_f$ crece, tanto $Q_{térmico}$ (por compensación del PID) como $\Delta P$ (por obstrucción) aumentan, generando una **doble penalización energética**.

#### E. Lógica del Agente de Control (Heurística PID de Lazo Abierto)

El simulador incluye un "agente" que aproxima, sin invertir el balance térmico, el comportamiento de un PLC industrial tradicional:

$$T_{servicio} = T_{base} + K_{f} \cdot R_f(t) + K_{flujo} \cdot (F_{flow} - F_{nominal}) + K_{Tin} \cdot (4 - T_{in\_leche}) + \eta_{PID}$$

con $T_{base}=79°C$, $K_f=25000$, $K_{flujo}=0.0010$, $K_{Tin}=0.12$ y $\eta_{PID} \sim N(0, 0.40)$.

1. **Objetivo aproximado:** Mantener $T_{out\_leche}$ cerca del umbral de pasteurización.
2. **Perturbación:** A medida que $R_f$ crece, la transferencia de calor cae y $T_{out\_leche}$ tiende a bajar.
3. **Reacción:** La heurística compensa aumentando $T_{servicio}$ en proporción al fouling acumulado, al caudal y a la temperatura de entrada.
4. **Consecuencia:** Aumenta el **Consumo Energético ($E_{consumo}$)** para lograr un resultado similar en la leche.
5. **Guardia final:** Al ser una heurística de lazo abierto (no una inversión exacta del balance), en los casos extremos donde $T_{out\_leche}$ caería por debajo de $72.3°C$ se recalcula $T_{servicio}$ invirtiendo el balance térmico hacia el setpoint de control ($72.6°C$). Esto no fuerza el cumplimiento al 100%: ~0.42% de los registros brutos (220 de 51 840) quedan por debajo de $72.3°C$, replicando el ruido de una planta real.

> **Objetivo del Proyecto Datagia:** Entrenar una IA que detecte este patrón y proponga configuraciones de flujo y temperatura que retrasen la formación de $R_f$, reduciendo así el consumo específico.

**Parámetros de simulación:**
- Duración: 180 días · Frecuencia: 5 min · Total: 51 840 registros
- $U_{limpio} = 3500$ W/m²K · Área = 15 m² · $c_p$ leche = 3890 J/kgK · $\rho$ leche = 1030 kg/m³
- Eficiencia caldera: 90% · Eficiencia bomba: variable según curva BEP (máx. 78% en 4800 L/h)
- Caudal: muestreo por rechazo, $N(5000, 400)$ L/h truncado a $[3500, 5500]$ L/h
- Pérdidas fijas independientes del caudal: $P_{fixed}=15$ kW · Penalización de canalización energética por encima de 5150 L/h
- CIP (limpieza): se dispara al cumplir 9h de ciclo o fouling > 0.0008 m²K/W
- Semilla: `np.random.seed(1)` → **resultados reproducibles**

### Dataset Generado

| Columna | Símbolo | Descripción | Tipo | Unidad |
|---------|---------|-------------|------|--------|
| `Time_min` | $t$ | Marca temporal de simulación | Índice | min |
| `T_in_leche` | $T_{c,in}$ | Temperatura de entrada de la leche cruda (estacional + ruido) | Feature — perturbación | °C |
| `F_flow` | $\dot{V}$ | Caudal volumétrico de leche | Feature — controlable | L/h |
| `T_servicio` | $T_{h,in}$ | Temperatura del fluido de calentamiento (manipulada por PID) | Feature — controlable | °C |
| `t_ciclo` | $t_{cip}$ | Tiempo acumulado sin limpieza CIP | Feature — estado | min |
| `Delta_P` | $\Delta P$ | Caída de presión en el intercambiador (indicador de fouling) | Feature — estado | bar |
| `E_consumo` | $P_{total}$ | Consumo energético instantáneo: térmico + bombeo | **Target (KPI)** | kW |
| `T_out_leche` | $T_{c,out}$ | Temperatura de salida de la leche pasteurizada | **Target / Restricción** (≥72.3°C — incluye margen PT100/PMO) | °C |
| `Is_Cleaning` | — | Flag de ciclo de limpieza química CIP | Auxiliar (filtrado) | 0/1 |

### Estructura de `data/` y Split Temporal

```
data/
├── raw/
│   └── pasteurizacion_dataset_simulado.csv    → 51 840 registros (incluye CIP)
├── processed/
│   └── final_data_sim.csv                     → 51 369 registros (solo producción, Is_Cleaning=0)
├── splits/                                    → Split temporal estratificado por cuartiles
│   ├── train.csv                              → 70% (35 957 registros)
│   ├── val.csv                                → 15% (7 704 registros)
│   └── test.csv                               → 15% (7 708 registros)
├── predictions/                               → Salidas de optimización y evaluación
│   ├── ga_v4_optimization_results.csv         → Setpoints óptimos por escenario de malla (histórico, no operativo)
│   └── evaluation_rt_hist_vs_ia.csv           → Comparativa punto a punto hist. vs IA (GA mono-objetivo, operativo)
└── images/                                    → Gráficos generados por los notebooks
```

**Estrategia de split:** Se divide temporalmente por **4 cuartiles** del dataset. Dentro de cada cuartil se toman 70% train / 15% val / 15% test. Esto garantiza que cada split contenga muestras representativas de **todos los estados del ciclo de fouling** (limpio → sucio), evitando el sesgo de un split secuencial simple.

**Normalización:** MinMaxScaler [0, 1] ajustado **únicamente sobre el conjunto de entrenamiento** y aplicado a val/test (sin data leakage).

---

## Modelo Predictivo (MLP)

Red neuronal multicapa (feed-forward) entrenada con PyTorch para actuar como **función surrogate** rápida del proceso físico.

### Arquitectura

| Parámetro | Valor |
|-----------|-------|
| Entradas | 5 (`T_in_leche`, `F_flow`, `T_servicio`, `t_ciclo`, `Delta_P`) |
| Salidas | 2 (`E_consumo`, `T_out_leche`) |
| Capas ocultas | Dinámicas (según `models/artifacts/model_config.json`) |
| Neuronas por capa | Dinámicas (según `models/artifacts/model_config.json`) |
| Activación | Dinámica (según `models/artifacts/model_config.json`) |
| Learning rate | Dinámico (según `models/artifacts/model_config.json`) |
| Normalización | MinMaxScaler [0, 1] |

**Notas de entrenamiento**

- **Modo tuneado (actual por defecto):** si existe `models/artifacts/model_config.json`, `src/main.py` entrena `DynamicMLP` con esos hiperparámetros, usando `batch_size=128` y `patience=15`.
- **Fallback por defecto:** si no existe `model_config.json`, se usan los valores por defecto del proyecto (2 capas, 128 neuronas, `ReLU`, `lr=0.0005`, `batch_size=128`, `patience=15`).

### Model Card

| Campo | Detalle |
|-------|---------|
| Nombre | DynamicMLP v1.0 |
| Formato | PyTorch `state_dict` (`.pt`) |
| Librería | PyTorch 2.5.1 |
| Parámetros totales | Variables (según arquitectura activa en `model_config.json`) |
| Tamaño en disco | Variable (según arquitectura activa) |
| Requisitos ejecución | Python 3.10+, CPU (GPU opcional con build PyTorch compatible) |
| Criterios de éxito | Cumplimiento T_out ≥ 72.3°C al 100% (incluye margen PT100/PMO), mejora consumo específico ≥ 3% vs baseline |

### Métricas de Entrenamiento (test set)

| Métrica | `E_consumo` | `T_out_leche` |
|---------|-------------|---------------|
| RMSE    | 5.3838 kW   | 0.0643 °C     |
| MAE     | 4.2571 kW   | 0.0473 °C     |
| R²      | 0.9779      | 0.3759        |

El modelo predice `E_consumo` con precisión alta (R²≈0.98). Para `T_out_leche` el R² es bajo (0.376) no porque el modelo prediga mal en términos absolutos (MAE≈0.047°C), sino porque el PID/supervisor mantiene esa variable casi constante (σ≈0.08°C en el dataset de producción) — con tan poca varianza que explicar, el R² es una métrica poco informativa; el MAE en unidades reales es la referencia más útil para esta variable.

---

## Optimización con Algoritmo Genético (GA mono-objetivo v4)

El algoritmo genético se desarrolló iterativamente; las versiones `v1`–`v3` (mono-escenario, bi-objetivo con frente de Pareto, y malla completa de escenarios con lookup table) quedan documentadas como material histórico en `notebooks/training/old_versions/` y `notebooks/training/optimization_ga_v3.ipynb`. **La versión actual y operativa es la v4, mono-objetivo**, implementada en `src/predict/optimization.py` y ejecutada en tiempo real por escenario (no precomputada en malla).

### Formulación del problema (v4)

- **Variables de decisión:** `F_flow` ∈ [3500, 5500] L/h, `T_servicio` ∈ [76, 95] °C.
- **Función objetivo:** minimizar el consumo específico `E_consumo / F_flow` (kW por L/h), donde `E_consumo` y `T_out` se predicen con el MLP surrogate.
- **Restricción (vía penalización, no rechazo duro):** si `T_out < 72.3°C`, se penaliza el fitness con `10000 + (E/F) × (1 + 10 × déficit)`, guiando al GA evolutivamente hacia la región factible en vez de descartar directamente los individuos.
- **Configuración DEAP:** población=150, generaciones=15, cruce `cxBlend` (α=0.5) con `cxpb=0.8`, mutación `mutGaussian` (μ=0, σ=0.2, `indpb=0.2`) con `mutpb=0.2`, selección por torneo (`tournsize=3`), elitismo con `HallOfFame(1)` (se conserva el mejor individuo, no un frente de Pareto).
- **Reproducibilidad por escenario:** para la fila `i` del CSV de entrada se fija `seed = seed_base + i` (por defecto `seed_base=1`), de modo que cada escenario es determinista de forma independiente.
- **Ejecución:** un GA completo (150 individuos × 15 generaciones) se ejecuta por cada fila del CSV de entrada, tardando ≈650-680 ms/instancia (depende de la máquina).

### Resultado sobre el conjunto de test completo (7 708 instancias)

Ver tabla de KPIs en "Paso 5 — Optimización GA" y en "Cómo leer `evaluation_rt_backtesting_report.json`": mejora del consumo específico de **+10.73%**, ahorro de energía absoluta de **+1.48%**, incremento de caudal de **+9.72%**, y cumplimiento de seguridad alimentaria del **100%** (vs 99.42% en la operación histórica).

---

## Notebooks — Documentación del Proceso Experimental

> **Nota para el auditor:** Los notebooks **no son el código del proyecto**. Son el registro del trabajo experimental e iterativo de desarrollo: exploración de datos, pruebas de arquitectura, análisis de convergencia del GA y visualizaciones de resultados intermedios. Están redactados con descripciones extensas y comentarios para documentar el razonamiento detrás de cada decisión técnica, pero **el código de producción está íntegramente en `src/`**.
>
> Para verificar el funcionamiento del sistema, ejecutar los scripts del pipeline (`scripts/`) tal como se describe en la sección anterior. Los notebooks sirven como contexto para entender por qué se tomaron determinadas decisiones de diseño.

### Orden cronológico del desarrollo experimental

> **Pipeline actual (4 notebooks):** `data_creation.ipynb` → `eda.ipynb` → `predict_model_t_h.ipynb` (referencia de hiperparámetros; **no volver a ejecutar la búsqueda**) → `ga_evaluation.ipynb`. El resto de notebooks listados abajo documentan iteraciones anteriores o análisis complementarios, y no es necesario ejecutarlos para reproducir el estado actual del proyecto.

| # | Notebook | Carpeta | Qué documenta |
|---|----------|---------|---------------|
| 1 | `data_creation.ipynb` | `EDA/` | Simulador físico V3.4 (heurística PID de lazo abierto) y verificación de las ecuaciones termodinámicas |
| 2 | `eda.ipynb` | `EDA/` | Exploración del dataset: distribuciones, correlaciones, ciclos de fouling, decisiones de preprocesado |
| 3 | `predict_model.ipynb` | `training/` | Primera versión exploratoria del modelo MLP (arquitectura fija, sin tuning) |
| 4 | `predict_model_t_h.ipynb` | `tuning_hyperparameters/` | Búsqueda aleatoria de hiperparámetros del MLP; el mejor modelo (3 capas, 64 neuronas, ReLU) se guardó en `model_config.json` |
| 5 | `tuning_params_ga.ipynb` | `tuning_hyperparameters/` | Análisis de convergencia del GA; justifica `n_gen=15` como valor operativo final |
| 6 | `optimization_ga_v4.ipynb` | `training/` | Ejecución del GA mono-objetivo v4 sobre una malla de escenarios (histórico/exploratorio, no operativo); genera `ga_v4_optimization_results.csv` |
| 7 | `baseline.ipynb` | `evaluacion/` | Cálculo del baseline histórico PID |
| 8 | `ga_evaluation.ipynb` | `evaluacion/` | Backtesting en tiempo real del GA mono-objetivo v4 sobre el conjunto de test (7 708 instancias) — es la referencia para `scripts/optimize.py` |

### Notebooks adicionales (material de referencia histórico)

| Notebook | Carpeta | Descripción |
|----------|---------|-------------|
| `optimization_ga.ipynb` | `training/old_versions/` | GA v1 — mono-objetivo, mono-escenario (versión descartada) |
| `optimization_ga_v2.ipynb` | `training/old_versions/` | GA v2 — bi-objetivo, mono-escenario (versión descartada) |
| `optimization_ga_v3.ipynb` | `training/old_versions/` | GA v3 — bi-objetivo (NSGA-II) sobre malla completa de escenarios con frente de Pareto (versión descartada) |
| `ej_alg_genetico.ipynb` | `opcional/` | Ejemplo didáctico de algoritmo genético mono-objetivo |
| `tuto_ml+GA.ipynb` | `opcional/` | Tutorial introductorio de ML + GA |

> Las mallas de escenarios precomputadas (`ga_v3_optimization_results.csv`, `ga_v4_optimization_results.csv`) son experimentos de precomputación offline de versiones anteriores/exploratorias. En el código existe una utilidad de consulta en `src/predict/lookup.py`, pero **no forma parte del pipeline operativo por defecto**. El flujo operativo auditado (`scripts/optimize.py`, documentado en `ga_evaluation.ipynb`) ejecuta el GA mono-objetivo v4 en tiempo real por escenario individual usando el MLP como surrogate.

---

## Limitaciones, Riesgos y Consideraciones

### Limitaciones
- **Datos sintéticos:** El modelo fue entrenado con datos generados por un simulador físico. Los resultados deben validarse con datos reales de planta (Sim2Real transfer) antes de cualquier despliegue.
- **Cobertura:** El simulador cubre un rango específico de condiciones (180 días, estacionalidad sinusoidal). Escenarios fuera de este rango no están garantizados.
- **Simplificaciones del simulador:** El modelo de fouling es una simplificación de Tarapata et al. (2025); el PID simulado no replica exactamente el controlador industrial real.

### Riesgos operativos
- **Data drift:** Si las condiciones reales de planta difieren significativamente de las simuladas, el rendimiento del modelo puede degradarse.
- **Sensibilidad a inputs:** Errores en las lecturas de sensores (T_in, Delta_P, t_ciclo) se propagan al modelo y al optimizador.
- **Dependencia de la calidad del modelo:** El GA usa la MLP como surrogate; si el modelo es impreciso en cierta región, los setpoints propuestos pueden ser subóptimos.

### Condiciones de uso recomendadas
- Usar con datos dentro del rango de entrenamiento: T_in ∈ [0, 8] °C, F_flow ∈ [3500, 5500] L/h, T_servicio ∈ [76, 95] °C.
- Verificar siempre que T_out ≥ 72.3 °C en la predicción antes de aplicar setpoints (incluye margen de seguridad para incertidumbre de sensor PT100 y tolerancia PMO-FDA).
- Recalibrar periódicamente con datos reales cuando estén disponibles.


---
