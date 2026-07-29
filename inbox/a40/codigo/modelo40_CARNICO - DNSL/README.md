# Sistema de Diagnóstico de Fallas (Refrigeración & Aireado)
> **Arquitectura Neurosimbólica para el Mantenimiento Predictivo Industrial**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework: Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Architecture: Neuro--Symbolic](https://img.shields.io/badge/Architecture-Neuro--Symbolic-red.svg)](#)

Este proyecto implementa una solución avanzada para el diagnóstico de fallas en sistemas industriales (Cárnico - DATAGIA). Combina la potencia estadística de **Machine Learning (Random Forest)** con la precisión determinista de **reglas físicas (Termodinámica)** y un sistema de consenso temporal (**Voto por Run**).

---

## Fuentes de Datos y Naturaleza del Dataset

El sistema se fundamenta en dos entornos de datos diferenciados para cubrir la complejidad de la cadena de frío y los procesos de curado:

* **Sistema de Refrigeración:** Utiliza el *Simulated Refrigerator Fault Diagnosis Dataset* (https://www.kaggle.com/datasets/samoilovmikhail/simulated-refrigerator-fault-diagnosis-dataset/). Contiene series temporales de 13 clases (Operación Normal + 12 tipos de fallos técnicos). Los datos provienen de simulaciones basadas en modelos físicos de primer principio que modelan el ciclo de compresión de vapor.
* **Sistema de Aireado:** Ante la ausencia de repositorios públicos específicos para secaderos cárnicos, se emplea un **Dataset Sintético de Alta Fidelidad** generado mediante el script `scripts/generate_dataset_aireado.py`. Este modelo parametriza fenómenos de transferencia de masa, humedad relativa y flujos de aire basados en literatura técnica especializada del sector cárnico.

---

## Estructura del Proyecto
```
├── config/             # Configuración centralizada (config.yaml)
├── data/               # Gestión de ciclos de datos
│   ├── raw/            # Datasets originales de planta
│   ├── processed/      # Datos con ingeniería de variables físicas
│   ├── to_predict/     # INPUT: Archivos crudos del cliente (input_{system}.csv)
│   └── predictions/    # OUTPUT: Resultados finales y diagnósticos por ciclo
├── logs/               # Registro de ejecución del sistema en archivos .txt
├── models/             
│   ├── artifacts/      # Modelos (.pkl), escaladores, parámetros óptimos (.yaml) y umbrales dinámicos (.yaml).
│   └── metrics/        # Matrices de confusión y reportes de desempeño
├── notebooks/          # Notebooks de experimentación y análisis detallado
│   ├── completo/       # Notebook con el flujo integral de ambos sistemas
│   ├── EDA/            # Análisis exploratorio de datos (Aireado y Refrigeración)
│   ├── evaluacion/     # Evaluación de sistemas y postprocesado neurosimbólico
│   └── tuning_hyperparameters/ # Validación cruzada y optimización de modelos
├── scripts/            # Scripts de automatización que invocan al orquestador (main.py)
├── src/                # Código fuente (Core)
│   ├── data_processing/# Lógica de sensores (Presiones, Lags, Ingeniería)
│   ├── training/       # Scripts de entrenamiento y tuning (RandomizedSearchCV)
│   ├── predict/        # Motor de inferencia y reglas neurosimbólicas
│   └── utils/          # Gestión de logs y utilidades del sistema
└── main.py             # Orquestador principal (Interfaz de Línea de Comandos)
```
## Arquitectura del Sistema (Pipeline de 3 Capas)

El diagnóstico se valida a través de tres capas de confianza para asegurar la explicabilidad del fallo:

1. **Capa Física (Preprocesamiento):** Transforma señales crudas en indicadores físicos (COP, Eficiencia Volumétrica, Ratio Aire/Carga, Delta Higroscópico...).
2. **Capa Estadística (Inferencia):** Un modelo **Random Forest** optimizado genera una clasificación basada en los patrones aprendidos durante el entrenamiento.
3. **Capa Neurosimbólica (Post-procesamiento):**
   * **Reglas Expertas:** Corrige al modelo si la predicción viola leyes físicas (ej. validación de subenfriamiento o coherencia de presiones).
   * **Voto por Run:** Aplica un consenso mayoritario sobre el ciclo de funcionamiento (`run_id`) para eliminar falsos positivos causados por ruido en sensores.

## Instalación y Configuración

### Requisitos previos
* **Python 3.9** o superior.
* **Hardware:** Optimizado para ejecución en **CPU**, lo que permite su despliegue en estaciones locales de procesamiento o dispositivos Edge sin necesidad de GPU dedicada.

### 1. Clonar el repositorio y preparar el entorno
```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno (Windows)
venv\Scripts\activate

# Activar el entorno (Linux/macOS)
source venv/bin/activate

### Instalar dependencias
pip install -r requirements.txt
```

### 2. Descarga Manual de Activos (Blob Storage)

Dado que varios de los archivos de datos superan los límites de tamaño de Git, estos deben descargarse manualmente desde el **Blob Storage** del proyecto antes de iniciar el sistema:

1.  **Datasets en crudo:** Descargue la carpeta `data/raw` y ubíquela en la ruta correspondiente (`data/raw`). Esta carpeta contiene los archivos `dataset_aireado.csv` y `dataset_refrigeracion.csv`.
2.  **Splits de entrenamiento/test generados:** Si sólo quiere realizar inferencia sin ejecutar el pipeline entero de entrenamiento, descargue la carpeta `data/splits` y ubíquela en la ruta correspondiente (`data/splits`). Esta carpeta contiene los archivos de splits `aireado_train(test).csv` y `refrigeracion_train(test).csv` que se han utilizado para el entrenamiento.


> **Nota:** La estructura final debería verse así: `data/raw/dataset_aireado(refrigeracion).csv`y `data/splits/aireado(refrigeracion)_train(test).csv`.


## Guía de Ejecución (CLI)

El sistema utiliza un único punto de entrada (`main.py`) para todas las operaciones. El sistema activo se define en el parámetro `selected_system` dentro de `config/config.yaml` (refrigeracion/aireado).

| Acción | Comando |
| :--- | :--- |
| **Procesar Datos** | `python -m src.main data_processing` |
| **Información estadística sobre los datasets** | `python -m src.main get_stats` |
| **Información sobre los datasets, caracterísicas y modelos utilizados** | `python -m src.main get_info` |
| **Optimizar (Tuning)** | `python -m src.main tuning` |
| **Entrenar Modelo** | `python -m src.main train` |
| **Calibrar** | `python -m src.main calibrate` |
| **Ejecutar Diagnóstico** | `python -m src.main predict` |
| **Evaluación con dataset supervisado** | `python -m src.main evaluate` |

---

## Flujo de Inferencia y Predicción

El sistema está diseñado para operar en dos escenarios: **Evaluación de rendimiento** (usando datos etiquetados) y **Diagnóstico en producción** (usando datos nuevos de planta).

### 1. Preparación de Entradas
Para realizar un diagnóstico sobre datos nuevos, el sistema busca archivos de entrada en la carpeta `data/to_predict/`.

| Sistema | Archivo de Entrada Requerido | Descripción |
| :--- | :--- | :--- |
| **Refrigeración** | `data/to_predict/input_refrigeracion.csv` | Datos crudos de sensores (Presiones, Temperaturas, Consumos). |
| **Aireado** | `data/to_predict/input_aireado.csv` | Datos crudos (HR, Temperaturas, Hz ventilación, Kg carga). |

> **Nota:** Si el archivo `input_{sistema}.csv` no existe, el motor de inferencia utilizará por defecto el set de test (`data/splits/{sistema}_test.csv`) para realizar una validación de control.

### 2. Pipeline de Inferencia "On-the-fly"
Al ejecutar el comando `predict`, el sistema activa un pipeline dinámico que asegura la consistencia entre el entrenamiento y la vida real:

* **Ingeniería Automática:** Se calculan indicadores físicos (VPD, COP, Deltas) y variables temporales (Lags/Rolling means) a partir de los datos crudos.
* **Alineación de Features:** Se seleccionan y ordenan las columnas exactamente como el modelo las espera.
* **Consenso Neurosimbólico:** Se aplican las reglas físicas y el voto por ciclo (`run_id`) para filtrar falsos positivos.

### 3. Salidas y Resultados
Los resultados se consolidan en `data/predictions/` y el estado de salud del modelo se monitoriza en los logs.

* **Archivo de Predicciones:** `data/predictions/predictions_{sistema}.csv`
    * `prediction`: Clase de falla detectada (o "Normal").
    * `confidence`: Nivel de certidumbre del modelo (0-100%).
* **Log de Monitorización:** En `logs/monitorization_{sistema}.csv` se centralizan los eventos críticos:
    * **Drift:** Registrado automáticamente tras cada ejecución de `calibrate`.
    * **Health/Status:** Registrado tras cada inferencia (`predict`), incluyendo la confianza del modelo y el estado de degradación.

---

## Trazabilidad y Evidencias

Tras cada ejecución, el sistema genera evidencias clave para garantizar la integridad y calidad del modelo:

* **Métricas de Rendimiento:** En `models/metrics/` se guardan reportes de clasificación y matrices de confusión generadas mediante `evaluate`.
* **Salud de los Modelos:** Los archivos `logs/monitorization_{system}.csv` actúan como el registro principal. Permiten correlacionar cronológicamente cuándo el modelo fue recalibrado (Drift) y cómo ha evolucionado su confianza predictiva (Health Status), facilitando la detección proactiva de degradación de sensores.
* **Artefactos dinámicos:** Los modelos, escaladores y los **umbrales dinámicos** (ajustados mediante la calibración automática) se persisten en `models/artifacts/`, garantizando que el sistema sea auditable y reproducible.
