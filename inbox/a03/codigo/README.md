# Predicción de Plagas y Enfermedades en la Vid mediante Deep Ensemble Learning

Este proyecto implementa y orquesta una solución de Inteligencia Artificial (Deep Learning) orientada al análisis de series temporales agroclimáticas para predecir la aparición y severidad de plagas y enfermedades críticas en el cultivo de la vid (**Botrytis, Oídio, Mildiu, Lobesia, Esca, etc.**).

El modelo central ensambla tres arquitecturas de redes neuronales complementarias (LSTM, CNN-1D, y Bi-GRU) bajo el método de *Soft Voting* y *Simple Averaging* para emitir pronósticos de alta confianza sobre ventanas de contexto de 7 días (editable desde config/config.yaml).

El modelo se ha desarrollado en Python version 3.10.11  ejecutado sobre Windows 11 Pro.

---

## Estructura de carpetas

El código está refactorizado siguiendo estándares de ingeniería de software y MLOps:

```text
modelo_3_enfermedades_plagas/
 ├─ requirements.txt     # Dependencias del sistema y librerías
 ├─ README.md            # Este fichero
 ├─ .gitignore           # Archivos omitidos de control de versiones
 │
 ├─ data/                # Bases de datos 
 │   ├─ raw/             # Crudos (Parquet, CSV)
 │   ├─ processed/       # Transformados y codificados
 │   ├─ predictions/     # Salidas de inferencia de la IA
 │   └─ splits/          # Archivos particionados (Train/Val/Test)
 │
 ├─ models/              # Serializados
 │   ├─ artifacts/       # Pesos .keras de los modelos, .pkl de scalers/encoders
 │   └─ metrics/         # Metricas de cada modelo y curvas de entrenamiento
 │
 ├─ config/              
 │   └─ config.yaml      # Parámetros, rutas dinámicas y definición de variables (raw/model)
 │
 ├─ src/                 # Lógica Interna del Código
 │   ├─ main.py          # CLI / Punto de entrada Orquestador Principal
 │   ├─ data_processing/ # Funciones de pipeline, limpieza y feature engineering
 │   ├─ training/        # Empaquetado de sensores 3D, arquitecturas de NN y Callbacks
 │   ├─ predict/         # Inferencia, formateo de la matriz y base de conocimiento (tratamientos)
 │   ├─ get_stats/       # Generador de metadatos del dataset
 │   └─ utils/           # Herramientas globales (Logger principal)
 │
 └─ scripts/             # Lanzadores fáciles (atajos para el usuario final)
     ├─ data/            # Scripts para generar el dataset crudo y procesado
     │   ├─ descarga_clima_historico.py # Descarga de datos meteorológicos históricos desde Open-Meteo
     │   ├─ data_cleaner.py  # Limpieza y preprocesamiento de datos meteorológicos
     │   └─ vid_simulator.py # Generador de dataset sintético con variables meteorológicas
     ├─ train.py             # Entrenamiento del modelo
     ├─ predict.py           # Inferencia del modelo
     ├─ get_stats.py         # Obtención de estadísticas del dataset
     └─ data_processing.py   # Preprocesamiento del dataset
```

---

## Guía de Ejecución Paso a Paso

El proyecto utiliza el orquestador principal `src/main.py` para todas sus fases.

### Paso 1 – Configuración del entorno
Tras situarse en la carpeta raíz del proyecto (`modelo_3_enfermedades_plagas`):

**Crear entorno virtual:**
```bash
python -m venv venv
```
**Activar entorno (Windows):**
```bash
venv\Scripts\activate
```
**Activar entorno (Linux/Mac):**
```bash
source venv/bin/activate
```
**Instalar dependencias:**
```bash
pip install -r requirements.txt
```

> ⚠️ **Nota GPU/CUDA**: Por defecto se instala `tensorflow`, que entrena en CPU si
> CUDA no está disponible a nivel de sistema (**x3-5 más lento**). Para aceleración GPU,
> sustituir `tensorflow>=2.15.0` por `tensorflow[and-cuda]>=2.15.0` en `requirements.txt`
> o instalar manualmente: `pip install tensorflow[and-cuda]>=2.15.0`.

### Paso 2 - Obtención del dataset 

Aunque los datos crudos y los splits se encuentran en un contenedor externo, si aún no se dispone del dataset inicial se puede generar con los siguientes scripts:

**Paso 2.1 - Obtener datos históricos:** Se descarga el conjunto de datos meteorológicos históricos desde la [API del clima histórico de Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) y se guardan en la carpeta `data/clima_real/clima_<parcela>.parquet`.
```bash
python scripts/data/descarga_clima_historico.py
```
**Paso 2.2 - Limpiar datos de la API climatologica:** Se eliminan los datos históricos que no son necesarios para el entrenamiento del modelo, la descripción de los campos eliminados se encuentra en el entregable. Se generan los conjuntos de datos limpios en `data/clima_real/clean/clima_<parcela>_clean.parquet`.
```bash
python scripts/data/data_cleaner.py
```
**Paso 2.3 - Generar dataset sintético:** Se genera un dataset sintético con variables meteorológicas para el cultivo de la vid. La lógica de generación del dataset se detalla en el entregable.
```bash
python scripts/data/vid_simulator.py
```

Estos scripts generarán el dataset en crudo en la carpeta `data/raw/data_vin_raw.parquet`. El fundamento de creación de este conjunto de datos se detalla tanto con comentarios de código como en el entregable. 

### Paso 3 – Obtención de datos (Procesamiento)
Tras garantizar que los datos crudos residen en `data/raw/data_vin_raw.parquet`, procedemos a estructurarlos limpiamente inyectando la ingeniería de variables temporales (`Hora_Sin`, `Hora_Cos`) y creacion de features (`GDD_Acumulado` y `Horas_Humedad_Foliar`):
```bash
python -m src.main data_processing
```
*(Este comando se encarga de generar el dataset base `data/processed/data_vin_processed.parquet`).*

### Paso 4 – Ejecución del pipeline de entrenamiento y evaluación
Este comando toma el dataset procesado, lo escala, genera las secuencias complejas 3D (ventanas temporales) y lanza el bucle de entrenamiento dinámico de los modelos LSTM, CNN y BiGRU. Añadiendo el flag `--metrics`, el sistema procederá a inferir con los mejores pesos sobre el conjunto de test inmaculado para generar resultados sin fuga de datos:
```bash
python -m src.main train --metrics
```
*(Nota: Si el sistema detecta que los splits `train.parquet`, `val.parquet` y `test.parquet` ya existen en la carpeta `data/splits/`, los cargará directamente saltándose la fase de preprocesamiento inicial. Si no existen, los generará a partir del dataset procesado y los guardará para futuras experimentaciones).*
El progreso se monitoriza por consola. Además de almacenar todos los artefactos de pesos `.keras` y el `scaler.pkl` en `models/artifacts/`, si se usa el flag `--metrics` se guardarán los archivos:
- **`models/metrics/<Modelo>/metrics_<Modelo>_test.csv`**: Reportes de clasificación y regresión por submodelo y ensamble (DEL).
- **`models/metrics/<Modelo>/confusion_matrix_<Modelo>_test.png`**: Matriz de confusión visual por submodelo.
- **`data/splits/`**: Almacena los conjuntos de datos `train.parquet`, `val.parquet` y `test.parquet` basados en la separación por `ID_Serie`.

### Paso 5 (OPCIONAL) – Extracción de estadísticas
Para obtener el balance y rangos de distribución técnica del dataset previos al entrenamiento:
```bash
python -m src.main get_stats
```
*(Generará un reporte estadístico exportado a `data/processed/estadisticas.csv`, una descripción de cada una de las variables del dataset en `data/processed/feature_descriptions.csv` y una breve descripción del modelo y su uso por consola).*

### Paso 6 – Ejecución de inferencia
Para realizar inferencia en producción sobre datos nuevos procedentes de los sensores:
```bash
python -m src.main predict
```
Donde se da la posibilidad de utilizar los siguientes argumentos (flags):
- `--input data/raw/datos_nuevos.parquet`: Ruta del dataset de entrada (Opcional, sobrescribe el de config que por defecto es `data/raw/data_vin_raw.parquet`).
- `--id_serie <ID>`: Filtra y predice sobre un único ciclo/parcela temporal en específico (Opcional).

*(La inferencia no utiliza la variable objetivo y guardará los resultados finales en `data/predictions/inferencia_vid.csv` detallando diagnósticos, confianza y severidad).*

*La solución guardará un fichero CSV en `data/predictions/inferencia_vid.csv` con columnas: [ID_Serie, Fecha_Evaluacion, Diagnostico_IA, Confianza_Clasificacion, Grado_Severidad, Tratamiento_Recomendado]*

---

## Estructura completa del proyecto

```text
modelo_3_enfermedades_plagas/
 ├── .gitignore           # Archivos omitidos de control de versiones
 ├── README.md            # Este fichero
 ├── requirements.txt     # Dependencias del sistema y librerías
 |
 ├── config/              # Configuración global
 │    ├── README.md
 │    └── config.yaml     # Parámetros, rutas y definición de variables (raw/model)
 |
 ├── data/                # Carpeta de datos (Estos datos no están en el estado inicial del repositorio, se deben generar con el script `vid_simulator.py` y siguiendo los pasos de la guía de ejecución, o descargar los datos crudos y splits del contenedor externo)
 │    ├── README.md
 │    ├── predictions/    # Salidas de inferencia de la IA
 │    │    └── inferencia_vid.csv
 │    ├── processed/      # Datos transformados y codificados
 │    |    ├── estadisticas.csv
 │    |    ├── feature_descriptions.csv
 │    │    └── data_vin_processed.parquet
 │    ├── clima_real/     # Datos meteorológicos históricos
 │    |    ├── clean/      # Datos meteorológicos históricos limpios
 │    |    └── clima_<parcela>.parquet
 │    ├── raw/            # Datos crudos (Salida del simulador)
 │    │    └── data_vin_raw.parquet
 │    └── splits/         # Particiones Train/Val/Test fijadas con semilla
 │         ├── test.parquet
 │         ├── train.parquet
 │         └── val.parquet
 |
 ├── models/              # Modelos y métricas
 │    ├── README.md
 │    ├── artifacts/      # Pesos .keras, Scaler .pkl y LabelEncoder .pkl
 │    │    ├── M1_LSTM.keras
 │    │    ├── M2_CNN.keras
 │    │    ├── M3_BiGRU.keras
 │    │    ├── label_encoder.pkl
 │    │    └── scaler.pkl
 │    └── metrics/     
 │         ├── history_M1_LSTM.csv
 │         ├── history_M2_CNN.csv
 │         ├── history_M3_BiGRU.csv
 │         ├── DEL/
 │         │   ├── confusion_matrix_DEL_test.png
 │         │   └── metrics_DEL_test.csv
 │         ├── LSTM/
 │         │   ├── confusion_matrix_LSTM_test.png
 │         │   └── metrics_LSTM_test.csv
 │         ├── CNN/
 │         │   ├── confusion_matrix_CNN_test.png
 │         │   └── metrics_CNN_test.csv
 │         └── BiGRU/
 │             ├── confusion_matrix_BiGRU_test.png
 │             └── metrics_BiGRU_test.csv
 |
 ├── notebooks/           # Experimentación y análisis
 │    ├── EDA/            # Análisis Exploratorio de Datos
 │    │    ├── EDA_Data_Vin.ipynb
 │    │    └── figuras/   # Visualizaciones generadas
 │    ├── Modelos/        # Notebooks de desarrollo de arquitecturas y experimentación
 │    │    ├── Modelo_DEL.ipynb
 │    │    ├── Modelo_DEL_sin_VOC.ipynb
 │    │    ├── Pruebas_Modelo_DEL.ipynb # Experimentación para elección de ventana de contexto, scaler, padding, etc.
 │    │    └── Modelo_XGBoost_Baseline.ipynb
 │    └── tuning_hyperparameters/ # Búsqueda de parámetros óptimos
 │         └── Optimizacion_DEL_CrossValidation.ipynb
 |
 ├── src/                 # Lógica Interna (Encapsulamiento)
 │    ├── __init__.py
 │    ├── main.py          # Orquestador Principal (CLI)
 │    ├── data_processing/ # Pipeline de limpieza y feature engineering
 │    │    ├── __init__.py
 │    │    └── preprocess.py
 │    ├── get_stats/       # Generador de metadatos y estadísticas
 │    │    ├── __init__.py
 │    │    └── column_info.py
 │    ├── predict/         # Lógica de inferencia y ensemble
 │    │    ├── __init__.py
 │    │    └── predictor.py
 │    ├── training/        # Bucle de entrenamiento y construcción de redes
 │    │    ├── __init__.py
 │    │    └── train.py
 │    └── utils/           # Utilidades transversales
 │         ├── __init__.py
 │         └── logging.py
 |
 └── scripts/             # Atajos CLI para el usuario
      ├── data_processing.py  # Preprocesamiento del dataset
      ├── get_stats.py        # Obtención de estadísticas del dataset
      ├── predict.py          # Inferencia del modelo
      ├── train.py            # Entrenamiento del modelo
      └── data/               # Scripts para generar el dataset crudo y procesado
         ├─ descarga_clima_historico.py # Descarga de datos meteorológicos históricos desde Open-Meteo
         ├─ data_cleaner.py  # Limpieza y preprocesamiento de datos meteorológicos
         └─ vid_simulator.py # Generador de dataset sintético con variables meteorológicas
```

---