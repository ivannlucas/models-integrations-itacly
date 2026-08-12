# m3-wine-disease-pest-forecast

Plugin de integración para el modelo DEL (Deep Ensemble Learning: LSTM + CNN-1D + BiGRU)
de predicción de enfermedades y plagas en viña.

## Integración: Original (a03) vs Plataforma (m3)

### Tabla Comparativa

| Paso | Original (a03) | Plataforma (m3) | Justificacion |
|------|----------------|-------------------|---------------|
| **Orquestación** | `scripts/train.py`/`predict.py` -> `src/main.py` (argparse, steps `data_processing`/`train`/`predict`/`get_stats`) | No existe. Endpoints `/models/ml3-wine-disease-pest-forecast/{health,predict,stats,train}` (plugin `ModelPluginPort`) | La plataforma expone el modelo como servicio HTTP; el pipeline se ejecuta bajo los casos de uso del router |
| **Data processing** | `run_data_processing()`: lee `data_vin_raw.parquet`, valida `raw_features`, reindexa alfabético, castea strings, aplica FE, guarda `data_vin_processed.parquet` | Eliminado. `run_retraining()` aplica FE directamente sobre el CSV/parquet etiquetado recibido | No hay fichero intermedio: el train es efímero (CSV -> artefactos -> MLflow). FE antes del split, semánticamente idéntico |
| **Split 70/15/15** | `train_test_split` estratificado por `ID_Serie` (etiqueta de serie), seed 42. Exporta splits a `data/splits/*.parquet` y los reutiliza como fallback si no hay procesado | `_split` en `run_retraining()`: mismo `train_test_split`, mismos tamaños y seed 42. Sin persistencia en disco | Mismo split para reproducibilidad; los datos se versionan en el CSV de entrada, no hace falta persistir índices |
| **Feature engineering** | `apply_feature_engineering()` (Hora_Sin/Cos, GDD_Acumulado base 10°C, Horas_Humedad_Foliar) en `preprocess.py`, flag `strict` | `feature_engineering.py`: portado 1:1 de `_calcular_gdd_serie`/`_calcular_horas_mojado_serie`, mismas fórmulas y umbrales | Evita train-serving skew. El plugin siempre exige `Fecha` (quita el flag `strict=False`) y no muta el df de entrada (copia) |
| **Escalado** | StandardScaler fit solo en train | Idéntico (fit en train, transform en train/val/test) | Sin cambios |
| **Ventanas (train)** | `crear_secuencias()`: stride 1, `range(len - window_size)`, dtype float32/int8 | `_crear_secuencias()`: portado 1:1 (mismos dtypes y rango) | Sin cambios |
| **Ventana (inferencia)** | `run_inference()`: tail a 168 o padding con la 1ª fila repetida, DESPUÉS FE y escalado | `build_window_tensor()`: `_tail_or_pad` idéntico; añade ordenación por `Fecha` antes del tail (robustez). FE y escalado después | Mismo comportamiento; el orden por fecha elimina dependencia del orden del fichero |
| **Arquitecturas** | `build_models()`: M1_LSTM (8->BN->Dropout .2->LSTM32->BN), M2_CNN (Conv16->BN->Conv32->BN->GAP->Dropout), M3_BiGRU (16->BN->Dropout->Dense8). Adam 1e-3, loss {crossentropy, mse} pesos 1:1 | `_build_models()`: portado 1:1 (mismas capas, nombres y salidas `out_class`/`out_reg`) | Sin cambios |
| **Entrenamiento** | EarlyStopping patience 2 + ModelCheckpoint (best val_loss), verbose=1. `set_random_seed` + intento `enable_op_determinism()`, 50 epochs, batch 512 | Idéntico (patience 2, checkpoint, seed 42) con verbose=0; sin `enable_op_determinism` | Mismo entrenamiento; el determinismo de op se omitió (dependía de GPU) |
| **Ensemble inferencia** | `predecir_ensemble()`: Soft Voting (media de probabilidades), argmax clase, severidad = media de regresiones sigmoid | `postprocessing.predict_ensemble()`: portado 1:1 (media de probs, argmax, media de regresiones) | Sin cambios |
| **Salida** | DataFrame -> `data/predictions/inferencia_vid.csv` (Confianza en string "96.08%", Grado a 4 decimales) | Respuesta JSON por ventana: floats a 6 decimales + `probabilidades_clases` (vector 11) + `xai_feature_values` (snapshot raw) | El API devuelve la predicción inline; la confianza como float en vez de string |
| **Tratamiento recomendado** | `base_conocimiento_tratamientos` del config.yaml, serializado `" | ".join(f"{k}: {v}")`, fallback `{"Aviso": "N/A"}` | `TREATMENT_KNOWLEDGE_BASE` en `constants.py` (verbatim), misma serialización (`build_treatment`) | KB editable, mismos textos |
| **MLflow / reentrenado por usuario** | No existía | `mlflow_utils.py`: train con `mlflow_run_id` sube artefactos a MLflow y predict puede cargar el bundle reentrenado (`_bundle_for`). Los artefactos fijos S3 nunca se sobrescriben | Convención obligatoria de la plataforma (todo plugin lleva `mlflow_utils.py`) |

### Features: El Original Ya Procesaba 8 Crudas -> 12 de Modelo

El `config.yaml` define `raw_features` (8 sensores + Fecha) y `model_features` (12): las 8 crudas
más `Horas_Humedad_Foliar`, `GDD_Acumulado`, `Hora_Sin` y `Hora_Cos`. Las 4 derivadas se
recalculan en inferencia sobre la ventana de 168 horas (no se piden al usuario).

**RAW_FIXED_COLUMNS en plataforma (idem):**
```python
RAW_FIXED_COLUMNS = ["Temp_Amb_C", "Hum_Rel_Pct", "Lluvia_mm", "Viento_kmh",
                     "CO2_ppm", "VOC_ppb", "Hum_Suelo_Pct", "pH_Suelo"]
```

**MODEL_FEATURES en plataforma (idem):**
```python
MODEL_FEATURES = ["Temp_Amb_C", "Hum_Rel_Pct", "Lluvia_mm", "Viento_kmh",
                  "Horas_Humedad_Foliar", "GDD_Acumulado", "Hum_Suelo_Pct", "pH_Suelo",
                  "CO2_ppm", "VOC_ppb", "Hora_Sin", "Hora_Cos"]
```

### Formato de Datasets

#### Original (raw, entrada al data_processing)

```
Fecha,Temp_Amb_C,Hum_Rel_Pct,Lluvia_mm,Viento_kmh,CO2_ppm,VOC_ppb,Hum_Suelo_Pct,pH_Suelo,
Clase_Entrenamiento,Etiqueta_Clase,Grado_Infeccion,ID_Serie,Parcela_ID
```

#### Original (procesado, entrada al train) — 20 columnas

```
Fecha,Temp_Amb_C,...,pH_Suelo,GDD_Acumulado,Horas_Humedad_Foliar,Hora,Hora_Sin,Hora_Cos,
Clase_Entrenamiento,Etiqueta_Clase,Etiqueta_Num,Grado_Infeccion,ID_Serie,Parcela_ID
```

#### Plataforma (CSV de entrenamiento — contrato `TRAIN_HARD_REQUIRED_COLUMNS`)

```
Fecha,Temp_Amb_C,Hum_Rel_Pct,Lluvia_mm,Viento_kmh,CO2_ppm,VOC_ppb,Hum_Suelo_Pct,pH_Suelo,
Clase_Entrenamiento,Grado_Infeccion,ID_Serie[,Etiqueta_Clase]
```

`Etiqueta_Clase` se recomienda (y es de facto necesaria) para el split estratificado por serie.
Las columnas derivadas (`GDD_Acumulado`, `Horas_Humedad_Foliar`, `Hora_Sin/Cos`) y `Etiqueta_Num`
se generan dentro del pipeline — el CSV no debe traerlas.

#### Plataforma (inline predict, via API)

JSON con la ventana horaria como lista de filas (mínimo operativo: 168 filas de una serie):

```json
{
  "mode": "inline",
  "rows": [
    {"Fecha": "2021-06-12 09:00:00", "Temp_Amb_C": 21.2, "Hum_Rel_Pct": 63.0, "Lluvia_mm": 0.8,
     "Viento_kmh": 16.1, "CO2_ppm": 410.57, "VOC_ppb": 18.25, "Hum_Suelo_Pct": 24.62, "pH_Suelo": 6.58},
    {"Fecha": "2021-06-12 10:00:00", "Temp_Amb_C": 21.8, "Hum_Rel_Pct": 61.0, "Lluvia_mm": 0.0, "...": "..."}
  ]
}
```

En inline todo el input se trata como UNA sola serie (si trae `ID_Serie`, solo se usa su primer
valor como nombre de la serie). El agrupado multi-serie es del modo batch. Con <168 filas hace
padding repitiendo la primera (predicción degradada).

### Resumen de Columnas

| Concepto | Original (a03) | Plataforma (m3) |
|----------|----------------|-------------------|
| Features crudas | raw_features (8 + Fecha) | RAW_FIXED_COLUMNS (8) + Fecha, mismos |
| Features de modelo | model_features (12) | MODEL_FEATURES (12), mismos |
| Derivadas | Hora_Sin, Hora_Cos, GDD_Acumulado, Horas_Humedad_Foliar | Idénticas (mismas fórmulas y umbrales) |
| Target clase (string) | Clase_Entrenamiento | Clase_Entrenamiento (mismo) |
| Target clase (num) | Etiqueta_Num (LabelEncoder) | Etiqueta_Num (LabelEncoder in-pipeline) |
| Target regresión | Grado_Infeccion | Grado_Infeccion (mismo) |
| Estratificación | Etiqueta_Clase | Etiqueta_Clase (mismo) |
| Metadato | Parcela_ID | Parcela_ID (no se usa como feature, igual) |
| Columnas extra del original | Hora, reindex alfabético, splits versionados | Hora (interna), sin splits en disco |

### Adapters Implementados en la Plataforma

| Adapter | Archivo | Funcion |
|---------|---------|---------|
| `build_raw_dataframe()` / `validate_raw_columns()` | preprocessing.py:24-33 | Construye el DataFrame desde las filas inline y valida el contrato de entrada (8 sensores + Fecha) |
| `_tail_or_pad()` | preprocessing.py:36-48 | Port de predictor.py: tail a 168 o padding con la 1ª fila repetida |
| `build_window_tensor()` | preprocessing.py:51-74 | FE + escalado + reshape (1, 168, 12) de la última ventana de la serie |
| `prepare_series_groups()` | preprocessing.py:77-93 | Agrupa por `ID_Serie` (mismo fallback "Única" que `run_inference`) |
| `apply_feature_engineering()` | feature_engineering.py:56-111 | Port 1:1 de preprocess.py (GDD base 10°C, mojado foliar >0.1 mm / HR>90%) |
| `predict_ensemble()` | postprocessing.py:26-57 | Port 1:1 de `predecir_ensemble` (Soft Voting + media severidad) |
| `build_treatment()` | postprocessing.py:60-63 | Serializa la KB igual que el original |
| `raw_snapshot()` | postprocessing.py:66-78 | Snapshot de la última fila cruda para el servicio de explicabilidad (XAI) |
| `run_retraining()` | training.py:152-265 | Port de `run_training` (split, escalado, ventanas, 3 arquitecturas, early stopping) |
| `_bundle_for()` / `mlflow_utils.py` | plugin.py:101-111 | Resuelve bundle reentrenado por usuario desde MLflow (o el fijo S3) |

### Diferencias por Diseño (no implementado a proposito)

- **Persistencia en disco**: No se generan `data/splits/*.parquet`, `data_vin_processed.parquet`,
  histories CSV por modelo, ni matrices de confusión/PNGs. El reentrenamiento es efímero (CSV ->
  artefactos user_* -> MLflow) y el endpoint `/stats` sirve `metrics_reported` del manifest
  (métricas DEL declaradas sobre test hold-out, Tabla 11/12 de la memoria). Las métricas del
  retrain se devuelven en el `TrainResponse` (accuracy, precision/recall/f1 macro y weighted,
  mae, mse, r2).

- **Métricas por modelo individual (LSTM/CNN/BiGRU)**: El original calculaba classification
  report + confusion matrix por red además del ensemble DEL. El plugin solo reporta métricas del
  ensemble (Soft Voting), que es la salida operativa del sistema.

- **Fallback de splits pre-existentes**: El original, si faltaba `data_vin_processed.parquet`,
  cargaba `data/splits/*.parquet`. El plugin siempre parte del CSV recibido en `/train`; no hay
  estado en disco.

- **`strict=False` y columna `Hora`**: El original podía omitir la fecha en inferencia
  (`strict=False`). El plugin exige siempre `Fecha` (es imprescindible para `Hora_Sin/Cos`).
  La columna `Hora` sigue generándose internamente, igual que el original.

- **Progress bars / verbosity**: Se eliminaron `tqdm` y `verbose=1`; el progreso se registra por
  logging (`Entrenando: M1_LSTM`, etc.).

- **`enable_op_determinism`**: No se intenta (dependía de la GPU y podía lanzar warning). El seed
  global (`set_random_seed(42)`) se conserva.

- **Orden de columnas**: El original reindexaba alfabéticamente todo el DataFrame antes del split.
  El plugin reindexa igual tras la FE (`reindex(sorted(df.columns))`), por lo que el tensor de
  entrada y el escalado son idénticos.

- **Golden dataset / reproducibilidad**: La inferencia del plugin es bit-a-exacta con el
  `run_inference` entregado sobre el parquet nativo (20/20 casos, tolerancia rtol 0.01). El único
  ruido observado (5º-6º decimal) aparece al hacer round-trip CSV por la conversión float32->float64
  de `pd.read_csv` — mismo comportamiento del código original.
