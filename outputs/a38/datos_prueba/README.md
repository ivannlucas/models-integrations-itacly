# Datos de prueba — ml38-wine-machinery-fault-detection

Diagnóstico jerárquico (DNLS-VINO): un Random Forest da la familia de fallo y, si no es `healthy`, el MLP
de esa familia da la severidad binaria (`leve` / `significativo`).

**Origen de los datos.** Todo sale de la entrega del equipo de IA y del manifest, sin valores inventados.
Las entradas son filas de `inbox/a38/codigo/data/processed/feature_dataset.parquet`: las 24 características
de cada ventana, que el equipo calculó desde el dataset público NLN-EMP con vibración ch1 y corriente ch1.
La revisión del 2026-11-24 recalculó esas 877 filas desde los ficheros raw y las reprodujo con una
diferencia relativa de 1,1e-12 (`outputs/a38/evidence/raw_dataset/`). Los valores esperados son los
`golden_cases` del manifest, es decir, la salida del modelo entregado en `data/predictions/predictions.csv`.
El script comprueba que los inputs del manifest son esas filas del parquet y que los esperados coinciden
con `predictions.csv`. Si algo no cuadra, se para sin generar nada.

Regenerar (lee solo la entrega, <5 s):
`.venv/bin/python -I outputs/a38/datos_prueba/generar_datos_prueba.py "$PWD"`

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_NNN_<familia>.json` (21) | `POST /models/ml38-wine-machinery-fault-detection/predict` | `{"mode": "inline", ...}` con las 24 características de un golden case |
| `predict_batch_golden.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 21 golden cases: `sample_id` + 24 características (las columnas que no son características se ignoran) |
| `expected_golden.csv` | — | Por caso: traza (motor, velocidad, fallo, familia y severidad reales) y `family`, `severity`, `severity_label` y `severity_confidence` esperados, con su tolerancia |
| `train_feature_dataset_877.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | La tabla completa con la que se entrenó el modelo servido: 877 ventanas, 114 grupos, `sample_id` + 24 características + `family`, `severity`, `group_id` (425 KB) |

**Selección.** Se usan los 21 golden cases del manifest: 3 por familia (alignment, bearing, cavitation,
electrical, healthy, mechanical_other, unbalance). Cubren las tres salidas de severidad que puede dar el
modelo entregado: `significativo` (14 casos), `leve` (4: casos 005, 007, 010 y 011) y `sin fallo` (los 3
healthy). La cuarta, `modelo no disponible`, no se da porque el modelo tiene MLP para las 6 familias con
fallo. Los casos
010 (`broken rotor bar`) y 018 (`loose foot motor`) no tienen severidad real (`-1`), pero el modelo la
estima igualmente porque predice una familia con fallo. Son casos in-sample: el modelo final se entrenó con
las 877 ventanas, así que comprueban la fidelidad de la integración y no la generalización (manifest
KI-02).

Tolerancias (manifest `tolerance_policy`): `family`, `severity` y `severity_label` exactos;
`severity_confidence` ±0,05.

**Modo raw no incluido.** El plugin también acepta `{"mode": "raw", "vibration_csv", "current_csv", "motor",
"speed"}` con los CSV originales de NLN-EMP, pero solo con canal 1 y ventanas completas de 240000/300000
muestras. Si no, responde 422 (KI-05, KI-06). El 2026-11-26 la copia local del dataset (84 GB) ya no estaba en
`/mnt/d`, así que no se han generado ficheros raw. El ejemplo recortado de la entrega
(`data/examples/raw/`, 4096 filas) ya no sirve: el plugin lo rechaza con 422. La paridad del modo raw quedó
comprobada en la revisión del 2026-11-24 (877/877, `outputs/a38/evidence/raw_dataset/`).

**Train.** Reentrena desde cero el RF y los 6 MLP de severidad, con el procedimiento original
(`scripts.train`). El resultado se guarda **solo en el run de MLflow** y no toca los artefactos base. Para
predecir con él hay que pasar `mlflow_run_id`. Mínimos del plugin: las 24 características, `family`,
`severity` (entera, `-1` si no aplica) y `group_id`; al menos 2 grupos y 2 familias. Una familia sin
2 niveles de severidad o sin 2 grupos válidos se queda sin MLP (`severity_skipped`). Se usa la tabla
completa y no un recorte porque es pequeña (877 filas), entrena en menos de 1 min en CPU y su CV de familia
tiene un valor de referencia fijo. El RF y los folds son deterministas: con scikit-learn 1.5.2 (el pin del
repo) da macro-F1 0,6064 ± 0,1105 y accuracy 0,6168 ± 0,1117 (manifest KI-14). Con la 1.7.2 del equipo
sería 0,6070 / 0,6180, como en la Tabla 9 de la memoria. Las métricas de CV de severidad **no** son valores
fijos: los MLP no son reproducibles de una ejecución a otra (memoria §6.3) y cambian en cada
reentrenamiento.

El CSV de train sirve también como entrada batch (las etiquetas se ignoran) para comparar las 877 filas con
`predictions.csv`. Los decimales están escritos con todas sus cifras. El lector CSV por defecto de pandas,
que es el que usa el plugin, redondea la última cifra de algunos valores (≤ 1e-12 relativo), sin efecto
en las predicciones (ver resultados).

## Resultados comprobados en local (2026-11-26)

Plugin real con `TestClient` sobre `main.app` (`MODEL=ml38-...`, CPU, backend Keras torch, scikit-learn 1.5.2,
keras 3.15.1), MLflow simulado con un tracker local en una carpeta temporal.

- Inline: 21/21 JSON con `family`, `severity` y `severity_label` idénticos; |Δ `severity_confidence`| máx.
  6,0e-8 (tolerancia 0,05).
- Batch (`predict_batch_golden.csv`): 21/21 dentro de tolerancia, |Δ| máx. 1,2e-7 (<0,1 s).
- Batch con `train_feature_dataset_877.csv`: 877/877 familias y 877/877 severidades iguales a
  `predictions.csv`, |Δ confianza| máx. 1,2e-7.
- Train (síncrono, MLflow simulado): 200, n_samples 877, n_groups 114, 7 familias, 6 MLP de severidad,
  ninguna familia saltada. CV familia: accuracy 0,6168 ± 0,1117 y macro-F1 0,6064 ± 0,1105, idéntico al
  valor de referencia (KI-14) en las 3 ejecuciones. Tiempo de entrenamiento: 23,6 s y 28,2 s (petición
  25-30 s). Macro-F1 de CV de severidad en tres ejecuciones: alignment 0,49 / bearing 0,45-0,46 /
  cavitation 0,56 / electrical 0,03-0,20 / mechanical_other 0,35-0,36 / unbalance 0,62. El equipo declara
  0,37 / 0,45 / 0,56 / 0,00 / 0,32 / 0,73. Electrical es la familia débil documentada (KI-08). En el run
  de MLflow se suben los 4 ficheros (`family_rf.pkl`, `family_scaler.pkl`, `severity_models.pkl`,
  `severity_scalers.pkl`).
- Predict con el modelo reentrenado (`mlflow_run_id`): 200. Caso 001 → alignment / significativo (0,63-0,73
  según la ejecución). Batch golden: 21/21 familias iguales a las esperadas.
- Predict sin `mlflow_run_id` tras el train: sigue dando el valor del modelo base (caso 001: 0,719051).
  md5 de `artifacts/ml38_wine_machinery_fault_detection/*` sin cambios (los del manifest).
- Memoria como en el pod: servidor uvicorn real en CPU dentro de un scope systemd (cgroup v2), MLflow
  sqlite local y `POST /train?wait=false` (el entrenamiento va en un proceso hijo). Serving: 634 MiB anon.
  Pico serving + hijo: 1460 MiB anon (`memory.peak` 1523 MiB), por debajo del límite de 2 GiB. Job
  `succeeded` en 45 s (32 s de entrenamiento).
