# Datos de prueba — ml30-meat-traceability-detection

Extraídos de la entrega del equipo de IA (`inbox/a30/codigo/data/splits/` y
`inbox/a30/golden_cases_test_selected.csv`), sin valores inventados. Regenerar:
`.venv/bin/python outputs/a30/datos_prueba/generar_datos_prueba.py "$PWD"` (ejecuta el predictor original
para los valores esperados del split de test).

**Solo hay predicción batch:** `mode: inline` está desactivado por decisión de producto (`PredictRequest =
PredictBatchRequest`, devuelve 422).

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_batch_golden.csv` | `POST /models/ml30-meat-traceability-detection/predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 18 golden cases (`row_id`, `event_uid` + 33 features) |
| `expected_predictions.csv` | — | Clase y score esperados de los 18 golden cases (pipeline original) + etiqueta real |
| `predict_batch_test.csv` | `POST .../predict` (batch) | Split de test completo: 379 eventos |
| `test_labels.csv` | — | Etiqueta real + predicción del pipeline original de las 379 filas; `sensor_numerico_vacio` marca filas con un sensor vacío |
| `train.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | Split de train: 1786 eventos, 33 features + `target_traceability_incident` (21.4 % incidencias) |

## Resultados comprobados en local (2026-11-04)

- **Golden (18 filas):** clase idéntica y score con diferencia máx. 1.6e-8.
- **Test (379 filas):** acc 0.8707, F1 0.6016 y AUC 0.7185, idénticos a la memoria. Hasta el
  2026-11-09 el AUC salía 0.7117: el plugin rellenaba con `0.0` los sensores numéricos vacíos
  (6 filas del test) antes del preprocesador, en vez de dejar que imputara la mediana como el
  original. Corregido; ver `verification_report.md`.
- **Train:** ~5 s, 1428/358 filas, acc 0.8324, F1 0.5833, AUC 0.7469. Con el relleno con 0
  anterior salía acc 0.8436 / F1 0.5942 / AUC 0.7675: el 0 (pH 0, 0 kg) delataba al MLP que
  faltaba la lectura, y en este CSV las filas con un sensor vacío tienen más incidencias (33 %
  frente a 21 %). El original no usa esa señal (calcula `miss_sensor_any` pero no es feature) e
  imputa la mediana, que es lo que hace ahora el plugin.
