# Datos de prueba — ml26-wine-sulfite-gru-pso-forecast

Extraídos del dataset del equipo de IA (`inbox/a26/codigo/data/raw/sequential.csv.gz`), sin valores
inventados. Regenerar: `.venv/bin/python outputs/a26/datos_prueba/generar_datos_prueba.py "$PWD"`.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_NNN.json` | `POST /models/ml26-wine-sulfite-gru-pso-forecast/predict` | Body inline (lote + historial de lecturas) de cada uno de los 18 golden cases del manifest |
| `predict_batch.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 18 lotes golden en un CSV; devuelve 1 predicción por lote |
| `expected_predictions.csv` | — | Salida esperada por caso (pipeline original) + target real (`y_true_*`, informativo) |
| `train.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | 12 lotes completos del split de **train** (4 red, 4 rose, 4 white), 1586 filas |

Lecturas en "modo bodega" (transformación del manifest): `ph_lab`/`free_sulfite_lab`/`total_sulfite_lab`
solo en filas con muestra de laboratorio y `dose_mg_l` solo cuando hubo adición.

Comprobado en local (2026-11-04): inline y batch reproducen `expected_predictions.csv` con diferencia
máx. 4e-6; `train` termina en ~9 s (8/1/3 lotes train/val/test). Las métricas de ese reentreno son
peores que las de la memoria porque se usan solo 12 lotes: el fichero es para probar el flujo, no la calidad.
