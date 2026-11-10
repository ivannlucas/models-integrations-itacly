# Datos de prueba — ml23-lactic-market-price-forecast

Extraídos de la entrega del equipo de IA (`inbox/a23/codigo/data/processed/dataset_forecast_ready.csv`,
`data/predictions/inference_predictions_gru.csv` y `golden_cases` del manifest), sin valores inventados.
Regenerar: `.venv/bin/python outputs/a23/datos_prueba/generar_datos_prueba.py "$PWD"`.

GRU que predice el precio medio de la leche (€/litro) a 6 meses por producto y canal, a partir de
ventanas de 6 meses.

> **Ojo con `/train`:** cada llamada publica una **nueva versión del modelo base** (`BASE_ARTIFACT_FOLDER`)
> y el servicio pasa a servirla. No lanzarlo contra un entorno compartido salvo que se quiera eso;
> para volver atrás, `POST /models/ml23-lactic-market-price-forecast/rollback`.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `dataset_forecast_ready.csv` | `POST /models/ml23-lactic-market-price-forecast/predict` con `{"mode": "batch", "data_path": "<ruta>"}` **y** `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": ""}` | Dataset del equipo de IA: 1440 filas (4 productos × 2 canales × meses) |
| `expected_batch.csv` | — | Predicción original de las 1360 ventanas completas (`esperado_y_pred`) |
| `expected_golden.csv` | — | Los 8 golden cases (fecha 2025-09-01, un producto × canal cada uno) |
| `predict_inline_<producto>_<canal>.json` | `POST .../predict` (inline) | Fila de features de cada golden case |

**Inline es aproximado por diseño:** recibe una sola fila y la replica 6 veces para formar la ventana
(documentado en el manifest y en `verification_report.md`), así que su salida no coincide con la de la
ventana real. Sirve para probar el endpoint, no la precisión; la comparación exacta es la del batch.

## Resultados comprobados en local (2026-11-09)

- Batch: 1360/1360 ventanas con diferencia máx. 5e-5 frente al original (redondeo a 4 decimales).
  La respuesta devuelve `fecha` como `2018-11-01 00:00:00`.
- Golden: 8/8 exactos.
- Inline: responde; p. ej. leche desnatada DISCOUNTS da 0.8934 frente a 0.9113 de la ventana real.
- Train (publicación del modelo base simulada, artefactos sin tocar): ~110 s en CPU, MAE 0.0135,
  RMSE 0.017, R² 0.8994, igual que el artefacto servido (`GRU_artifact_metrics`).
