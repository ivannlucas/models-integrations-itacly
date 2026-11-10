# Datos de prueba — ml24-dairy-spatial-price-forecast

Extraídos de la entrega del equipo de IA (`inbox/a24/codigo/data/processed/series_consumo/`,
`data/predictions/spatial_price_model/inference_smoke.csv` y `golden_cases` del manifest), sin valores
inventados. Regenerar: `.venv/bin/python outputs/a24/datos_prueba/generar_datos_prueba.py "$PWD"`.

CatBoost + baseline calibrado que predice el precio medio de TOTAL LECHE LIQUIDA por comunidad autónoma
a 1, 3 y 6 meses. **Inline no recibe datos:** filtra por `ccaa`, `horizons` y `as_of_period` (mes de
corte, para backtesting) sobre la serie del modelo base. Batch y train reciben la serie en formato largo.

> **Ojo con `/train`:** cada llamada publica una **nueva versión del modelo base**, y la serie enviada pasa
> a ser la del modelo base con la que se predice. No lanzarlo contra un entorno compartido salvo que se
> quiera eso; para volver atrás, `POST /models/ml24-dairy-spatial-price-forecast/rollback`.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `serie_5_productos.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` **y** `POST .../train` | Serie MAPA 1999-2026 de los 5 productos que usa el modelo (la entregada pesa 337 MB con todos los productos) |
| `expected_batch_ultimo_mes.csv` | — | Predicción original del último mes (17 CCAA × 3 horizontes) |
| `predict_inline_h<h>_<mes>_<CCAA>.json` | `POST .../predict` (inline) | 18 golden cases de backtesting sobre el test 2020 |
| `expected_golden.csv` | — | Salida del predictor original (`esperado_y_pred_original`) y precio real (`y_true`) |
| `predict_inline_rechazo_*.json` | ídem | 2 casos que deben responder 422 (`expected_rechazos.csv`) |

Train sin más parámetros usa como test el último año completo de la serie (2025) y como validación el
anterior. Para reproducir la memoria, añadir `"validation_year": 2019, "test_year": 2020` al body.

## Resultados comprobados en local (2026-11-09)

- Golden inline: diferencia máx. 4.8e-7 con el predictor original en los 18. Frente al precio real,
  13/18 dentro de tolerancia, como en `verification_report.md` (los 5 restantes están investigados allí).
- Rechazos: `ESP` → `UnknownCcaaError`; `as_of_period` 1999-06 → `InsufficientPriceHistoryError`.
- Batch: 51/51 predicciones idénticas a `inference_smoke.csv` (~7 s).
- Train (publicación del modelo base simulada, artefactos sin tocar), ~70 s en CPU:
  - con 2019/2020: filas 3655/3621/3366 + 204 + 204 (= memoria Tabla 5); sMAPE test 1.91 / 1.95 / 2.09.
  - por defecto (test 2025): sMAPE test 5.21 / 4.27 / 3.37, peor que el baseline (3.90 / 3.51 / 2.99),
    con `ml_alpha` 0.0 en h1 y h3.
