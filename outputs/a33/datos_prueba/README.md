# Datos de prueba — ml33-cereals-reuse-strategy-optimizer

Extraídos de la entrega del equipo de IA (`inbox/a33/codigo/data/split/…_test_raw.csv`,
`data/predictions/inference_with_constraints.csv` y `golden_cases` del manifest), sin valores inventados.
Regenerar: `.venv/bin/python outputs/a33/datos_prueba/generar_datos_prueba.py "$PWD"`.

**No hay datos de entrenamiento:** ml33 es un optimizador MILP determinista sin pesos; `POST /train`
responde 501 (`TrainingNotSupportedError`).

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_bloque_001_filas_1_15.json`, `…_002_filas_16_30.json` | `POST /models/ml33-cereals-reuse-strategy-optimizer/predict` | Golden cases inline: 15 lotes cada uno, capacidades por defecto (90/140/45/10000 t, 15 lotes/día) |
| `predict_batch_150.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Primeras 150 filas del test (10 bloques), ~2 s |
| `predict_batch_test_completo.csv` | ídem | Split de test completo, 10 000 lotes, **~2 min** |
| `expected_predictions.csv` | — | Estrategia esperada por fila (las 10 000) del pipeline original |
| `expected_batch_test_completo.json` | — | Agregado esperado del batch completo (distribución, emisiones, fallbacks) |

No se incluyen escenarios con capacidades alteradas: los de `data/predictions/capacity_sensitivity/`
salen del motor NEAT anterior (`inference_base.csv` solo coincide en un 39 % con el MILP desplegado), así
que no sirven como referencia.

## Resultados comprobados en local (2026-11-04)

- Inline: los 2 bloques coinciden 15/15 con el manifest, sin fallbacks.
- Batch 150: 150/150 filas coinciden.
- Batch completo: 10 000/10 000 filas coinciden; distribución 37.18 / 27.27 / 17.86 / 17.69 %
  (composting / animal feed / biomass combustion / biochar), emisiones 30 893 908.4 kg y 0 fallbacks,
  idéntico a `inference_evaluation_report.json`.
