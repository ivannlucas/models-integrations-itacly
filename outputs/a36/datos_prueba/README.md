# Datos de prueba — ml36-dairy-dnl-co2-emissions-optimizer

Extraídos de la entrega del equipo de IA (`inbox/a36/codigo/data/splits/`, `data/predictions/`,
`data/processed/df_pasteurizacion_master.csv` y `golden_cases` del manifest), sin valores inventados.
Regenerar: `.venv/bin/python outputs/a36/datos_prueba/generar_datos_prueba.py "$PWD"`.

El modelo tiene dos operaciones en `POST /models/ml36-dairy-dnl-co2-emissions-optimizer/predict`:
gemelo digital (MLP: `T_out` y CO2 a partir de las 8 variables de proceso) y `model_key: "optimize"`
(GA: recomienda `T_serv`, `Delta_P`, `Regeneration_perc` minimizando CO2 con `T_out >= 72.5 °C`).

| Fichero | Petición | Contenido |
|---|---|---|
| `predict_inline_predict_001…018.json` | inline (MLP) | 18 golden cases del split de test |
| `predict_inline_optimize_001…004.json` | inline, `model_key: "optimize"` | 4 golden cases de validación (modo `hybrid`, semilla original `43 + fila`, con la operación actual) |
| `predict_batch_mlp_test.csv` | `{"mode": "batch", "data_path": "<ruta>"}` | Split de test completo, 7704 filas (8 variables base), ~30 s |
| `expected_mlp_test.csv` | — | Entrada + predicción original (`esperado_*`) de las 7704 filas |
| `predict_batch_optimize_val20.csv` | `{"mode": "batch", "model_key": "optimize", "data_path": "<ruta>"}` | 20 primeras filas de validación con `seed`, `decision_mode` y operación actual, ~17 s (el GA va fila a fila) |
| `expected_optimize_val20.csv` | — | Decisiones originales de esas 20 filas (`realtime_decisions.csv`) |
| `train_40ciclos.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | 40 ciclos completos del dataset maestro (4542 filas, con `cycle_id`) |

## Resultados comprobados en local (2026-11-04)

- MLP inline (18) y batch (7704): diferencia máx. 6e-5 con la predicción original (redondeo a 4 decimales).
- Optimize inline (4) y batch (20): setpoints, CO2 y `T_out` con diferencia máx. 5e-5.
- Train: ~20 s, 3167/688/687 filas train/val/test, 400 épocas; R² `T_out` 0.986 y CO2 0.906. Las métricas
  no son las de la memoria porque se usan 40 de los 452 ciclos: el fichero sirve para probar el flujo.
  Para reproducir el modelo entregado, usar `inbox/a36/codigo/data/processed/df_pasteurizacion_master.csv`
  completo (51 389 filas; tarda bastante más).
