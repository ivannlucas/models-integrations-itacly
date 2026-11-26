# Datos de prueba — ml37-wine-press-predictive-maintenance

LSTM multitarea que clasifica 4 componentes (`cooler`, `valve`, `pump`, `accumulator`) a partir de un
ciclo de 14 sensores físicos, con la capa neurosimbólica de alertas y la explicabilidad de a39
(Integrated Gradients por defecto, SHAP opcional). Extraídos de la entrega del equipo de IA
(`inbox/a37/codigo/data/`), sin valores inventados. Regenerar:
`.venv/bin/python -I outputs/a37/datos_prueba/generar_datos_prueba.py "$PWD"`

Datos del banco hidráulico UCI (proxy técnico, no telemetría de bodega). Cada ciclo son las 600
lecturas a 10 Hz del tensor del equipo de IA (`data/processed/tensors/X.npy`, los 14 sensores
físicos) con `t` = paso / 10 s: la misma entrada que generó los golden cases.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_cycle_NNNN.json` | `POST /models/ml37-wine-press-predictive-maintenance/predict` | 20 golden cases del manifest (split de test): `readings` con 600 lecturas `{t, PS1…VS1}` e `include_xai: true` (IG, ≈2 s) |
| `predict_inline_shap_cycle_0033.json` | ídem | Ciclo 33 con `include_shap: true` (≈65-70 s en CPU) |
| `predict_batch_golden.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 20 ciclos en formato long (`cycle_id`, `t`, 14 sensores); IG incluido por defecto (≈1,7 s por ciclo) |
| `expected_predictions.csv` | — | Por ciclo: etiqueta real (`real_*`, `y.npy`), clase predicha y `risk` (manifest), probabilidades (`predictions_test.parquet` del equipo de IA) y alertas `componente:fuente:severidad` (manifest) |
| `expected_ig_top3.json` | — | Top-3 sensores por \|IG\| y cabeza (golden cases de `inbox/a39/manifest.yaml`, 6 decimales) |
| `train_100ciclos.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | Los 100 primeros ciclos de `train_idx` (`splits.json` de la variante servida) con `y_cooler`, `y_valve`, `y_pump`, `y_accumulator` e `y_stable` |

Las predicciones esperadas son las del modelo, no la etiqueta real: solo 5 de los 20 ciclos aciertan
las 4 cabezas (columna `heads_acertadas`; valve/pump/accumulator rondan 47-65 % de accuracy en el
test oficial). Las alertas se comparan como conjunto (el orden de la respuesta puede variar). No hay
golden por ciclo de SHAP: el entregable solo publica su concordancia agregada con IG.

## Resultados comprobados en local (2026-11-26)

Con el plugin real y `TestClient` (router de `make_model_router`, plugin importado vía
`app.registry`, así que Keras va con backend torch como en el pod) y CPU (`CUDA_VISIBLE_DEVICES=`).

- Inline: 20/20 ciclos con las 4 cabezas idénticas al manifest, 20/20 con el mismo conjunto de alertas,
  `risk` con diferencia 0 y probabilidades con diferencia máxima 1e-6 (el redondeo del parquet).
  `xai_error` null en los 20. Tiempo medio por llamada con IG: 1,9 s.
- IG frente a a39: ranking top-3 idéntico en 80/80 cabezas; 230/240 valores dentro del 1 %. Los 10
  restantes (ciclos 1126, 1175 y 1275, magnitudes de 8e-6 a 3e-5) coinciden exactamente al redondear a
  los 6 decimales del manifest: es la precisión del valor de referencia, no una divergencia.
- Batch: 200 en 34 s, `n_rows=20`, `n_predictions=20`; mismos resultados que inline (risk e IG
  idénticos bit a bit en los 20 ciclos).
- SHAP (ciclo 33): 200 en 62-73 s, `xai_error` null y dos llamadas con la misma explicación
  (`rseed=42`). Top-3 de referencia del checkpoint servido: cooler TS4 1,21e-4 · PS6 1,09e-4 · PS5
  1,07e-4; valve TS1 3,99e-5 · PS4 2,52e-5 · TS3 2,36e-5; pump TS1 6,67e-5 · TS4 5,82e-5 · VS1 5,34e-5;
  accumulator VS1 1,17e-4 · TS4 1,02e-4 · PS5 9,9e-5.
- Train (MLflow simulado, artefacto base sin tocar): 200 en 48 s en CPU; split 70/15/15; 15 épocas
  (early stopping). Test (15 ciclos): accuracy cooler 0,867 · valve 0,533 · pump 0,400 · accumulator
  0,333; F1 macro 0,861 · 0,174 · 0,269 · 0,250. Sube `best_model.keras`, `scaler.json`,
  `class_mappings.json` y `shap_background.npy` al run, y `/predict` con ese `mlflow_run_id` responde
  200 con IG. Es una prueba de humo con 100 ciclos; el modelo servido se entrenó con 1.543.
