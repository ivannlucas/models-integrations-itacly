# Datos de prueba — ml48-dairy-dnsl-critical-point-detection

Extraídos de la entrega del equipo de IA (`inbox/a48/codigo/data/processed/hydraulic_10hz_raw.csv`,
`golden_cases` y `golden_cases_xai` del manifest), sin valores inventados. Regenerar:
`.venv/bin/python outputs/a48/datos_prueba/generar_datos_prueba.py "$PWD"`. El manifest cita
`codigo/golden/golden_cycles_10hz.csv`, que no viene en la entrega: los mismos ciclos se toman del CSV a 10 Hz.

Datos del banco hidráulico UCI (aceite, TS1 ≈ 35-45 °C). **Los ficheros de predicción están en bruto y
se envían con `apply_digital_twin: true`** (ya incluido en los JSON inline; en batch hay que añadirlo al
body). Sin él las predicciones no son válidas para estos datos.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_cycle_NNNN.json` | `POST /models/ml48-dairy-dnsl-critical-point-detection/predict` | 20 golden cases (ciclos de test que cubren las 12 parejas componente/clase) |
| `predict_inline_xai_local_cycle_123.json` | ídem | Ciclo 123 con `include_xai` + `include_shap` (~5 s) |
| `expected_xai_local_cycle_123.json` | — | Explicación esperada por componente (memoria v1.3, Tabla 3) |
| `predict_batch_golden.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>", "apply_digital_twin": true}` | Los 20 ciclos en un CSV; Grad-CAM incluido por defecto (`include_xai`) |
| `expected_predictions.csv` | — | Estado esperado por componente (= etiqueta real de `profile.txt`) |
| `train_100ciclos.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | 100 ciclos del split de train **ya en dominio planta** (offset del gemelo digital aplicado: `fine_tune` no lo aplica) |

No se incluye el caso XAI global (`xai_global_50_test_cycles`): ejecuta SHAP sobre todo el test (~10 min).

## Resultados comprobados en local (2026-11-04)

- Inline y batch: 20/20 ciclos con los 4 componentes idénticos al manifest.
- XAI ciclo 123: 37/37 campos idénticos (predicción, confianza, sensor más relevante, importancia SHAP,
  ventana crítica, urgencia, riesgo).
- Train (`fine_tune` por defecto): ~4 s, 70/15/15 ciclos, exact_match 1.0 en los 15 de test.
