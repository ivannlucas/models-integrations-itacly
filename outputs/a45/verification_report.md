# Verificación — ml45-cereals-dnsl-critical-point-detection (a45)

## Checklist técnico

- [x] flake8: 0 errores (`app/plugins/ml45_cereals_dnsl_critical_point_detection/`, `app/registry.py`,
      `app/domain/services/exceptions.py`, `tests/conftest.py`, `tests/unit/test_ml45_cereals_dnsl_critical_point_detection.py`)
- [x] pytest: 333/333 passed (suite completa del repo). El test dedicado del modelo
      (`tests/unit/test_ml45_cereals_dnsl_critical_point_detection.py`, 7 casos) usa `FakePlugin`
      — valida wiring HTTP (health/stats/predict inline+batch/train/mapeo de excepciones a 422),
      no correctitud. La correctitud contra el golden dataset se valida en la Parte B de este
      informe, con el plugin real y los artefactos reales.
- [x] pylint: `app/plugins/ml45_cereals_dnsl_critical_point_detection/` puntúa 8.79/10. Sin
      issues bloqueantes; los avisos restantes (too-many-locals/branches/arguments, nombres tipo
      `T`/`F`/`X_bg` que no son snake_case, líneas largas) están todos en `_vendor/` — son
      inherentes al código matemático vendorizado verbatim del equipo de IA, mismo patrón y
      orden de magnitud que el precedente ya aceptado en este repo (`ml46_dairy_fouling_clog_detection`
      puntúa 8.14/10 con el mismo tipo de avisos en su propio `_vendor/`).
- [x] pip-audit: sin CVEs nuevas atribuibles a este cambio. `pip-audit -r requirements.txt`
      reporta 3 vulnerabilidades preexistentes del entorno (torch 2.11.0 → PYSEC-2025-194,
      setuptools 81.0.0 → PYSEC-2026-3447 x2), ninguna relacionada con `shap` (la única
      dependencia nueva añadida por este modelo) ni con el resto de paquetes ya fijados en
      `requirements.txt` para otros plugins.
- [x] Arranque local + health + predict + stats: OK. Servido con
      `CUDA_VISIBLE_DEVICES="" STORAGE_BUCKET="" MODEL=ml45-cereals-dnsl-critical-point-detection
      uvicorn main:app` (CPU + artefactos locales, sin S3). `/health` → `loaded: true`.
      `/stats` → metadata + métricas reales del checkpoint. `/predict` (batch, CSV real del test
      split) → reproduce exactamente el JSON del PCC monitor original (ver Parte B).
- [x] /train: 200 con `TrainResponse` (fine-tuning real ejecutado sobre un ciclo de prueba,
      `accuracy=0.75 f1=0.8571 n_windows=4 n_epochs=30` — números bajos porque el smoke test usó
      un único ciclo de 4 ventanas, no un dataset de fine-tuning real; el endpoint y el pipeline
      funcionan correctamente). `manifest.training.supported=true`, comportamiento esperado.
      **Nota de proceso**: `/train` sobrescribe el checkpoint local en
      `artifacts/ml45_cereals_dnsl_critical_point_detection/best_dnf_model.pt` — se restauró el
      checkpoint original entregado (`inbox/a45/codigo/models/artifacts/best_dnf_model.pt`,
      md5 `5eb670db5f01d91299fe34d3151389e7`) después de cada prueba de `/train`, verificado con
      md5sum antes de ejecutar la Parte B.

## Correctitud (golden dataset)

Fuente: los 6 `golden_cases` de `inbox/a45/manifest.yaml`, cada uno un ciclo completo (600 filas)
de `data/splits/test.csv` del equipo de IA → 4 ventanas de salida por ciclo (24 ventanas en
total). El manifest ya documentaba valores de referencia obtenidos ejecutando el pipeline
**original** del equipo de IA (`scripts/predict.py`, checkpoint entregado, CPU, seed=42). Esta
verificación reproduce esos mismos 6 casos, pero contra el **plugin real** integrado en este
repo, servido vía HTTP (`POST /predict`, `mode=batch`) — no se reutiliza el cálculo del
manifest, se vuelve a invocar el endpoint real.

| Caso | Ventana | Esperado (clase, prob) | Obtenido (clase, prob) | Diferencia | ¿OK? |
|---|---|---|---|---|---|
| cycle_2400_normal | 1 | 0, 0.4300 | 0, 0.4300 | 0.0000 | ✅ |
| cycle_2400_normal | 2 | 0, 0.6315 | 0, 0.6315 | 0.0000 | ✅ |
| cycle_2400_normal | 3 | 0, 0.0863 | 0, 0.0863 | 0.0000 | ✅ |
| cycle_2400_normal | 4 | 0, 0.0325 | 0, 0.0325 | 0.0000 | ✅ |
| cycle_2405_discharge_jam | 1 | 1, 0.9576 | 1, 0.9576 | 0.0000 | ✅ |
| cycle_2405_discharge_jam | 2 | 1, 1.0000 | 1, 1.0000 | 0.0000 | ✅ |
| cycle_2405_discharge_jam | 3 | 1, 1.0000 | 1, 1.0000 | 0.0000 | ✅ |
| cycle_2405_discharge_jam | 4 | 1, 1.0000 | 1, 1.0000 | 0.0000 | ✅ |
| cycle_2407_humidity_sensor_drift | 1 | 0, 0.3549 | 0, 0.3549 | 0.0000 | ✅ |
| cycle_2407_humidity_sensor_drift | 2 | 0, 0.2031 | 0, 0.2031 | 0.0000 | ✅ |
| cycle_2407_humidity_sensor_drift | 3 | 0, 0.2975 | 0, 0.2975 | 0.0000 | ✅ |
| cycle_2407_humidity_sensor_drift | 4 | 0, 0.1375 | 0, 0.1375 | 0.0000 | ✅ |
| cycle_2412_filter_clogged | 1 | 0, 0.4072 | 0, 0.4072 | 0.0000 | ✅ |
| cycle_2412_filter_clogged | 2 | 1, 0.9998 | 1, 0.9998 | 0.0000 | ✅ |
| cycle_2412_filter_clogged | 3 | 1, 1.0000 | 1, 1.0000 | 0.0000 | ✅ |
| cycle_2412_filter_clogged | 4 | 1, 1.0000 | 1, 1.0000 | 0.0000 | ✅ |
| cycle_2423_burner_degraded | 1 | 0, 0.4486 | 0, 0.4486 | 0.0000 | ✅ |
| cycle_2423_burner_degraded | 2 | 1, 0.7942 | 1, 0.7942 | 0.0000 | ✅ |
| cycle_2423_burner_degraded | 3 | 1, 0.9999 | 1, 0.9999 | 0.0000 | ✅ |
| cycle_2423_burner_degraded | 4 | 1, 1.0000 | 1, 1.0000 | 0.0000 | ✅ |
| cycle_2440_plenum_thermal_leak | 1 | 0, 0.3478 | 0, 0.3478 | 0.0000 | ✅ |
| cycle_2440_plenum_thermal_leak | 2 | 0, 0.6668 | 0, 0.6668 | 0.0000 | ✅ |
| cycle_2440_plenum_thermal_leak | 3 | 1, 0.9999 | 1, 0.9999 | 0.0000 | ✅ |
| cycle_2440_plenum_thermal_leak | 4 | 1, 0.9999 | 1, 0.9999 | 0.0000 | ✅ |

Tolerancia usada: `atol=0.005` en `anomaly_probability` + coincidencia exacta en
`predicted_anomaly_class` (definida en `manifest.golden_cases[*].tolerance`, no un default del
5% — el manifest ya traía una tolerancia explícita basada en reproducibilidad exacta del
checkpoint, no en el error de test reportado).

Resultado: **24/24 ventanas dentro de tolerancia** (diferencia real = 0.0000 en todos los casos —
reproducción bit-a-bit, no solo "dentro de tolerancia").

Nota sobre `cycle_2407_humidity_sensor_drift`: este ciclo SÍ tiene una falla real
(HUMIDITY_SENSOR_DRIFT) pero el modelo entregado clasifica las 4 ventanas como "No Fallo" — un
falso negativo real y ya documentado en el manifest (consistente con `fallo_recall=0.656` en
`metrics_reported`). El plugin reproduce fielmente ese comportamiento (correcto = fiel al
modelo entregado, no que el modelo acierte siempre); no se ha ajustado nada para "corregirlo".

## Estado final

**LISTO PARA PR** — checklist técnico en verde, 24/24 golden cases reproducidos exactamente
contra el plugin real servido por HTTP. Puntos a destacar para el revisor humano:

1. Este plugin porta la capa XAI/PCC (SHAP + reglas fuzzy) **inline**, añadiendo `shap` a
   `requirements.txt` del proyecto raíz — desviación deliberada del patrón habitual del repo
   (donde SHAP se delega al microservicio externo de explicabilidad), documentada en
   `inbox/a45/manifest.yaml::known_issues` y confirmada explícitamente por el usuario antes de
   implementarla.
2. No hay memoria (`.docx`) del equipo de IA para este modelo — todo el manifest y esta
   verificación se basan en el código entregado y en resultados reales ya calculados
   (`models/metrics/results.json`), nunca en contenido inventado.
3. `train()` implementa fine-tuning simplificado (mismo loss/optimizador que el original, epochs
   fijos, sin early stopping/scheduler/checkpointing en disco) — el pipeline original de 268
   líneas es un entrenamiento desde cero con esa maquinaria, no aplicable a un fine-tuning sobre
   un CSV de usuario. Sigue el mismo patrón ya establecido en `ml34`/`ml35`.
4. No existe un modo `predict_inline` de "una fila = una predicción": la unidad mínima de
   inferencia es una ventana de 240 lecturas consecutivas, igual que en el precedente
   `m47_dnsl_fallas_maquinaria_pasteurizado`.
