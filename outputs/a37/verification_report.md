# Verificación — ml37-wine-press-predictive-maintenance

Fecha: 2026-11-18

## Checklist técnico

- [x] **flake8** (`app/plugins/ml37_wine_press_predictive_maintenance/`, tests nuevos):
  0 errores. (El `flake8 .` a nivel de repo muestra ruido pre-existente en
  `outputs/*/datos_prueba/*.py`, ajeno a este plugin.)
- [x] **pytest**: 38/38 pasados
  - `tests/unit/test_ml37_wine_press_predictive_maintenance.py` (9 tests, wiring con
    `FakePlugin`): health, stats, predict inline, predict inline sin `readings` → 422,
    `DataContractError` → 422 (inline y batch), predict batch, train 200, train sin
    `mlflow_run_id` → 422.
  - `tests/unit/test_ml37_preprocessing.py` (21 tests, código real — no `FakePlugin`):
    `resample_cycle_to_fixed_length` (interpolación, caso de 1 muestra, NaN/Inf, timestamps
    duplicados), `build_schema_v1_tensor`, `apply_scaler`, `compute_cycle_stats`,
    `parse_readings`, las 9 reglas neurosimbólicas una a una, el colapso legacy de
    `accumulator` (ver más abajo), `check_model` (sin downgrade / suprimido / degradado) y
    `fuse_alerts` (escalado por acuerdo regla+modelo).
  - `tests/unit/test_user_model_isolation.py` (4/4, fixture `ml37` nueva): ninguna petición
    sin `mlflow_run_id` ve el modelo de otro usuario, ni concurrente ni tras una petición con
    run solapada.
  - `tests/unit/test_user_model_download_hygiene.py` (4/4, aplica solo sin registrar):
    `_resolve_bundle` nunca asigna a `self`, `download_user_model_from_mlflow` limpia su
    temporal cuando el run no trae modelo.
  - Suite completa: 1525 pasados, 40 skipped, 3 fallos **pre-existentes y ajenos** a este
    plugin (`test_ml10_dairy_disease_vector_detection_unit`, `test_numba_cache`,
    `test_user_model_download_hygiene[ml3...]` — versiones de sklearn/numba y `tensorflow`
    no instalado en este entorno, nada relacionado con ml37).
- [x] **pylint** (`app/plugins/ml37_wine_press_predictive_maintenance/`): **10.00/10**. La
  primera pasada daba 9.53/10 (líneas largas, `invalid-name` en variables `X`/`X_scaled`,
  `too-many-branches` en el motor de reglas, `too-many-locals`/`too-many-arguments` en el
  entrenamiento) — se corrigió el código (renombrado a minúsculas, descomposición de
  `check_rule_evidence` en un helper por componente, descomposición de
  `train_lstm_multitask` en `_split_cycle_indices`/`_encode_targets`/`_fit_scaler`/
  `_build_sample_weights`/`_compile_and_fit`/`_evaluate_test`, `LstmTrainResult` con
  `metrics`/`split_sizes` anidados en vez de 10 campos sueltos), no se relajó ninguna regla.
  Único disable añadido: `too-many-instance-attributes` en `_MultiMetricEarlyStopping` (8
  atributos de estado genuino de early-stopping: config + mejor-hasta-ahora + contador de
  espera + pesos guardados, no un smell de modelado). Se re-verificaron los 20 golden cases y
  el smoke test de `train()` después del refactor: resultados idénticos.
- [x] **pip-audit**: no disponible en este entorno (`ensurepip`/`python3-venv` no instalado,
  pip-audit no puede crear su venv aislado) — misma limitación ya documentada para otros
  modelos de este repo, no específica de ml37.
- [x] **Arranque local real** (`MODEL=ml37-wine-press-predictive-maintenance python main.py`):
  1/1 modelo listo, sin warnings.
- [x] **`GET /health`**: `{"status":"ok","model":"ml37-wine-press-predictive-maintenance","version":"1.0.0","loaded":true}`
- [x] **`POST /predict` (inline)**: probado con el ciclo real 33 (600 lecturas reales de
  `data/processed/tensors/X.npy`) → predicciones, riesgo, alertas y `rule_evidence` idénticos
  al golden case (ver tabla abajo).
- [x] **`GET /stats`**: `model_name`, `task_type=classification_multitask_sequence + neurosymbolic_rule_fusion`,
  1 input (`readings`), métricas reales de `METRICS_REPORTED`.
- [x] **`POST /train` sin `mlflow_run_id`**: 422 (`Field required`) — `TrainRequest.mlflow_run_id`
  es obligatorio.
- [x] **`POST /train` con `mlflow_run_id` y 60 ciclos reales** (20 por clase `cooler`, para que
  el split estratificado no falle): entrena de verdad (349 s en CPU) y responde **502**
  (`ModelPersistenceError`) porque este entorno no tiene MLflow accesible
  (`mlflow.mlflow:5000` no resuelve) — comportamiento correcto según el contrato: un fallo de
  subida nunca debe devolver 200. `training.supported: true`, confirmado.

## Correctitud (golden dataset)

Fuente: 20 ciclos reales de `data/predictions/no_virtual_v1__rand_scool/predictions_test.parquet`
(predicción real del equipo de IA, no la etiqueta — las 4 cabezas rara vez coinciden todas con
el ground truth real, por eso el golden case compara contra la salida real del modelo
entregado) + `alerts_all.parquet` (alertas reales) + `data/splits/.../splits.json::test_idx`
(confirmación de que son ciclos de test). Tolerancia: coincidencia exacta en las 4 clases
predichas (son códigos categóricos reales, no continuos) y en `risk_*` a 3 decimales (1e-3,
más estricto que el 5% por defecto porque el riesgo se calcula determinísticamente a partir
del mismo softmax); el conjunto de alertas fusionadas (componente, fuente, severidad) debe
coincidir exactamente con `alerts_all.parquet`.

| Ciclo | cooler | valve | pump | accumulator | risk OK | alerts OK |
|---|---|---|---|---|---|---|
| 33   | 3   | 100 | 0 | 130 | ✓ | ✓ |
| 220  | 3   | 100 | 1 | 130 | ✓ | ✓ |
| 340  | 3   | 100 | 0 | 130 | ✓ | ✓ |
| 556  | 3   | 100 | 1 | 100 | ✓ | ✓ |
| 593  | 3   | 100 | 0 | 130 | ✓ | ✓ |
| 688  | 3   | 100 | 1 | 100 | ✓ | ✓ |
| 858  | 20  | 100 | 0 | 90  | ✓ | ✓ |
| 1126 | 20  | 73  | 1 | 115 | ✓ | ✓ |
| 1175 | 20  | 73  | 1 | 115 | ✓ | ✓ |
| 1206 | 20  | 100 | 2 | 100 | ✓ | ✓ |
| 1275 | 20  | 80  | 2 | 100 | ✓ | ✓ |
| 1471 | 100 | 100 | 0 | 90  | ✓ | ✓ |
| 1722 | 100 | 100 | 1 | 100 | ✓ | ✓ |
| 1742 | 100 | 100 | 1 | 100 | ✓ | ✓ |
| 1792 | 100 | 90  | 1 | 100 | ✓ | ✓ |
| 1821 | 100 | 100 | 1 | 100 | ✓ | ✓ |
| 1842 | 100 | 90  | 1 | 100 | ✓ | ✓ |
| 1876 | 100 | 90  | 1 | 100 | ✓ | ✓ |
| 1919 | 100 | 100 | 1 | 100 | ✓ | ✓ |
| 2169 | 100 | 100 | 1 | 100 | ✓ | ✓ |

**Resultado: 20/20 ciclos con las 4 cabezas exactas (80/80 clases individuales exactas),
20/20 con riesgo dentro de tolerancia y 20/20 con el conjunto de alertas idéntico.**

Verificado dos veces: antes y después del refactor de pylint (misma salida exacta, ver
checklist técnico arriba).

## Notas adicionales

- **El dataset es un proxy técnico, no telemetría real de bodega** (declarado explícitamente
  por el equipo de IA en la memoria y replicado en `constants.py`/`plugin.py`): el modelo usa
  el dataset público UCI "Condition Monitoring of Hydraulic Systems" como sustituto funcional
  de prensas/bombeo/filtración/embotellado. No es un hallazgo a corregir, es la base declarada
  del modelo.
- **`risk_*` es incertidumbre predictiva (1 − max(softmax)), no riesgo físico** — documentado
  en `constants.py`, `plugin.py` y `inference.py` para que nadie lo interprete como severidad
  de fallo.
- **Quirk real del colapso de alertas de `accumulator`, replicado a propósito**: la vista
  "legacy" (`alerts`, lo que alimenta `alerts_all.parquet` en el entregable original) colapsa
  varias alertas de regla del mismo componente con un diccionario simple — gana la última
  evaluada. Como `accumulator` puede disparar independientemente la regla de `PS1_mean` (p.ej.
  CRITICAL) y la de `PS1_std` (p.ej. WARNING), la vista legacy puede perder silenciosamente la
  más grave. El propio equipo de IA es consciente de esto (`would_be_overwritten_in_legacy` en
  su detalle de evidencia) — no se ha "arreglado": se replica el colapso en `alerts` tal cual,
  y se expone la evidencia completa sin colapsar en `rule_evidence` para que el consumidor
  pueda ver ambas alertas si lo necesita. Cubierto por
  `test_accumulator_legacy_collapse_can_silently_drop_the_critical_alert` y documentado en
  `inbox/a37/manifest.yaml::known_issues` y en el docstring de `rules.py`.
- **Duración real de entrenamiento medida en este entorno**: 305–349 s en CPU para un smoke
  test de 60 ciclos (20 por clase `cooler`, 36 000 filas de sensor). La estimación inicial del
  manifest ("minutos, no horas") no estaba verificada y se corrigió con esta medición real —
  el entrenamiento de referencia completo (1543 ciclos) no se ha medido, pero no debe asumirse
  que escale linealmente a "minutos"; se recomienda `/train?wait=false` (infraestructura
  asíncrona ya disponible en el repo) en vez de asumir que la llamada síncrona es aceptable en
  producción.
- **Verificación con Keras 3 standalone, sin TensorFlow**: `tensorflow==2.19.0` no tiene wheel
  para Python 3.14 en este entorno (solo hay `2.22.0rc0`). Se verificó que
  `best_model.keras` carga sin pérdida de fidelidad con el paquete `keras` 3.15.1 +
  `KERAS_BACKEND=torch` (cero dependencia de TensorFlow): arquitectura exacta
  (Masking → LSTM(64) → Dense(64) → 4 heads softmax, 31 950 parámetros) y los 20 golden cases
  reproducidos de forma exacta.
- **Reentrenamiento (`train()`)**: `training.supported: true`. Refit completo desde cero
  (sin fine-tuning, igual que el original) con el mismo split
  (`random_stratified_cooler`, 70/15/15, seed=42), mismos hiperparámetros (`TRAIN_*` en
  `constants.py`, citando `train_config.json` real) y los mismos dos *callbacks*
  (`_MultiMetricEarlyStopping` sobre `val_valve_loss+val_pump_loss+val_accumulator_loss`,
  `ReduceLROnPlateau`). Verificado end-to-end con un smoke test real de 60 ciclos: entrena,
  evalúa en el holdout de test y solo falla al subir a MLflow (502, por la falta de MLflow en
  este entorno) — nunca se guarda localmente ni sobrescribe el artefacto base.
- **Artefactos**: copia local en `artifacts/ml37_wine_press_predictive_maintenance/`
  (`best_model.keras` 434 943 bytes, `scaler.json`, `class_mappings.json`, `train_config.json`)
  desde `inbox/a37/codigo/models/artifacts/no_virtual_v1__rand_scool/` — variante de
  referencia elegida por el propio equipo de IA.

## Estado final

**[ LISTO PARA PR ]**
