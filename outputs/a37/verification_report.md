# Verificación — ml37-wine-press-predictive-maintenance

## [FUSIÓN a39 — 2026-11-24] Reverificación con la capa XAI integrada

ml37 sirve ahora también el entregable a39 (Integrated Gradients + SHAP por ciclo,
`app/plugins/ml37_wine_press_predictive_maintenance/xai.py`); el plugin
`ml39-wine-predictive-maintenance-xai` se ha eliminado. Ver la nota [FUSIÓN a39] de
`inbox/a37/manifest.yaml`. Versión 1.0.0 → 1.1.0.

### Checklist técnico

- [x] **flake8** (plugin, tests de ml37, `conftest.py`, `registry.py`): 0 errores.
- [x] **pylint** (plugin + tests de ml37): **10.00/10**.
- [x] **pytest, suite completa**: **1693 passed, 0 failed** (los dos fallos previos ajenos
  de ml10/numba_cache que citaba el informe de ml39 ya no aparecen en este entorno).
  - `test_ml37_xai.py` (17 tests, código real): IG frente a una referencia por diferencias
    finitas, lote = ciclo a ciclo, agregación por sensor, secuencia de `rng` del fondo SHAP,
    carga/escalado/rechazo del fondo, `xai_error`, tope `n_samples` de SHAP en batch,
    reentreno real de 1 época que genera su fondo, `train()` sube `shap_background.npy`, y
    2 golden cases de a39 con el plugin y el artefacto reales.
  - Pasan con Keras en backend **torch y tensorflow** (50/50 tests de ml37 en ambos).
- [x] **`from app.registry import REGISTRY`**: 38 entradas, sin ml39.
- [x] **Arranque real** (`MODEL=ml37-wine-press-predictive-maintenance`): 1/1 modelo listo;
  `/health` → `version 1.1.0, loaded: true`; `/stats` con `include_xai`/`include_shap` y las
  métricas XAI publicadas; `/models/ml39-…/health` → 404.
- [x] **`/predict` inline sin XAI, 20 golden cases de a37**: `predicho` 20/20, `risk` 20/20
  (±1e-4), `alertas` 20/20 — la parte predictiva no cambia.
- [x] **`/predict` inline por defecto (IG), 20 golden cases de a39**: ranking top-3 **80/80**
  cabezas; valor dentro del 1 % en 74/80. Las 6 restantes (ciclos 1126, 1175, 1275, de
  magnitud ~1e-5–1e-6) son el redondeo a 6 decimales del valor de referencia del manifest
  (p. ej. 7,73e-6 frente a "0.000008"): el mismo resultado que el informe de ml39.
- [x] **`include_shap=true`**: 200, `xai_error` null, y dos llamadas iguales dan la misma
  explicación (`rseed=42`). En el plugin ml39 esto devolvía 500 en todos los casos.
- [x] **`include_shap` sin `include_xai`** → 422. **`/train` sin `mlflow_run_id`** → 422.
- [x] **`include_xai=true` por defecto también en batch** (decisión del usuario, 2026-11-24; en batch suma ≈3 s por ciclo, usar `wait=false` para lotes grandes).
- [x] **Batch** con 2 ciclos (uno con PS1 = NaN) e `include_xai=true`: `n_rows=2`,
  `n_predictions=1`, el ciclo bueno con IG y el roto con su `error`.
- [x] **Coste en CPU en un cgroup** (`CUDA_VISIBLE_DEVICES=`, como los pods): sin XAI 0,5 s,
  IG 2,2–2,8 s, SHAP ≈65 s; memoria 0,61 GiB cargado, pico 1,41 GiB con SHAP (límite del pod
  2 GiB: dos SHAP simultáneos probablemente no caben).

### SHAP frente al entregable (perfil `full`)

Reproducción completa del barrido del perfil `full` de `xai_analysis.py` con las funciones del
plugin (`xai.integrated_gradients`, `xai.shap_attributions(rseed=None)`): misma muestra de 150
ciclos de test para IG, el mismo fondo de 80 ciclos de train (el `shap_background.npy`
publicado; comprobado que es exactamente `train_idx[rng…]` del original) y los mismos 30 ciclos
explicados, con `np.random.seed(42)` como hace `set_random_seed(42)`. Gradientes con torch, no
con TensorFlow.

| Cabeza | r Spearman IG-SHAP (plugin) | r del entregable (`xai_metrics.csv`) | p (plugin) |
|---|---|---|---|
| cooler | 0.9165 | 0.9165 | 4.084e-06 |
| valve | 0.6484 | 0.6484 | 0.01214 |
| pump | 0.6044 | 0.6044 | 0.02206 |
| accumulator | 0.7978 | 0.7978 | 0.0006283 |

Coinciden a 4 decimales, y los p-valores también. Es la primera verificación numérica de SHAP
frente al entregable: el informe de ml39 la había dejado pendiente.

### Pendiente fuera del repo

- Subir a `s3://<STORAGE_BUCKET>/artifacts/fixed/ml37_wine_press_predictive_maintenance/`
  `best_model.keras`, `scaler.json`, `class_mappings.json`, `train_config.json` (los del
  entregable, md5 en el manifest) y `shap_background.npy` (md5
  89f4838a69e2f940b0beef18a20f1a2f). Hoy no existe en S3 ni la carpeta de ml37 ni la de ml39.
- Fichas: `outputs/a39/datos_a39.json` sigue apuntando a `ml39-wine-predictive-maintenance-xai`
  y la ficha de a37 no menciona la explicabilidad; regenerar con `docs-generation`.

---

*Informe original (2026-11-18), sin reescribir:*


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
