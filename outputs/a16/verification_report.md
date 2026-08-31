# Verificación — ml16-meat-raw-material-price-alert (a16)

Fecha: 2026-08-31 (checklist técnico re-ejecutado el mismo día sobre `.venv/` del repo, con
`torch`/`tensorflow`/`cv2` ya instalados — sustituye la primera pasada, hecha en un sandbox sin
esas dependencias)
Plugin: `app/plugins/ml16_meat_raw_material_price_alert/`
Manifest: `inbox/a16/manifest.yaml`
Entorno de verificación: `.venv/` del repo — pandas 2.3.3, scikit-learn 1.9.0, xgboost 3.4.1

## Checklist técnico

- [x] **flake8** (`flake8 . --extend-exclude=dist,build,.venv --show-source --statistics`):
      **0 errores** en `app/plugins/ml16_meat_raw_material_price_alert/*.py`, `app/registry.py`,
      `tests/conftest.py`, `tests/unit/test_ml16_*.py`. El repo completo sí reporta cientos de
      avisos, pero todos caen en `inbox/*/codigo/` (código crudo entregado por los equipos de IA
      de otros modelos, nunca pasado por lint) — ninguno en ficheros tocados por esta integración.
      Config del repo: `max-line-length=120`, `extend-ignore=E501`.
- [x] **pytest** (`python -m pytest tests/unit/ --cov=app -v`): **400/400 passed**, 0 fallos, 0
      errores de colección (con `torch` instalado ya no quedan los 13 fallos de
      `ModuleNotFoundError: torch` de la primera pasada). `tests/unit/test_ml16_meat_raw_material_price_alert.py`
      (10) + `tests/unit/test_ml16_preprocessing.py` (8) → 15/15 dentro de ese total.
      Cobertura del plugin vía la suite de endpoints (`FakePlugin`, valida wiring/contrato HTTP,
      no artefactos reales — **mismo patrón que el resto de los ~25 plugins del repo**, todos con
      0% en su `plugin.py` bajo esta suite): `preprocessing.py` 75%, `postprocessing.py` 56%
      (cubiertos por tests unitarios de lógica pura), `plugin.py`/`model_loader.py`/
      `mlflow_utils.py`/`training.py` 0% en esa suite (esperado: no cargan artefactos reales). La
      correctitud contra artefactos reales se verificó aparte — ver Parte B — con un harness que
      monta el router real (`router_factory` + `PredictModelUseCase`/`GetStatsUseCase`/
      `TrainModelUseCase`) sobre el plugin real y los artefactos entregados, sin mocks.
- [x] **pylint**: `app/plugins/ml16_meat_raw_material_price_alert/` → **9.26/10** (idéntico a la
      primera pasada). Avisos: `line-too-long` (>100, dentro del límite real de 120 del repo),
      `too-many-locals`/`too-many-branches`/`too-many-statements` en
      `training.py::train_models` (función de 7 fases, deliberadamente no fragmentada para que
      sea trazable 1:1 contra `trainer.py::run_training()` del código original), `duplicate-code`
      entre `mlflow_utils.py`/`model_loader.py` (misma estructura de carga, justificado), y un
      falso positivo `E1101` sobre `numpy.random.RandomState` (existe; problema de stubs de
      pylint). Mismo perfil de avisos que plugins ya integrados —
      `ml40_meat_refrigeration_aeration_fault_diagnosis` puntúa 9.53/10 con las mismas
      categorías. Sin categorías de aviso nuevas para el repo.
- [x] **pip-audit** (`pip-audit -r requirements.txt`), ahora sí ejecutable: 4 vulnerabilidades
      conocidas en 3 paquetes — `torch==2.11.0` (PYSEC-2025-194, fix en 2.13.0), `setuptools==81.0.0`
      (PYSEC-2026-3447, fix en 83.0.0) y `cryptography==49.0.0` (PYSEC-2026-3552, fix en 50.0.0).
      **Ninguna involucra a este plugin**: `ml16` solo usa `pandas`/`numpy`/`scikit-learn`/
      `xgboost`/`joblib`, ya pinneados en el repo y sin CVEs reportadas; no añade ninguna
      dependencia nueva. `requirements.txt` fija `torch>=2.6.0` (mínimo abierto, resuelto aquí a
      2.11.0) y no pinea `setuptools`/`cryptography` explícitamente — es deuda técnica
      preexistente del repo, no introducida por esta integración; se señala para que el equipo
      valore actualizar esos pines de forma independiente a este PR.
- [x] **Arranque local + health + predict + stats**: OK. En vez de `main.py` completo (arranca
      los ~25 plugins del registro; probar solo lo que aporta valor a esta verificación) se montó
      un router real equivalente (mismo `router_factory.make_model_router` + casos de uso
      genéricos de `app/application/`) para **ml16 en solitario**, con el plugin real y los
      artefactos entregados cargados desde `artifacts/ml16_meat_raw_material_price_alert/` (sin
      `FakePlugin`, sin mocks). Resultados:
      - `GET /health` → `200 {"status":"ok","loaded":true}`
      - `GET /stats` → `200`, con `task_type`, contrato de inputs/outputs y métricas reales
      - `POST /predict` (batch, `dataset_clasificacion_base.csv` completo) → `200`, 47 predicciones
      - `POST /predict` (inline, 56 filas) → `200`, coincide exactamente con la última fila del batch
      - `POST /predict` (inline, 5 filas) → `422` (Pydantic `min_length=10`, antes de tocar el plugin)
- [x] **/train**: `manifest.training.supported = true` → `POST /train` responde `200` con
      `TrainResponse` real (ver Parte B).

## Correctitud (golden dataset)

Los 12 `golden_cases` del manifest cubren el bloque de test hold-out completo (2023-09 a
2024-08). Se llamó a `POST /predict` (batch) sobre
`inbox/a16/codigo/data/processed/dataset_clasificacion_base.csv` a través del router real
descrito arriba, y se comparó cada fila de salida (fecha = mes de entrada + horizonte(4), ver
manifest `known_issues` sobre esta convención) contra el valor esperado de
`data/predictions/predicciones_test.csv` (salida real del propio `trainer.py::run_training()`
del equipo de IA, ejecutado durante la extracción del manifest — no inventado).

| Caso (mes entrada) | Mes objetivo | Esperado (animales pred/proba) | Obtenido | Esperado (insumos pred/proba) | Obtenido | ¿OK? |
|---|---|---|---|---|---|---|
| 2023-09 | 2024-01-01 | 0 / 0.2503 | 0 / 0.2503 | 1 / 0.3581 | 1 / 0.3581 | ✅ |
| 2023-10 | 2024-02-01 | 0 / 0.1201 | 0 / 0.1201 | 1 / 0.3531 | 1 / 0.3531 | ✅ |
| 2023-11 | 2024-03-01 | 0 / 0.0696 | 0 / 0.0696 | 1 / 0.3116 | 1 / 0.3116 | ✅ |
| 2023-12 | 2024-04-01 | 0 / 0.3257 | 0 / 0.3257 | 0 / 0.2881 | 0 / 0.2881 | ✅ |
| 2024-01 | 2024-05-01 | 1 / 0.7624 | 1 / 0.7624 | 0 / 0.2914 | 0 / 0.2914 | ✅ |
| 2024-02 | 2024-06-01 | 0 / 0.4404 | 0 / 0.4404 | 0 / 0.2902 | 0 / 0.2902 | ✅ |
| 2024-03 | 2024-07-01 | 0 / 0.1711 | 0 / 0.1711 | 0 / 0.2859 | 0 / 0.2859 | ✅ |
| 2024-04 | 2024-08-01 | 1 / 0.5704 | 1 / 0.5704 | 0 / 0.2808 | 0 / 0.2808 | ✅ |
| 2024-05 | 2024-09-01 | 1 / 0.7445 | 1 / 0.7445 | 1 / 0.3276 | 1 / 0.3276 | ✅ |
| 2024-06 | 2024-10-01 | 1 / 0.7193 | 1 / 0.7193 | 1 / 0.3161 | 1 / 0.3161 | ✅ |
| 2024-07 | 2024-11-01 | 1 / 0.9334 | 1 / 0.9334 | 1 / 0.3524 | 1 / 0.3524 | ✅ |
| 2024-08 | 2024-12-01 | 1 / 0.9339 | 1 / 0.9339 | 1 / 0.3345 | 1 / 0.3345 | ✅ |

Tolerancia usada: clase exacta (pred) + probabilidad ±0.0005 absoluto. No se usó el 5% por
defecto ni una tolerancia derivada de MAE porque el modelo es un clasificador (no hay MAE en la
memoria aplicable a probabilidades); se optó por una tolerancia estricta porque el plugin
reproduce el pipeline de inferencia original (`predictor.py::prepare_input`/`run_inference`)
literalmente — un desajuste de wiring habría producido diferencias muy superiores a 0.0005 (las
features rolling/lag son sensibles a errores de alineación temporal), así que este umbral
ajustado es un canario más sensible, no una relajación.

**Resultado: 12/12 casos dentro de tolerancia (coincidencia exacta a 4 decimales en las 24
probabilidades comparadas).**

### Verificación adicional — `train()` (más allá de lo exigido por golden_cases, que solo cubren predict)

`POST /train` sobre el mismo CSV base reproduce el procedimiento de `trainer.py::run_training()`
(walk-forward CV, búsqueda de umbral, ajuste manual de insumos, bagging bootstrap) entrenando
modelos nuevos desde cero (nunca se sobrescriben los artefactos fijos — se guardan como
`user_*.joblib` en local, o se suben a MLflow si se pasa `mlflow_run_id`):

- `n_train_rows=35`, `n_test_rows=12` — coincide exactamente con la memoria ("el entrenamiento
  final usa ... 35 secuencias efectivas de train tras lookback=3 y 12 en test").
- `target_animales`: accuracy=0.833, precision=0.833, recall=0.833, f1=0.833, auc=0.917 —
  **coincide exactamente** con `metrics_reported.target_animales.test` del manifest (artefacto
  real, no la memoria v1.5 que reporta cifras distintas — ver `known_issues`).
- `target_insumos`: accuracy=0.667, precision=0.429, recall=1.0, f1=0.6, auc=0.741 — **coincide
  exactamente** con `metrics_reported.target_insumos.test`.
- El único valor que difiere del artefacto entregado es el umbral óptimo de `target_animales`
  (0.44 aquí vs. 0.48 en el artefacto original) — esperable: la búsqueda de umbral por F1-máximo
  sobre predicciones OOF de un walk-forward CV reentrenado desde cero con una versión de
  XGBoost/scikit-learn distinta a la usada originalmente (ver `known_issues` del manifest sobre
  las `InconsistentVersionWarning`) puede desplazar el punto de empate de F1 en un paso de grid
  (0.01–0.04) sin cambiar las métricas de test resultantes, como se observa aquí. No indica un
  error de wiring — las métricas de test, que sí son sensibles al umbral, coinciden exactamente.

## Hallazgos y decisiones documentadas (no bloqueantes, ya en `manifest.yaml` → `known_issues`)

1. Discrepancia entre la memoria v1.5 (Tabla 9) y los artefactos realmente entregados (umbral y
   métricas de test de `target_animales`) — se usan las cifras del artefacto real en
   `metrics_reported`, `stats()` y este informe.
2. Convención de fecha distinta entre `predicciones_test.csv` (fecha = mes de entrada) y el
   plugin/`predictor.py` (fecha = mes objetivo, entrada + horizonte) — el plugin usa la
   convención de producción (mes objetivo) y lo documenta en `predict_dto.py` y `stats()`.
3. `precip_max` y `wash_days` forman parte del esquema de entrada exigido pero no influyen en la
   predicción (no aparecen en `input_cols_per_target`) — se siguen aceptando/exigiendo por
   paridad de esquema con `dataset_clasificacion_base.csv`.
4. `month` es funcionalmente necesaria (usada en `month_sin`/`month_cos`) pero el código original
   nunca la deriva de `fecha` — el plugin sí la deriva automáticamente si falta
   (`preprocessing.ensure_month_column`), documentado como adición del plugin, no como
   comportamiento heredado.
5. El pipeline multi-horizonte (h=1/h=2, `scripts/train_multi_horizon.py`) queda fuera de alcance
   de este plugin — no forma parte del modo de despliegue oficial descrito en la memoria.

Ninguno de estos hallazgos requirió ajustar tolerancias ni silenciar un caso fallido — los 12
golden cases pasaron con coincidencia exacta.

## Estado final

**LISTO PARA PR.** Checklist técnico completo en verde (flake8, pytest 400/400, pylint, pip-audit
sin CVEs propias de este plugin) y 12/12 golden cases con coincidencia exacta contra el pipeline
original, verificado a través del endpoint HTTP real. Única nota no bloqueante: las 4 CVEs de
`pip-audit` (torch/setuptools/cryptography) son deuda técnica del repo, ajena a este plugin — se
deja constancia para que se valore actualizar esos pines en un cambio independiente.

Esta verificación no abre PR ni hace merge — queda pendiente de revisión humana del plugin,
este informe y `inbox/a16/manifest.yaml` antes de abrir el PR.
