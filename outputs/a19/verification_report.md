# Verificación — ml19-cereals-cost-forecast

## Nota sobre el entorno de ejecución

Esta verificación se ejecutó en un entorno de desarrollo cuyo intérprete Python pasó de 3.12 a
3.14 a mitad de sesión (actualización del sistema, ajena a este cambio), lo que dejó sin efecto
los paquetes previamente instalados. `torch` y `tensorflow` no tienen todavía rueda (`wheel`)
publicada para Python 3.14, y el módulo `venv` del sistema está incompleto (falta
`python3.14-venv`), así que:

- No fue posible arrancar `main.py` tal cual (importa `app/registry.py`, que carga TODOS los
  plugins — incluidos los que dependen de `torch`/`tensorflow` — antes de aplicar el filtro
  `MODEL`).
- No fue posible ejecutar `pip-audit -r requirements.txt` (necesita crear un entorno virtual
  aislado para resolver dependencias).

Ninguna de las dos limitaciones es específica de ml19 ni la introduce este cambio — afectan por
igual a cualquier plugin de este repo en este entorno concreto. Para no dejar la Parte A sin
verificar, se montó una app FastAPI mínima que registra **únicamente** el router real de ml19
(mismo `make_model_router` + `ModelContainer` que usa `main.py`, mismas clases
`Ml19CerealsCostForecastPlugin`/`PredictRequest`/`PredictResponse` reales, mismo
`DataContractError` en `extra_predict_exceptions`) y se golpeó con peticiones HTTP reales vía
`TestClient` (Starlette) — equivalente a `curl` contra un servidor real, sin pasar por el import
conflictivo de `app/registry.py`. Script: ver comando reproducible más abajo.

## Checklist técnico

- [x] `flake8 . --extend-exclude=dist,build,inbox --show-source --statistics`: **0 errores**
      (el repo tiene `extend-ignore = E501` en `.flake8`, por eso no se reporta longitud de línea).
- [x] `pytest tests/unit/test_ml19_cereals_cost_forecast.py --cov=app.plugins.ml19_cereals_cost_forecast`:
      **8/8 passed**. Cobertura de línea del módulo real ml19: 21% — esperado y consistente con
      el resto del repo: estos tests usan `FakePlugin` (wiring: esquemas, DTOs, mapeo de
      excepciones a HTTP), nunca cargan el plugin real/los artefactos reales. La correctitud
      real se valida en la Parte B de este informe, no aquí.
- [x] `pytest tests/unit/ -q` (repo completo, excluyendo los módulos que ya fallan en main por
      falta de `torch` en este entorno — 9 ficheros no se pueden ni *importar*, ver nota de
      entorno arriba): **600 passed, 76 failed, 4 skipped**. Los 76 fallos son *todos*
      `ModuleNotFoundError: No module named 'torch'` en `ml9`, `ml10`, `ml26`, `ml45`, `ml46`,
      `modelo10_lacteo` y `test_user_model_isolation.py` (parametrizado sobre varios modelos que
      cargan torch) — verificado caso por caso, ninguno menciona ni toca `ml19`. 0 fallos nuevos.
- [x] `pylint app/plugins/ml19_cereals_cost_forecast/ --disable=import-error`: **9.63/10**
      (solo `line-too-long` — el repo no fija `max-line-length` en pylint y su convención real,
      la de `.flake8`, ignora E501 — y un `too-many-locals` en `inference.py::predict_next_month`,
      que es una copia fiel línea a línea del `predict_next_month()` original del equipo de IA).
      Comparado contra `ml18` (ya en producción) como referencia: 9.42/10 con el mismo patrón de
      avisos — consistente con el resto del repo, no es una regresión de calidad.
- [ ] `pip-audit -r requirements.txt`: **no ejecutable en este entorno** (ver nota arriba — el
      módulo `venv` de Python 3.14 está incompleto en esta máquina). Pendiente de ejecutar en el
      pipeline de CI real, donde si corre con normalidad.
- [x] Arranque + health + predict + stats + train, vía app FastAPI mínima con el router/plugin
      reales (ver nota de entorno):
      - `GET /models/ml19-cereals-cost-forecast/health` → `200 {"status": "ok", "loaded": true}`
      - `GET /models/ml19-cereals-cost-forecast/stats` → `200`, contrato completo
      - `POST /predict {"mode": "inline"}` (sin fecha) → `200`, usa la fila más reciente del
        dataset empaquetado (`date_label: "2026-01"`)
      - `POST /predict {"mode": "inline", "date": "1999-01"}` (fecha fuera de rango) → `422`
        con `DataContractError` y mensaje explícito del rango disponible
      - `POST /train` → **501** (`TrainingNotSupportedError`) — correcto: `manifest.yaml`
        declara `training.supported: false`, nunca se implementó `train()` como stub silencioso.

## Correctitud (golden dataset)

Los 18 `golden_cases` de `inbox/a19/manifest.yaml` se generaron en la fase de
`manifest-extraction` ejecutando `load_artifacts_src()` + `predict_next_month()` del **código
original entregado, sin modificar**, contra los 6 artefactos `.pkl` reales y el
`dataset_v7_fe.csv` real. Esta verificación los reproduce ahora contra el **plugin integrado**
(vía HTTP, `POST /predict {"mode": "inline", "date": "<caso>"}`), comparando cada horizonte
(`reg.pred`, `reg.signal_str`, `clf.prob_up`, `clf.signal_str`, `ensemble_str`, `confidence`).

| Caso | HTTP | reg.pred (h1/h2/h3) diff | clf.prob_up (h1/h2/h3) diff | señales/confianza | ¿OK? |
|---|---|---|---|---|---|
| caso_2020-01 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2020-04 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2020-07 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2020-10 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2021-01 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2021-04 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2021-07 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2021-10 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2022-01 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2022-04 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2022-07 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2022-10 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2023-01 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2023-04 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2023-07 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2023-10 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2024-01 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |
| caso_2024-04 | 200 | 0 / 0 / 0 | 0 / 0 / 0 | coinciden | ✅ |

Tolerancia usada: **1e-4 absoluto** en `reg.pred` y `clf.prob_up` (y coincidencia exacta de
`signal_str`/`ensemble_str`/`confidence`) — no 2× el error reportado en la memoria, porque aquí
no estamos comparando contra la realidad del mercado (eso es lo que mide `metrics_reported`,
Pearson r / Direction Accuracy / AUC-ROC de la Tabla 38-39 de la memoria), sino verificando que
el plugin reproduce **bit a bit** la salida del código original entregado para el mismo input —
es un test de wiring/fidelidad, no de acierto del modelo. Diferencia de 0 en todos los casos:
el plugin es un port fiel, no introduce ninguna desviación numérica.

Resultado: **18/18 casos dentro de tolerancia.**

## Hallazgos registrados durante la integración (no bloqueantes, ver `inbox/a19/manifest.yaml`)

- Decisión de arquitectura: el plugin responde contra un dataset de referencia empaquetado y
  congelado a la fecha de entrega (última fila real: 2026-01), no contra histórico que aporte el
  cliente — confirmado tanto por el diseño del `predict.py` original como por la memoria
  (sección 4.4) y por la descripción del propio ticket de Jira (DATAPLAT-464: "construir con la
  selección del dataset fuera del flujo de predicción").
- Dos discrepancias memoria-vs-código documentadas y **no implementadas** en el plugin (no se
  inventan mecanismos que el código entregado no tiene): un "gate de volatilidad" y un "filtro
  de confianza 0.65/0.35" que la Tabla 48 de la memoria describe pero `predict.py` no contiene.
- Una discrepancia de nomenclatura de sector en el propio ticket Jira DATAPLAT-464 ("Sector
  cárnico" en la descripción, cuando a19 es cerealista) — señalada al usuario, pendiente de que
  él la corrija en Jira si procede.

## Estado final

**LISTO PARA PR**, con una salvedad explícita: `pip-audit` no se pudo ejecutar en este entorno de
desarrollo concreto (limitación de sandbox, no del código) — debe correr sin problemas en el
pipeline de CI real y se recomienda confirmarlo ahí antes del merge final.

## Correcciones tras revisión externa (post-merge, PR #20)

Un compañero probó la rama desplegada y reportó tres síntomas. Investigados y confirmados los
tres contra el código real (no solo contra el informe del compañero):

1. **Toda predicción fallaba con el `requirements.txt` real del repo.** La verificación original
   de este informe se ejecutó con una versión de scikit-learn distinta a la realmente pineada
   en el repo (`scikit-learn==1.5.2`, fijada para no romper la integración con
   `retech-lote2-xai-explicabilidad`). Los artefactos se serializaron con `scikit-learn==1.8.0`.
   Reproducido: con `scikit-learn==1.5.2` real, `best_clf_h1m_logreg.pkl` lanza
   `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'` en **toda**
   petición (h1 de clasificación está siempre presente). **Corregido** en `model_loader.py`:
   se restaura el atributo a `'auto'` tras cargar el modelo si falta — matemáticamente
   equivalente para un clasificador binario, verificado reproduciendo los 18 golden cases con
   `scikit-learn==1.5.2` real tras el parche (0 diferencia, igual que antes).
2. **Una fecha vacía en el CSV de `predict_batch` tiraba todo el lote.** Confirmado: pandas
   parsea una celda vacía como `NaN` (float) pese a `dtype=str`; ese valor llegaba a
   `BatchPrediction.date_requested` (tipo `str`), Pydantic lo rechazaba con `ValidationError`
   (no capturado por el `except DataContractError` de la fila), y la excepción se propagaba
   fuera de la función — abortando también las filas válidas ya procesadas, al contrario de lo
   que decía el propio docstring. **Corregido** en `plugin.py`: la fecha de cada fila se sanea
   antes de usarse; una celda vacía se reporta como error de esa fila sin afectar al resto.
3. **`mlflow_run_id` se ignoraba siempre.** Confirmado: `predict_inline`/`predict_batch` hacían
   `_ = mlflow_run_id` y nunca llamaban a `download_user_model_from_mlflow`; cualquier run_id
   (válido, inexistente o con typo) servía el modelo fijo con HTTP 200 sin avisar. **Corregido**:
   `mlflow_utils.py` ahora usa el decorador compartido `@require_user_model`
   (`app/domain/services/mlflow_tracker.py`), y los dos métodos lo invocan cuando `mlflow_run_id`
   no está vacío — un run_id inexistente da ahora `422 UserModelUnavailableError`.

Re-verificación tras los tres parches: `flake8` 0 errores, `pylint` 9.65/10, 8/8 tests de wiring,
18/18 golden cases exactos vía HTTP real con **`scikit-learn==1.5.2` real** (no una versión
aproximada), las dos fecha-vacía sin afectar al resto del lote, y `mlflow_run_id` inexistente
devolviendo 422.

**Nota para ml18**: el mismo patrón de `mlflow_run_id` ignorado (`_ = mlflow_run_id` sin llamar a
`download_user_model_from_mlflow`) existe en `app/plugins/ml18_meat_spatial_price_forecast/plugin.py`
(ya en `main`) — fuera del alcance de este PR, señalado para que su autor lo revise.

## Actualización 19/10/2026 — reentreno que actualiza el modelo base

`/train` pasa de 501 a real y sigue la lógica de los modelos de series temporales
(`BASE_ARTIFACT_FOLDER`; spec `docs/superpowers/specs/2026-10-14-reentreno-actualiza-base-design.md`,
§10.1). Entorno: venv del repo, Python 3.12, scikit-learn 1.5.2 (pin del repo), xgboost 3.4.1.

| Comprobación | Resultado |
|---|---|
| Inferencia con los artefactos reales (copia local de `models/artifacts/` + `dataset_v7_fe.csv`) | 18/18 golden cases idénticos (máx. 5e-7) |
| Port `training.py` frente a `src/training/train.py` original, mismo dataset y entorno | Features, modelos elegidos, métricas walk-forward y predicciones de los 6 `.pkl` idénticos bit a bit; ~70 s |
| Funciones vendorizadas (`_vendor/training_core.py`) | AST idéntico al entregable (test) |
| `/train` real vía `ModelContainer` con el dataset entregado | v1 publicada, instancia nueva cargada desde `versions/v1/`, predicción de 2020-01 h1 = golden de v0 (1,024653) |
| Tests | `test_ml19_cereals_cost_forecast.py`, `test_ml19_base_promotion.py`, recarga en `test_base_reload_per_model.py` |

Hallazgo: el reentreno en Linux elige LogReg en h2 de clasificación en lugar del XGBoost entregado.
AUC walk-forward del XGBoost: 0,7538 con xgboost 3.2.0, 0,7672 con 3.4.1, 0,7782 entregado (LogReg
0,7731 en todos). Los `.pkl` se entrenaron con xgboost 3.2.0 en Windows; reajustar el XGBoost de h2
con la misma versión, datos, scaler (idéntico) y parámetros en Linux da árboles distintos (hasta
0,095 en probabilidad): es una diferencia de plataforma, no de versión. Se fija `xgboost==3.2.0` en
`requirements.txt` por el resto de modelos (§10.3 de la spec), pero no la elimina. La inferencia de
los artefactos entregados es idéntica.

Pendiente de acción humana: subir los artefactos a `s3://<STORAGE_BUCKET>/artifacts/fixed/ml19_cereals_cost_forecast/`
(a 19/10/2026 el prefijo está vacío: el modelo da 503 al arrancar).
