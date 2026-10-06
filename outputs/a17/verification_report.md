# Verificación — ml17-meat-market-price-analysis

## Contexto

Este plugin (`app/plugins/ml17_meat_market_price_analysis/`) ya estaba integrado y
registrado en `app/registry.py` ANTES de que existiera `inbox/a17/manifest.yaml` — igual
que ocurrió con ml21 y ml23, violando el orden de trabajo estándar del repo
(`manifest-extraction` siempre antes de `plugin-integration`). Este ciclo cubre, en orden:
manifest-extraction retroactivo, auditoría del plugin ya integrado contra el código real
entregado (`inbox/a17/codigo/`), corrección de los bugs encontrados, y verificación técnica +
de correctitud contra el servidor real.

## Resumen de la auditoría (manifest-extraction + revisión de plugin.py)

- **`training.supported` — corregido de `false` (asumido) a `true` (con evidencia real).**
  El plugin ya integrado tenía `train()` lanzando `TrainingNotSupportedError` incondicionalmente,
  con un mensaje que afirmaba "modelo entrenado externamente". Esto era **falso**: el código
  entregado (`src/cu05/training/service.py::run_training` + `src/cu05/models/ridge.py`) trae un
  procedimiento de entrenamiento real, reproducible y versionado (Ridge sobre el split train
  oficial, mismo preprocesado que el artefacto fijo desplegado). Corregido: `train()` ahora
  refit del Ridge pipeline sobre el CSV del usuario, `train_dto.py` añadido con
  `mlflow_run_id: str` **obligatorio** (sin default), y `train_request_type`/
  `train_response_type` wireados en `app/registry.py` (antes ausentes — el mismo patrón de bug
  #3 encontrado en ml21: sin esto, `/train` habría caído al DTO genérico aunque `train()` se
  hubiese implementado).
- **Bug tipo ml21 #1 (dict crudo en vez de Pydantic tipado) — NO presente.** `predict_inline`/
  `predict_batch` ya devolvían `PredictInlineResponse`/`PredictBatchResponse` tipados.
- **Bug tipo ml21 #2 (zero-fill silencioso de features) — NO presente.** Las 8 features que
  construye `_build_frame()` coinciden exactamente, columna a columna y en el mismo orden, con
  `feature_columns` del `.meta.json` del artefacto real y con `dataset.final_feature_columns`
  de `config/official_v1_4.yaml`. Este modelo solo tiene 8 features totales (no ~90 como ml21),
  todas explícitas en el request — no hay dataset de referencia oculto que reindexar.
- **Bug nuevo encontrado y corregido (no catalogado en ml21): `mlflow_run_id` ignorado en
  predict.** `predict_inline`/`predict_batch`/`stats` llamaban a
  `download_user_model_from_mlflow(mlflow_run_id)` pero **descartaban el resultado** — nunca
  sustituían `self._model` por el modelo de usuario descargado. Inofensivo mientras
  `mlflow_utils.py` devolvía siempre `None` (sin entrenamiento soportado), pero se habría
  convertido en un bug real de aislamiento entre usuarios en cuanto `train()` empezara a
  producir artefactos reales en MLflow. Corregido: ambos métodos ahora intercambian
  `self._model` por el modelo descargado dentro de un `try/finally` que restaura el modelo base
  y hace `shutil.rmtree` del directorio temporal, siguiendo el patrón de `ml35`/`ml25`.
- `mlflow_utils.py` reescrito: antes siempre devolvía `None` (comentario "no hay artefacto de
  usuario que buscar"); ahora descarga el pipeline Ridge reentrenado desde el `artifact_path`
  MLflow del run, siguiendo el patrón de `ml35_dairy_ann_cleaning_cost/mlflow_utils.py`.
- `stats()` enriquecido con métricas/params de MLflow cuando se pasa `mlflow_run_id`, igual que
  `ml35`.

Detalle completo y trazabilidad campo a campo en `inbox/a17/manifest.yaml`.

## Parte A — Checklist técnico

- [x] `flake8` sobre `app/plugins/ml17_meat_market_price_analysis/`, `app/registry.py`,
      `tests/conftest.py`, `tests/unit/test_ml17_meat_market_price_analysis.py`: **0 errores**.
- [x] `pytest tests/unit/ -q`: **500/500 passed** (suite completa, no se rompió ningún otro
      plugin). Nota: al arrancar esta verificación la suite tenía 2 fallos preexistentes en
      `test_modelo10_lacteo_unit.py`, confirmados **no relacionados** con este trabajo
      (`git stash` de todos los cambios no commiteados — incluyendo cambios de ml17 — seguía
      sin tocar esos ficheros, y por tanto la causa es ajena a esta tarea); otro proceso en
      este mismo entorno los corrigió en paralelo durante la verificación, de ahí que la
      ejecución final dé 500/500.
- [x] `pylint app/plugins/ml17_meat_market_price_analysis/ --disable=import-error`: 9.45/10.
      Únicos hallazgos: `line-too-long`, `invalid-name` (`X`, `X_train`),
      `too-many-locals`, `import-outside-toplevel` — las mismas categorías, con la misma
      frecuencia relativa, que ya tolera el baseline del repo en los plugins de referencia
      citados por el skill (`ml35_dairy_ann_cleaning_cost`: 8.93/10 con hallazgos idénticos).
      Sin issues nuevos de severidad (ningún `error`, ningún `warning` de lógica).
- [x] `pip-audit -r requirements.txt`: 2 vulnerabilidades reportadas, ambas en `setuptools`
      80.9.0 (`PYSEC-2026-3447`) — dependencia transitiva compartida por todo el repo, no
      introducida ni agravada por este trabajo.
- [x] Arranque local (`MODEL=ml17-meat-market-price-analysis ./.venv/bin/python main.py`,
      artefacto real descargado/cacheado desde S3) + `/health` + `/predict` (inline y batch) +
      `/stats`: **OK, sin 500s**.
- [x] `/train`: `supported: true` en el manifest.
  - Sin `mlflow_run_id` → **422** confirmado (`Field required` — el campo no tiene default,
    por la convención de todo el repo: un retrain siempre debe atarse a un run de MLflow
    concreto y nunca sobrescribir el artefacto fijo de S3).
  - Con `mlflow_run_id` real vía HTTP → **no completable en este entorno**: ver nota de
    entorno debajo. La lógica de negocio de `train()` se validó de forma aislada (ver abajo) y
    reproduce EXACTAMENTE las métricas de train de la memoria (Tabla 7).

### Nota de entorno — `/train` vía HTTP con `mlflow_run_id`

`BaseMLflowTracker` (compartido por todo el repo, `app/domain/services/mlflow_tracker.py`) usa
por defecto `MLFLOW_TRACKING_URI=http://mlflow.mlflow:5000` (hostname interno de Kubernetes).
En este entorno de verificación local ese host no es alcanzable, y `tracker.log_params()` —la
primera llamada de red dentro de `train()`, antes incluso de leer el CSV— cuelga sin timeout y
bloquea el único worker de `uvicorn` (confirmado: `/health` dejó de responder tras una llamada
a `/train` con `mlflow_run_id` mientras el proceso seguía colgado). Esto **no es un bug
introducido en ml17**: es el mismo patrón (`tracker.log_params()` como primera operación) ya
presente en `ml35`/`ml25`, así que afecta a cualquier plugin con reentrenamiento real cuando se
prueba fuera de un clúster con MLflow accesible — no se tocó `mlflow_tracker.py` (infraestructura
compartida, fuera del alcance de este plugin).

Para no dejar `train()` sin verificar, se probó la lógica de negocio de forma aislada,
sustituyendo `BaseMLflowTracker` por un stub no-op en memoria (sin tocar el código de
`plugin.py`, solo el objeto importado en un script de verificación) y llamando a
`plugin.train(data_path=".../data/splits/official_v1_4/train.csv", mlflow_run_id="isolated-test-run")`
directamente:

```
mae=5.2979 rmse=7.1946 mase=0.7607 r2_train=0.9404 directional_accuracy=0.7833 n_samples=120
```

Coincide **exactamente, a 4 decimales**, con la Tabla 7 de la memoria ("Métricas descriptivas
de entrenamiento sobre conjuntos train persistidos", línea official_v1_4: MAE=5.2979,
RMSE=7.1946, MASE=0.7607, R²_train=0.9404, Directional Accuracy=0.7833) — confirma que el refit
del pipeline Ridge, la selección de columnas y el cálculo de métricas en `train()` replican
bit-a-bit el procedimiento original. También se verificaron los dos `raise ValueError`
(columnas requeridas faltantes; menos de 2 filas) — ambos disparan correctamente con el mensaje
esperado.

## Parte B — Correctitud contra golden dataset

18 casos extraídos de la serie temporal continua train+test de `official_v1_4`
(`data/splits/official_v1_4/{train,test}.csv`), fila *t* → precio real observado en *t+1*. El
tramo cubierto (*t+1* entre 2023-06-01 y 2025-12-01) es el mismo tramo fuera de muestra que
`metrics_reported` (MAE=4.4605, n=31), así que son comparables 1:1 contra el benchmark oficial.
Tolerancia = 2× MAE reportado = **8.9210**.

| Caso | Fecha input (t) | Esperado (t+1) | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|---|
| caso_001 | 2023-05-01 | 247.4030 | 251.3831 | 3.9801 | ✅ |
| caso_002 | 2023-07-01 | 243.1503 | 245.2389 | 2.0886 | ✅ |
| caso_003 | 2023-09-01 | 216.1513 | 223.3221 | 7.1708 | ✅ |
| caso_004 | 2023-10-01 | 201.9853 | 208.3176 | 6.3323 | ✅ |
| caso_005 | 2023-12-01 | 199.3013 | 200.5501 | 1.2488 | ✅ |
| caso_006 | 2024-02-01 | 217.5032 | 207.9529 | 9.5503 | ❌ |
| caso_007 | 2024-04-01 | 219.0539 | 225.2197 | 6.1658 | ✅ |
| caso_008 | 2024-05-01 | 220.7743 | 223.2243 | 2.4500 | ✅ |
| caso_009 | 2024-07-01 | 222.3274 | 219.0372 | 3.2902 | ✅ |
| caso_010 | 2024-09-01 | 201.8261 | 204.8455 | 3.0194 | ✅ |
| caso_011 | 2024-11-01 | 198.2364 | 192.1772 | 6.0592 | ✅ |
| caso_012 | 2024-12-01 | 193.3597 | 197.4828 | 4.1231 | ✅ |
| caso_013 | 2025-02-01 | 204.8623 | 201.7839 | 3.0784 | ✅ |
| caso_014 | 2025-04-01 | 217.8161 | 222.5526 | 4.7365 | ✅ |
| caso_015 | 2025-06-01 | 220.4077 | 220.2508 | 0.1569 | ✅ |
| caso_016 | 2025-07-01 | 211.1016 | 215.9280 | 4.8264 | ✅ |
| caso_017 | 2025-09-01 | 182.6052 | 188.6164 | 6.0112 | ✅ |
| caso_018 | 2025-11-01 | 140.2247 | 166.8263 | 26.6016 | ❌ |

Tolerancia usada: **8.9210** (2× MAE=4.4605 de `metrics_reported`, memoria Tabla 8 /
`data/benchmarks/official_v1_4/full_benchmark.csv`).

**Resultado: 16/18 casos dentro de tolerancia (88.9%).**

### Investigación de los 2 casos fuera de tolerancia

Se comparó la predicción devuelta por `/predict` contra una invocación directa del mismo
artefacto `.pkl` fuera de la API (mismo `FEATURE_COLUMNS`, mismo cálculo de `month_sin`/
`month_cos`), para los 18 casos. **Los 18 valores coinciden bit a bit** — descarta
definitivamente un bug de wiring/preprocesado en el plugin; el modelo que sirve la API es
exactamente el modelo original.

- **caso_006**: diferencia 9.55 vs. tolerancia 8.92 — excede por un margen pequeño (6.6%
  relativo). Dentro de la variabilidad esperable de un modelo con RMSE=6.41 > MAE=4.46 (indica
  cola pesada de errores, típico de un problema con shocks de mercado ocasionales).
- **caso_018** (2025-11-01 → 2025-12-01): diferencia 26.60, el error más grande del dataset
  completo (31 casos de test). El precio real cae de 170.45 a 140.22 (**-17.7% intermensual**),
  la mayor caída mes a mes de toda la serie disponible (2013-2025). Un modelo Ridge lineal, sin
  señales de ruptura de régimen, estructuralmente no puede anticipar un shock de esta magnitud a
  partir únicamente de covariables de oferta/coste/estacionalidad observadas en *t*. Esto
  coincide textualmente con la limitación ya documentada por el propio equipo de IA en la
  memoria (sección "SESGOS CONOCIDOS DEL MODELO", riesgo 3: *"riesgo de degradación ante
  cambios de distribución (drift) o episodios de mercado no representados adecuadamente en la
  historia reciente"*).

**No se ajustó la tolerancia ni se descartó ningún caso.** Ambos fallos son error de modelo
genuino sobre meses concretos fuera de muestra, no un defecto de integración — quedan
documentados aquí y en `inbox/a17/manifest.yaml::known_issues` para la revisión humana.

## Estado final

**LISTO PARA PR**, con una salvedad a revisar por una persona: la prueba end-to-end de `/train`
con un `mlflow_run_id` real no pudo completarse en este entorno por falta de acceso de red al
MLflow tracking server (ver nota de entorno arriba) — la lógica de negocio de `train()` sí se
validó de forma aislada y reproduce exactamente las métricas oficiales de entrenamiento. Se
recomienda una prueba manual de `/train` contra un clúster con MLflow accesible antes de dar el
reentrenamiento por completamente verificado en producción.
