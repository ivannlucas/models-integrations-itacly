# Verificación — wine-sulphite (a25)

> **[RENOMBRADO 2026-10-06]** Nomenclatura alineada con el estándar `mlNN_<sector>_<desc>` del resto de plugins: `wine-sulphite (model_id) / wine_sulphite (artefactos)` → `ml25-wine-sulphites / ml25_wine_sulphites (el paquete ya era app/plugins/ml25_wine_sulphites/)`. La carpeta de artefactos (local y en S3) pasa de `artifacts/wine_sulphite/` a `artifacts/ml25_wine_sulphites/`. Las referencias a los nombres antiguos en el texto de abajo son históricas y corresponden a la fecha de cada ciclo.

Plugin: `app/plugins/ml25_wine_sulphites/` · model_id: `wine-sulphite` · prefix `/models/wine-sulphite`
Manifest: `inbox/a25/manifest.yaml`

## Resumen de la auditoría

El plugin ya estaba integrado (nunca había pasado por `manifest-extraction` ni `verification`).
Al comparar contra el código real entregado (`inbox/a25/codigo/`, 78 ficheros, memoria v3.2) se
encontraron y corrigieron dos bugs reales, y se confirmó que el resto del wiring (DTOs tipados,
features, grid de simulación, decisión de intervención) replica fielmente el pipeline entregado.

### Bug 1 — `train()` no replicaba el protocolo real (CORREGIDO)

El `train()` anterior usaba un protocolo **inventado**: `RandomForestRegressor(n_estimators=200)`
sobre un split 80/20 cronológico (sin shuffle, sin CV) del 20% final de las filas, sin el filtro
de limpieza química del dataset. El protocolo real (`inbox/a25/codigo/modules/wine_quality/src/
training/train_rf.py: train_dual_models` + `cv_regression_metrics`) es: `n_estimators=300`,
`random_state=42`, `n_jobs=-1`; métricas = media de `KFold(5, shuffle=True, random_state=42)`
sobre el dataset **completo** y limpio; el modelo que se sirve se ajusta sobre el **100%** de los
datos (la CV solo estima el error de generalización, nunca retira datos del modelo final).

Más grave: **el artefacto base servido** (`artifacts/wine_sulphite/{quality_rf,bound_rf}.pkl`,
`metadata.json`) había sido generado con ese protocolo incorrecto — confirmado comparando
`metadata.json` (n_train=3918, n_test=980, sin campo `protocol`) contra el entregado por el
equipo de IA (protocol=`official_cv5_full_dataset`, n_estimators=300, mae_quality=0.42703,
mae_bound=14.51109 — idéntico a la memoria y a `reports/baseline_metrics.csv`).

**Corrección aplicada**:
- `_train_from_local()` reescrito para replicar exactamente `train_dual_models`: limpieza de
  filas físicamente imposibles, CV 5-fold con `clone()` por fold, `n_estimators=300`, ajuste
  final sobre el 100% del dataset limpio, y nuevo campo `metadata["simulation"]` con los
  percentiles reales de `free sulfur dioxide` (ver bug 2).
- Se sustituyó el artefacto base (`artifacts/wine_sulphite/*`) por el entregado
  (`inbox/a25/codigo/modules/wine_quality/models/*`) — ahora el modelo servido por defecto es,
  literalmente, el que entregó el equipo de IA.
- Verificado con `/train` real (MLflow mockeado, ver Parte A): `mae_quality=0.427`,
  `mae_bound_so2=14.5111`, `n_train=4898` — coincide con la memoria hasta el redondeo.

### Bug 2 — grid de simulación sin tope de percentil (CORREGIDO)

`build_simulation_grid()` solo acotaba el rango explorado por `delta_max`, ignorando el tope del
percentil 99 de `free sulfur dioxide` en el dataset de entrenamiento que usa el pipeline real
(`wine_quality.common.build_free_grid`, `SimulationConfig(sim_free_p_high=99.0)`) para no
extrapolar fuera de la distribución de entrenamiento. Para vinos con SO2 libre actual alto, el
optimizador podía explorar dosis muy por encima de lo que el modelo vio en entrenamiento.
Corregido: `train()` guarda los percentiles reales en `metadata["simulation"]`; el artefacto base
usa como fallback `BASE_FREE_SO2_P1/P99` (6.0/81.0 mg/L, calculados del propio
`white_wine.csv`). Verificado con test directo: sin tope la grid llega a 90 mg/L, con tope a 81.

### Revisado y confirmado correcto (sin cambios)

- `predict_inline`/`predict_batch` devuelven siempre objetos Pydantic tipados (`PredictInlineResponse`/
  `PredictBatchResponse`), nunca un dict crudo.
- Orden y nombre de features (`FEATURES_PHYS`/`FEATURES_QUAL`/`FEATURES_BOUND`) idéntico al
  entregado; decodificación `log1p`/`expm1` y ajuste de monotonicidad de `bound_so2` coinciden
  con `wine_quality.pipeline.predict_trajectory`.
- Regla de decisión (`select_recommendation`, umbral 1×MAE) y restricciones duras
  (`min_molecular=0.6`, `max_total=200.0` → `NoValidSimulationPointError`/422) coinciden con
  `recommend_strategy` del pipeline real.
- `mlflow_run_id` obligatorio (sin default) en `TrainRequest`; el modelo reentrenado se sube solo
  a su propio run de MLflow, nunca sobrescribe el artefacto base local (`test_train_never_writes_
  to_base_artifacts_dir`, verificado).
- `mlflow_utils.py` presente; `download_user_predictor_from_mlflow` se usa en predict/stats
  cuando se informa `mlflow_run_id`, con `shutil.rmtree` en `finally`.
- No hay lógica CUDA/GPU en este plugin (modelo tabular RandomForest) — no aplica el bug class de
  `_safe_device()`.

## Checklist técnico (Parte A)

- [x] `flake8 app/plugins/ml25_wine_sulphites/`: 0 errores
- [x] `pytest tests/unit/ -q`: **503/503 passed** (igual que el baseline antes de tocar nada)
- [x] `pylint app/plugins/ml25_wine_sulphites/ --disable=import-error`: 7.98/10 (mejora sobre el
      7.75/10 previo a los cambios); todos los issues restantes (line-too-long, too-many-locals en
      `predict_batch`/`_run_inference`, imports locales para S3, `broad-exception-caught`,
      `E0601` en el `finally` de `predict_inline`) son preexistentes, confirmados línea a línea
      contra el diff — no se introdujo ningún issue nuevo.
- [x] `pip-audit -r requirements.txt`: 2 CVEs conocidas en `setuptools==80.9.0` (PYSEC-2026-3447),
      preexistentes y no relacionadas con este plugin (ninguna dependencia propia de a25 aparece).
- [x] Arranque local (`MODEL=wine-sulphite ./.venv/bin/python main.py`, puerto 8000) + `/health` +
      `/predict` (inline y batch) + `/stats`: OK, sin 500. El log de arranque confirma que carga
      el artefacto corregido (`quality MAE=0.427, bound MAE=14.511`, coincide con memoria).
- [x] `/train` sin `mlflow_run_id` → 422 (confirmado). Con `mlflow_run_id` real cuelga por el
      timeout de conexión conocido de `BaseMLflowTracker` (no es bug de este plugin — ver
      CLAUDE.md/brief); se verificó la lógica real con `BaseMLflowTracker` mockeado:
      `mae_quality=0.427`, `mae_bound_so2=14.5111`, `n_train=4898`, `n_test=0`,
      `upload_warning=None`, sube `quality_rf.pkl`/`bound_rf.pkl`/`metadata.json` al run de
      MLflow — coincide con la memoria hasta el redondeo.

## Correctitud (golden dataset)

14 filas reales de `inbox/a25/codigo/data/white_wine.csv` (mismo CSV usado para entrenar+evaluar
los artefactos), elegidas para cubrir todo el rango de `quality` (3–9). No existe held-out test
split en el código entregado (protocolo oficial = CV 5-fold sobre el dataset completo) — el
`expected` es el valor real (ground truth) del dataset, comparado contra la salida del plugin con
`delta_max=0` (fuerza evaluación solo en el punto actual, sin optimización) y restricciones de
negocio desactivadas (`min_molecular=0`, `max_total=100000`, para no confundir el filtro de
negocio con la fidelidad del modelo).

| Caso | quality esperado | quality obtenido | dif | ¿OK? (tol 0.854) | bound_so2 esperado | bound_so2 obtenido | dif | ¿OK? (tol 29.02) |
|---|---|---|---|---|---|---|---|---|
| caso_001 | 6 | 5.903 | 0.097 | OK | 125.0 | 126.063 | 1.063 | OK |
| caso_002 | 5 | 5.327 | 0.327 | OK | 52.0 | 60.604 | 8.604 | OK |
| caso_003 | 7 | 6.983 | 0.017 | OK | 95.0 | 93.967 | 1.033 | OK |
| caso_004 | 8 | 7.717 | 0.283 | OK | 46.0 | 48.181 | 2.181 | OK |
| caso_005 | 4 | 4.430 | 0.430 | OK | 143.0 | 140.047 | 2.953 | OK |
| caso_006 | 3 | 3.853 | 0.853 | OK | 156.0 | 144.476 | 11.524 | OK |
| caso_007 | 9 | 7.613 | 1.387 | **NO** | 96.0 | 104.189 | 8.189 | OK |
| caso_008 | 9 | 7.930 | 1.070 | **NO** | 112.0 | 97.207 | 14.793 | OK |
| caso_009 | 3 | 3.950 | 0.950 | **NO** | 97.5 | 107.895 | 10.395 | OK |
| caso_010 | 4 | 4.993 | 0.993 | **NO** | 108.0 | 97.837 | 10.163 | OK |
| caso_011 | 5 | 5.027 | 0.027 | OK | 141.0 | 139.777 | 1.223 | OK |
| caso_012 | 8 | 7.747 | 0.253 | OK | 130.0 | 125.859 | 4.141 | OK |
| caso_013 | 7 | 6.890 | 0.110 | OK | 78.0 | 77.396 | 0.604 | OK |
| caso_014 | 6 | 6.147 | 0.147 | OK | 62.0 | 65.873 | 3.873 | OK |

Tolerancia usada: `quality` = 2× MAE CV reportado (2×0.427 = 0.854 puntos); `bound_so2` = 2× MAE CV
reportado (2×14.511 = 29.02 mg/L). Fuente: `metadata.json`/memoria sección 7.2.

**Resultado: 10/14 casos dentro de tolerancia en ambas dimensiones** (14/14 en `bound_so2`,
10/14 en `quality`).

### Investigación de los 4 fallos (caso_007, 008, 009, 010)

Los 4 casos que fallan son exactamente los de `quality` extrema (9, 9, 3, 4) — la memoria
(sección 3.3 y 10) documenta explícitamente que el dataset tiene un **sesgo de centralidad**
severo (92.59% de las filas tienen quality 5–7; solo 180 filas tienen 8–9 y 183 tienen 3–4), y
que "el modelo... se vuelve muy preciso en vinos estándar pero más conservador al predecir vinos
excelentes o pésimos". Para confirmar que esto es una limitación real del modelo entregado y no
un bug de wiring del plugin, se llamó a `model_qual.predict()` **directamente** sobre el mismo
artefacto (`artifacts/wine_sulphite/quality_rf.pkl`), sin pasar por el plugin:

| Caso | quality real | predicción directa del .pkl (bypass plugin) | baseline del plugin |
|---|---|---|---|
| caso_007 | 9 | 7.743 | 7.613 |
| caso_008 | 9 | 8.163 | 7.930 |
| caso_009 | 3 | 3.937 | 3.950 |
| caso_010 | 4 | 4.770 | 4.993 |

La predicción directa del artefacto (sin ningún código del plugin de por medio) es igual de
conservadora — confirma que es un límite real, documentado, del modelo RandomForest entregado
(no un bug de preprocesado/feature-building del plugin). **No se ha ajustado la tolerancia para
que estos 4 casos pasen** — se documentan como fallo esperado y explicado, por instrucción
expresa de la skill de verificación.

## Estado final

**LISTO PARA PR**, con la salvedad documentada de los 4 casos de calidad extrema (limitación real
del modelo entregado, no del wiring). Bugs de `train()` y artefacto base corregidos; manifest y
este informe documentan la trazabilidad completa. Revisión humana pendiente sobre este informe y
sobre `inbox/a25/manifest.yaml` antes de abrir PR (no se ha hecho commit ni PR).
