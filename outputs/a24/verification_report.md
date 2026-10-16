# Verificación — ml24-dairy-spatial-price-forecast (a24)

Fecha: 2026-10-14
Plugin: `app/plugins/ml24_dairy_spatial_price_forecast/`
Manifest: `inbox/a24/manifest.yaml`
Entorno: Windows 11, Python 3.12, catboost 1.2.10, pandas 3.0.6, numpy 2.4.4, scikit-learn 1.5.2,
GPU NVIDIA RTX 3060. El código original se ejecutó en un venv aparte con las versiones de su
`requirements.txt` (pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0).

> **Resumen.** La integración reproduce exactamente al código del equipo de IA, tanto en la
> inferencia (paridad 1e-16 con `inference_smoke.csv`) como en el reentrenamiento (métricas
> idénticas a 6 decimales en CPU y GPU en la misma máquina). El checklist técnico está en verde
> para ml24. **Listo para PR.** Los 5 golden cases fuera de tolerancia están investigados y se
> deben al modelo, no a la integración (el modelo de etapa del propio equipo falla igual). La
> degradación fuera de muestra 2021-2026 (KPI incumplido en h3 en 2025-2026) es un problema del
> modelo entregado que se traslada al equipo de IA; no bloquea la integración.

## Checklist técnico

- [x] **flake8** (`flake8 . --extend-exclude=dist,build,inbox,artifacts`): 0 errores.
- [x] **pytest** `tests/unit/ --cov=app`: 1014 passed, 2 skipped, **7 failed, ninguno de ml24**,
      cobertura total 53 %. Los 36 tests de ml24 (wiring, aislamiento concurrente, higiene de
      MLflow, `preprocessing`) pasan todos. Los 7 fallos ya estaban en `main` y no tocan ficheros
      de esta rama:
  - 5 de **ml36** (`test_user_model_download_hygiene`, `test_mlflow_persistence_contract`):
    `predict_inline`/`predict_batch` asignan el modelo del usuario a `self._model` (fuga entre
    peticiones concurrentes) y `download_user_model_from_mlflow` no da 422 ni limpia el temporal
    si el run está incompleto. **Bug real en `main`**, fuera del alcance de esta integración.
  - 2 de **ml10** (`test_training_robustness`): el test compara rutas con `/`, lo que falla en
    Windows (`\`). Probablemente pasa en CI (Linux).
- [x] **pylint** `app/plugins/ml24_dairy_spatial_price_forecast/ --disable=import-error`: **10.00/10**.
      Hay `line-too-long` desactivado a nivel de módulo (el límite del repo es 120 en `.flake8`, y
      flake8 lo vigila) y `too-many-locals/arguments` puntuales en las funciones portadas de
      `train.py`, que conservan la estructura del original. `_vendor/` lleva `pylint: skip-file`
      por ser código del equipo copiado literalmente.
- [x] **pip-audit** `-r requirements.txt`: 1 vulnerabilidad, `setuptools==80.9.0`
      (PYSEC-2026-3447, se corrige en 83.0.0). Ese pin ya estaba en `main`; `catboost==1.2.10`,
      la dependencia añadida, no introduce CVEs.
- [x] **Arranque local** (`MODEL=ml24-dairy-spatial-price-forecast uvicorn main:app`):
      `from app.registry import REGISTRY` correcto (33 modelos); `/health` devuelve `loaded: true`
      (carga en ~3 s, leyendo por bloques la serie de 338 MB y quedándose con los 5 productos del
      modelo).

| Endpoint / caso | HTTP | Tiempo |
|---|---|---|
| `GET /health`, `GET /stats` | 200 | — |
| `POST /predict` inline sin filtros (51 filas) | 200 | 5,0 s |
| `POST /predict` batch con el CSV largo oficial (51 filas) | 200 | 5,2 s |
| inline `ccaa: [ESP]` | 422 (`UnknownCcaaError`) | — |
| inline `as_of_period: 1999-06` (6 meses de histórico) | 422 (`InsufficientPriceHistoryError`) | — |
| inline `horizons: [2]` | 422 (DTO) | — |
| batch con `data_path` inexistente | 422 (`SeriesContractError`; antes de corregirlo daba 500) | — |
| inline con `mlflow_run_id` cuyo run no carga | 422 (`UserModelUnavailableError`) | 2 s |
| `POST /train` sin `mlflow_run_id` / con `" "` | 422 | — |
| `POST /train` con fichero inexistente | 400 | — |
| `POST /train` real con MLflow inalcanzable | **502** (`ModelPersistenceError`) tras entrenar | 50,7 s (GPU) |

- [x] **/train → 200 con métricas** (`manifest.training.supported: true`): con MLflow no
      disponible en local, el flujo se comprobó por partes. (1) Entrenamiento real con el servidor
      hasta la subida, que falla con 502 como exige el contrato. (2) `save_training_artifacts` →
      `load_bundle_from_dir` → inferencia con `mlflow_run_id`, simulando MLflow con un directorio
      local: el modelo reentrenado se sirve (CYL h1 0,8209 frente al base 0,8765) y el base no
      se modifica. (3) `TrainResponse` con FakePlugin en los tests de wiring.

### Fidelidad del reentrenamiento (`training.py` frente a `src/training/train.py`)

| Comparación (misma máquina) | Resultado |
|---|---|
| Frame supervisado por horizonte (`prepare_frame` frente a `_prepare_frame`) | Idéntico: mismas filas, columnas y dtypes. Con paneles en memoria solo difiere en 1 ULP (el original los escribe y relee por CSV); no cambia ninguna métrica |
| Plugin CPU frente a original CPU | train/val/test sMAPE y MAE **idénticos a 6 decimales**; mismos ml_alpha (0,6/0,5/0,5), pesos y features |
| Plugin GPU frente a original GPU | **Idénticos a 6 decimales**; ml_alpha 0,4/0,5/0,55 |
| Original GPU aquí frente a memoria (GPU del equipo) | h1 test sMAPE 2,019 frente a 1,871; h3 1,966 frente a 1,979; h6 2,079 frente a 2,046. Diferencia de hardware GPU: el artefacto entregado no se reproduce bit a bit en otra GPU |
| Duración de `/train` | ~50 s en GPU, ~65 s en CPU |

**Bug del original encontrado durante esta comparación.** `config.yaml` declara
`bootstrap_type: No`. `yaml.safe_load` lo convierte en el booleano `False` y `build_model` solo
pasa el parámetro si es truthy, así que el equipo de IA (y el artefacto servido) entrenó con el
bootstrap **por defecto** de CatBoost, no con "No". La primera versión del port usaba "No" y se
desviaba hasta 0,02 pp. Se corrigió para reproducir el comportamiento real
(`TRAINING_CFG["bootstrap_type"] = False`, comentado). Arreglarlo en el original cambiaría el
modelo: decide el equipo de IA.

## Correctitud (golden dataset)

**Origen de los casos.** `data/splits/` no se entrega; el README dice que es reproducible. Se
regeneró con `train.py::_prepare_frame` y las ventanas de `config.yaml`, y los recuentos coinciden
con la Tabla 5 de la memoria (train 3655/3621/3366, validación y test 204). Se tomaron 18 filas
del split de test (2020) con una regla fija: ene/mar/may/jul/sep/nov × 17 CCAA, con Castilla y
León dos veces. La fila con mes base t usa dato MAPA hasta t−12 (`publication_lag_months=12`):
`input.as_of_period = t−12` y `expected = y real del mes t+h`.

**Tolerancia**: 2 × MAE de test / media de y en test 2020 (0,695), con el MAE de
`metrics_reported`: **h1 3,742 %, h3 3,962 %, h6 4,098 %**. No se ha ajustado.

| Caso | Esperado (y real) | Obtenido (API) | Diferencia | Error rel. | Tol. | ¿OK? |
|---|---|---|---|---|---|---|
| h1_2020-01_CYL | 0.663144 | 0.676657 | +0.013513 | +2.04 % | 3.742 % | ✅ |
| h1_2020-03_AND | 0.673777 | 0.676795 | +0.003018 | +0.45 % | 3.742 % | ✅ |
| h1_2020-05_CAN | 0.687172 | 0.682664 | −0.004508 | −0.66 % | 3.742 % | ✅ |
| h1_2020-07_CAT | 0.699593 | 0.726119 | +0.026526 | +3.79 % | 3.742 % | ❌ |
| h1_2020-09_RIO | 0.735028 | 0.690182 | −0.044846 | −6.10 % | 3.742 % | ❌ |
| h1_2020-11_GAL | 0.688321 | 0.689119 | +0.000798 | +0.12 % | 3.742 % | ✅ |
| h3_2020-01_MAD | 0.695547 | 0.690697 | −0.004850 | −0.70 % | 3.962 % | ✅ |
| h3_2020-03_CYL | 0.655166 | 0.674480 | +0.019314 | +2.95 % | 3.962 % | ✅ |
| h3_2020-05_BAL | 0.732004 | 0.704992 | −0.027012 | −3.69 % | 3.962 % | ✅ |
| h3_2020-07_VAL | 0.687657 | 0.681060 | −0.006597 | −0.96 % | 3.962 % | ✅ |
| h3_2020-09_PVA | 0.716457 | 0.713772 | −0.002685 | −0.37 % | 3.962 % | ✅ |
| h3_2020-11_EXT | 0.683649 | 0.677014 | −0.006635 | −0.97 % | 3.962 % | ✅ |
| h6_2020-01_AST | 0.686111 | 0.683273 | −0.002838 | −0.41 % | 4.098 % | ✅ |
| h6_2020-03_MUR | 0.652137 | 0.707416 | +0.055279 | +8.48 % | 4.098 % | ❌ |
| h6_2020-05_ARA | 0.732200 | 0.711625 | −0.020575 | −2.81 % | 4.098 % | ✅ |
| h6_2020-07_NAV | 0.732923 | 0.699076 | −0.033847 | −4.62 % | 4.098 % | ❌ |
| h6_2020-09_CTB | 0.646598 | 0.707589 | +0.060991 | +9.43 % | 4.098 % | ❌ |
| h6_2020-11_CLM | 0.678695 | 0.677581 | −0.001114 | −0.16 % | 4.098 % | ✅ |
| rechazo_ccaa_espana | HTTP 422 | HTTP 422 | — | — | — | ✅ |
| rechazo_historial_insuficiente | HTTP 422 | HTTP 422 | — | — | — | ✅ |

Tolerancia usada: 2 × MAE relativo de test (memoria, Tabla 10 / `metrics.csv`).
**Resultado: 15/20 casos dentro de tolerancia (13/18 numéricos + 2/2 rechazos).**

Paridad adicional (no son golden cases):

- Frente al **predictor original sin modificar** con el mismo input (CSV truncado en
  `as_of_period`): los 18 casos coinciden, con diferencia máxima 4,8e-7 (el redondeo a 6
  decimales del manifest).
- Frente a **`inference_smoke.csv`** (referencia oficial del equipo; `verify_repro.py` da PASS
  en este entorno): 51/51 filas, `max|Δ y_pred| = 1,1e-16`, `y_baseline_calibrated` 2,2e-16,
  `y_persist_1/12` 0. `feature_period` y `period` idénticos.

### Investigación de los 5 casos fuera de tolerancia

1. **No es un fallo de integración.** El plugin coincide con el predictor original a 5e-7 en los
   18 casos, y los paneles y features coinciden.
2. **El propio equipo falla igual.** El modelo de etapa del equipo (`evaluation_predictions.csv`,
   split test) da en esos casos +4,49 % (CAT h1), −5,81 % (RIO h1), +8,34 % (MUR h6), −4,75 %
   (NAV h6) y +10,28 % (CTB h6), casi lo mismo que el artefacto servido.
3. **Una tolerancia por caso de 2 × error medio deja fuera, por construcción, la cola de errores.**
   Sobre las 204 filas de test, quedan fuera de esa tolerancia el 14,7 / 12,7 / 11,8 % (h1/h3/h6)
   con el modelo de etapa del equipo, el 13,7 / 12,3 / 9,8 % con el artefacto desplegado por la
   API y el 18,1 / 16,7 / 15,7 % con la persistencia ingenua. Aquí salen 5/18 (28 %) porque 3 de
   los 6 casos de h6 caen en la cola. MUR 2020-03 y CTB 2020-09 son caídas puntuales del precio
   MAPA de ~0,70 a ~0,65 que ningún predictor anticipa.
4. Test completo por la API (612 filas): sMAPE **1,859 / 1,850 / 1,937 %** (h1/h3/h6), en línea
   con lo declarado (1,871 / 1,979 / 2,046 %). Ojo: es *in-sample* para el artefacto, que se
   reentrenó con todo hasta 2020.

## Evidencia fuera de muestra 2021-2026 (predictor original, sin modificar)

Se ejecutó `run_prediction` con la serie oficial truncada en cada mes de 2020-12 a 2026-05
(66 cortes) y se comparó con el precio real publicado para el `period` de la salida.

| h | Exactitud 2021-26 | Persistencia ingenua | Exactitud 2025-26 | Sesgo 2023-26 |
|---|---|---|---|---|
| 1 | 90,96 % | 97,61 % | 88,71 % | −7 a −12 % |
| 3 | 85,32 % | 96,60 % | **79,48 % (< KPI 80 %)** | −15 a −19 % |
| 6 | 85,59 % | 94,55 % | 82,19 % | −12 a −17 % |

- 2021: exactitud de ~97 %, como en la memoria. Desde 2022 (precios de ~0,95-1,12 EUR/l frente
  a ≤ 0,98 en 1999-2020, y 0,63-0,75 en 2020) **infraestima de forma sistemática**: CatBoost no
  extrapola y el blend arrastra la predicción hacia el rango histórico. La propia
  `inference_smoke.csv` (dato hasta 2026-06) predice de media un 8,7 / 15,1 / 11,3 % por debajo
  del último precio observado.
- Desde 2022 es peor que repetir el último precio en los tres horizontes.
- La regla de reentrenamiento de la memoria (sMAPE > 5 % durante 6 meses seguidos) se habría
  activado en **2022-06** (h3, h6) y **2022-12** (h1), y sigue activa (h1 25, h3 46 y h6 54 meses consecutivos con sMAPE > 5 %).

## Otros hallazgos sobre el código original (detalle en `known_issues` del manifest)

- **Exógenas desalineadas y etiqueta temporal.** Con `publication_lag_months=12`, el
  entrenamiento retrasa las features MAPA pero no las exógenas. Además, la salida etiqueta
  `period = último mes MAPA + h`, aunque el modelo aprendió a predecir 12 + h meses por delante.
  En 2020 ambas lecturas dan el mismo error (1,86 frente a 1,82 de sMAPE en h1).
- En inferencia, la imputación por mediana usa la mediana del propio frame de inferencia
  (replicado).
- Una CCAA con menos de 13 meses de histórico se predecía con valores imputados. El plugin la
  omite con aviso, como dice la memoria (8.4).
- `validate_inference_input` no exigía `es_reconstruido`, aunque se necesita después (el plugin
  la exige).

## Cambios del plugin frente al original (deliberados)

- No se porta `--refresh-source`: un pod no debe descargar de webs externas en cada petición.
- El batch siempre devuelve 17 × 3 filas, porque el caso de uso genérico no reenvía filtros a
  `predict_batch`.
- `/train` admite `validation_year`/`test_year` opcionales (por defecto 2019/2020, que es el
  procedimiento original) y no recalcula el rolling summary.

## Pendiente de acción humana

- Subir los 10 ficheros de `artifacts/ml24_dairy_spatial_price_forecast/` (md5 en el manifest) a
  `s3://<STORAGE_BUCKET>/artifacts/fixed/ml24_dairy_spatial_price_forecast/`.
- Decidir con el equipo de IA: reentrenar con datos posteriores a 2020; aclarar la etiqueta
  temporal y la alineación de las exógenas; si se corrige `bootstrap_type`.

## Estado final

**LISTO PARA PR.** El plugin reproduce fielmente el código entregado (inferencia y
reentrenamiento) y el checklist técnico está en verde para ml24. Los defectos del código original
se replican a propósito y quedan documentados en `known_issues`.

Constan en el PR y se trasladan al equipo de IA (no bloquean la integración):

1. 5/18 golden cases están fuera de tolerancia, sin silenciarlos ni ajustar la tolerancia. Se
   deben al modelo: el modelo de etapa del equipo falla igual en los mismos casos.
2. El modelo entregado está degradado fuera de muestra desde 2022 e incumple el KPI en h3 en
   2025-2026 (79,5 %): hay que reentrenarlo o aceptar el riesgo.
3. Aclarar con el equipo la etiqueta temporal y la alineación de las exógenas, y el
   `bootstrap_type: No` parseado como `False`.
4. Los 7 fallos de la suite son de otros plugins y ya estaban en `main` (ml36, bug real de
   aislamiento; ml10, separador de rutas en Windows). Si el pipeline de Bitbucket ejecuta la suite
   completa, fallará por ellos y no por ml24.
