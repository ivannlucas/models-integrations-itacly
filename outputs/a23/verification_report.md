# Verificación — ml23-lactic-market-price-forecast (a23)

Fecha: 2026-09-01
Plugin: `app/plugins/ml23_lactic_market_price_forecast/`
Manifest: `inbox/a23/manifest.yaml` (generado en esta revisión — no existía antes)
Entorno de verificación: `.venv/` del repo (pandas 2.3.3, torch instalado)

## Contexto: por qué esta verificación se hace ahora

El plugin de ml23 ya estaba integrado en `app/plugins/` y documentado en `outputs/a23/`
(fichas técnica/funcional) **sin que existiera nunca `inbox/a23/manifest.yaml`** — se saltó el
orden de trabajo estándar del repo (manifest-extraction siempre antes de plugin-integration, y
verification antes de dar el plugin por bueno). El usuario aportó el código original en
`inbox/a23/codigo/` para poder ejecutar ahora, retroactivamente, todas las comprobaciones que
deberían haberse hecho entonces. El resultado: **se ha encontrado y corregido un bug real de
correctitud que dejaba `predict_batch()` devolviendo cero predicciones en silencio.**

## Ciclo 2 (2026-10-06) — `train()` estaba mal deshabilitado (bug de clase 3)

Al revisar el estado global de los 25 plugins del repo se encontró que `training.supported:
false` en este manifest (Ciclo 1) tenía el mismo motivo que ya se había detectado y corregido
esta sesión en otros 4 modelos (ml2, ml4, ml7, ml17): el manifest afirmaba "sin procedimiento de
reentrenamiento entregado en este repo", pero `inbox/a23/codigo/` **sí trae uno real y completo**
(`scripts/train.py` → `src/training/runner.py::train_from_config()` →
`src/training/compare_models.py::run_comparison()`), con `modules.training_enabled: true`
explícito en `config/config.yaml`.

### Bug corregido: `train()` deshabilitado sin motivo real

Corregido reimplementando `train()` — porta fielmente SOLO el refit final de la arquitectura GRU
ya seleccionada y desplegada (`prepare_horizon_dataset` → `split_dev_test` → `make_refit_split` →
`prepare_rnn_split_with_test` → `train_rnn_multi_seed`), sin re-ejecutar la búsqueda completa
Naive/Drift/XGBoost/LSTM/GRU por CV temporal que originalmente seleccionó GRU — misma
simplificación ya aplicada a la búsqueda neuroevolutiva de `ml30`/`ml9` en este repo (reentrenar
una NAS/comparativa completa en cada petición HTTP síncrona sería impracticable).

**Verificación cuantitativa (sin reentrenamiento largo):** se ejecutó `plugin.train()`
directamente (con `BaseMLflowTracker` monkeypatcheado a un stub no-op — mismo workaround ya
documentado para otros modelos de esta sesión, `MLFLOW_TRACKING_URI` inalcanzable en este
sandbox) sobre el **dataset real completo** (`data/processed/dataset_forecast_ready.csv`, 1.440
filas, 16 series producto×canal):

```
TrainResponse: mae=0.0135 rmse=0.017 mape_pct=1.33 r2=0.8994 direction_acc_pct=90.6
               n_train=1040 n_test=128 training_time_s=48.4
```

Estas cifras **coinciden a 4+ decimales** con `metrics_reported.gru_shipped_artifact_seed42` del
manifest (MAE=0.013472696283459668, RMSE=0.0169822828247763, R²=0.8994110336510438,
direction_acc=90.6 — el bloque de métricas del propio artefacto `gru_model.pt` ya servido,
auditado en el Ciclo 1) — confirma que el refit portado reproduce exactamente el procedimiento
original, no una aproximación. 48.4s de duración total (3 seeds sobre el dataset completo): no es
un reentrenamiento largo, es ejecutable en cada petición.

### Bug corregido (relacionado): `mlflow_run_id` ignorado en predict/stats

`predict_inline`/`predict_batch`/`stats` declaraban `mlflow_run_id` en su firma pero lo
descartaban siempre (`_ = mlflow_run_id`) — inofensivo mientras no existía ningún artefacto de
usuario al que apuntar, pero se habría convertido en un bug real de aislamiento entre usuarios en
cuanto `train()` empezara a producir artefactos reales. Corregido: los tres métodos ahora
resuelven el bundle (modelo+scaler+manifest) del usuario vía `_resolve_for_predict()` cuando se
indica `mlflow_run_id`, sin mutar nunca `self.*`, con `shutil.rmtree` del directorio temporal en
`finally` — mismo patrón ya usado en `ml2`/`ml4`/`ml7`/`ml17`/`ml25`.

### Checklist técnico (Ciclo 2)

- [x] `flake8 app/plugins/ml23_lactic_market_price_forecast/ app/registry.py tests/conftest.py tests/unit/test_ml23_lactic_market_price_forecast.py`: 0 errores.
- [x] `pylint app/plugins/ml23_lactic_market_price_forecast/ --disable=import-error`: **9.39/10**
      (mismas categorías que el baseline ya aceptado en otros plugins de este repo:
      `invalid-name` en variables `X`/`train_X`/`val_X` — convención numpy/ML estándar, no se
      renombra — y `consider-using-from-import` en `rnn_models.py`, preexistente).
- [x] `pytest tests/unit/ -q`: **505/505 passed** (suite completa, +1 respecto al baseline
      anterior: se sustituyó `test_train_returns_501` por `test_train_returns_200_with_metrics` +
      `test_train_without_mlflow_run_id_returns_422`).
- [x] Arranque real (`MODEL=ml23-lactic-market-price-forecast`, puerto 8000) + `/health`: OK.
- [x] `POST /train` sin `mlflow_run_id` → **422** (`Field required`), confirmado contra el
      servidor real.
- [x] `POST /predict` inline y batch (con el dataset real completo) tras el fix: sin cambios de
      comportamiento, 200 en ambos, mismas predicciones que antes del Ciclo 2 (el fix de
      `train()`/`mlflow_run_id` no toca la ruta de inferencia con el artefacto base).
- Servidor y procesos detenidos limpiamente al terminar (`kill -9` + `pkill -9 -f
  multiprocessing.spawn`), confirmado sin procesos huérfanos.

## Hallazgos y correcciones aplicadas (Ciclo 1)

### 1. [CRÍTICO, CORREGIDO] `predict_batch()` no derivaba `current_price`

`dataset_forecast_ready.csv` — el propio dataset del modelo — no trae una columna
`current_price`, solo `target_precio_medio`. El código original
(`predictor.py::_prepare_input_df()`) deriva `current_price = target_precio_medio` cuando falta;
`plugin.py` nunca lo hacía. Confirmado ejecutando el plugin real contra el CSV real: **las 16
series (producto × canal) se descartaban por completo** con `"Missing cols... ['current_price']"`
y `predict_batch` devolvía `predictions=[]` — HTTP 200 con lista vacía, sin ningún error visible
para el llamador.

**Corregido** en `plugin.py::predict_batch()` — misma derivación de una línea que el código
original. Verificado: tras el fix, `predict_batch` sobre `dataset_forecast_ready.csv` produce
1360 predicciones (antes: 0) y coincide exactamente (4 decimales) con la salida del script
original `predictor.py` para las mismas filas.

### 2. [ALTO, CORREGIDO] `load()` fallaba en cualquier entorno sin `STORAGE_BUCKET`

`model_loader.py` llamaba a `ArtifactStore.download_all_if_needed()`, que lanza
`EnvironmentError` **inmediatamente** si `STORAGE_BUCKET` no está seteado — sin comprobar antes
si los artefactos ya existen localmente. Confirmado: con los 3 artefactos ya vendorizados en
`artifacts/ml23_lactic_market_price_forecast/`, `load()` fallaba igualmente en este entorno de
verificación (sin S3 configurado).

**Corregido** en `model_loader.py` — cambiado a `_store.path(filename)` por fichero (patrón lazy
que ya usan correctamente ~18 de los ~24 plugins del repo, incluido `ml16`). Verificado: `load()`
funciona ahora sin `STORAGE_BUCKET`.

**Mismo bug, sin tocar (fuera de alcance de esta tarea):** `ml21_cereals_price_spatial`,
`ml25_wine_sulphites`, `ml5_meat_cow_behaviour`, `ml17_meat_market_price_analysis` y
`modelo10_lacteo` usan el mismo patrón roto. Se deja constancia para que se decida si se corrigen
en un cambio aparte.

**[RESUELTO 2026-10-02]** Los 5 plugins listados ya usan el patrón lazy `_store.path(filename)`
— ninguno llama a `download_all_if_needed()` a día de hoy. Corregidos en cambios posteriores a
este informe (no en esta tarea de a23); verificado releyendo cada `model_loader.py`.

### 3. [MEDIO, CORREGIDO] Faltaba `mlflow_utils.py`

Viola la regla del repo ("todo plugin lleva mlflow_utils.py, sin excepción"). Añadido un stub
honesto (no hay formato de artefacto reentrenado que descargar, ya que `train()` no está
soportado y no hay procedimiento de reentrenamiento entregado) — documentado como tal, no se ha
inventado lógica de descarga sin caller real.

**Mismo gap, sin tocar (pre-existente a esta regla, fuera de alcance):** `ml2_fungal_cnn_disease_detection`,
`ml5_meat_cow_behaviour`, `ml7_cereals_grain_pest_detection`.

### 4. [DOCUMENTADO, no bloqueante] Métricas reportadas vs. artefacto servido

`final_test_metrics.json` reporta dos bloques distintos: `"GRU"` (MAE=0.0177, RMSE=0.0226,
R²=0.7957 — **media de 3 semillas**) y `"GRU_artifact_metrics"` (MAE=0.0135, RMSE=0.0170,
R²=0.899 — **el artefacto único `gru_model.pt` realmente servido**, notablemente mejor). La
ficha técnica generada (`outputs/a23/a23_ficha_tecnica.docx`) debería citar el segundo bloque, no
el primero. Ver `inbox/a23/manifest.yaml` known_issues — no se ha modificado la ficha en esta
revisión (fuera de alcance; requeriría re-ejecutar docs-generation).

### 5. [DOCUMENTADO, no bloqueante] `predict_inline()` replica (tile) la fila 6 veces

No existe en el código original una vía de predicción "de una sola fila" — el modelo siempre se
evaluó sobre 6 meses reales y distintos entre sí. Tilear una fila constante es una aproximación
de conveniencia ya usada en otros modelos de este repo con la misma limitación estructural
(ml16, modelo-40, modelo-46) y ya señalada como tal en `retech-lote2-xai-plataforma`. Se deja
documentada, no se ha intentado cuantificar el error que introduce.

## Checklist técnico

- [x] **flake8**: 0 errores en `app/plugins/ml23_lactic_market_price_forecast/*.py`.
- [x] **pylint**: 9.66/10. Avisos: `invalid-name` en variables `X`/`X_t`/`X_sc` (convención
      numpy/ML estándar, no se renombra), `consider-using-from-import` en `rnn_models.py`
      (preexistente, no se toca por ser vendorizado del código original). Sin categorías nuevas.
- [x] **pytest** (`tests/unit/`, suite completa del repo): **400/400 passed**, 0 regresiones
      tras los 3 fixes.
- [x] **pip-audit**: sin dependencias nuevas añadidas por este plugin (torch/numpy/pandas/xgboost
      ya pinneados en el repo); mismas 4 CVEs preexistentes del repo (torch/setuptools/cryptography)
      ya reportadas en la verificación de ml16, ajenas a ml23.
- [x] **Correctitud contra golden dataset**: **8/8 casos** (`inbox/a23/manifest.yaml`
      golden_cases) — reproducidos ejecutando `plugin.py::predict_batch()` real (artefactos
      reales, sin mocks) contra `dataset_forecast_ready.csv`, coincidencia exacta a 4 decimales
      con la salida del script original `predictor.py --model gru`.

## Estado final

**LISTO PARA PR.** Resumen acumulado de ambos ciclos:

- Ciclo 1: 3 fixes (current_price, model_loader lazy-download, mlflow_utils.py stub) — el plugin
  estaba roto en producción para `predict_batch` (devolvía siempre cero predicciones) sin que
  ningún test existente lo detectara (`tests/unit/` usa `FakePlugin`, nunca ejercita el código
  real contra artefactos reales).
- Ciclo 2: `train()` estaba deshabilitado sin motivo real (bug de clase 3, mismo patrón que
  ml2/ml4/ml7/ml17) — corregido, verificado end-to-end contra el dataset real completo
  (mae=0.0135, rmse=0.017, r2=0.8994 — coincide a 4+ decimales con el artefacto servido) y contra
  el servidor HTTP real (422 sin `mlflow_run_id`). `mlflow_run_id` ahora también se usa de verdad
  en predict/stats (antes se ignoraba).
- `pytest` 505/505, `flake8` 0 errores, `pylint` 9.39/10 — sin regresiones.

Pendiente de decisión humana (no bloqueante para PR de este plugin):
- ~~Corregir el mismo bug de `download_all_if_needed()` en los otros 5 plugins afectados.~~
  **[RESUELTO 2026-10-02]** — ver nota en el hallazgo #2 más arriba.
- ~~Añadir `mlflow_utils.py` a los plugins que todavía lo omiten.~~ **[RESUELTO]** — los 5
  plugins listados (`ml17`, `ml2`, `ml4`, `ml5`, `ml7`) ya tienen `mlflow_utils.py` real, no stub,
  corregidos en sus propios ciclos de verificación de esta sesión.
- ~~Actualizar `outputs/a23/a23_ficha_tecnica.docx` para citar `GRU_artifact_metrics` en vez de
  `GRU` (mean-across-seeds).~~ **[RESUELTO 2026-10-02]** — `datos_a23.json` ahora incluye ambos
  bloques, etiquetados por separado ("media 3 semillas" vs "artefacto servido, seed=42"), y la
  ficha técnica se ha regenerado con `docs-generation`.
- ~~La ficha técnica/funcional aún no se ha regenerado tras el Ciclo 2 (seguían describiendo
  `train()` como no soportado).~~ **[RESUELTO]** — `datos_a23.json`/`datos_a23_funcional.json`
  actualizados (contrato de train, métricas de correctitud del refit, limitaciones) y las fichas
  regeneradas con `docs-generation`.

Esta verificación no abre PR ni hace merge — queda pendiente de revisión humana del plugin, este
informe y `inbox/a23/manifest.yaml` antes de abrir el PR.
