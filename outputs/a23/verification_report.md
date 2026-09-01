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

## Hallazgos y correcciones aplicadas

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

**Mismo bug, sin tocar (fuera de alcance de esta tarea):** `m21_cereal_price_spatial`,
`ml25_wine_sulphites`, `ml5_meat_cow_behaviour`, `ml17_meat_market_price_analysis` y
`modelo10_lacteo` usan el mismo patrón roto. Se deja constancia para que se decida si se corrigen
en un cambio aparte.

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

**LISTO PARA PR** tras los 3 fixes aplicados (current_price, model_loader lazy-download,
mlflow_utils.py). Antes de esta revisión el plugin estaba **roto en producción para
predict_batch** (devolvía siempre cero predicciones) sin que ningún test existente lo detectara
— los tests de `tests/unit/test_ml23_lactic_market_price_forecast.py` usan `FakePlugin` y nunca
ejercitan el código real. Se recomienda añadir un test con artefactos reales (mismo patrón que
`test_ml16_meat_raw_material_price_alert` si existiera) para que esta clase de bug no vuelva a
pasar desapercibida.

Pendiente de decisión humana (no bloqueante para PR de este plugin):
- Corregir el mismo bug de `download_all_if_needed()` en los otros 5 plugins afectados.
- Añadir `mlflow_utils.py` a los otros 3 plugins que también lo omiten.
- Actualizar `outputs/a23/a23_ficha_tecnica.docx` para citar `GRU_artifact_metrics` en vez de
  `GRU` (mean-across-seeds).

Esta verificación no abre PR ni hace merge — queda pendiente de revisión humana del plugin, este
informe y `inbox/a23/manifest.yaml` antes de abrir el PR.
