# Verificación — ml21-cereals-price-spatial

Modelo espacial de predicción de precios cerealísticos (ESP-CEREAL). Esta verificación se
lanzó a partir de una auditoría manual (manifest + plugin + documentación) que encontró tres
bugs de integración reales, no solo de wiring superficial. Los tres se corrigieron antes de
completar esta verificación — ver `inbox/a21/manifest.yaml` → `known_issues` para el detalle
completo de cada uno, y el resumen más abajo.

## Checklist técnico

- [x] flake8: 0 errores en `app/plugins/ml21_cereals_price_spatial/` y `app/registry.py`
- [x] pytest: 496/496 passed (suite completa del repo, no solo este modelo) — cobertura de
      `app/plugins/ml21_cereals_price_spatial/` 30% (los tests de `tests/unit/` validan wiring
      vía `FakePlugin`, no correctitud del modelo; la correctitud se valida en la Parte B)
- [x] pylint: 9.18/10 sobre el paquete (baseline previo a esta auditoría: 9.05/10 — sin
      regresión; los avisos nuevos son de complejidad razonable, p. ej. `too-many-instance-
      attributes` por los 3 atributos de cache que añade el fix del panel lookup)
- [x] pip-audit: 2 CVEs en `setuptools==80.9.0` (PYSEC-2026-3447) — preexistente y transversal
      a todo el repo, no introducido por ni específico de este modelo
- [x] Arranque local (`MODEL=ml21-cereals-price-spatial python main.py`) + health + predict
      (inline y batch) + stats + train: **ver hallazgos — health/stats ya funcionaban, predict
      y train estaban rotos en producción antes de esta verificación**

## Hallazgos corregidos durante esta verificación

| # | Severidad | Síntoma real | Causa | Estado |
|---|---|---|---|---|
| 1 | **Crítica** | `/predict` (inline y batch) devolvía **500** en toda llamada real, sin excepción | `predict_inline`/`predict_batch` devolvían un `dict` Python en vez de `PredictInlineResponse`/`PredictBatchResponse`; el use case genérico hace `type(result).model_validate(...)`, y `dict` no tiene ese método | ✅ Corregido — ambos envuelven ahora el resultado en su DTO tipado |
| 2 | **Crítica** | `predict_inline` producía predicciones silenciosamente incorrectas con el contrato documentado (provincia+cereal+date) | Las ~90 columnas de ingeniería de features (lat_centroide, month_sin/cos, fase_*, lags, clima, índices MAPA...) se rellenaban con 0 en vez de buscarse en el dataset base, como hace el código original (`run_single()`) | ✅ Corregido — `preprocessing.lookup_panel_row()` busca la fila real; `dataset_entrenamiento_final.csv` añadido como artefacto obligatorio |
| 3 | **Alta** | `/train` devolvía 422 si no se enviaba `mlflow_run_id` (aun siendo opcional para el plugin), y descartaba silenciosamente `mae_h*/pearson_h*/da_h*/auc_h*/n_train/n_test` de la respuesta real | `app/registry.py` no conectaba `train_request_type`/`train_response_type` para este modelo pese a tener `train()` implementado de verdad (GridSearchCV) — caía al DTO genérico | ✅ Corregido — registry.py ahora usa los DTOs reales del plugin |
| 4 | Media | `causal_drivers` era siempre el string fijo "No hay importancias disponibles..." | El modelo original calcula feature importance real + ranking por z-score (`_extract_feature_importance`/`_top_causal_drivers`); nunca se replicó | ✅ Corregido — `explain.py` replica la lógica; se calcula en `load()` y se persiste de nuevo en cada `train()` |
| 5 | Baja | Nomenclatura `m21_cereal_price_spatial` no seguía el estándar `mlNN_<sector>_<desc>` del resto de plugins cerealistas (`cereals` en plural) | — | ✅ Renombrado a `ml21_cereals_price_spatial` (carpeta, clase, model_id, tests, manifest) |

Los hallazgos #1 y #2 significan que, **antes de esta verificación, el modelo nunca había
funcionado a través de su API real** — los 8 tests que pasaban en verde en `tests/unit/`
usaban `FakePlugin` (factories que devuelven directamente el DTO tipado correcto) y nunca
ejercitaban el `return` real del plugin, por lo que ninguno de los dos bugs era visible en CI.
Confirmado end-to-end contra el servidor real antes y después de cada fix (ver manifest
known_issues para los curls exactos).

## Correctitud (golden dataset)

18 casos extraídos de `dataset_entrenamiento_final.csv` con `date >= CUT_DATE (2021-01-01)`,
estratificados 6 por cereal (trigo/cebada/maíz), `random_state=42`. `expected` es el **retorno
real realizado** por horizonte (`(TARGET_H{h} - precio_provincial_lag_1) / precio_provincial_lag_1`),
no una predicción del propio modelo — así la verificación compara contra la realidad, no
contra sí misma.

Tolerancia de magnitud: 2× el MAE reportado en la memoria por horizonte
(H1=0.1016, H2=0.1348, H3=0.2024). Dirección: se reporta el % de aciertos por separado, sin
usarlo como criterio pass/fail por caso individual — con DA reportada de 55.9%-66.1% en la
memoria, fallar la dirección en casos puntuales es un comportamiento esperado del modelo, no
un síntoma de wiring roto.

| Caso | Provincia / Cereal / Mes | H1 diff (tol) | H2 diff (tol) | H3 diff (tol) | ¿Magnitud OK? | Dirección (3 horiz.) |
|---|---|---|---|---|---|---|
| caso_001 | Valladolid / cebada / 2021-02 | 0.044 (0.102) | 0.085 (0.135) | 0.127 (0.202) | ✅ | 0/3 |
| caso_002 | Toledo / cebada / 2023-07 | 0.016 (0.102) | 0.086 (0.135) | 0.033 (0.202) | ✅ | 3/3 |
| caso_003 | Zaragoza / cebada / 2023-07 | 0.009 (0.102) | 0.058 (0.135) | 0.009 (0.202) | ✅ | 3/3 |
| caso_004 | Lleida / cebada / 2023-11 | 0.007 (0.102) | 0.008 (0.135) | 0.043 (0.202) | ✅ | 3/3 |
| caso_005 | Lleida / cebada / 2023-12 | 0.028 (0.102) | 0.028 (0.135) | 0.027 (0.202) | ✅ | 2/3 |
| caso_006 | Cuenca / cebada / 2024-01 | 0.038 (0.102) | 0.079 (0.135) | 0.032 (0.202) | ✅ | 3/3 |
| caso_007 | León / maíz / 2021-01 | 0.077 (0.102) | 0.127 (0.135) | 0.166 (0.202) | ✅ | 3/3 |
| caso_008 | León / maíz / 2021-06 | 0.001 (0.102) | 0.052 (0.135) | 0.040 (0.202) | ✅ | 2/3 |
| caso_009 | León / maíz / 2022-02 | 0.196 (0.102) | 0.300 (0.135) | 0.418 (0.202) | ❌ | 0/3 |
| caso_010 | León / maíz / 2024-01 | 0.034 (0.102) | 0.062 (0.135) | 0.048 (0.202) | ✅ | 3/3 |
| caso_011 | León / maíz / 2024-10 | 0.065 (0.102) | 0.078 (0.135) | 0.131 (0.202) | ✅ | 0/3 |
| caso_012 | León / maíz / 2025-07 | 0.039 (0.102) | 0.006 (0.135) | 0.077 (0.202) | ✅ | 2/3 |
| caso_013 | Salamanca / trigo / 2021-05 | 0.102 (0.102) | 0.033 (0.135) | 0.123 (0.202) | ✅ | 0/3 |
| caso_014 | Palencia / trigo / 2021-06 | 0.004 (0.102) | 0.058 (0.135) | 0.083 (0.202) | ✅ | 1/3 |
| caso_015 | Salamanca / trigo / 2022-05 | 0.102 (0.102) | 0.064 (0.135) | 0.083 (0.202) | ✅ | 0/3 |
| caso_016 | Cádiz / trigo / 2023-01 | 0.053 (0.102) | 0.026 (0.135) | 0.005 (0.202) | ✅ | 1/3 |
| caso_017 | Navarra / trigo / 2023-05 | 0.085 (0.102) | 0.072 (0.135) | 0.049 (0.202) | ✅ | 3/3 |
| caso_018 | Valladolid / trigo / 2024-02 | 0.092 (0.102) | 0.058 (0.135) | 0.034 (0.202) | ✅ | 3/3 |

**Resultado magnitud: 17/18 casos dentro de tolerancia (94%).**
**Resultado dirección: 32/54 horizonte-llamadas correctas (59.3%)** — en línea con la DA
reportada en la memoria (55.9%-66.1% según horizonte), no es una regresión del plugin.

### caso_009 (único fallo de tolerancia) — investigado, no silenciado

León/maíz/2022-02 tuvo un movimiento real extremo (+17% a +35% según horizonte) durante el
pico de precios de cereales de 2022 (shock de materias primas post-COVID / inicio guerra en
Ucrania). El modelo predijo un retorno plano/ligeramente negativo en los tres horizontes. Esto
**no es un bug de wiring** — es la limitación esperable de un modelo entrenado con MAE como
pérdida (optimiza para el caso típico, no para eventos extremos) sobre un periodo de
entrenamiento pre-2021 sin precedente de un shock de esa magnitud. Coincide con el patrón ya
documentado en `datos_ml21_cereals_price_spatial.json` → `limitaciones` ("no anticipa eventos
de impacto inmediato / cisnes negros"). Se deja documentado, no se ajusta la tolerancia.

### Patrón general observado

El modelo tiende a predecir retornos pequeños y ligeramente negativos para la mayoría de los
inputs de este sample (2021-2025), subestimando sistemáticamente la magnitud de los
movimientos alcistas grandes (casos 007, 009, 011, 013, 015). Es coherente con un Pearson
débil (0.21-0.38) y una DA apenas por encima del azar en H1 (55.9%) reportados en la propia
memoria — el modelo es direccionalmente mediocre mientras acierta razonablemente la
*magnitud típica* del movimiento (94% dentro de 2×MAE). Esto es una limitación del modelo
entregado por el equipo de IA, ya reflejada en `metrics_reported` y en la documentación
generada — no algo que competa corregir a la integración del plugin.

## Estado final

**LISTO PARA PR** — los 3 bugs críticos de wiring/arquitectura encontrados durante la
auditoría están corregidos y verificados end-to-end contra el servidor real (no solo contra
`FakePlugin`). La correctitud contra golden dataset es consistente con las métricas ya
reportadas en la memoria; el único caso fuera de tolerancia es una limitación documentada del
modelo (evento extremo), no un defecto de integración.

Pendiente de acción humana antes de mergear:
1. Confirmar que `data/processed/dataset_entrenamiento_final.csv` se sube a
   `s3://<STORAGE_BUCKET>/artifacts/fixed/ml21_cereals_price_spatial/` junto con los 6
   `.joblib` y `model_metadata.json` — sin él, `predict_inline` falla en `load()`.
2. Revisión humana de este informe y de los 3 documentos institucionales
   (`outputs/a21/ml21_*`), pendientes de regenerar con la nomenclatura nueva (`docs-generation`).
