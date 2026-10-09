# Verificación — a13 (`ml13-wine-price-fluctuation-prediction`)

- **Modelo**: Modelo para la predicción de fluctuaciones de precios basado en redes neuronales recurrentes (RNN-VINO), sector vitivinícola
- **Plugin**: `app/plugins/ml13_wine_price_fluctuation_prediction/` · rama `feature/model-a13-integration`
- **Manifest**: `inbox/a13/manifest.yaml` · memoria v1.10 (23/06/2026) · código commit `50bf1214d89c`
- **Modelo servido**: Regresión Logística (`models/prod/model_config.json = logreg`). Pese al título, no se integra ninguna RNN: la rama GRU es experimental, no trae artefacto y la memoria la excluye de producción (ver `known_issues.titulo_rnn_vs_modelo_produccion`).
- **Fecha**: 2026-10-06 · Entorno: WSL Ubuntu, Python 3.12.3, scikit-learn 1.5.2 (pin del repo), pandas 3.0.6, numpy 2.5.3, xgboost 3.4.1, fastapi 0.136.1

> **[CORREGIDO 2026-10-08] Contrato de reentrenamiento.** Al integrar `main` en la rama de
> auditoría, ml13 se alinea con el contrato común:
> - `mlflow_run_id` es obligatorio en `/train`; si falta o va vacío, la respuesta es 422.
> - El modelo reentrenado se guarda **solo en MLflow**. Ya no deja `user_*` en `artifacts/`.
> - Si la subida a MLflow falla, `/train` responde 502. Antes respondía 200 con `upload_warning`.
> - `/predict` con un run sin modelo cargable responde 422. Antes respondía 200 con el modelo base;
>   las filas de la tabla de abajo que dicen "fallback al artefacto fijo" y "user_\*" son del
>   comportamiento anterior.
>
> Re-verificación con `mapa_wine_prices_raw.csv`: `/train` obtiene hold-out AUC 0.8438, F1 0.5926,
> P 0.4211, R 1.0 y Acc 0.5417 (idéntico a lo de abajo). La predicción base no cambia (0.4621).
>
> **Pendiente (acción humana):** los artefactos no están en
> `s3://…/artifacts/fixed/ml13_wine_price_fluctuation_prediction/`. Hay que subir
> `inbox/a13/codigo/models/prod/`.

## Checklist técnico

- [x] flake8 (repo completo, `--extend-exclude=dist,build,inbox,outputs,artifacts`): 0 errores
- [x] pytest `tests/unit/`: **547/547 passed** (20 tests nuevos en `test_ml13_wine_price_fluctuation_prediction.py`). Cobertura global 37%. En los módulos ml13: preprocessing 90%, training 91%, DTOs y constants 100%. `plugin.py`, `model_loader.py` y `mlflow_utils.py` quedan al 0% en unit tests porque el patrón del repo prueba los endpoints con FakePlugin; el plugin real se ejercita E2E abajo.
  - 6 warnings `OptimizeWarning: Unknown solver options: iprint`, de sklearn 1.5.2 con scipy reciente. Son inocuos y salen en `test_train_models_end_to_end_on_synthetic_series`.
- [x] pylint `app/plugins/ml13_wine_price_fluctuation_prediction/ --disable=import-error`: **10.00/10**
- [x] pip-audit `-r requirements.txt`: **sin CVEs nuevas**. Sigue presente una CVE **previa** del repo: `setuptools==80.9.0` → PYSEC-2026-3447 (fix 83.0.0). ml13 no añade dependencias; no se toca el pin y queda como decisión humana.
- [x] Arranque local (`MODEL=ml13-wine-price-fluctuation-prediction uvicorn main:app`) + llamadas reales:

| Llamada | Resultado |
|---|---|
| `GET /health` | 200 `{status: ok, loaded: true, version: 1.0.0}` |
| `GET /stats` | 200, `model_type_served = logreg`, métricas declaradas (Tablas 4-6) |
| `POST /predict` inline, historial completo (174 filas MAPA) | 200 → semana 2025-11-24, `pred_proba_up = 0.4621`, `alerta_subida = 0`, 154 semanas predecibles |
| `POST /predict` batch (`data/raw/mapa_wine_prices_raw.csv`) | 200 → 174 filas, 154 con predicción (warm-up de 19 semanas + 1 fila sin precio = null), última fila idéntica al inline |
| `POST /predict` inline solo con esquema `bulletin` | 200, misma probabilidad (0.4621) que con `campaign`+`week` |
| `POST /predict` inline con 19 semanas válidas | **422** `InsufficientDataError`: "...at least 20 are required..." |
| `POST /predict` inline con 10 filas | **422** (validación del DTO, `min_length=20`) |
| `POST /predict` con `mlflow_run_id` no resoluble | 200 con fallback al artefacto fijo y WARNING en log (1,0 s) |
| `POST /train` (`mapa_wine_prices_raw.csv`) | **200** en 0,7 s → `best_model_type=logreg`, 126 trainval / 24 test, hold-out AUC 0.8438 · F1 0.5926 · P 0.4211 · R 1.0 · Acc 0.5417; CV logreg AUC 0.6564±0.2704, xgboost 0.4328±0.0685; Smart Score 0.3796 vs 0.3251 |
| `POST /train` con CSV sin `campaign`/`week`/`bulletin` | **400**, mensaje con las columnas requeridas |

`/train` responde 200 como exige `manifest.training.supported = true`. Los artefactos de usuario (`user_*`) que creó la prueba en `artifacts/` se han borrado después; los artefactos fijos no se modifican nunca.

> **Corrección de repo detectada.** `app/registry.py` en `main` (commit 68987d7) **no compilaba**: tenía un `),` sobrante entre las entradas ml16 y ml15 y un `]` duplicado al final, restos del merge de ml15. Se corrigió en esta rama al registrar ml13, con un cambio mínimo. Hace falta revisión humana: implica que `main` no arranca tal cual.

## Correctitud (golden dataset)

**Fuente**: los 24 golden cases son el hold-out completo del split entregado (`data/processed/wine_prices_test_processed.csv`, 2025-05-19..2025-10-27, 8 positivos). Cada caso se ejecuta contra el endpoint real `/predict` (inline), enviando el historial crudo MAPA hasta la fecha del caso.

**Tolerancias** (derivadas de `metrics_reported`; el modelo es un clasificador, así que no hay MAE):
1. **Agregado frente a la Tabla 6**: `atol = 0.0005`, medio paso del último decimal publicado.
2. **Etiqueta por caso** (`alerta_subida` frente al target real): comparación exacta. Los fallos se esperan y están cuantificados por la propia Tabla 6 (Precision 0,421 implica 11 FP; Recall 1,0 implica 0 FN).
3. **Cross-check de wiring**: probabilidad por caso frente al código original del equipo de IA (`inference.py::evaluate_on_last_weeks`) ejecutado con sus versiones exactas (sklearn 1.3.2, pandas 2.1.4, numpy 1.26.4) sobre los artefactos entregados, con `atol = 1e-9`. No es un golden inventado: es la salida del pipeline del propio equipo, porque no entregaron `output/test_predictions.csv`.

### Agregado frente a la Tabla 6 de la memoria

| Métrica | Memoria (Tabla 6) | Obtenido vía API | Diferencia | ¿OK? (atol 0.0005) |
|---|---|---|---|---|
| AUC-ROC | 0.8438 | 0.84375 | 5.0e-05 | ✅ |
| F1 | 0.593 | 0.59259 | 4.1e-04 | ✅ |
| Precision | 0.421 | 0.42105 | 5.3e-05 | ✅ |
| Recall | 1.000 | 1.00000 | 0 | ✅ |
| Accuracy | 0.542 | 0.54167 | 3.3e-04 | ✅ |

Matriz de confusión obtenida: **TP 8 · FP 11 · TN 5 · FN 0**. Es idéntica a la deducida de la Tabla 6.

### Por caso

| Caso | Target real | pred_proba_up (plugin) | Proba código original | Δ abs. proba | alerta_subida | ¿Etiqueta = target? |
|---|---|---|---|---|---|---|
| caso_2025-05-19 | 0 | 0.2892 | 0.2892 | 5.6e-17 | 0 | ✅ |
| caso_2025-05-26 | 0 | 0.4523 | 0.4523 | 0.0e+00 | 0 | ✅ |
| caso_2025-06-02 | 0 | 0.4820 | 0.4820 | 0.0e+00 | 0 | ✅ |
| caso_2025-06-09 | 1 | 0.7199 | 0.7199 | 0.0e+00 | 1 | ✅ |
| caso_2025-06-16 | 1 | 0.6552 | 0.6552 | 0.0e+00 | 1 | ✅ |
| caso_2025-06-23 | 0 | 0.5463 | 0.5463 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-06-30 | 0 | 0.3907 | 0.3907 | 5.6e-17 | 0 | ✅ |
| caso_2025-07-07 | 0 | 0.5169 | 0.5169 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-07-14 | 0 | 0.7524 | 0.7524 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-07-21 | 0 | 0.6994 | 0.6994 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-07-28 | 0 | 0.6269 | 0.6269 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-08-04 | 1 | 0.7895 | 0.7895 | 0.0e+00 | 1 | ✅ |
| caso_2025-08-11 | 1 | 0.7924 | 0.7924 | 0.0e+00 | 1 | ✅ |
| caso_2025-08-18 | 0 | 0.6348 | 0.6348 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-08-25 | 0 | 0.5275 | 0.5275 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-09-01 | 0 | 0.7123 | 0.7123 | 1.1e-16 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-09-08 | 1 | 0.7775 | 0.7775 | 0.0e+00 | 1 | ✅ |
| caso_2025-09-15 | 0 | 0.7371 | 0.7371 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-09-22 | 0 | 0.5740 | 0.5740 | 1.1e-16 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-09-29 | 1 | 0.7293 | 0.7293 | 0.0e+00 | 1 | ✅ |
| caso_2025-10-06 | 0 | 0.6188 | 0.6188 | 0.0e+00 | 1 | ❌ FP (declarado en Tabla 6) |
| caso_2025-10-13 | 1 | 0.6987 | 0.6987 | 0.0e+00 | 1 | ✅ |
| caso_2025-10-20 | 1 | 0.5593 | 0.5593 | 1.1e-16 | 1 | ✅ |
| caso_2025-10-27 | 0 | 0.4749 | 0.4749 | 1.1e-16 | 0 | ✅ |

Además, en los 24 casos:
- fecha de salida = fecha del caso;
- `price` = `price_red` del split;
- features (`xai_feature_values`) frente a `features_split` del manifest: error máximo 5,0e-09, que corresponde al redondeo a 10 cifras significativas del manifest;
- `/predict` batch frente a inline: diferencia máxima 1,1e-16.

**Resultado:**
- Agregado (Tabla 6): **5/5 dentro de tolerancia**
- Probabilidad frente al código original (wiring): **24/24 dentro de tolerancia** (diferencia máxima 1,1e-16)
- Etiqueta frente al target real: **13/24 coinciden; 11/24 NO coinciden**

### Investigación de los 11 casos con etiqueta distinta del target (no se silencian)

Casos afectados: `caso_2025-06-23`, `07-07`, `07-14`, `07-21`, `07-28`, `08-18`, `08-25`, `09-01`, `09-15`, `09-22` y `10-06`. Todos son **falsos positivos**: target real 0, el plugin predice 1, con probabilidades entre 0.517 y 0.752.

- **No es un fallo de wiring.** La probabilidad del plugin coincide con el pipeline original del equipo de IA en todos los casos (diferencia ≤ 1e-16), las features coinciden con el split entregado y el agregado reproduce la Tabla 6.
- **Es el comportamiento declarado del modelo entregado.** Tabla 6: Precision 0,421 y Recall 1,000 sobre 8 positivos implican exactamente 19 alertas, de las que 11 son falsas. El modelo alerta en 19 de 24 semanas (79%), así que alcanza el Recall de 1,0 a costa de la precisión.
- **Contexto de la memoria (sec. 7.3).** El test cae en un régimen de volatilidad doble que en entrenamiento y tiene un 33,3% de positivos frente al 20% de la CV. La propia memoria presenta estas métricas como "límite superior alcanzable bajo condiciones de mercado favorables".
- **Decisión pendiente (humana).** Es aceptable como herramienta de apoyo a la decisión, pero **no** como señal automática: aproximadamente 6 de cada 10 alertas son falsas en el mejor periodo documentado. No se ha ajustado el umbral de decisión (0,5, el del código original) ni ninguna tolerancia.

## Reproducibilidad (evidencia adicional)

| Comprobación | Resultado |
|---|---|
| Artefactos entregados con sklearn 1.3.2 (versión de serialización) frente a 1.5.2 (pin del repo) | Probabilidades idénticas (diferencia 0,0). El `InconsistentVersionWarning` al cargar es inocuo |
| Reentrenamiento con el código original (sklearn 1.3.2, mismo CSV crudo) | Coeficientes **idénticos** al `ml_model.pkl` entregado (diferencia de probabilidades 2,2e-16). El artefacto es reproducible |
| CV walk-forward reproducida frente a Tablas 4/5 | logreg AUC 0.6564 ± 0.2704, F1 0.1714, P 0.3211, R 0.290 ✅. xgboost AUC 0.4328 ± 0.0685 ✅. Smart Score 0.3796 / 0.3251 ✅. **Accuracy media CV = 0.6600; la Tabla 5 dice 0.659** (errata de 0,001 en la memoria) |
| `/train` del plugin (sklearn 1.5.2, xgboost 3.4.1) | Mismas métricas hold-out y CV que el original. Coeficientes con diferencia ≤ 2,6e-4 en probabilidad, por versiones de librería |
| XGBoost en CV (alternativa) | F1 = 0 en los 5 folds: con umbral 0,5 nunca predice la clase positiva. La memoria no lo menciona |
| Fold 1 de la CV | Entrena con un único positivo (AUC 0.274). La memoria lo reconoce como "estructural" |

## Cambios tras la revisión del PR (2026-10-08)

**Hallazgo del revisor:** no se validaba `price > 0` ni la finitud de los precios. Un precio 0 genera `logret = -inf`, que `dropna()` no filtra.

**Reproducción antes del cambio:**
- Precio 0 en las últimas semanas: 500 (el `StandardScaler` rechaza el `inf`).
- Precio 0 a mitad de la serie: 200 silencioso, con indicadores corruptos (RSI 89,6 frente a 58,3).
- Precio negativo o `inf` en la última semana: 200 silencioso, devolviendo la predicción de la semana anterior.

El código original del equipo de IA se comporta igual.

**Cambio:** `preprocessing.validate_prices()` rechaza la serie completa si algún precio informado no es numérico, finito y mayor que 0. Responde **422** en `/predict` y 400 en `/train`, con `DataContractError` y un mensaje que indica las semanas afectadas. Los precios vacíos se siguen omitiendo, como en el ETL original. Además hay una comprobación `isfinite` de las features antes del escalado.

**Re-verificación:**
- flake8: 0 errores · pylint: 10/10 · pytest: **561/561** (14 tests nuevos de regresión).
- Golden: probabilidades idénticas a las de antes del cambio (diferencia 0,0), mismas etiquetas y Tabla 6 reproducida.
- Endpoints: los mismos códigos HTTP que antes.

## Estado final

**LISTO PARA PR.** El wiring está verificado y es reproducible, y el checklist técnico está en verde.

**Decisión humana (2026-10-07):** se aceptan los 11/24 golden cases con etiqueta distinta del target real como comportamiento documentado del modelo entregado. Son los falsos positivos declarados en la Tabla 6 de la memoria (Precision 0,421) y no errores de integración. Se aceptan también las métricas de validación cruzada por debajo de los criterios de éxito de la memoria, tal como las reporta el propio equipo de IA. No se ha modificado ninguna tolerancia ni el umbral de decisión.

Puntos a tener en cuenta en el PR (no bloquean):

1. La corrección de `app/registry.py` (en `main` no compilaba) va incluida en este diff.
2. CVE previa en `setuptools==80.9.0` (PYSEC-2026-3447, fix 83.0.0). No la introduce ml13.
3. El título del proyecto dice "RNN", pero el modelo servido es una Regresión Logística. Está explicado en las fichas.
4. Convención del año en `bulletin` (segundo año de la campaña). A tener en cuenta en la integración con el front.
