# Verificación — ml30-meat-traceability-detection

Plugin: `app/plugins/ml30_meat_traceability_detection/`
Manifest: `inbox/a30/manifest.yaml` (generado en esta sesión — el plugin ya estaba integrado y
registrado en `app/registry.py`, pero nunca había pasado por `manifest-extraction` ni
`verification`. Código delivery real auditado: `inbox/a30/codigo/`, 101 ficheros, memoria
`inbox/a30/entregable/30 - ... v1.1.docx`).

## Resumen de la auditoría (contexto: bugs de ml21)

Se auditó el plugin con el mismo rigor que destapó 3 bugs severos en `ml21_cereals_price_spatial`
(dict crudo en vez de DTO tipado; zero-fill silencioso de ~90 features; `train_request_type`/
`train_response_type` ausentes en el registro). Resultado para ml30:

| Clase de bug (ml21) | ¿Presente en ml30? | Evidencia |
|---|---|---|
| 1. `predict_inline`/`predict_batch` devuelven `dict` crudo en vez de DTO tipado | **No.** | `plugin.py` devuelve `PredictInlineResponse(...)` / `PredictBatchResponse(...)` explícitamente en ambos métodos (líneas 128 y 172). `grep -n "return {" plugin.py` → sin resultados. |
| 2. Zero-fill silencioso de features que el modelo necesita | **No.** | Las 33 `FEATURE_COLUMNS` del plugin (`constants.py`) coinciden EXACTAMENTE (mismo orden, mismo conteo 19 numéricas + 14 categóricas) con `feature_columns` de `metadata.json`/`model_card.json` del artefacto entregado. El propio modelo original (`scripts/predict.py`) tampoco deriva features desde eventos crudos en inferencia — las exige ya calculadas (memoria 8.1: "despliegue de tipo batch"), así que el contrato del plugin es fiel al original, no una aproximación. Además `contract.py` emite warning explícito (no 0 silencioso) si falta una columna. Verificado numéricamente: ver sección "Correctitud" — diferencia máxima entre pipeline original y plugin: **1.6e-8** en `pred_score` sobre 18 casos reales, con `pred_traceability_incident` idéntico en el 100% de los casos. |
| 3. `train_request_type`/`train_response_type` ausentes en `app/registry.py` | **No** (ya estaba bien antes de esta sesión). | `app/registry.py:346-356` — `ModelEntry` trae ambos tipos y `extra_predict_exceptions=(DataContractError,)`. Verificado consistente. |

**Conclusión de la auditoría de bugs: no se encontró ningún bug de las 3 clases de ml21 en ml30.**
No fue necesario modificar ningún fichero de `app/plugins/ml30_meat_traceability_detection/`.

### Otros hallazgos (no bloqueantes, documentados)

- `train()` no re-ejecuta la búsqueda neuroevolutiva completa original (32 configuraciones MLP
  evaluadas: población=8 × generaciones=4) — reutiliza el genoma ganador (`best_genome` de
  `metadata.json`) y reentrena solo esa arquitectura fija con Adam/BCEWithLogitsLoss y los mismos
  hiperparámetros (`lr=0.0556514207154333`, `l2=0.0001236243687952708`, `epochs=147`,
  `batch_size=128`, `seed=42`) — verificado byte a byte contra `best_genome` del artefacto
  entregado. Es una simplificación deliberada y documentada en el propio código
  (`plugin.py` docstring + `constants.py`), razonable porque re-correr una NAS completa en cada
  request HTTP síncrona de `/train` sería impracticable. El algoritmo de ajuste de pesos sí es
  idéntico al original. Documentado en `manifest.yaml:training.hyperparams`.
- `predict_inline` existe (`ModelPluginPort` lo exige) pero el router solo acepta
  `"mode": "batch"` — decisión de producto documentada en `predict_dto.py`, coherente con la
  memoria ("El despliegue actual es de tipo batch"). Se verificó igualmente que `predict_inline`
  reproduce la misma predicción que el pipeline batch (ver más abajo).
- `pip-audit` reporta `setuptools==80.9.0` (PYSEC-2026-3447) — vulnerabilidad preexistente a nivel
  de repo (`requirements.txt` raíz), no introducida por ni específica de este plugin.
- `tests/unit/test_modelo10_lacteo_unit.py::test_no_center_crop` falla en la suite completa —
  preexistente, no relacionado con ml30 ni tocado en esta sesión.
- El entorno de verificación local no tiene un servidor MLflow alcanzable
  (`MLFLOW_TRACKING_URI` por defecto apunta a `http://mlflow.mlflow:5000`, un DNS interno de
  Kubernetes). Al llamar `/train` con un `mlflow_run_id` no vacío, `BaseMLflowTracker` intenta
  conectar y la petición queda colgada (sin timeout corto) — bloqueando el único worker de
  uvicorn. Esto es un comportamiento de `app/domain/services/mlflow_tracker.py` (infraestructura
  compartida, no se toca por plugin) reproducible en cualquier plugin con `train()` + MLflow, no
  específico de ml30. En clúster real, ese DNS resuelve al servicio MLflow interno y conecta sin
  problema. Se deja constancia para que el equipo de la plataforma lo evalúe (p. ej. timeout
  corto en `_build_client`), pero no bloquea este plugin.

## Checklist técnico
- [x] flake8: 0 errores (`app/plugins/ml30_meat_traceability_detection/`)
- [x] pytest: 498/499 passed en la suite completa (`tests/unit/`) — el único fallo
      (`test_modelo10_lacteo_unit.py::test_no_center_crop`) es preexistente y ajeno a ml30. Tests
      propios de ml30: **13/13 passed** (`test_ml30_meat_traceability_detection.py` +
      `test_ml30_contract.py`).
- [x] pylint: 9.64/10, sin issues nuevos de severidad relevante (solo `line-too-long`/
      `too-many-locals`/`broad-exception-caught` — mismo perfil que otros plugins ya
      mergeados, p. ej. `ml35_dairy_ann_cleaning_cost` puntúa 8.93/10 con el mismo tipo de avisos)
- [x] pip-audit: sin CVEs nuevas atribuibles a este plugin (ver `setuptools` arriba, preexistente)
- [x] Arranque local (`MODEL=ml30-meat-traceability-detection ./.venv/bin/python main.py`) +
      `/health` + `/predict` (batch) + `/stats`: **OK**, artefactos cargados desde
      `artifacts/ml30_meat_traceability_detection/` (verificados byte-idénticos — mismo SHA-256 —
      a los artefactos delivery de `inbox/a30/codigo/models/artifacts/neuroevolution_mlp/`)
- [x] `/train` sin `mlflow_run_id` → **422** (`Field required`), como exige la regla repo-wide de
      este sprint
- [x] `/train` con `mlflow_run_id=""` → **200**, `TrainResponse` con métricas coherentes
      (`accuracy=0.8436, f1=0.5942, roc_auc=0.7675, n_train=1428, n_test=358,
      training_time_s=3.5`) sobre un split 80/20 de `data/splits/train.csv`;
      `upload_warning="Sin run de MLflow: el modelo reentrenado no se ha guardado."` — correcto,
      sin `mlflow_run_id` no debe subir nada. **Artefacto base verificado intacto tras el
      entrenamiento** (mismo SHA-256 de `model_state.pt`/`preprocessor.pkl` antes y después) —
      cumple la regla de no sobreescribir nunca el artefacto fijo de S3/local.
- [~] `/train` con `mlflow_run_id` no vacío (subida real a MLflow): no verificable end-to-end en
      este entorno — no hay servidor MLflow alcanzable (ver "Otros hallazgos" arriba). No es un
      fallo del plugin.

## Correctitud (golden dataset)

18 casos extraídos de `inbox/a30/codigo/data/splits/test.csv` (fuente: split de test real
entregado, filas 7/10 estratificadas por `target_traceability_incident`, incluyendo casos con
`process_route`/`packaging_type` nulos por visibilidad progresiva de etapa). `expected` =
predicción REAL del pipeline original (`src/predict/predictor.py` del código entregado, ejecutado
sobre los artefactos delivery) — no la etiqueta ground-truth del dataset sintético, siguiendo la
convención ya usada en otros manifests de este repo para modelos de clasificación (p. ej. a45):
se verifica reproducibilidad del puerto, no acierto del modelo. CSV completo con las 18 filas:
`inbox/a30/golden_cases_test_selected.csv`.

Sanity check previo: ejecutando el pipeline original completo sobre las 379 filas de
`test.csv` se reproduce exactamente `accuracy=0.870712, f1=0.601626, roc_auc=0.718529` —
idéntico a `metadata.json`/memoria (Tabla 4), confirmando que el artefacto y el pipeline de
referencia usados para generar `expected` son los correctos.

| Caso (row_id) | Esperado (pred / score) | Obtenido — servidor real (pred / score) | Diferencia (score) | ¿OK? |
|---|---|---|---|---|
| OBS0000269 | 0 / 0.199724 | 0 / 0.199724 | ~4e-9 | ✅ |
| OBS0000325 | 0 / 0.041464 | 0 / 0.041464 | ~5e-10 | ✅ |
| OBS0000475 | 0 / 0.143161 | 0 / 0.143161 | ~1e-9 | ✅ |
| OBS0000675 | 0 / 0.114083 | 0 / 0.114083 | ~1e-9 | ✅ |
| OBS0000721 | 0 / 0.033278 | 0 / 0.033278 | ~4e-10 | ✅ |
| OBS0000795 | 0 / 0.113355 | 0 / 0.113355 | ~3e-10 | ✅ |
| OBS0001067 | 0 / 0.081986 | 0 / 0.081986 | ~3e-9 | ✅ |
| OBS0001335 | 1 / 0.992749 | 1 / 0.992749 | ~2e-8 | ✅ |
| OBS0001351 | 0 / 0.000002 | 0 / 0.0000019 | ~4e-14 | ✅ |
| OBS0001565 | 1 / 0.992749 | 1 / 0.992749 | ~2e-8 | ✅ |
| OBS0001779 | 0 / 0.032766 | 0 / 0.032766 | ~8e-11 | ✅ |
| OBS0001880 | 0 / 0.006805 | 0 / 0.006805 | ~2e-11 | ✅ |
| OBS0001947 | 0 / 0.003004 | 0 / 0.003004 | ~6e-12 | ✅ |
| OBS0001997 | 0 / 0.043638 | 0 / 0.043638 | ~2e-10 | ✅ |
| OBS0002013 | 0 / 0.131462 | 0 / 0.131462 | ~1e-9 | ✅ |
| OBS0002150 | 0 / 0.004608 | 0 / 0.004608 | ~1e-10 | ✅ |
| OBS0002190 | 1 / 0.966893 | 1 / 0.966893 | ~5e-10 | ✅ |
| OBS0002226 | 1 / 0.992749 | 1 / 0.992749 | ~2e-8 | ✅ |

Tolerancia usada: `prediction_exact=true` (clase predicha debe coincidir exactamente) +
`probability_atol=0.005` en `pred_score` (análoga al criterio ya usado en `inbox/a45/manifest.yaml`
para clasificación binaria; 2× no aplica directamente porque `metrics_reported` son métricas de
clasificación, no un error relativo continuo). Diferencia real observada: máximo 2e-8 (ruido de
punto flotante entre ejecuciones CPU, pytorch 2.11.0 original vs 2.13.0 de este entorno) — muy por
debajo de la tolerancia.

**Resultado: 18/18 casos dentro de tolerancia (100%)**, tanto vía `predict_batch` directo contra
artefactos locales (offline, antes de levantar servidor) como vía el **servidor HTTP real**
(`POST /models/ml30-meat-traceability-detection/predict`, `mode: batch`). También se verificó
`predict_inline` (método interno, aunque desconectado del router) sobre 1 caso adicional:
coincide exactamente con el pipeline original (`pred=1, score=0.9927489`).

De los 18 casos, 5 tienen `target_traceability_incident` real = 1 pero el modelo predice 0
(falsos negativos: OBS0000475, OBS0001351, OBS0001880, OBS0002150 — consistente con el recall de
test reportado, 0.5068, el modelo detecta ~la mitad de incidencias reales). Esto **no es un fallo
de la integración** — es el comportamiento documentado del modelo original, reproducido fielmente.

## Estado final

**LISTO PARA PR** (pendiente de revisión humana, como exige el skill). No se ha modificado ningún
fichero de `app/plugins/ml30_meat_traceability_detection/` ni `app/registry.py` — la auditoría no
encontró necesidad de cambios. Entregables nuevos de esta sesión:
- `inbox/a30/manifest.yaml`
- `inbox/a30/golden_cases_test_selected.csv` (18 filas de `data/splits/test.csv` + predicciones
  esperadas, soporte de los golden_cases del manifest)
- `outputs/a30/verification_report.md` (este fichero)
