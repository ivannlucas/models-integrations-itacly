# Verificación — modelo10-lacteo (a10, detección de vectores en ganado vacuno)

> **[RENOMBRADO 2026-10-06]** Nomenclatura alineada con el estándar `mlNN_<sector>_<desc>` del resto de plugins: `modelo10_lacteo / modelo10-lacteo` → `ml10_dairy_disease_vector_detection / ml10-dairy-disease-vector-detection`. La carpeta de artefactos (local y en S3) pasa de `artifacts/modelo10_lacteo/` a `artifacts/ml10_dairy_disease_vector_detection/`. Las referencias a los nombres antiguos en el texto de abajo son históricas y corresponden a la fecha de cada ciclo.

## Contexto de este ciclo (re-auditoría completa desde cero)

Este informe **sustituye por completo** la versión anterior. Motivo: `inbox/a10/codigo/`
pasó de 219 a **49.695 ficheros** desde la última auditoría — el usuario añadió los 3
datasets públicos reales completos que antes solo existían como instrucciones de descarga
en el README:

1. `data/detection/raw/insects/` (Roboflow, YOLO) — detección de vectores.
2. `data/classification/raw/fly_mos/` (Kaggle, bbox) — mosca/mosquito.
3. `data/classification/raw/ticks/` (Roboflow, segmentación) — garrapatas.

Las dos auditorías previas (manifest + verificación inicial, y un fix posterior de
`_safe_device()`) solo tuvieron acceso a **1 imagen real de ejemplo** (`data/test.jpg` +
`data/predictions/test_results.json`) para validar correctitud. Este ciclo repite el
proceso de `manifest-extraction` → re-auditoría del plugin → `verification` desde cero,
con los 3 datasets reales disponibles, para ver qué seguía siendo válido y qué solo era
visible ahora.

## Resumen de resultados frente al ciclo anterior

| | Ciclo anterior (1-2) | Este ciclo (3) |
|---|---|---|
| `inbox/a10/codigo/` | 219 ficheros, solo artefactos + 1 imagen | 49.695 ficheros, 3 datasets reales completos |
| golden_cases | 1 (`caso_001_test_jpg`) | **25**, estratificados fly/mos/tick + crops + imágenes crudas + 1 heredado |
| Correctitud golden dataset | 1/1 | **25/25** (100%) |
| Evaluación agregada del pipeline completo | No medible (1 imagen) | **394/415 = 94.94%** sobre split test real reproducido (nuevo hallazgo, ver abajo) |
| Bugs nuevos encontrados | — | **0** (los 3 bugs del ciclo 1 + el de `_safe_device()` del ciclo 2 se reconfirmaron corregidos contra datos/carga reales, no solo releyendo el código) |
| `/train` con datos reales | No disponible (solo ZIP sintético de colores) | Datos reales (75 imágenes reales, 25/clase), 16 épocas, best_val_acc=88.9% |
| GPU | No confirmado contra carga real | Confirmado: RTX 2070 SUPER, `cuda`, 415 predicciones + entrenamiento real en ~15s totales |

**No se encontró ningún bug nuevo.** Los 5 patrones de bug transversales de la tarea se
re-verificaron explícitamente contra datos reales (detalle en "Re-auditoría del plugin"
más abajo) y todos siguen correctos. El hallazgo principal de este ciclo no es un bug de
wiring, sino una **métrica real antes invisible**: el pipeline completo mide
accuracy=94.94%, por debajo del 0.981 oficial de la memoria (que nunca evaluó el pipeline
completo, solo el clasificador aislado — ver known_issues del manifest).

## Re-auditoría del plugin contra los 5 patrones de bug conocidos

1. **Tipado de respuesta (`predict_inline`/`predict_batch`)**: ambos construyen
   `PredictInlineResponse(**result)` / `PredictBatchResponse(...)` — nunca devuelven un
   dict crudo. Confirmado leyendo `plugin.py` línea por línea y verificado en vivo contra
   el servidor real (ver Parte A). Sin cambios necesarios.
2. **Preprocesado (resize/crop/normalize, device, class mapping)**: re-diffeado línea por
   línea `app/plugins/modelo10_lacteo/preprocessing.py` contra
   `src/predict/predictor.py::predict_full_pipeline` (y `_build_classifier_model` /
   `_get_classification_transforms` en `trainer.py`/`predictor.py`) del código real
   entregado — siguen coincidiendo exactamente
   (`Resize(256)+CenterCrop(224)+Normalize(ImageNet)` para producción,
   `Resize((224,224))` plano solo para el propio `train()` del plugin, replicando
   `_get_classification_transforms`). Validado ahora empíricamente contra 415 imágenes
   reales con ground truth (no solo 1 como antes): 394/415 correctas, con el detalle
   completo de las 25 imágenes en la tabla más abajo.
3. **`train()` real**: hiperparámetros (`Adam`, `lr=0.00798`, `weight_decay=0.000344`,
   `batch_size=64`, `StepLR(step_size=10, gamma=0.1)`, `patience=15`, split 70/15/15)
   siguen coincidiendo con `config/config.yaml`/`trainer.py`. Verificado en este ciclo por
   primera vez con **datos reales** (antes solo con un ZIP sintético de colores sólidos,
   ver detalle en "Parte A"): entrenó 16 épocas sobre 75 imágenes reales, `best_val_acc=88.9%`.
4. **Device placement**: `safe_device()` (fix del ciclo 2) sigue devolviendo `cuda` y,
   crucialmente, se confirmó esta vez contra **carga de trabajo real** (415 predicciones +
   1 entrenamiento completo), no solo contra una llamada aislada de autotest — sin
   regresiones. El detector YOLO sigue recibiendo `device=self._device` explícito en
   `_run_pipeline` (fix del ciclo 1).
5. **`mlflow_run_id`**: `TrainRequest.mlflow_run_id: str` sigue sin default (obligatorio);
   confirmado en vivo que `/train` sin ese campo devuelve 422. `predict_inline`/
   `predict_batch`/`stats` siguen usando el resultado de
   `download_user_classifier_from_mlflow()` cuando se informa (`user_clf`/`user_cls_names`
   pasados a `_run_pipeline`, nunca descartados).

Artefactos vendorizados verificados **bit a bit idénticos** (md5sum) a los del entregable:
`best_classifier.pth`, `detector_best.pt`, `class_names.json` — no hay desviación.

## Parte A — Checklist técnico

- [x] `flake8` (`app/plugins/modelo10_lacteo/` + test): **0 errores**
- [x] `pytest tests/unit/ -q`: **503/503 passed** (suite completa, coincide con el número
      esperado; no se rompió nada fuera del plugin)
- [x] `pylint app/plugins/modelo10_lacteo/ --disable=import-error`: **9.07/10**, sin issues
      nuevos (los `line-too-long`/`too-many-locals`/`broad-exception-caught` son
      preexistentes; `flake8`, el gate real de CI, está limpio)
- [x] `pip-audit -r requirements.txt`: 2 CVEs en `setuptools==80.9.0` (`PYSEC-2026-3447`),
      preexistentes, no relacionadas con las dependencias de este plugin
      (torch/torchvision/ultralytics/opencv/pillow)
- [x] Arranque local (`MODEL=modelo10-lacteo ./.venv/bin/python main.py`, puerto 8000,
      GPU real `cuda`) + `/health` → `{"status":"ok","loaded":true}`
- [x] `/predict` inline (base64, `data/test.jpg`) → `prediction=tick, confidence=0.9996,
      bbox={755,280,840,374}` — coincide exactamente con el golden case heredado
- [x] `/predict` batch (ZIP de 3 imágenes reales del split test de `fly`) → 3/3 clasificadas
      correctamente como `fly` (confidence 0.59–1.00)
- [x] `/stats` → responde con `runtime_stats.total_predictions` incrementado correctamente
- [x] `/train` sin `mlflow_run_id` → **422** (`Field required`), como exige la regla repo-wide
- [x] `/train` con `mlflow_run_id` real contra servidor real → **cuelga** (timeout de cliente
      a los 8s, `/health` sigue respondiendo en paralelo) — **reconfirmado, no es un bug del
      plugin**: `BaseMLflowTracker` (`app/domain/services/mlflow_tracker.py`, infraestructura
      compartida) no tiene connect timeout contra el `MLFLOW_TRACKING_URI` por defecto
      (host interno de k8s no resoluble en este sandbox). Mismo caveat ya documentado en el
      ciclo 1, fuera del alcance de este plugin.
- [x] `/train` con `mlflow_run_id` + lógica real verificada **en proceso**, monkeypatcheando
      `BaseMLflowTracker` a un stub no-op (workaround indicado en la tarea) con **datos
      reales** (75 imágenes reales, 25 por clase, del split train real reproducido — no
      sintéticas como en el ciclo 1): auto-split 70/15/15 interno (51/9/15), 16 épocas,
      `best_val_acc=88.9%`, `upload_artifacts` del stub recibió `class_names.json` +
      `best_classifier.pth` correctamente, en 3.0s totales. Confirma que la lógica de
      entrenamiento del plugin funciona de extremo a extremo; el único bloqueo real es de
      infraestructura compartida.

## Parte B — Correctitud contra golden dataset (25 casos)

Metodología (detalle completo en `inbox/a10/manifest.yaml` → `golden_cases`): split real
70/15/15 reproducido con el código original (`set_seed(42)` +
`create_detection_splits`/`create_classification_splits`) sobre `data/*/processed` ya
presente en el entregable. 18 casos (`crop_*`) son crops del split test real (ground truth =
carpeta); 6 casos (`fullraw_*`) son las imágenes RAW originales (sin recortar) de las que
salieron 6 de esos crops, ejercitando el pipeline completo (detector+crop+clasificador) sobre
una imagen entera, no solo sobre un crop ya hecho; 1 caso (`caso_001_test_jpg`) es el heredado
del ciclo 1, re-verificado. `expected` en todos los casos = salida real de `predict_inline()`
del plugin integrado contra el checkpoint vendorizado (no hand-typed).

| Caso | Ground truth | Obtenido (prediction) | cls_conf | det_conf | ¿OK? |
|---|---|---|---|---|---|
| crop_fly_01..06 | fly | fly (6/6) | 0.88–1.00 | 0.76–0.87 | ✅ 6/6 |
| crop_mos_01..06 | mos | mos (6/6) | 0.976–1.00 | 0.39–0.79 | ✅ 6/6 |
| crop_tick_01..06 | tick | tick (6/6) | 0.9999–1.00 | 0.68–0.94 | ✅ 6/6 |
| fullraw_fly_01..02 | fly | fly (2/2) | 0.9999–1.00 | 0.84–0.90 | ✅ 2/2 |
| fullraw_mos_03..04 | mos | mos (2/2) | 0.9995–1.00 | 0.81–0.82 | ✅ 2/2 |
| fullraw_tick_05..06 | tick | tick (2/2) | 1.00 | 0.74–0.88 | ✅ 2/2 |
| caso_001_test_jpg | tick | tick | 0.9996 | 0.6819 | ✅ |

Tolerancia usada: 0.01 absoluto sobre `det_conf`/`cls_conf` (sin cambios respecto al ciclo
anterior — no hay MAE/tolerancia reportada en la memoria para una confianza de caso
individual).

**Resultado: 25/25 casos dentro de tolerancia (100%), `prediction` == ground truth en los
25.** Ningún caso falló — no hay nada que silenciar.

### Hallazgo adicional (no un golden case, una métrica agregada nueva)

Además de los 25 casos puntuales, se corrió el pipeline completo (`predict_inline()` real,
sin mocks) sobre **las 415 imágenes completas** del split de test de clasificación
reproducido (no solo la muestra de 18), para cuantificar por primera vez la brecha entre el
accuracy=0.981 oficial (clasificador aislado) y el pipeline completo real:

| | n | Correctas | Accuracy | Sin detección (vectors_count=0) |
|---|---|---|---|---|
| **Total** | 415 | 394 | **94.94%** | 10 |
| fly | 106 | 101 | 95.28% | 2 |
| mos | 98 | 85 | **86.73%** | 6 |
| tick | 211 | 208 | 98.58% | 2 |

También se corrió `model.val()` de `ultralytics` sobre el split de detección test
reproducido (653 imágenes): mAP50=0.933, precisión=0.947, recall=0.860 — distinto tanto del
0.790/0.824/0.710 de la memoria como del 0.885/–/0.818 de los JSON del código entregado
(ninguno de los tres es directamente comparable: cada uno usa una partición distinta del
dataset, y no hay garantía de que la reproducida en este ciclo sea exactamente la misma que
usó el equipo de IA para entrenar los artefactos servidos — ver el caveat de
reproducibilidad del split en `manifest.yaml → metrics_reported`). Este número NO sustituye
al `metrics_reported` de la memoria como fuente de tolerancia; se reporta como contexto
adicional, con su caveat, no como benchmark autorizado.

Esto **no es un bug** — es la cuantificación, con datos reales, de un riesgo que el ciclo 1
ya había señalado por lectura de código pero no podía medir: el accuracy oficial de la
memoria nunca evaluó el pipeline completo con detección real, y en producción (vía
`predict_inline`/`predict_batch`) el accuracy efectivo es más bajo, especialmente para
mosquitos. Recomendación para revisión humana: considerar si merece la pena re-entrenar o
afinar el detector específicamente para mejorar el recall en `mos` antes de un uso en
producción con alto impacto, o documentar el 94.94% (no el 0.981) como la expectativa real
de precisión del endpoint.

## Estado final

**LISTO PARA PR**, con las mismas dos salvedades ya conocidas y reconfirmadas (no
bloqueantes para el wiring, sí relevantes para la revisión humana):

1. `/train` con `mlflow_run_id` contra un `MLflow` real no se pudo verificar end-to-end en
   este sandbox (sin red hacia `MLFLOW_TRACKING_URI`) — la lógica de entrenamiento en sí se
   verificó completa con datos reales vía stub; recomendado repetir contra un MLflow
   accesible antes de mergear.
2. El accuracy real del pipeline completo (94.94%, 86.73% en `mos`) es más bajo que el
   0.981 oficial de la memoria — no es un bug del plugin (reproduce fielmente el
   comportamiento del código original), pero es información nueva y relevante para quien
   apruebe el PR, sobre todo para `mos`.

Ningún caso del golden dataset falló tolerancia; no se silenció ni ajustó ninguna
tolerancia para forzar un verde.
