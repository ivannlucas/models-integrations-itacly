# Verificación — ml8_cereals_img_anomaly_detector

> **[RENOMBRADO 2026-10-06]** Nomenclatura alineada con el estándar `mlNN_<sector>_<desc>` del resto de plugins: `(carpeta de artefactos) modelo8_cereales` → `ml8_cereals_img_anomaly_detector (model_id sin cambios: ml8-cereals-img-anomaly-detector)`. La carpeta de artefactos (local y en S3) pasa de `artifacts/modelo8_cereales/` a `artifacts/ml8_cereals_img_anomaly_detector/`. Las referencias a los nombres antiguos en el texto de abajo son históricas y corresponden a la fecha de cada ciclo.

Plugin ya existía en `app/plugins/ml8_cereals_img_anomaly_detector/` y estaba registrado en
`app/registry.py` (`model_id="ml8-cereals-img-anomaly-detector"`), pero nunca había pasado por
`manifest-extraction` ni `verification`. Esta auditoría genera el manifest a partir de
`inbox/a08/codigo/` + `inbox/a08/entregable/` y verifica el plugin contra él.

**Actualización (2026-10-05):** el equipo de IA añadió los datasets raw completos
(`data/raw/{Wheat_Coccinellid,archive,archive (1)..(5)}`, ~177k imágenes) y el `data/processed/`
ya generado por el pipeline original, y el sandbox pasó de una GPU incompatible (GTX 1060) a
una GPU real y compatible (RTX 2070 SUPER). Esta sección documenta el re-ciclo completo:
golden_cases reales (antes vacío), y la corrección de un bug de `_safe_device()` que hacía caer
a CPU incluso con CUDA funcional. El resto del informe original (checklist Parte A, auditoría
de bugs clase 1-3) se conserva sin cambios porque sigue siendo válido.

## Checklist técnico
- [x] flake8 `app/plugins/ml8_cereals_img_anomaly_detector/`: 0 errores (re-ejecutado tras el
      fix de `_safe_device()`, sigue en 0)
- [x] pytest `tests/unit/`: 503/503 passed (suite completa del repo, sin romper nada; la suite
      creció a 503 desde los 500 de la primera pasada por trabajo concurrente de otros agentes
      sobre otros plugins, no por este ciclo)
- [x] pylint `app/plugins/ml8_cereals_img_anomaly_detector/ --disable=import-error`: 8.51/10
      (sin cambio respecto a la primera pasada — el fix de `_safe_device()` no introduce
      issues nuevos; mismas categorías de aviso que el resto del repo: docstrings, líneas
      largas, imports diferidos a propósito para no cargar torch en el arranque)
- [x] pip-audit -r requirements.txt: 1 vulnerabilidad preexistente (`setuptools 80.9.0`,
      PYSEC-2026-3447) — pertenece a `requirements.txt` compartido del repo, no introducida
      por este plugin, fuera de alcance de esta tarea
- [x] Arranque local + health + predict (inline + batch) + stats: OK, sin 500. Re-verificado
      en esta pasada contra CUDA real (ver sección CUDA abajo) además de CPU.
- [x] /train sin `mlflow_run_id` → 422 (coherente con `training.supported=true` + `mlflow_run_id`
      obligatorio sin default en `train_dto.py`) — re-confirmado en esta pasada.
- [x] /train con `mlflow_run_id` real: no probado end-to-end contra el server HTTP por el
      caveat conocido de `BaseMLflowTracker` (sin timeout de conexión, cuelga contra el
      `MLFLOW_TRACKING_URI` interno de k8s inalcanzable en este sandbox) — sigue vigente, no ha
      cambiado con la GPU nueva. En esta pasada se re-verificó `plugin.train()` llamándolo
      directamente en Python con `mlflow_run_id=""` (salta las llamadas a MLflow), pero ahora
      con **datos reales** (108 imágenes reales de `data/processed/<cereal>/{train,validation}/<categoria>/`,
      no el dataset sintético de PIL de la primera pasada) y **sobre CUDA real**: terminó sin
      excepciones (`train_samples=64, val_samples=36, fase1_epochs=10, fase2_epochs=5,
      best_val_acc_cat=77.8, best_val_acc_cer=83.3` — valores bajos por ser solo 64 imágenes de
      smoke test, no representativos de accuracy real), sin ningún mismatch de device/dtype
      entre `balance_hongos`, `class_weights`/`cereal_weights.to(device)`,
      `WeightedRandomSampler` y el `accumulation_steps=2` de `_train_epoch` — las 4 piezas
      corrigidas en la primera pasada funcionan igual en GPU que en CPU.

## Auditoría del plugin ya integrado contra el código entregado

**Bug clase 1 (dict crudo en vez de Pydantic) — NO presente.**
`grep -n "return {"` en `plugin.py` solo aparece en helpers internos (`model_loader.py`,
`postprocessing.py`); `predict_inline`/`predict_batch` devuelven
`PredictInlineResponse(**...)` / `PredictBatchResponse(**...)` ya tipados. Sin riesgo de 500
vía `type(result).model_validate()`.

**Bug clase 2 (preprocesado divergente) — NO presente.**
`preprocessing.py` (`Resize((224,224)) + ToTensor + Normalize(ImageNet mean/std)`) coincide
exactamente con `src/predict/inference.py:predict_image` del código entregado. El mapeo
índice→etiqueta de `constants.py` (`CATEGORY_NAMES`/`CEREAL_NAMES`, orden alfabético) coincide
con el `idx_to_class`/`idx_to_cereal` reales embebidos en el checkpoint de producción
(verificado cargando `artifacts/modelo8_cereales/mobilenet_v3_large_cereales_multitask.pth`).
Los 3 artefactos en `artifacts/modelo8_cereales/` son bit-a-bit idénticos (md5) a los
entregados en `inbox/a08/codigo/.../models/`.

**Bug clase 3 (train() no replica el procedimiento real) — PRESENTE, corregido.**
El `train()` ya existente tenía una implementación de fine-tuning funcional (dos fases,
early stopping, arquitectura correcta) pero con divergencias reales respecto a
`src/main.py:run_training` + `src/training/*` + notebook `04_MobileNetV3.ipynb` + memoria
sección 6.2 ("todos los modelos compartieron el mismo esquema de balanceo en entrenamiento"):

| Aspecto | Antes (plugin) | Código real entregado | Fix aplicado |
|---|---|---|---|
| Backbone weights | `IMAGENET1K_V2` | `IMAGENET1K_V1` (model.py, notebook celda 13) | `_load_mobilenetv3_backbone()` usa V1, con fallback a `weights=None` si no hay red |
| `balance_hongos` | Ausente | Recorta 'hongos' al promedio de las otras categorías (`max_samples_hongos=null`) | `_balance_hongos()` añadido, aplicado solo sobre train |
| `class_weights`/`cereal_weights` | `CrossEntropyLoss()` sin pesos | Inverse-frequency normalizado, memoria 6.2 | Añadido, con guarda `max(count,1)` para no crashear si el ZIP de un usuario no trae las 4 clases |
| `WeightedRandomSampler` | `shuffle=True` plano | Sampler balanceado por categoría | Añadido sobre `train_loader` |
| Gradient accumulation | Ausente (step por batch) | `accumulation_steps=2` (config.yaml, train.py) | `_train_epoch` reescrito para acumular/hacer step cada 2 batches |
| Seed | No fijada en train() | `torch.manual_seed(42)` + `np.random.seed(42)` | `torch.manual_seed(42)` + `random.seed(42)` añadidos |

Archivo modificado: `app/plugins/ml8_cereals_img_anomaly_detector/plugin.py` (helpers
`_balance_hongos`, `_load_mobilenetv3_backbone`, `_train_epoch` con `accumulation_steps`,
y el cuerpo de `train()`). `mlflow_run_id` ya era obligatorio (sin default) en
`train_dto.py:TrainRequest` desde antes de esta sesión — verificado consistente, no tocado.

**Bug clase 4 (falta `device=` en hardware real) — PRESENTE, corregido en esta pasada.**
En la primera pasada, con la GTX 1060 incompatible del sandbox, `model_loader._safe_device()`
caía a CPU de forma "segura" — pero al investigar por qué sigue cayendo a CPU con la **RTX 2070
SUPER real y compatible** de este sandbox (`torch.cuda.is_available()==True`, capability (7,5),
perfectamente soportada por `torch==2.13.0+cu130`), se encontró que el propio self-test de
`_safe_device()` estaba roto:

```python
# Antes (bug):
torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())
# El módulo Conv2d se queda en CPU (nunca se le llama .cuda()); el tensor de entrada SÍ
# se mueve a CUDA → RuntimeError: "Input type (torch.cuda.FloatTensor) and weight type
# (torch.FloatTensor) should be the same" → capturado por el except genérico → SIEMPRE
# devuelve CPU, incluso con una GPU real y funcional. No es una detección de incompatibilidad
# de hardware: es un bug en el propio test, que daba un falso negativo permanente.
```

Reproducido de forma aislada (`python -c "torch.nn.Conv2d(1,1,1)(torch.zeros(1,1,4,4).cuda())"`
→ el mismo `RuntimeError`) antes de tocar el código, para confirmar que no era un problema de
instalación de CUDA/drivers. Mismo patrón de bug encontrado en paralelo por otro agente en
`app/plugins/ml2_fungal_cnn_disease_detection/model_loader.py` (ya corregido ahí) — se aplicó el
mismo fix aquí:

```python
# Después (fix):
probe = torch.nn.Conv2d(1, 1, 1).cuda()
probe(torch.zeros(1, 1, 4, 4).cuda())
```

Archivo modificado: `app/plugins/ml8_cereals_img_anomaly_detector/model_loader.py:_safe_device()`.

Verificado end-to-end tras el fix:
- `_safe_device()` → `cuda` (antes: `cpu`).
- `load_model_bundle()['device']` → `cuda`, modelo con parámetros en `cuda:0`.
- Servidor real (`MODEL=ml8-cereals-img-anomaly-detector python main.py`) loguea al arrancar:
  `"... bundle ready — arch=mobilenet_v3_large device=cuda test_acc_cat=92.14969758064517
  test_acc_cer=98.87852822580645"`.
- Los 19 `golden_cases` (ver abajo), `predict_batch` (ZIP de 3 imágenes) y `plugin.train()`
  (con datos reales, ver checklist arriba) corren sin errores de device/dtype sobre CUDA real.

**Nota de alcance:** `app/plugins/{modelo10_lacteo, ml41_meat_curing_machinery_acoustic_anomaly,
ml5_meat_cow_behaviour, ml7_cereals_grain_pest_detection,
ml4_lactic_cnn_thermal_early_disease_detection}/model_loader.py` tienen el mismo patrón de bug
(`torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())` sin `.cuda()` en el módulo) sin
corregir — fuera de alcance de esta tarea (solo `ml8`), señalado en `known_issues` del manifest
para que se corrija en sus propios ciclos de verificación.

**Otros hallazgos (menores, documentados, no bloqueantes):**
- `constants.py:IMAGE_EXTENSIONS` incluye `.webp`, que no está en
  `config.yaml:data.valid_image_extensions` — superset inofensivo, se deja así.
- El `test_accuracy` embebido en el checkpoint de producción (cat=92.15%, cer=98.88%,
  both=91.33%) difiere ~0.4-0.5pp de la Tabla 12 de la memoria (92.58%/98.98%/91.82%) —
  probablemente un run distinto; ambos superan los KPI (90/95/85), no bloqueante.
- `PredictBatchResponse.predictions` es `list[dict]` (no un sub-modelo Pydantic por item) —
  razonable porque cada item puede ser éxito o error con forma distinta; el contrato externo
  sigue siendo un modelo tipado, no un dict crudo en el nivel que valida
  `predict_model_use_case.py`.

## Correctitud (golden dataset)

**Actualización (2026-10-05):** con los datasets raw + `data/processed/` ya entregados, se
construyeron 19 `golden_cases` reales en `inbox/a08/manifest.yaml`, reemplazando el
`golden_cases: []` de la primera pasada (motivo documentado en el primer bullet, histórico, de
`known_issues`). Fuente: `data/processed/<cereal>/test/<categoria>/` — el split de test real ya
generado por `src/data_processing/preprocess.py:procesar_dataset` (train_test_split
estratificado, seed=42, 70/15/15), el mismo que consume
`src/training/dataset.py:cargar_todos_los_datos(..., 'test')` en la evaluación original. No se
ha inventado ni reconstruido ningún split propio.

19 imágenes seleccionadas con `random.Random(42).sample()` sobre las 12 combinaciones
cereal×categoría SÍ presentes en el test set (de las 16 posibles — sorgo solo tiene 'hongos' en
test, maíz no tiene 'otros'; así es el dataset real entregado, no un error de carga),
estratificadas con refuerzo en 'hongos' (clase mayoritaria, 2 por cereal) e 'insectos' (peor f1
reportado en memoria, 0.76; 2 por cereal donde hay datos).

`expected` se obtuvo **ejecutando el checkpoint real servido**
(`artifacts/modelo8_cereales/mobilenet_v3_large_cereales_multitask.pth`, cargado vía
`load_model_bundle()`) directamente en proceso sobre **CUDA real**, con el mismo
preprocesado/postprocesado que usa el plugin (`preprocessing.image_path_to_tensor` +
`postprocessing.build_inline_response`) — nunca a mano. Esto convierte cada caso en una
detección de regresión de *serving* (preprocesado/pesos/postprocesado/device), independiente de
si el modelo acierta o no contra la etiqueta real — eso se mide aparte (ver tabla).

### Resultado — Parte B (server real, HTTP, `/predict` modo inline, imagen → base64)

| # | Caso | Esperado (cat/cereal) | Obtenido (cat/cereal) | Δ confianza | ¿OK vs. expected? | Ground truth (cat/cereal) |
|---|---|---|---|---|---|---|
| 1 | caso_001_arroz_hongos | hongos / arroz | hongos / arroz | 0.0 / 0.0 | OK | hongos / arroz ✓ |
| 2 | caso_002_arroz_hongos | sano / arroz | sano / arroz | 0.0 / 0.0 | OK | hongos / arroz (cat✗, modelo real) |
| 3 | caso_003_maiz_hongos | hongos / maiz | hongos / maiz | 0.0 / 0.0 | OK | hongos / maiz ✓ |
| 4 | caso_004_maiz_hongos | hongos / maiz | hongos / maiz | 0.0 / 0.0 | OK | hongos / maiz ✓ |
| 5 | caso_005_sorgo_hongos | hongos / sorgo | hongos / sorgo | 0.0 / 0.0 | OK | hongos / sorgo ✓ |
| 6 | caso_006_sorgo_hongos | hongos / sorgo | hongos / sorgo | 0.0 / 0.0 | OK | hongos / sorgo ✓ |
| 7 | caso_007_trigo_hongos | insectos / trigo | insectos / trigo | 0.0 / 0.0 | OK | hongos / trigo (cat✗, modelo real) |
| 8 | caso_008_trigo_hongos | insectos / trigo | insectos / trigo | 0.0 / 0.0 | OK | hongos / trigo (cat✗, modelo real) |
| 9 | caso_009_arroz_sano | sano / arroz | sano / arroz | 0.0 / 0.0 | OK | sano / arroz ✓ |
| 10 | caso_010_maiz_sano | sano / maiz | sano / maiz | 0.0 / 0.0 | OK | sano / maiz ✓ |
| 11 | caso_011_trigo_sano | sano / trigo | sano / trigo | 0.0 / 0.0 | OK | sano / trigo ✓ |
| 12 | caso_012_arroz_insectos | insectos / arroz | insectos / arroz | 0.0 / 0.0 | OK | insectos / arroz ✓ |
| 13 | caso_013_arroz_insectos | insectos / arroz | insectos / arroz | 0.0 / 0.0 | OK | insectos / arroz ✓ |
| 14 | caso_014_maiz_insectos | insectos / maiz | insectos / maiz | 0.0 / 0.0 | OK | insectos / maiz ✓ |
| 15 | caso_015_maiz_insectos | insectos / maiz | insectos / maiz | 0.0 / 0.0 | OK | insectos / maiz ✓ |
| 16 | caso_016_trigo_insectos | insectos / trigo | insectos / trigo | 0.0 / 0.0 | OK | insectos / trigo ✓ |
| 17 | caso_017_trigo_insectos | insectos / trigo | insectos / trigo | 0.0 / 0.0 | OK | insectos / trigo ✓ |
| 18 | caso_018_arroz_otros | otros / arroz | otros / arroz | 0.0 / 0.0 | OK | otros / arroz ✓ |
| 19 | caso_019_trigo_otros | otros / trigo | otros / trigo | 0.0 / 0.0 | OK | otros / trigo ✓ |

Tolerancia usada: coincidencia exacta de etiqueta (`categoria`/`cereal`) + diferencia de
confianza < 1e-3 (softmax determinista en inferencia, sin augmentation). **Resultado: 19/19
casos dentro de tolerancia** (match exacto, Δ confianza = 0.0 en los 19 — el server real sobre
CUDA reproduce bit-a-bit la salida del checkpoint cargado en proceso).

**Accuracy real vs. ground truth (secundario, no es el criterio de esta tabla):** 16/19 (84%)
aciertan la categoría ground truth, 19/19 (100%) aciertan el cereal — coherente con el
test_accuracy agregado del checkpoint (cat=92.15%, cer=98.88%) dado el tamaño pequeño de la
muestra (19 casos). Los 3 fallos de categoría (caso_002, caso_007, caso_008) son errores reales
del modelo sobre 'hongos' (confunde con 'sano' o 'insectos' en casos visualmente ambiguos, con
confianza baja en caso_008: 0.59) — **no son bugs de wiring del plugin**, están documentados
aquí y en el manifest, y no se han silenciado ni se ha ajustado ninguna tolerancia para
ocultarlos.

**Metrics_reported (agregado, auditado — se mantiene como evidencia complementaria):** el
checkpoint de producción servido reporta test_acc_cat=92.15%, test_acc_cer=98.88%,
test_acc_both=91.33% — coherente (±0.5pp) con la Tabla 12 de la memoria y por encima de los 3
umbrales KPI (≥90/≥95/≥85).

**Smoke tests funcionales adicionales contra el server real:** `predict_batch` con ZIP de 3
imágenes reales de test (`arroz/sano`) → 200, 3/3 predicciones `sano` (correcto), sin errores
por imagen; `/stats` refleja el contador actualizado; `/train` sin `mlflow_run_id` → 422.

## Estado final
LISTO PARA PR. Ambos pendientes de la pasada anterior quedan resueltos en esta: (1)
`golden_cases` ya no está vacío — 19 casos reales, trazables a `data/processed/.../test/`, con
19/19 de acierto contra el comportamiento real del checkpoint servido (criterio de wiring) y
16/19 contra ground truth (criterio de accuracy, documentado aparte, consistente con el 92%
agregado); (2) CUDA real confirmado funcionando end-to-end para predict_inline, predict_batch y
train() tras corregir el bug de `_safe_device()`. Pendiente no bloqueante y ya señalado: el
mismo bug de `_safe_device()` existe sin corregir en otros 5 plugins de visión, fuera de alcance
de esta tarea.
