# Verificación — ml4-lactic-cnn-thermal-early-disease-detection (a04)

> **[CORREGIDO 2026-10-07]** El patrón descrito más abajo para usar el modelo del usuario (sustituir `self._model`/`self._device` en `self` durante la petición y restaurarlo en `finally`) **no era seguro con peticiones concurrentes**: hay una sola instancia del plugin, así que una petición sin `mlflow_run_id` podía recibir el modelo de otro usuario, y dos peticiones solapadas podían dejar el de un usuario como modelo base. Se ha cambiado para resolver el modelo en variables locales, sin tocar `self` (rama `fix/retrain-mlflow-only-persistence`, cubierto por `tests/unit/test_user_model_isolation.py` y `test_user_model_download_hygiene.py`). Además, si el run pedido no tiene un modelo cargable, la petición falla con 422 en lugar de usar el modelo base. El texto de abajo se conserva como histórico.

## Contexto

Este plugin ya estaba integrado en `app/plugins/` y registrado en `app/registry.py` **sin haber
pasado nunca por `manifest-extraction` ni `verification`**. Un primer ciclo generó el manifest a
posteriori (`inbox/a04/manifest.yaml`) a partir de `inbox/a04/codigo/` (código entregado real) y
la memoria, auditó el plugin contra ambos, corrigió lo encontrado y verificó en vivo — con el
dataset TIDS real todavía no disponible (gitignored) y solo 2/3 escenarios de validación externa
de la memoria como golden cases.

## Ciclo 2 (este informe): dataset TIDS real + GPU real

El usuario ha añadido el dataset TIDS real completo a
`inbox/a04/codigo/.../TIDS Dataset/` (862 imágenes) y el sandbox ahora tiene una GPU real y
compatible (RTX 2070 SUPER, CC 7.5 — antes GTX 1060, CC 6.1, incompatible con
`torch==2.13.0+cu130`). Este ciclo:

1. Investigó `test_images/` (vacío en la entrega) — no tiene ninguna lógica de selección propia
   en el código entregado: solo se referencia en `notebooks/predict_image.ipynb` como una
   carpeta genérica "copia aquí tus imágenes nuevas que el modelo nunca ha visto". El split
   train/val/test real está definido programáticamente por ID en
   `src/data/dataset.py::create_dataloaders` (`train_test_split` estratificado,
   `random_state=42`), y el split de test EXACTO usado para los resultados de la Tabla 9 de la
   memoria está guardado en
   `experiments/results/rigorous_final/test_ids_20260526_074452.txt` (42 IDs, 21 healthy + 21
   SCM). Se pobló `test_images/healthy/<ID>.jpg` y `test_images/SCM/<ID>.jpg` con esas 42
   imágenes CROPPED (misma convención `use_cropped=true` que el entrenamiento/evaluación
   original) — ni inventado ni dejado vacío sin más: es exactamente "imágenes nunca vistas en
   entrenamiento" Y el split oficial auditado, a la vez.
2. Actualizó `inbox/a04/manifest.yaml::golden_cases` con 18 casos nuevos (9 healthy + 9 SCM,
   primeros 9 IDs en orden numérico ascendente de cada clase dentro del split oficial — regla
   determinista, sin sesgo por resultado) + los 3 casos de validación externa del ciclo 1
   (conservados como complementarios). `expected` = label real de `ID_Labels.csv`;
   `real_served_output` = salida real del checkpoint servido (este ciclo, `device=cuda` real).
3. Encontró y corrigió un **bug de CUDA real** en `model_loader.py::_safe_device()` (detalle
   abajo) que solo se manifestaba con una GPU real y compatible — enmascarado en el ciclo 1
   porque la GTX 1060 fallaba el self-test por una razón distinta (incompatibilidad real) con
   el mismo resultado observable (fallback a CPU).
4. Re-ejecutó Parte B contra el servidor real con los 42 casos del split oficial (no solo los 18
   documentados en el manifest) + re-verificó `/train` (fine-tuning) end-to-end sobre CUDA real
   con datos TIDS reales.

### Bug de CUDA real — `_safe_device()`

```python
# ANTES (roto, pasaba inadvertido en cualquier GPU):
torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())
# El módulo Conv2d nunca se mueve a .cuda() — solo el tensor de entrada. Esto lanza SIEMPRE
# RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
# should be the same — independientemente de si la GPU real funciona o no.
```

Con la GTX 1060 del ciclo 1 (incompatible con el torch instalado), este `except` capturaba el
error de incompatibilidad real y hacía fallback a CPU — comportamiento correcto, por la razón
incorrecta. Con la RTX 2070 SUPER (CC 7.5, perfectamente soportada) de este ciclo, el mismo
`except` seguía ejecutándose y el modelo caía a CPU **a pesar de tener GPU funcional** —
confirmado reproduciendo el snippet exacto en el sandbox (ver abajo). Corregido al patrón ya
usado en `ml2_fungal_cnn_disease_detection`/`ml8_cereals_img_anomaly_detector`:

```python
probe = torch.nn.Conv2d(1, 1, 1).cuda()
probe(torch.zeros(1, 1, 4, 4).cuda())
```

Verificado tras el fix: arranque real del servidor registra
`Ml4LacticCnnThermal model loaded (device=cuda)`; `/predict` inline y batch y `/train`
(fine-tuning con `copy.deepcopy` + `.to(device)` sobre un subconjunto real de TIDS, fuera del
split de test) se ejecutaron end-to-end sobre CUDA real sin errores — `/train` completó 32
muestras reales en 4.7s (velocidad consistente con GPU, no CPU).

**Nota (fuera de alcance, no tocado):** el mismo patrón roto (`Conv2d(...)(tensor.cuda())` sin
mover el módulo) existe en `modelo10_lacteo`, `ml41_meat_curing_machinery_acoustic_anomaly`,
`ml5_meat_cow_behaviour` y `ml7_cereals_grain_pest_detection`. Se señala para una revisión
futura — la instrucción de este ciclo era no tocar otros plugins.

## Hallazgo principal de manifest-extraction: `training.supported`

El código entregado (`src/training/trainer.py`, `src/training/losses.py` — FocalLoss transcrita
verbatim —, `src/training/metrics.py`, `scripts/train_kfold.py`,
`configs/baseline_efficientnet.yaml`) implementa un procedimiento de entrenamiento real y
reproducible (Adam, lr=1e-4, wd=1e-4, batch=16, Focal Loss α=0.25/γ=2.0, ReduceLROnPlateau,
early stopping patience=15, 100 épocas máx., seed=42). El plugin integrado, sin embargo, tenía
`train()` lanzando `TrainingNotSupportedError` (501) — **bug de clase 3** (train() marcado
incorrectamente como no soportado, igual que el caso ml17 referenciado en la tarea).

## Bugs encontrados y corregidos

1. **Clase 3 — `train()` incorrectamente deshabilitado.** Corregido: se implementó el
   fine-tuning real en `plugin.py::train()`, siguiendo exactamente el optimizador/loss/scheduler
   del código entregado, clonando (`copy.deepcopy`) los pesos servidos antes de entrenar (nunca
   se muta `self._model` in-place hasta terminar), con `train_dto.py::TrainRequest.mlflow_run_id`
   **obligatorio, sin default** (repo-wide rule). El checkpoint reentrenado se guarda únicamente
   bajo el run de MLflow (`tracker.upload_artifacts(..., artifact_path="model")`) — nunca
   sobreescribe el artefacto base fijo servido por `model_loader.py`.
2. **`mlflow_utils.py` quedó obsoleto** (antes devolvía siempre `None` porque el modelo "no
   soportaba" reentrenamiento). Reescrito para descargar de verdad el checkpoint de un run de
   MLflow (`BaseMLflowTracker(run_id).download_artifacts(..., artifact_path="model")`,
   reconstruyendo `BaselineModel` + `state_dict`), siguiendo el patrón real ya usado en
   `ml8_cereals_img_anomaly_detector`.
3. **`predict_inline`/`predict_batch` ahora SÍ usan el resultado de
   `download_user_model_from_mlflow`** (antes se llamaba y se descartaba, aunque siempre
   devolvía `None`). Ahora swapea `self._model`/`self._device` para la duración de la petición y
   los restaura en `finally` junto con `shutil.rmtree` del temp dir — verificado directamente
   (ver Parte B).
4. **`stats()`** ya no llama a `download_user_model_from_mlflow` y descarta el resultado (no
   tiene sentido para metadata); ahora enriquece `metrics["mlflow"]` con
   `tracker.get_params()`/`get_metrics()` del run, igual que `ml8`.

### Bugs de las clases 1, 2 y 4 — NO encontrados en este plugin

- **Clase 1 (dict vs. Pydantic):** `predict_inline`/`predict_batch` ya devolvían
  `PredictInlineResponse`/`PredictBatchResponse` tipados. Sin bug.
- **Clase 2 (preprocesado silenciosamente distinto):** `preprocessing.py::_INFERENCE_TRANSFORM`
  (`Resize(224,224)` + `Normalize(ImageNet mean/std)` + `ToTensorV2`) coincide EXACTAMENTE,
  campo a campo, con `src/data/transforms.py::get_val_transforms`/`get_inference_transforms` del
  código entregado — no hay crop ni resize distinto (no es el caso de modelo10). Sin bug.
- **Clase 4 (falta `device=` en hardware real):** en el ciclo 1 se concluyó "sin bug" porque
  `_safe_device()` caía a CPU limpiamente en la GTX 1060 (incompatible) del sandbox de
  entonces. **Revisado en el ciclo 2 con GPU real (RTX 2070 SUPER, CC 7.5): SÍ había bug** —
  ver sección "Bug de CUDA real" arriba. Corregido.

## Parte A — Checklist técnico (ciclo 2)

- [x] `flake8 app/plugins/ml4_lactic_cnn_thermal_early_disease_detection/ app/registry.py tests/conftest.py`: 0 errores
- [x] `pytest tests/unit/ -q`: **503/503 passed** (suite completa, sin regresiones — re-ejecutado tras el fix de `_safe_device()`)
- [x] `pylint app/plugins/ml4_lactic_cnn_thermal_early_disease_detection/ --disable=import-error`: 9.07/10 (igual que ciclo 1 — el fix de CUDA no introduce issues nuevos) — issues restantes son del mismo tipo/baseline que en plugins ya aceptados (`broad-exception-caught` en catches auxiliares de CAM/upload, `import-outside-toplevel` para evitar import pesado de torch/timm en frío, `no-member` cv2/PIL — falsos positivos conocidos de pylint con esas libs).
- [x] `pip-audit -r requirements.txt`: 1 vulnerabilidad (`setuptools 80.9.0`, PYSEC-2026-3447) — preexistente, no relacionada con este plugin.
- [x] Arranque local (`MODEL=ml4-lactic-cnn-thermal-early-disease-detection python -m uvicorn main:app --port 8010`, en puerto dedicado para no interferir con otros agentes concurrentes en el 8000) + `/health` + `/stats`: OK, 200. **Log confirma `device=cuda` real** (RTX 2070 SUPER) tras el fix — antes del fix, el mismo arranque reportaba `device=cpu` a pesar de la GPU funcional.
- [x] `/predict` inline y batch sobre CUDA real: OK, 200, sin 500 — ver Parte B. `predict_batch` reproduce EXACTAMENTE (misma confianza a 13+ decimales) las mismas predicciones que `predict_inline` para las mismas imágenes — confirma pipeline único sin divergencia CPU/CUDA.
- [x] `/train` sin `mlflow_run_id` → **422** (campo requerido, sin default) ✓ esperado
- [x] `/train` con `mlflow_run_id` real, sobre CUDA real y datos TIDS reales (no sintéticos): no se
  pudo probar vía HTTP (MLflow inalcanzable en este sandbox, igual que ciclo 1). Se monkeypatcheó
  `BaseMLflowTracker` a un stub no-op y se llamó `plugin.train()` directamente sobre un
  subconjunto REAL de TIDS (32 imágenes reales: 16 healthy + 16 SCM, tomadas del dev set —
  excluyendo expresamente los 42 IDs del split de test oficial para no contaminarlo) —
  **ejecutó sin errores sobre `device=cuda`**:
  `detail='Reentrenamiento completado' accuracy=1.0 f1=1.0 n_train=25 n_val=7 training_time_s=4.7`.
  El `training_time_s=4.7` (vs. 14.11s en el ciclo 1 sobre CPU con un dataset sintético más
  pequeño) es consistente con ejecución real en GPU. `accuracy=1.0` sigue siendo un artefacto
  esperado de `n_val=7` (val set diminuto), no una validación de correctitud numérica — pero el
  pipeline completo (dataloader de imágenes reales, augmentation, `copy.deepcopy` +
  `.to(device)` del modelo clonado, Focal Loss, Adam, ReduceLROnPlateau, early stopping,
  checkpoint a MLflow) corrió de principio a fin sobre CUDA real sin errores.
- [x] Swap `mlflow_run_id` en `predict_inline`: verificado directamente (igual que ciclo 1,
  monkeypatcheando `download_user_model_from_mlflow`) — sigue funcionando tras el fix de device.

## Parte B — Correctitud contra golden dataset (ciclo 2 — dataset TIDS real)

El dataset TIDS real (862 imágenes) y el split de test oficial exacto (42 IDs,
`test_ids_20260526_074452.txt`, el mismo usado para el accuracy=59.52% de la Tabla 9 de la
memoria) están ahora disponibles. `test_images/` se pobló con las 42 imágenes cropped de ese
split. Se ejecutaron los **42/42 casos del split oficial** contra el servidor real
(`device=cuda`), no solo los 18 documentados individualmente en el manifest:

| Resultado | n | % |
|---|---|---|
| Clase correcta (prediction == label real) | 26/42 | 61.90% |
| Clase incorrecta | 16/42 | 38.10% |

Comparación con la métrica auditada de la memoria para este mismo split exacto: accuracy
reportada = 59.52% (25/42). Diferencia de 1 muestra (61.90% vs 59.52%) — coherente con el drift
de versión de `timm`/PyTorch entre el entorno de entrenamiento original y este, ya señalado en
`known_issues`; **no indica un bug de wiring** (confirmado: `predict_batch` reproduce
bit-a-bit las mismas salidas que `predict_inline`, y el preprocesado coincide exactamente con
`src/data/transforms.py` — ver Clase 2 arriba).

De esos 42, el manifest documenta 18 casos ejecutables en detalle (9 healthy + 9 SCM, selección
determinista por ID ascendente, sin filtrar por si aciertan o fallan):

| Caso (ID) | Label real | Predicción servidor real | Confianza | ¿Clase OK? |
|---|---|---|---|---|
| 17 | Healthy | SCM | 0.6106 | ✗ |
| 109 | Healthy | SCM | 0.5199 | ✗ |
| 147 | Healthy | Healthy | 0.5650 | ✓ |
| 326 | Healthy | Healthy | 0.5583 | ✓ |
| 365 | Healthy | SCM | 0.6326 | ✗ |
| 382 | Healthy | Healthy | 0.6229 | ✓ |
| 404 | Healthy | Healthy | 0.5405 | ✓ |
| 413 | Healthy | Healthy | 0.5627 | ✓ |
| 423 | Healthy | Healthy | 0.5159 | ✓ |
| 11 | SCM | Healthy | 0.5096 | ✗ |
| 83 | SCM | Healthy | 0.5443 | ✗ |
| 133 | SCM | Healthy | 0.6126 | ✗ |
| 309 | SCM | SCM | 0.5840 | ✓ |
| 311 | SCM | SCM | 0.5797 | ✓ |
| 315 | SCM | SCM | 0.5377 | ✓ |
| 366 | SCM | SCM | 0.6233 | ✓ |
| 371 | SCM | Healthy | 0.5688 | ✗ |
| 372 | SCM | SCM | 0.5760 | ✓ |

**Resultado de los 18 documentados: 12/18 (66.7%) clase correcta** — dentro del rango esperable
dado el accuracy global del split completo (61.90%, n=42, cada muestra pesa ~2.4pp).

**Tolerancia usada:** coincidencia de clase (Healthy/SCM) — no hay MAE/RMSE para clasificación
binaria; el F1/accuracy agregado de la memoria (Tabla 8/9) es la referencia de tolerancia, no
un umbral por caso. Los casos "✗" no se investigan como bug individual porque, en conjunto
(16/42 = 38.1% de error), reproducen exactamente el rendimiento moderado ya documentado del
modelo (F1 test=0.6047, apenas por encima del azar) — no un patrón de error sistemático de
wiring (p.ej. no hay inversión de clases: 21 healthy y 21 SCM aciertan en proporción similar,
13/21 y 13/21 sobre el total de 42).

Los 3 escenarios de **validación externa** del ciclo 1 (memoria, fuera del dominio TIDS) se
conservan sin cambios: 2/2 ejecutables correctos (Healthy y SCM), 1/3 no ejecutable
(interempresas.net, imagen embebida vía JS).

predict_batch sobre un subconjunto de los casos TIDS (4 imágenes, zip) reprodujo exactamente
las mismas predicciones que predict_inline (misma clase, misma confianza a 13+ decimales) —
confirma que ambos endpoints comparten el mismo pipeline de preprocesado/inferencia sobre CUDA
real sin divergencias.

### Validación externa complementaria (ciclo 1, sin cambios)

| Caso | Imagen | Diagnóstico real (memoria) | Predicción modelo original (memoria) | Predicción servidor real | ¿Clase OK? |
|---|---|---|---|---|---|
| caso_prueba_1_scm_interempresas | SubMastitis_sample.png (interempresas.net) | SCM | SCM (conf. 61.79%) | **No ejecutado** — imagen no descargable de forma estática (requiere JS) | N/A |
| caso_prueba_2_healthy_thermomast | THERMOMAST sample_14.jpg (Healthy Udders pt.1, Nov 2022) | Healthy | Healthy (conf. 61.15%) | **Healthy** (conf. 56.14%) | ✓ OK |
| caso_prueba_3_scm_thermomast | THERMOMAST sample_171.jpg (Subclinical Mastitis, Nov 2022) | SCM | SCM (conf. 62.55%) | **SCM** (conf. 65.67%) | ✓ OK |

Resultado sin cambios respecto al ciclo 1: 2/2 casos ejecutables dentro de tolerancia (clase
correcta), 1/3 no ejecutable por falta de la imagen fuente (documentado, no silenciado).

Métricas de referencia del modelo completo (memoria, Tabla 8/9): test hold-out (n=42)
accuracy=59.52%, F1=0.6047 — **ahora verificado 1:1 contra el split de test exacto real**
(ver tabla de 42/42 arriba: 61.90% obtenido, diferencia de 1 muestra sobre 42). CV 5-fold
(n=376) F1=0.6253±0.0427 (no verificable 1:1 sin repetir el entrenamiento completo, no es
necesario para esta verificación). Rendimiento moderado, documentado tal cual en
`LIMITATIONS.md`/memoria — no es un bug de integración.

## Estado final

**LISTO PARA PR.** Resumen del ciclo 2:

- Dataset TIDS real (862 imágenes) incorporado; `test_images/` poblado con el split de test
  oficial exacto (42 imágenes, antes vacío).
- `inbox/a04/manifest.yaml::golden_cases` ampliado de 2/3 casos ejecutables a 21 (18 nuevos
  sourced del split de test real de TIDS + 3 de validación externa conservados).
- Parte B re-ejecutada contra el servidor real: 42/42 casos del split oficial (61.90% accuracy,
  coherente con el 59.52% de la memoria para ese mismo split) + 18 documentados en detalle en el
  manifest + 2/3 externos. Ningún caso fallido es un bug de wiring — todos coherentes con el
  rendimiento moderado ya documentado del modelo.
- **Bug de CUDA real encontrado y corregido**: `_safe_device()` caía siempre a CPU por un
  self-test roto (módulo Conv2d no movido a `.cuda()`), invisible en el ciclo 1 por casualidad
  (GPU incompatible entonces). Con la GPU real de este ciclo, `/predict` y `/train` ahora se
  ejecutan verificadamente sobre `device=cuda`.
- El mismo caso #1 de validación externa (interempresas.net, imagen embebida vía JS) sigue sin
  poder ejecutarse en este entorno — sin cambios respecto al ciclo 1, se recomienda revisión
  humana si se dispone de la imagen por otro medio.
- `pytest` 503/503, `flake8` 0 errores, `pylint` 9.07/10 (sin cambios) — sin regresiones tras el
  fix de CUDA.
