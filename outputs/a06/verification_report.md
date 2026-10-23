# Verificación — ml6-meat-poultry-disease-classification (a06)

- Plugin: `app/plugins/ml6_meat_poultry_disease_classification/` · rama `feature/model-a06-integration`
- Manifest: `inbox/a06/manifest.yaml` (63 golden cases, `training.supported: true`)
- Fecha: 2026-10-16 · Entorno: WSL2 Ubuntu 24.04, Python 3.12.3, torch 2.13.0+cpu, torchvision 0.28.0,
  opencv-python(-headless) 5.0.0, **CPU** (la GPU RTX 3060 del equipo no se usó: el venv es el del
  `requirements.txt` del repo con índice CPU, igual que en las integraciones anteriores).
- Artefactos base: `artifacts/ml6_meat_poultry_disease_classification/` — md5 **bit a bit** igual a
  `inbox/a06/codigo/models/{thermal,fecal}/artifacts/*.pt` (6/6), y los 6 `state_dict` cargan con
  `strict=True` sobre las arquitecturas replicadas.

> **Resumen.** El plugin reproduce exactamente al código del equipo de IA. En inferencia, 63/63 golden
> cases y 0 diferencias de voto en las 232 imágenes de test (Tablas 4, 5 y 6 de la memoria a 4 decimales).
> En reentrenamiento, el port da métricas de validación **idénticas a 4 decimales** a las de
> `src/training/trainer.py` en los 3 modelos térmicos, en la misma máquina. El checklist técnico está en verde
> para ml6 (la única CVE es preexistente y ajena). **Listo para PR**, con las decisiones de la sección
> «Pendiente de acción humana».

## Checklist técnico

- [x] **flake8**: 0 errores en el código del repo. El comando del skill (`flake8 . --extend-exclude=dist,build`)
  reporta 61 avisos, **todos** en `inbox/a06/codigo/` (código del equipo de IA, ignorado por git y ausente en CI);
  no se ha tocado.
- [x] **pytest**: **1132 passed, 9 skipped**, cobertura total `app/` 60 % (suite completa, ejecución final tras todos los cambios).
  Los 9 skips son previos y ajenos a ml6 (artefactos de ml15-18/21/23 que viven en S3 y `inbox/a15`, `a36` ausentes).
  ml6: 19 tests propios (endpoint + funciones puras + `train()` real con MLflow simulado) + higiene (`test_user_model_download_hygiene.py`) + aislamiento concurrente
  (`test_user_model_isolation.py`, ml6 añadido) + contrato MLflow (`test_mlflow_persistence_contract.py`).
- [x] **pylint** `app/plugins/ml6_meat_poultry_disease_classification/`: **10.00/10** (ml4 9.03, ml2 9.08 como
  referencia). Único `disable` de módulo: `no-member` de cv2 (extensión C, mismo falso positivo que ml4/ml2).
- [ ] **pip-audit** `-r requirements.txt`: **1 CVE PREEXISTENTE, no nueva** — `setuptools==80.9.0`
  PYSEC-2026-3447 (fix 83.0.0). ml6 no añade dependencias. No se ha subido el pin porque el Dockerfile compila
  detectron2 con `--no-build-isolation` y el cambio afecta a todo el repo → **decisión humana**.
- [x] **Arranque local** `MODEL=ml6-meat-poultry-disease-classification python main.py` → `Startup complete. 1/1 models ready`.
- [x] `/health` → `{"status":"ok","loaded":true}` · `/stats` → 200 (métricas declaradas por tarea).
- [x] `/predict` inline (thermal y rgb) y batch (directorio) → 200. Errores: `model_key` ausente, inválido,
  imagen no decodificable o ruta inexistente → 422.
- [x] `/train` sin `mlflow_run_id` / con `"  "` / sin `task` → 422.
- [x] `/train` real **thermal** contra un MLflow local (`mlflow server` en 127.0.0.1:5055) con el dataset térmico
  completo (254 imágenes) → **200** en 860 s (CPU), ver detalle abajo.
- [x] `/predict` con `mlflow_run_id` del run entrenado → 200 con **el ensemble reentrenado** (probabilidades
  distintas de las del base); mismo run pedido como `rgb` → 422; run inexistente → 422. 0 temporales huérfanos.
- [x] `/stats?mlflow_run_id=` → incluye params y métricas por época del run.

### Bug encontrado y corregido durante la verificación

`PredictInlineRequest`/`PredictBatchRequest` no declaraban `mlflow_run_id`: Pydantic descartaba el campo, el caso
de uso pasaba `""` y **una petición con un run inexistente respondía 200 con el modelo base** (el fallo silencioso
que prohíbe el skill). Detectado al llamar al servidor real; los tests con FakePlugin no lo veían. Corregido en
`predict_dto.py` y cubierto con `test_predict_request_forwards_mlflow_run_id_to_plugin`.

> **Hallazgo transversal (no corregido, fuera de alcance):** los `predict_dto.py` de **ml4, ml14, ml17, ml18,
> ml21 y ml23** tampoco declaran `mlflow_run_id`, así que por HTTP su `/predict` ignora el run del usuario y
> sirve el base. Requiere revisión por separado.

### /train — detalle

**thermal** (run `b95842ae6896451891be48a0719018e3`, dataset completo `data/thermal/raw`, split 177/37/40 = el
mismo split que el original):

| Modelo | Épocas (early stopping) | Mejor val_acc | Test accuracy | Test f1_w |
|---|---|---|---|---|
| densenet121 | 25 | 0.8378 | 0.7250 | 0.7160 |
| efficientnet_v2s | 13 | 0.8649 | 0.7250 | 0.7270 |
| resnet50 | 21 | 0.9189 | 0.7750 | 0.7604 |
| **ensemble** | — | — | **0.8000** | **0.7905** |

Ensemble declarado por el equipo en ese test: accuracy 0.8000, f1_w 0.7960. Artefactos subidos al run:
`model/{densenet121,efficientnet_v2s,resnet50}.pt` + `metadata.json`. Nota: el reentreno parte del checkpoint
servido (regla del repo), no de ImageNet como el original, así que no se espera reproducir bit a bit las métricas
de entrenamiento del equipo; sí el procedimiento (Adam + StepLR(7, 0.1) + early stopping sobre val_loss,
paciencia 10, mejor val_acc, WeightedRandomSampler, seed 42).

**rgb** (run `8781ec6b25504a58b977c9132005851e`, prueba de humo con 10 imágenes reales por clase del split de train
original, `pcr*` como carpetas): **200** en 286 s (CPU), split 28/4/8, épocas 15/11/13 (vgg19_bn /
efficientnet_b7 / mobilenet_v3l), `/predict` con ese run → 200. Con 8 imágenes de test las métricas devueltas
(accuracy 1.0) **no son representativas**: solo prueban el circuito (carpetas `pcr*` → folder_map, batch 64,
subida a MLflow, recarga del ensemble del usuario). El reentreno completo del ensemble rgb (1 255 imágenes) dura
~2 h 45 min en GPU según la memoria (Tabla 3) y no se ha ejecutado aquí.

Cobertura de `app/plugins/ml6_meat_poultry_disease_classification/` con sus tests unitarios: 64 %. El resto
(construcción de las 6 redes, bucle real de entrenamiento, CLAHE) se ha ejercido contra el servidor real en esta
verificación, no en la suite.

### Fidelidad del reentrenamiento (`training.py` frente a `src/training/trainer.py`)

Se ejecutó el código original (`scripts/train.py --task thermal`, sobre una copia en un directorio temporal con
el dataset térmico completo) y, en paralelo, el port del plugin (`split_dataset` + `make_loaders` +
`train_model`), ambos desde la misma inicialización ImageNet (builders originales), en CPU, 8 hilos cada uno y la
misma máquina. Métricas del mejor checkpoint sobre validación (n=37):

| Modelo | Épocas reales (port / original) | Port: acc · f1_w · f1_macro | Original aquí | Entregado por el equipo (GPU RTX 4070) |
|---|---|---|---|---|
| densenet121 | 19 / 19 | 0.7838 · 0.7838 · 0.7533 | **idéntico** | 0.7838 · 0.7838 · 0.7533 |
| efficientnet_v2s | 22 / 22 | 0.8378 · 0.8424 · 0.8276 | **idéntico** | 0.8919 · 0.8938 · 0.8814 |
| resnet50 | 21 / 21 | 0.8378 · 0.8424 · 0.8276 | **idéntico** | 0.8919 · 0.8891 · 0.8706 |

- **Port frente a original: idénticos** en accuracy, precision/recall/f1 weighted y macro de los 3 modelos, y en
  el punto de parada. El original imprime «after 20/23/22 epochs» por un fallo cosmético del mensaje (una época
  de más); las dos ejecuciones paran en la misma época real. Duración del original en CPU: 1 470 s.
- **Re-ejecución frente a lo entregado:** densenet121 coincide exacto; efficientnet_v2s y resnet50 no (val_acc
  0.8378 frente a 0.8919). El equipo entrenó en GPU: cuDNN no es determinista y CPU/GPU difieren, así que **los
  checkpoints entregados no se reproducen bit a bit reentrenando en otra máquina**. No afecta a la inferencia,
  que usa sus checkpoints tal cual (63/63). No se ha podido comparar en GPU porque el venv es CPU.
- La prueba se hizo desde ImageNet para comparar con el original. En producción, `/train` parte del checkpoint
  servido (regla del repo); esa es la única diferencia con este harness.

## Correctitud (golden dataset)

Tolerancia usada (fijada en `manifest.verification_tolerance` **antes** de ejecutar):

- **Por caso: 0** — la clase del ensemble y el voto de cada uno de los 3 modelos base deben coincidir exactamente
  con lo que registró el pipeline original para esa imagen (`{tarea}_predictions.csv` y `{modelo}_predictions.csv`
  del entregable). Es una clasificación determinista: no hay un MAE del que derivar una tolerancia relativa.
- **Agregada:** accuracy del ensemble en el test completo, |obtenida − declarada| ≤ 1 muestra (1/40 = 0.025
  thermal; 1/192 = 0.0052 rgb).

**Resultado: 63/63 casos dentro de tolerancia** (40 thermal + 20 rgb + 3 empates a tres rgb), 189/189 votos de
modelo base idénticos. Llamadas reales a `/predict` inline con `image_path`.

### Split de test completo vía `/predict` batch (además de los golden)

| Tarea | Imágenes | Diferencias de voto vs CSV del equipo | Tiempo (CPU) |
|---|---|---|---|
| thermal | 40 | **0** (ensemble y 3 modelos, 160 decisiones) | 6.4 s |
| rgb | 192 | **0** (ensemble y 3 modelos, 768 decisiones) | 74.4 s |

| Métrica (test) | Declarada (memoria) | Obtenida (plugin) | Δ |
|---|---|---|---|
| thermal ensemble accuracy | 0.8000 | 0.8000 | 0 |
| thermal ensemble precision_w / recall_w / f1_w | 0.7964 / 0.8000 / 0.7960 | 0.7964 / 0.8000 / 0.7960 | 0 |
| thermal densenet121 / efficientnet_v2s / resnet50 accuracy | 0.6750 / 0.6750 / 0.7750 | 0.6750 / 0.6750 / 0.7750 | 0 |
| thermal recall sick / healthy / macro | 0.8846 / 0.6429 / 0.7637 | 0.8846 / 0.6429 / 0.7637 | 0 |
| rgb ensemble accuracy | 0.8490 | 0.8490 | 0 |
| rgb ensemble precision_w / recall_w / f1_w | 0.8505 / 0.8490 / 0.8484 | 0.8505 / 0.8490 / 0.8484 | 0 |
| rgb vgg19_bn / efficientnet_b7 / mobilenet_v3l accuracy | 0.8490 / 0.8125 / 0.8490 | 0.8490 / 0.8125 / 0.8490 | 0 |
| rgb por clase f1 (healthy / cocci / ncd / salmo), Tabla 6 | 0.7358 / 0.8704 / 0.9667 / 0.8727 | 0.7358 / 0.8704 / 0.9667 / 0.8727 | 0 |

Las Tablas 4, 5 y 6 de la memoria se reproducen **a 4 decimales** con el plugin servido.

### Casos golden

| Caso | Esperado (ensemble · votos) | Obtenido (ensemble · votos) | Diferencia | Clase real | ¿OK? |
|---|---|---|---|---|---|
| thermal_test_001 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | healthy | ✅ |
| thermal_test_002 | healthy · sick/healthy/healthy | healthy · sick/healthy/healthy | 0 votos | healthy | ✅ |
| thermal_test_003 | healthy · healthy/sick/healthy | healthy · healthy/sick/healthy | 0 votos | healthy | ✅ |
| thermal_test_004 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | healthy | ✅ |
| thermal_test_005 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | healthy | ✅ |
| thermal_test_006 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | healthy | ✅ |
| thermal_test_007 | healthy · healthy/sick/healthy | healthy · healthy/sick/healthy | 0 votos | healthy | ✅ |
| thermal_test_008 | sick · healthy/sick/sick | sick · healthy/sick/sick | 0 votos | healthy | ✅ |
| thermal_test_009 | healthy · healthy/healthy/sick | healthy · healthy/healthy/sick | 0 votos | healthy | ✅ |
| thermal_test_010 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | healthy | ✅ |
| thermal_test_011 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | healthy | ✅ |
| thermal_test_012 | healthy · sick/healthy/healthy | healthy · sick/healthy/healthy | 0 votos | healthy | ✅ |
| thermal_test_013 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | healthy | ✅ |
| thermal_test_014 | healthy · healthy/healthy/sick | healthy · healthy/healthy/sick | 0 votos | healthy | ✅ |
| thermal_test_015 | sick · healthy/sick/sick | sick · healthy/sick/sick | 0 votos | sick | ✅ |
| thermal_test_016 | healthy · healthy/healthy/sick | healthy · healthy/healthy/sick | 0 votos | sick | ✅ |
| thermal_test_017 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | sick | ✅ |
| thermal_test_018 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | sick | ✅ |
| thermal_test_019 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | sick | ✅ |
| thermal_test_020 | sick · healthy/sick/sick | sick · healthy/sick/sick | 0 votos | sick | ✅ |
| thermal_test_021 | sick · healthy/sick/sick | sick · healthy/sick/sick | 0 votos | sick | ✅ |
| thermal_test_022 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_023 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_024 | sick · healthy/sick/sick | sick · healthy/sick/sick | 0 votos | sick | ✅ |
| thermal_test_025 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | sick | ✅ |
| thermal_test_026 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_027 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_028 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_029 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_030 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | sick | ✅ |
| thermal_test_031 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_032 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_033 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_034 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_035 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | sick | ✅ |
| thermal_test_036 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_037 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_038 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_039 | sick · sick/sick/sick | sick · sick/sick/sick | 0 votos | sick | ✅ |
| thermal_test_040 | sick · sick/healthy/sick | sick · sick/healthy/sick | 0 votos | sick | ✅ |
| rgb_test_001 | cocci · cocci/cocci/cocci | cocci · cocci/cocci/cocci | 0 votos | cocci | ✅ |
| rgb_test_002 | cocci · cocci/cocci/salmo | cocci · cocci/cocci/salmo | 0 votos | cocci | ✅ |
| rgb_test_003 | cocci · cocci/cocci/cocci | cocci · cocci/cocci/cocci | 0 votos | cocci | ✅ |
| rgb_test_004 | cocci · cocci/cocci/cocci | cocci · cocci/cocci/cocci | 0 votos | cocci | ✅ |
| rgb_test_005 | cocci · cocci/cocci/cocci | cocci · cocci/cocci/cocci | 0 votos | cocci | ✅ |
| rgb_test_006 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | healthy | ✅ |
| rgb_test_007 | healthy · cocci/healthy/healthy | healthy · cocci/healthy/healthy | 0 votos | healthy | ✅ |
| rgb_test_008 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | healthy | ✅ |
| rgb_test_009 | salmo · salmo/healthy/salmo | salmo · salmo/healthy/salmo | 0 votos | healthy | ✅ |
| rgb_test_010 | healthy · healthy/healthy/healthy | healthy · healthy/healthy/healthy | 0 votos | healthy | ✅ |
| rgb_test_011 | ncd · ncd/ncd/ncd | ncd · ncd/ncd/ncd | 0 votos | ncd | ✅ |
| rgb_test_012 | ncd · ncd/ncd/ncd | ncd · ncd/ncd/ncd | 0 votos | ncd | ✅ |
| rgb_test_013 | ncd · ncd/ncd/ncd | ncd · ncd/ncd/ncd | 0 votos | ncd | ✅ |
| rgb_test_014 | ncd · ncd/ncd/ncd | ncd · ncd/ncd/ncd | 0 votos | ncd | ✅ |
| rgb_test_015 | ncd · ncd/ncd/ncd | ncd · ncd/ncd/ncd | 0 votos | ncd | ✅ |
| rgb_test_016 | healthy · healthy/salmo/healthy | healthy · healthy/salmo/healthy | 0 votos | salmo | ✅ |
| rgb_test_017 | salmo · salmo/salmo/salmo | salmo · salmo/salmo/salmo | 0 votos | salmo | ✅ |
| rgb_test_018 | salmo · salmo/salmo/salmo | salmo · salmo/salmo/salmo | 0 votos | salmo | ✅ |
| rgb_test_019 | salmo · salmo/salmo/salmo | salmo · salmo/salmo/salmo | 0 votos | salmo | ✅ |
| rgb_test_020 | salmo · salmo/salmo/salmo | salmo · salmo/salmo/salmo | 0 votos | salmo | ✅ |
| rgb_tie_001 | healthy · healthy/cocci/salmo | healthy · healthy/cocci/salmo | 0 votos | cocci | ✅ |
| rgb_tie_002 | salmo · salmo/healthy/cocci | salmo · salmo/healthy/cocci | 0 votos | healthy | ✅ |
| rgb_tie_003 | cocci · cocci/ncd/healthy | cocci · cocci/ncd/healthy | 0 votos | healthy | ✅ |

Aciertos frente a la etiqueta real (informativo, es el rendimiento del modelo, no un fallo de integración):
32/40 thermal golden y 18/23 rgb golden. Latencia inline en CPU: mediana 163 ms (thermal) y 370 ms (rgb), con un
`/train` concurrente en el mismo proceso durante parte de la medición.

### Observaciones de la investigación (sin casos fuera de tolerancia)

1. **Votos al borde en thermal.** El margen top-2 de softmax de algunos modelos base es mínimo:
   0.0019 (`thermal_test_035`, efficientnet_v2s), 0.0026 (`thermal_test_023`, efficientnet_v2s), 0.0066
   (`thermal_test_016`, densenet121). Hoy coinciden con el original, pero el entorno de verificación (opencv 5.0.0,
   torch 2.13) **no es el del equipo** (opencv 4.13.0.92, torch 2.12.0). Una deriva numérica del CLAHE o de las
   convoluciones podría voltear esos votos. `opencv-python-headless` no está fijado en el `requirements.txt` del
   repo: se recomienda fijarlo. En rgb el margen mínimo es 0.061 (mucho más estable).
2. **Desempate a tres** (`rgb_tie_001..003`): el original devuelve el voto de vgg19_bn (primer modelo de
   `config.yaml`) por cómo funciona `Counter.most_common`; el plugin lo reproduce en los 3 casos. Si se quiere otro
   criterio (p. ej. media de probabilidades) es una decisión del equipo de IA, no de la integración.
3. **Posible etiqueta errónea en test thermal:** `A1_1_124_3_0_مصاب.bmp` ('infectado') está en la carpeta
   Healthy del dataset público; el ensemble lo predice healthy (cuenta como acierto). Si la etiqueta real fuese
   sick, la accuracy del ensemble térmico bajaría a 0.775. No se ha alterado ninguna etiqueta.
4. **Datos de verificación** descargados de las fuentes públicas citadas en la memoria (no venían en la entrega);
   ambos splits regenerados con `ensure_data_splits` del equipo coinciden fichero a fichero con sus CSV.
5. **No ejercitado:** la ruta CUDA (`safe_device` con la sonda `Conv2d(1,1,1).cuda()`, módulo y tensor en GPU)
   no se ha ejecutado porque el venv es CPU. El entrenamiento completo rgb tampoco (ver arriba).

## Cambios del plugin frente al original (deliberados)

- Un único plugin sirve los dos ensembles; `model_key` (`thermal` | `rgb`) es obligatorio porque las modalidades
  no son intercambiables (el original lo elige con `--task`).
- Las redes se construyen con `weights=None` y `state_dict` estricto: el original descarga los pesos de ImageNet
  incluso para predecir.
- La respuesta añade el softmax de cada modelo y avisos de entrada (imagen térmica que no es 240x320). La
  decisión sigue siendo el voto duro del original, con su mismo desempate.
- `/train` reentrena los 3 modelos del ensemble pedido (el original permite `--model`), parte del checkpoint
  servido en vez de ImageNet, aísla el RNG con `fork_rng` + `manual_seed(42)` y devuelve las métricas del split
  de test (el original solo guarda las de validación). No repite la búsqueda Optuna.
- No se sirven Grad-CAM ni la evaluación de atención (fases offline según la memoria §6.4).

## Pendiente de acción humana

- Subir los 6 ficheros de `artifacts/ml6_meat_poultry_disease_classification/` (md5 en el manifest) a
  `s3://<STORAGE_BUCKET>/artifacts/fixed/ml6_meat_poultry_disease_classification/`.
- Decidir sobre la CVE de `setuptools==80.9.0` y sobre fijar `opencv-python-headless` en el repo.
- Revisar `mlflow_run_id` en los predict DTOs de ml4, ml14, ml17, ml18, ml21 y ml23.
- Con el equipo de IA: la etiqueta de `A1_1_124_3_0_مصاب.bmp`, el criterio de desempate y que los checkpoints
  thermal no se reproducen reentrenando fuera de su GPU.

## Estado final

**LISTO PARA PR — pendiente de revisión humana**, con estas decisiones abiertas (ninguna es un fallo de
correctitud del plugin):

- CVE preexistente `setuptools==80.9.0` (pip-audit), ajena a ml6.
- Hallazgo transversal: `mlflow_run_id` ausente en los predict DTOs de ml4, ml14, ml17, ml18, ml21 y ml23.
- Fijar la versión de opencv del repo (votos al borde en thermal).
- Validar con el equipo de IA la etiqueta de `A1_1_124_3_0_مصاب.bmp` y el criterio de desempate.
- `/train` rgb completo (~2 h 45 min en GPU) es síncrono: depende del timeout de la plataforma.
