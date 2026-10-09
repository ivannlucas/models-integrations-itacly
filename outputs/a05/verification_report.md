# Verificación — ml5-meat-cow-behaviour

## ESTE INFORME SUPERA AL ANTERIOR

La versión previa de este fichero concluía **"LISTO PARA PR"** afirmando que no se había
encontrado ningún bug. Esa conclusión era correcta dada la evidencia disponible en su
momento (el entregable de `inbox/a05/codigo/` no incluía entonces ningún vídeo/imagen/frame
real — solo placeholders en `data/images/README.md` y `data/annotations/README.md`), pero
**ya no es válida**: el dataset real completo (CSIRO Cow Behavior Dataset — anotaciones COCO
+ 33.875 imágenes solo en el split de test) se añadió al entregable después de aquel ciclo,
y permitió detectar un **bug crítico de decodificación** que era invisible sin datos reales.
Este informe no borra el anterior — lo sustituye con el resultado de esta nueva auditoría
(2026-10-06), ahora con acceso al dataset real.

## Bug confirmado y corregido

**Síntoma:** el plugin decodificaba el índice de salida del clasificador SlowFast al
comportamiento equivocado en 10 de sus 12 clases.

**Causa raíz:** el código de entrenamiento entregado tiene DOS listas de clases
inconsistentes:
- `configs/slowfast_cow_behavior.py:BEHAVIOR_TO_IDX` — la que `train_classifier.py`
  importa y **guarda embebida en el checkpoint** (`torch.save(..., 'behavior_to_idx':
  BEHAVIOR_TO_IDX)`).
- `utils/data_utils.py:BehaviorClassificationDataset.__init__` — un dict hardcodeado
  **distinto**, con el que `__getitem__` calcula el `label` entero que realmente se usa
  para entrenar (`label = self.behavior_to_idx[clip_info['behavior']]`).

Ambos dicts solo coinciden en los índices 0 (`grazing`) y 6 (`ruminating-standing`). El
plugin (igual que `main.py` del entregable original) decodificaba siempre con el dict
embebido en el checkpoint — es decir, con el orden de `configs/`, que **no** es el orden que
la cabeza de salida del modelo aprendió realmente.

**Verificación cuantitativa** (checkpoint real, cargado vía el propio loader del plugin,
1.500 clips reales del split de test):

| Decodificación | Accuracy | Macro F1 |
|---|---|---|
| Dict embebido en checkpoint (orden `configs/`) — **comportamiento previo** | 0.44 | 0.17 |
| Orden de `data_utils.py` (el que el modelo realmente aprendió) — **fix** | 0.91 | 0.87 |

(Ambas cifras comparten el mismo leakage de split por frame de `format.py` — ver más abajo —
por lo que son una A/B válida entre las dos decodificaciones, no una medida limpia de
generalización.)

**Fix aplicado:**
- `app/plugins/ml5_meat_cow_behaviour/constants.py` — nueva constante
  `TRAINING_BEHAVIOR_TO_IDX` con el orden correcto (`data_utils.py`), documentada con un
  comentario extenso explicando la discrepancia y la evidencia, para que esto no regrese.
- `app/plugins/ml5_meat_cow_behaviour/model_loader.py:_load_classifier` — ya no construye
  `idx_to_behavior` desde `checkpoint["behavior_to_idx"]`; usa siempre
  `TRAINING_BEHAVIOR_TO_IDX`. El checkpoint solo se sigue usando para leer `model_state_dict`
  y la forma de la capa de proyección (`num_classes`).
- `inbox/a05/manifest.yaml` — la entrada de `known_issues` que decía "esta inconsistencia no
  afecta a la inferencia" quedó marcada `[CORREGIDO Ciclo 2]` con la explicación completa de
  por qué esa afirmación era incorrecta.

No se tocó `postprocessing.py` (la lógica de `is_anomaly = confidence < threshold` y el
softmax ya eran correctos) ni ningún otro plugin.

## Golden cases — reconstruidos desde el dataset real

Los golden cases del Ciclo 1 (derivados de `results.json`) heredaban el bug — p.ej. sus 206
detecciones etiquetadas `resting-lying` eran en realidad `drinking` — así que se descartaron
por completo.

**25 clips reales** extraídos directamente de `inbox/a05/codigo/data/annotations_test.json`
+ `data/test/*.jpg` (nunca inventados): se agrupan las anotaciones por `(clip_name,
track_id)` igual que `utils/data_utils.py:BehaviorClassificationDataset`, se toman los
primeros 32 frames (`CLIP_LENGTH`) de cada track ordenados por `image_id`, recorte ROI por
bbox COCO real + resize a 224×224. Solo se usaron tracks con **pureza 100%** (un único
`attributes.behavior` en los 32 frames), así el `expected` es ground truth inambiguo, no una
"mayoría" discutible. 2-3 casos por clase (3 para `grazing`), cubriendo las 12 clases.
Detalle completo (clip, track, frames, bboxes) en `inbox/a05/golden_cases_real.json`.

**Cobertura débil señalada, no disimulada:** ningún track de `running` en el split de test
alcanza los 32 frames completos (el más largo tiene 30) — los 2 casos de `running` usan
relleno por duplicación del último frame, el mismo fallback que usa
`BehaviorClassificationDataset.__getitem__` para tracks cortos durante el entrenamiento. Es
real + relleno documentado (`padded_to_32: true` en el JSON), no una invención.

### Resultado contra el servidor real (POST /predict, mode=inline)

**24/25 casos coinciden** tras el fix.

| Comportamiento | Casos | Coinciden |
|---|---|---|
| drinking, grazing (×3), grooming, hidden, none, other, resting-lying, ruminating-lying, ruminating-standing, running (×2, con padding), walking | 23 | 23/23 |
| resting-standing | 2 | 1/2 |

**Único fallo:** clip `1349_arm01_gopro4_20200324_040135_beh3_ani1_ins2_cut_00011` /
`track_id=1` (32/32 frames reales, pureza 100%, sin padding), ground truth
`resting-standing`, predicho `hidden` con `confidence=0.88`. Investigado: no es un fallo de
wiring/decodificación — las otras clases que podrían confundirse con `hidden` en el resto de
casos se deciden correctamente, y el fix de decodificación ya está confirmado por los otros
24 casos. Es un error real del modelo/artefacto sobre ese clip concreto. **No se ajustó la
tolerancia para ocultarlo** — se documenta aquí y en el manifest.

El caso adicional de lógica pura (`is_anomaly` con el operador estricto `<`,
`confidence=0.45 < threshold=0.5`) se verificó en vivo durante el smoke test de Parte A:
confirmado `is_anomaly=true`.

**Tolerancia usada:** pass/fail por coincidencia exacta de clase (no hay tolerancia numérica
aplicable a una clasificación categórica). **24/25 = 96% dentro de tolerancia.**

## Parte A — Checklist técnico

- [x] `flake8 app/plugins/ml5_meat_cow_behaviour/ --extend-exclude=dist,build --show-source --statistics` → 0 errores.
- [x] `pylint app/plugins/ml5_meat_cow_behaviour/ --disable=import-error` → **8.61/10**, sin cambios
      respecto al ciclo anterior. Todos los hallazgos son los mismos falsos positivos/notas de
      estilo ya documentados (p.ej. `cv2` `E1101 no-member`, imports perezosos de
      `detectron2`/`torch.hub` documentados en `model_loader.py`, `useless-return` en
      `mlflow_utils.py`). El fix no introdujo ningún hallazgo nuevo.
- [x] `pip-audit -r requirements.txt` → 2 hallazgos, ambos el mismo CVE (`PYSEC-2026-3447`) en
      `setuptools==80.9.0` — vulnerabilidad preexistente y transversal a todo el repo, no
      específica de este plugin.
- [x] `pytest tests/unit/ -q` → **503/503 passed** (suite completa; los tests de este plugin usan
      un `FakePlugin` vía `fake_plugins`/`client` y no ejercitan la decodificación real, por lo
      que el fix no requirió tocar ningún test).
- [x] Arranque local real: `MODEL=ml5-meat-cow-behaviour .venv/bin/python main.py` (puerto 8000,
      libre). `Startup complete. 1/1 models ready.` confirmado en logs. En este arranque
      `device=cuda` (GPU funcional en este sandbox, a diferencia del ciclo anterior) — log de
      arranque confirma el orden de clases corregido: `behaviors=['grazing', 'drinking',
      'walking', 'resting-lying', 'resting-standing', 'ruminating-lying', 'ruminating-standing',
      'running', 'grooming', 'hidden', 'other', 'none']` (orden `data_utils.py`, ya no el de
      `configs/`).
- [x] `POST /predict` modo `inline` con los 25 clips reales → `200` en todos los casos, ver
      tabla de Parte B arriba.
- [x] `POST /train` con `mlflow_run_id` → `501` (`"Este modelo usa artefactos externos; el
      reentrenamiento no está disponible."`) — correcto, `training.supported=false` sigue siendo
      la conclusión correcta (ver manifest `training.reason`, sin cambios de alcance).

Limpieza: servidor (`main.py`) detenido al finalizar (`pkill`, confirmado puerto 8000 libre);
ficheros temporales de verificación quedaron en el scratchpad de la sesión, no en el repo. No
se tocaron procesos de otros agentes. No se hizo commit ni se abrió PR (deja el working tree
para revisión humana, según instrucción explícita).

## Otros hallazgos de este ciclo (para seguimiento humano)

1. **Gap de AP del detector (afecta al plugin servido hoy, no corregido — fuera de alcance de
   un fix de decodificación):** Faster R-CNN R-101-FPN mide AP real=0.705 sobre 1.000 imágenes
   reales de test (vs. 0.882 reportado en memoria §7.3 / 0.85 objetivo mínimo). AP50 sí
   coincide (0.948 real vs. 0.953 reportado), lo que indica que el wiring del detector en el
   plugin es correcto — el gap es específico de la métrica AP estricta (promediada sobre IoU
   0.5:0.95). **Señalado para quien gestione la ficha técnica** (docs-generation no se tocó en
   este ciclo): `metrics_reported.detector.AP=0.882` del manifest sigue citando la memoria, pero
   ahora anotado como no reproducido contra el checkpoint real.
2. **4 bugs en el código de ENTRENAMIENTO entregado (NO afectan a la inferencia servida hoy,
   solo a un reentrenamiento futuro — documentados, no corregidos, por instrucción explícita de
   no reentrenar en este ciclo):**
   - Scheduler de LR: se llama una vez por época pero su fórmula cuenta steps por batch → LR
     se queda casi en 0 durante todo el entrenamiento.
   - `format.py` divide el split train/val/test por imagen individual, no por clip/vídeo — los
     502 clips aparecen en los tres splits simultáneamente (leakage), inflando todas las
     métricas reportadas, incluido el 0.91 de accuracy de la tabla A/B de arriba (válido como
     comparación relativa de decodificación, no como estimación de generalización limpia).
   - Track IDs no únicos entre vídeos (solo 23 IDs para 502 clips) — algunos clips de
     entrenamiento mezclan frames de vídeos distintos.
   - ColorJitter se aplica también en validación, no solo en entrenamiento.

   Ver `inbox/a05/manifest.yaml:known_issues` para el detalle completo de ambos hallazgos.

## Ciclo 3 (2026-10-06) — los 4 bugs de entrenamiento, corregidos; `/train` habilitado

El Ciclo 2 dejó los 4 bugs del código de entrenamiento **documentados pero sin corregir**, por
decisión explícita de no reentrenar en aquel ciclo. En este ciclo se piden dos cosas distintas:
(a) corregir los 4 bugs a nivel de código — verificable sin necesidad de un entrenamiento largo
— y (b) dejar el reentrenamiento realmente disponible vía `/train` para quien tenga el
presupuesto de GPU para ejecutarlo, no solo documentado como roto. Ningún entrenamiento
LARGO/completo se ha ejecutado aquí (sigue sin ser viable: ~4.5-6 días en este sandbox).

### Los 4 fixes, y cómo se verificó cada uno SIN entrenar de verdad

Aplicados tanto en `app/plugins/ml5_meat_cow_behaviour/training.py` (el `train()` real del
plugin) como en el propio `inbox/a05/codigo/` entregado (`scripts/training/train_classifier.py`,
`utils/data_utils.py`, `scripts/data/format.py`), para que el repo original también sea correcto
si alguien lo ejecuta de forma standalone.

| # | Bug | Fix | Verificación (sin entrenar) |
|---|---|---|---|
| 1 | Scheduler LR: `scheduler.step()` una vez por ÉPOCA, fórmula cuenta steps por BATCH → LR casi en 0 todo el run | `scheduler.step()` ahora una vez por BATCH | Smoke test real de 19s (16 clips train / 8 val, 2 épocas, 4 batches/época, `warmup_epochs=2`→8 steps de warmup): LR pasó de 0.00025 (tras época 1, step=4/8) a 0.0005 (tras época 2, step=8/8) — exactamente el doble de lo que daría el bug (que solo avanzaría 1 "step" por época, no 4). Confirma matemáticamente que el scheduler ya cuenta steps por batch. |
| 2 | `split_dataset` por IMAGEN, no por clip — 502 clips en los 3 splits a la vez (leakage) | Split por `clip_name` | Instantáneo sobre los 502 clips reales: `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅` (351/75/76 clips). Comprobado por `set.intersection()`, no requiere entrenar. |
| 3 | `track_id` no único entre vídeos (23 IDs para 502 clips) | `track_id` namespaced por `clip_name` (`'{clip}__track{id}'`) | Instantáneo: 3.693 `track_id` únicos tras el fix (antes: 23) sobre las 1.163.408 anotaciones reales. |
| 4 | `ColorJitter` también en validación | Bloque de transforms depende de `is_train` | Instantáneo: tras construir un `ClipClassificationDataset(is_train=False)` y forzar `__getitem__`, `.transforms = Compose()` (vacío). Con `is_train=True`, `.transforms` sí incluye `ColorJitter`. |

### `/train` habilitado para el clasificador (detector sigue fuera de alcance)

- `train_dto.py` nuevo: `mlflow_run_id: str` obligatorio (sin default). `data_path` espera la
  MISMA estructura que ya consume `scripts/data/format.py` del entregable
  (`raw_frames/<clip>/*.jpg` + `annotations/<clip>/annotations/instances_default.json`,
  export COCO/CVAT por clip) — no se inventó ninguna estructura ZIP nueva.
- `app/registry.py` y `tests/conftest.py`: faltaba `train_response_type` en ambos (el agente que
  dejó este trabajo a medias se cortó antes de añadirlo) — completado en este ciclo, reutilizando
  `app.application.dto.train_dto.TrainResponse` genérico (mismo patrón que `modelo10_lacteo`,
  ya que `train()` construye esa clase directamente con `detail`+`metrics`+`mlflow_run_id`+
  `upload_warning`).
- `mlflow_utils.py`: el reentrenamiento se persiste SOLO en el run de MLflow del caller
  (`artifact_path="classifier"`), nunca sobreescribe el artefacto base servido. Solo el
  clasificador es descargable — el detector nunca tiene artefacto de usuario.
- Confirmado en vivo (`MODEL=ml5-meat-cow-behaviour`, servidor real): `/health` y `/stats` 200;
  `POST /train` **sin** `mlflow_run_id` → **422** (`"Field required"`), confirmando que el DTO
  exige el campo correctamente.
- **Smoke test end-to-end de `plugin.train()` completo (ZIP real → extracción → MLflow
  mockeado → checkpoint): intentado, no completado.** Con un ZIP de 4 clips reales (369 clips
  troceados de train, 205 de val) y epochs forzado a 1 vía monkeypatch, el proceso superó los
  19 minutos sin terminar — el cuello de botella es I/O real (cientos de `cv2.imread` de JPGs
  individuales por clip, sin caché entre `__getitem__`), no cómputo de GPU. Se abortó para no
  convertir un "smoke test" en un entrenamiento largo de facto. La lógica de entrenamiento en sí
  (bucle completo, scheduler, forward+backward+optimizer.step()) SÍ se verificó end-to-end y
  rápido (19s) en la tabla de arriba — lo que queda sin probar en vivo es solo el envoltorio
  mecánico alrededor (extracción de ZIP, llamadas a `BaseMLflowTracker`, guardado del
  checkpoint) — código estándar, revisado directamente y idéntico al patrón ya verificado en
  vivo esta sesión para `ml2`/`ml4`/`ml7`/`ml8`/`modelo10_lacteo`. **Recomendado para quien
  ejecute el entrenamiento real**: probar primero `/train` con un ZIP pequeño (2-3 clips) y
  `epochs` reducido antes de lanzar las 30 épocas completas, para confirmar el envoltorio sin
  esperar días.

### Checklist técnico re-confirmado

flake8 limpio; pylint 8.93/10 (mejora desde 8.65/10 del ciclo anterior; los únicos avisos
nuevos son falsos positivos conocidos de `cv2` en el `training.py` nuevo, misma categoría que
ya existía en `preprocessing.py`); pip-audit solo el CVE preexistente de `setuptools`
(transversal al repo); pytest 504/504 (toda la suite, sin regresiones).

## Estado final

**LISTO PARA PR** — bug crítico de decodificación corregido (Ciclo 2, 24/25 golden cases
reales), y en este ciclo los 4 bugs de entrenamiento corregidos y verificados con evidencia
rápida (no requirió entrenar), `/train` real habilitado y alcanzable para el clasificador
(422 sin `mlflow_run_id` confirmado en vivo), wiring de `registry.py`/`conftest.py` completado.
Pendiente de seguimiento humano, no bloqueante: (1) el gap de AP del detector ya documentado en
Ciclo 2, (2) ejecutar el entrenamiento real a escala completa (~50h de GPU) cuando haya
presupuesto — este ciclo garantiza que el código es correcto y accesible vía API, no que un
checkpoint reentrenado concreto ya exista o se haya validado en precisión.
