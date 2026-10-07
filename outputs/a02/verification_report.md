# Verificación — ml2-fungal-cnn-disease-detection

> **[CORREGIDO 2026-10-07]** El patrón descrito más abajo para usar el modelo del usuario (sustituir `self._bundle` en `self` durante la petición y restaurarlo en `finally`) **no era seguro con peticiones concurrentes**: hay una sola instancia del plugin, así que una petición sin `mlflow_run_id` podía recibir el modelo de otro usuario, y dos peticiones solapadas podían dejar el de un usuario como modelo base. Se ha cambiado para resolver el modelo en variables locales, sin tocar `self` (rama `fix/retrain-mlflow-only-persistence`, cubierto por `tests/unit/test_user_model_isolation.py` y `test_user_model_download_hygiene.py`). Además, si el run pedido no tiene un modelo cargable, la petición falla con 422 en lugar de usar el modelo base. El texto de abajo se conserva como histórico.

## Contexto

Este plugin ya estaba integrado y registrado en `app/registry.py` antes de esta sesión,
pero nunca había pasado por `manifest-extraction` ni `verification`. Se generó
`inbox/a02/manifest.yaml` a partir de `inbox/a02/codigo/` (código real entregado) e
`inbox/a02/entregable/*.docx` (memoria), y se auditó el plugin ya integrado contra ambos.

## Actualización (2026-10-05) — datos reales ya disponibles + GPU real

Dos cosas cambiaron desde la primera verificación de este plugin (ver secciones
originales más abajo, que se conservan sin editar por trazabilidad):

1. **El equipo de IA añadió los datasets reales** a `inbox/a02/codigo/` que antes solo
   tenían un README apuntando a descargas externas: `data/raw/datasets/{NGLD, Grape Plant
   from Plant Village Dataset}/`, `data/processed/Images_256/<clase>/` (~12.067 imágenes,
   5 clases) y, sobre todo, `data/splits/{train,val,test}.csv` ya generados por el pipeline
   original (test.csv: 528 filas — black_rot=118, downy_mildew=97, healthy=135,
   powdery_mildew=40, trunk_disease=138). Esto permite por fin construir golden cases
   reales (ver "Correctitud" más abajo, que sustituye la sección "0 casos" anterior).
2. **La GPU del sandbox cambió** de una GTX 1060 (CUDA no funcional, forzaba CPU) a una
   RTX 2070 SUPER real con CUDA funcional (`torch.cuda.is_available()==True`,
   `torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 2070 SUPER'`). Esto expuso un bug
   real en `_safe_device()` que nunca se había podido ejercitar contra CUDA de verdad —
   ver "Bug nuevo encontrado" más abajo.

## Checklist técnico

**Re-ejecutado el 2026-10-05 contra el código actual (con el fix de `_safe_device()` ya
aplicado) y manteniendo el checklist original (histórico) debajo sin editar.**

- [x] flake8 (`app/plugins/ml2_fungal_cnn_disease_detection/`): 0 errores (exit 0).
- [x] pytest `tests/unit/` (suite completa): **503/503 passed** (77.95s). No se rompió
  ningún test existente de otros plugins ni de los agentes que están reverificando ml4/ml8
  en paralelo.
- [x] pylint `app/plugins/ml2_fungal_cnn_disease_detection/plugin.py --disable=import-error`:
  **9.74/10** (idéntico al run anterior, "+0.00" respecto al run previo — el fix de
  `_safe_device()` vive en `model_loader.py` y no tocó `plugin.py`).
  `model_loader.py` solo: 9.62/10, con los mismos 2 warnings preexistentes
  (`R0402 consider-using-from-import`, `W0718 broad-exception-caught` en el propio
  `except Exception` de `_safe_device()`) que ya estaban antes del fix de 2 líneas — el fix
  no introdujo warnings nuevos. Los `E1101` de `postprocessing.py` (cv2/PIL) siguen siendo
  el mismo patrón preexistente ya documentado abajo.
- [x] pip-audit `-r requirements.txt`: mismo único aviso preexistente y no relacionado
  (`setuptools 80.9.0`, `PYSEC-2026-3447`).
- [x] Arranque local (`MODEL=ml2-fungal-cnn-disease-detection uvicorn main:app --port 8020`,
  puerto no estándar para no chocar con los agentes que están reverificando ml4 (puerto
  8010) y otro modelo en el puerto 8000 por defecto en paralelo en este mismo sandbox):
  log de arranque confirma **`device=cuda`** explícitamente
  (`Ml2FungalCnnDiseaseDetectionPlugin bundle ready — device=cuda`). `/health`: `{"status":
  "ok","model":"ml2-fungal-cnn-disease-detection","version":"1.0.0","loaded":true}`.
  `/stats`: 200 con el esquema completo. `/predict` inline (18/18 casos reales, ver abajo),
  batch (3 imágenes reales con subcarpeta ZIP, predicciones + heatmap CAM generado
  correctamente con tensores en CUDA) y caso de imagen inválida → 422 confirmado.
- [x] `/train` vía HTTP sin `mlflow_run_id` → 422 confirmado otra vez contra el servidor
  real. Servidor (PID 483013, puerto 8020) detenido limpiamente al terminar — comprobado
  que no quedó ningún proceso vivo tras el `kill`.
- [x] `/train` — **ejecución real en proceso sobre CUDA** (no vía HTTP, por el mismo motivo
  ya documentado antes: `MLFLOW_TRACKING_URI` no está seteado en `.env` y el tracker real
  apunta a un host k8s inalcanzable desde este sandbox, así que se monkeypatchea
  `BaseMLflowTracker` a un stub no-op, exactamente como en la verificación anterior — el
  resto de la ejecución es 100% código real sin mocks). Esta vez con un ZIP de **imágenes
  reales del dataset** (30 imágenes: 10 por clase de black_rot/healthy/trunk_disease,
  copiadas de `data/processed/Images_256/`, no sintéticas) en vez del smoke test sintético
  anterior. Se instrumentó un spy sobre `model_loader._safe_device()` para capturar cada
  valor devuelto durante `load()` + `train()`: **las 2 llamadas devolvieron `cuda`** — cero
  fallback a CPU. El entrenamiento corrió 4 épocas (max_epochs/patience reducidos solo para
  esta prueba de humo de dispositivo, vía monkeypatch de las constantes ya importadas en
  `plugin.py`, nunca tocando `constants.py` real) con forward+backward+optimizer.step() en
  GPU sin ningún error de dtype/device, terminando con accuracy=0.8333/f1=0.8222 sobre el
  split de validación interno (6 imágenes) — de nuevo, es una prueba de wiring/dispositivo,
  NO un golden case de accuracy.

### Checklist técnico — histórico (primera verificación, antes de tener datos reales/GPU)
- [x] flake8 (`app/plugins/ml2_fungal_cnn_disease_detection/`, `tests/conftest.py`,
  `tests/unit/test_ml2_fungal_cnn_disease_detection.py`, `app/registry.py`): 0 errores.
- [x] pytest `tests/unit/` (suite completa): 503/503 passed (502 antes de añadir el test
  `test_train_without_mlflow_run_id_returns_422`, que se suma al añadir cobertura real de
  `/train`). No se rompió ningún test existente de otros plugins.
- [x] pylint `app/plugins/ml2_fungal_cnn_disease_detection/ --disable=import-error`:
  9.74/10 en `plugin.py` tras el fix (sin warnings nuevos de complejidad: se añadieron
  `# pylint: disable=too-many-locals,too-many-branches,too-many-statements` al nuevo
  `train()`, igual que ya hace `ml30_meat_traceability_detection/plugin.py` en su propio
  `train()`). Los `broad-exception-caught`/`import-outside-toplevel`/`E1101` restantes en
  `postprocessing.py`/`model_loader.py` ya existían antes de esta sesión (CAM/cv2/PIL,
  patrón idéntico en otros plugins de imagen del repo) — no son nuevos.
- [x] pip-audit `-r requirements.txt`: 1 paquete con aviso (`setuptools 80.9.0`,
  `PYSEC-2026-3447`), preexistente y no relacionado con este plugin (ninguna dependencia de
  torch/torchvision/pillow/opencv tiene CVEs).
- [x] Arranque local (`MODEL=ml2-fungal-cnn-disease-detection ./.venv/bin/python main.py`):
  "Startup complete. 1/1 models ready." Artefacto `leafcnn_best.pth` descargado de S3 sin
  errores. `/health`: `{"status":"ok","loaded":true}`. `/stats`: 200 con el esquema
  completo. `/predict` inline y batch: 200, sin 500, con un caso de imagen inválida
  verificado como 422 (`InvalidImageError` → `extra_predict_exceptions`).
- [x] `/train`: `training.supported=true` en el manifest → **200 con métricas**, no 501.
  Sin `mlflow_run_id` → 422 (`Field required`), confirmando que el campo es obligatorio.
  Servidor y procesos de verificación (PIDs 418260/418654, incl. su
  `multiprocessing.spawn` hijo) detenidos al terminar — comprobado que no quedó ninguno
  vivo.

## Bugs encontrados y corregidos en el plugin ya integrado

0. **Nuevo (2026-10-05) — `_safe_device()` siempre fuerza CPU, incluso con CUDA real y
   funcional.** Solo detectable ahora que el sandbox tiene una RTX 2070 SUPER real
   (antes tenía una GTX 1060 con CUDA ya no-funcional por otro motivo, así que el bug
   quedaba enmascarado — el resultado final, CPU, era "casualmente" el mismo).
   `model_loader.py:_safe_device()` probaba la disponibilidad real de CUDA así:
   `torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())` — el `Conv2d` de prueba se
   crea en CPU (dispositivo por defecto) y nunca se mueve a CUDA, pero se le pasa un tensor
   de entrada que sí está en `.cuda()`. Esto **siempre** lanza
   `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
   should be the same`, que el `except Exception` interpretaba como "CUDA detectada pero no
   funcional", forzando CPU incondicionalmente sin importar si la GPU funcionaba de verdad.
   Confirmado reproduciendo la excepción exacta de forma aislada antes de tocar nada.
   Corregido moviendo también el módulo de prueba a CUDA:
   `probe = torch.nn.Conv2d(1, 1, 1).cuda(); probe(torch.zeros(1, 1, 4, 4).cuda())`.
   Verificado end-to-end tras el fix (ver checklist técnico arriba): `load_model_bundle()`
   y `train()` resuelven ambos a `cuda` (instrumentado con un spy sobre `_safe_device()` que
   capturó las 2 llamadas reales), el modelo y los tensores de entrada/logits quedan en
   `cuda:0`, y tanto `/predict` (inline+batch, incluyendo el CAM de `postprocessing.py`, que
   ya movía explícitamente a `.cpu()` antes de `numpy()` — sin cambios necesarios ahí) como
   `train()` (forward+backward+optimizer.step() sobre tensores CUDA) corren sin ningún error
   de dtype/device.
   **Mismo patrón de `_safe_device()` (bug idéntico, NO corregido aquí — fuera de alcance de
   esta sesión)** está presente en `ml4_lactic_cnn_thermal_early_disease_detection`,
   `ml5_meat_cow_behaviour`, `ml8_cereals_img_anomaly_detector` y
   `ml41_meat_curing_machinery_acoustic_anomaly/model_loader.py` — se señala para quien
   verifique esos modelos (ml4 y ml8 ya están siendo reverificados en paralelo por otros
   agentes en esta misma sesión).

1. **Clase 3 — `train()` marcado como no soportado sin respaldo del manifest.**
   `plugin.py` lanzaba `TrainingNotSupportedError` (501) incondicionalmente. El código
   entregado (`inbox/a02/codigo/.../src/training/train.py` + `src/training/model.py` +
   `config/config.py`) contiene un procedimiento de entrenamiento real, completo y
   reproducible (AdamW, lr=0.001065, weight_decay=1e-6, ReduceLROnPlateau(0.3,3),
   CrossEntropyLoss, batch=16, hasta 100 épocas con early stopping patience=10) que el
   equipo de IA ejecutó para producir `leafcnn_best.pth`. La memoria (P102) solo dice que
   reentrenar NO es necesario para usar el sistema porque ya se entrega el artefacto — no
   que esté deshabilitado — y P303 recomienda explícitamente reentrenar con imágenes del
   cliente para un despliegue fiable. Corregido: `training.supported=true` en el manifest
   y `train()` reimplementado fielmente (mismo optimizador/scheduler/loss/hiperparámetros,
   misma inicialización de pesos vía `model_loader.create_model`), aceptando un ZIP con
   subcarpetas por clase (mismo patrón de `modelo10_lacteo`/`ml8_cereals_img_anomaly_detector`,
   ya que no hay forma de pasar imágenes por un CSV). `mlflow_run_id` es obligatorio
   (sin default) en `train_dto.TrainRequest`, y el checkpoint reentrenado se sube
   ÚNICAMENTE al run de MLflow indicado (`artifact_path="model"`) — nunca se escribe al
   artefacto base S3/local `leafcnn_best.pth`.
   Verificado end-to-end: `plugin.train()` ejecutado directamente (con `BaseMLflowTracker`
   monkeypatcheado a un stub no-op, ya que `MLFLOW_TRACKING_URI` no está seteado en `.env`
   y apunta por defecto a un host k8s inalcanzable desde este sandbox — mismo workaround ya
   documentado para otros modelos) sobre un ZIP sintético de 3 clases / 8 imágenes por
   clase: entrena 39 épocas con early stopping, accuracy=1.0/f1=1.0 en el split de
   validación (trivialmente separable por diseño, es un smoke test de wiring, NO un golden
   case), y sube un checkpoint `leafcnn_best.pth` con `classes=["black_rot","healthy",
   "powdery_mildew"]` al directorio simulado de MLflow.
   Vía HTTP: `/train` sin `mlflow_run_id` → 422 confirmado contra el servidor real.

2. **Clase 1 (bug de otros plugins, ausente aquí) — comprobado explícitamente que NO
   aplica.** `predict_inline`/`predict_batch` ya devolvían `PredictInlineResponse(...)` /
   `PredictBatchResponse(...)` tipados, nunca un `dict` crudo. Confirmado leyendo el
   `return` real y ejecutándolo contra el servidor vivo (no solo contra `FakePlugin`).

3. **Bug de "discard" de `mlflow_run_id` (mismo patrón que el bug ya conocido de ml17) —
   estaba presente aquí también.** `predict_inline`/`predict_batch`/`stats` llamaban a
   `download_user_model_from_mlflow(mlflow_run_id)` pero descartaban el resultado sin
   usarlo nunca — además `mlflow_utils.py` devolvía siempre `None` (el plugin no soportaba
   reentrenamiento, así que no había nada que descargar). Corregido: `mlflow_utils.py`
   ahora descarga de verdad el checkpoint desde el run de MLflow, reconstruye `LeafCNN`
   dinámicamente con el número de clases del propio checkpoint (no asume
   `constants.CLASS_NAMES`, por si un reentrenamiento futuro cambia las clases), y
   `predict_inline`/`predict_batch`/`stats` intercambian `self._bundle` por el bundle del
   usuario con guardado/restauración (`saved_bundle = self._bundle` ... `finally:
   self._bundle = saved_bundle`) y limpian el directorio temporal — mismo patrón ya probado
   en `ml8_cereals_img_anomaly_detector` y `ml25_wine_sulphites`. Verificado con un
   monkeypatch de `download_user_model_from_mlflow`: el `model_id` devuelto refleja el
   bundle de usuario cuando se pasa `mlflow_run_id`, y el bundle base se restaura
   exactamente después de la llamada, con el directorio temporal eliminado.

4. **Preprocesado de imagen — verificado línea a línea, SIN divergencias.**
   `preprocessing.py` (`Resize((224,224))`, `ToTensor()`, `Normalize(0.5,0.5,0.5)`)
   coincide exactamente con `config/config.py:VAL_TRANSFORMS` del código entregado. El
   orden de clases (`CLASS_NAMES = ["black_rot","downy_mildew","healthy","powdery_mildew",
   "trunk_disease"]`) coincide con `sorted(df["label"].unique())` tal y como construye las
   clases el código original (`src/predict/predict.py:load_model`,
   `src/training/train.py:run_training`), confirmado también contra el mapeo real de
   carpetas→etiqueta en `src/data_processing/load_data.py` (`FOLDER_TO_CLASS_PLANT_VILLAGE`
   / `NGLD_MAP`). No se encontró ningún bug de clase 2 en este plugin.

## Correctitud (golden dataset)

**ACTUALIZADO (2026-10-05) — 18/18 casos reales dentro de tolerancia.** La limitación
histórica de abajo ya no aplica: `inbox/a02/codigo/data/splits/test.csv` (528 filas,
columnas `path,label`, generado por el pipeline de entrenamiento original) es un split de
test real con las 5 clases representadas. Se seleccionaron 18 imágenes
(`random.seed(42)`, estratificadas: 4 black_rot, 3 downy_mildew, 4 healthy,
3 powdery_mildew, 4 trunk_disease — proporcional al tamaño de cada clase en el test set) y
se verificó que las 18 rutas existen en disco. `expected` se generó ejecutando el
checkpoint real (`artifacts/ml2_fungal_cnn_disease_detection/leafcnn_best.pth`) en proceso
con el mismo código de pre/postprocesado del plugin — nunca a mano.

| Caso | Esperado (pred/conf) | Obtenido vía HTTP (pred/conf) | Diferencia confianza | ¿OK? |
|---|---|---|---|---|
| caso_001_black_rot | black_rot / 0.999735 | black_rot / 0.999735 | 0 | ✅ |
| caso_002_black_rot | black_rot / 0.999979 | black_rot / 0.999979 | 0 | ✅ |
| caso_003_black_rot | black_rot / 0.999964 | black_rot / 0.999964 | 0 | ✅ |
| caso_004_black_rot | black_rot / 0.969956 | black_rot / 0.969956 | 0 | ✅ |
| caso_005_downy_mildew | downy_mildew / 0.989935 | downy_mildew / 0.989935 | 0 | ✅ |
| caso_006_downy_mildew | downy_mildew / 0.999940 | downy_mildew / 0.999940 | 0 | ✅ |
| caso_007_downy_mildew | downy_mildew / 0.999888 | downy_mildew / 0.999888 | 0 | ✅ |
| caso_008_healthy | healthy / 0.999701 | healthy / 0.999701 | 0 | ✅ |
| caso_009_healthy | healthy / 0.999681 | healthy / 0.999681 | 0 | ✅ |
| caso_010_healthy | healthy / 0.999975 | healthy / 0.999975 | 0 | ✅ |
| caso_011_healthy | healthy / 0.999244 | healthy / 0.999244 | 0 | ✅ |
| caso_012_powdery_mildew | powdery_mildew / 0.998776 | powdery_mildew / 0.998776 | 0 | ✅ |
| caso_013_powdery_mildew | powdery_mildew / 0.961269 | powdery_mildew / 0.961269 | 0 | ✅ |
| caso_014_powdery_mildew | powdery_mildew / 0.999012 | powdery_mildew / 0.999012 | 0 | ✅ |
| caso_015_trunk_disease | trunk_disease / 0.999995 | trunk_disease / 0.999995 | 0 | ✅ |
| caso_016_trunk_disease | trunk_disease / 0.984186 | trunk_disease / 0.984186 | 0 | ✅ |
| caso_017_trunk_disease | trunk_disease / 0.999808 | trunk_disease / 0.999808 | 0 | ✅ |
| caso_018_trunk_disease | trunk_disease / 0.999289 | trunk_disease / 0.999289 | 0 | ✅ |

Tolerancia usada: coincidencia exacta de `prediction` + diferencia de `confidence` < 1e-4
(no hay métrica de tolerancia relativa de la memoria aplicable a un clasificador — el
objetivo es detectar regresiones de *serving*, igual que el golden dataset está pensado:
`expected` ES la salida del modelo real, no la etiqueta del dataset).
**Resultado: 18/18 casos dentro de tolerancia**, con el servidor real corriendo sobre CUDA
(confirmado en el log de arranque: `device=cuda`).

Adicionalmente, y solo como dato de contexto (no es lo que mide la tolerancia de arriba):
las 18 predicciones del modelo **también coinciden con la etiqueta real del dataset**
(`ground_truth_label` en el manifest) — 18/18. Esto es una muestra pequeña y no sustituye
una medición de accuracy formal sobre las 528 filas del test set completo, pero es
consistente con el `accuracy_test≈0.98` reportado en la memoria (P198) y no contradice al
KPI de negocio (accuracy > 0.95, P40/P184).

Validación estructural adicional (igual que antes, sigue cumpliéndose):
- Clase predicha siempre ∈ las 5 clases esperadas.
- Probabilidades por clase suman 1.0 (softmax válido).
- Orden/nombres de clase coinciden exactamente con el código de entrenamiento original.
- Imagen inválida → 422 (`InvalidImageError`), no 500.
- `predict_batch` con ZIP de imágenes reales: mismas predicciones/confianzas que inline
  para las mismas imágenes, heatmap CAM generado correctamente (tensores CUDA → `.cpu()`
  antes de `numpy()`, sin errores).
- `/train` con imágenes reales (no sintéticas) sobre CUDA: entrena, converge, devuelve
  métricas — ver checklist técnico arriba para el detalle del spy de dispositivo.

### Correctitud — histórico (primera verificación, antes de tener datos reales)

0 golden cases ejecutados — limitación real de las entradas, documentada y no silenciada
en su momento. `inbox/a02/codigo/.../data/` solo contenía un `README.md` con instrucciones
para descargar externamente Kaggle Plant Village + NGLD; no se había entregado ninguna
imagen real ni ningún `data/splits/*.csv`. La memoria tampoco incluye una tabla de
escenarios tipo "Tabla 6" con pares imagen→resultado auditados (las Tablas 3/4/5 que
menciona el texto son capturas de pantalla embebidas — comparativa de arquitecturas,
hiperparámetros y folds de CV — no casos individuales). En su lugar, en aquel momento solo
se pudo verificar correctitud estructural (clase válida, softmax suma 1, orden de clases,
422 en imagen inválida, train sintético de wiring).

## Estado final

**LISTO PARA PR.** La salvedad de la verificación anterior (correctitud numérica contra un
golden dataset real pendiente de datos) queda resuelta: 18/18 golden cases reales dentro de
tolerancia contra el servidor real, y el path de CUDA (predict inline, predict batch con
CAM, y train) queda confirmado end-to-end sobre la GPU real del sandbox tras corregir el
bug de `_safe_device()` documentado arriba. Pendiente solo de revisión humana de este
informe y de `inbox/a02/manifest.yaml` antes de abrir PR — no se ha hecho merge ni se ha
abierto PR desde esta sesión.
