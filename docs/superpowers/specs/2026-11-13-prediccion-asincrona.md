# Predicción asíncrona (vídeo de modelo-5) — especificación

Fecha: 2026-11-13 · Estado: diseño · Alcance: solo código (servicio de modelos, orquestador y
plataforma). Patrón de referencia: `2026-10-09-reentrenamiento-asincrono.md`.

## Problema

La predicción por lotes de modelo-5 (plugin `ml5_meat_cow_behaviour`) recorre el vídeo entero con
detección + seguimiento + SlowFast: ~10 s por fotograma en CPU con 8 vacas, ~14 veces menos en GPU.
Un vídeo de pocos minutos ya son horas de CPU y decenas de minutos de GPU, y toda la cadena es
síncrona:

- **Servicio de modelos:** `POST /predict` no responde hasta acabar el vídeo.
- **Orquestador:** el worker espera con `MODEL_HTTP_TIMEOUT_SECONDS` (60 min) y, con
  `visibility_timeout: 3600`, una tarea de más de 1 h se reentrega y lanza otra predicción igual.
- **Plataforma:** `PredictTaskManager.pollUntilDone` abandona a los 65 min
  (`ORCH_TASK_TIMEOUT_MINUTES`), en memoria: un reinicio pierde el seguimiento.

Producción tiene GPU para este modelo; la vía asíncrona es por robustez y por experiencia de usuario
(el usuario lanza el vídeo, sigue trabajando y recibe un aviso al terminar).

## Diseño

Igual que el entrenamiento asíncrono, aplicado a la predicción por lotes y **activado por modelo**
(hoy solo modelo-5).

1. **Servicio de modelos** (capa común; ningún plugin cambia su `predict_batch`):
   - Opt-in por modelo: `ModelEntry(async_predict=True)` en `app/registry.py` (solo ml5).
   - `POST /predict?wait=false&job_id=<id>` con cuerpo `mode=batch` → `202` + `PredictJob`. Lanza
     `PredictModelUseCase(plugin).execute(body)` en un **proceso aparte** (`spawn`) que construye
     su propio plugin (`plugin_class()` + `load()`): mismo enrutado de argumentos y misma
     normalización JSON que el `/predict` síncrono, así el resultado es idéntico.
   - Sin `wait` (o `wait=true`) → síncrono, sin cambios (despliegue escalonado). `wait=false` en un
     modelo sin opt-in → `501` (el orquestador vuelve al síncrono). `wait=false` con `mode=inline`
     → se responde en síncrono (`200`), es instantáneo.
   - `GET /predict/jobs/{job_id}` → estado y, si `succeeded`, el resultado.
   - **Identidad:** a diferencia del entrenamiento, no hay un run de MLflow previo, así que
     `job_id` lo pone el cliente (el orquestador manda su `task_id`, estable ante reentregas y
     reenvíos de Celery); si falta, el servicio genera un UUID (sin idempotencia). Formato
     `[A-Za-z0-9._-]{1,128}` (va en un filtro de búsqueda de MLflow).
   - **Estado:** un run de MLflow por trabajo en el experimento `PREDICT_JOB_EXPERIMENT`
     (`datagia-async-predict`), `run_name = job_id` y etiquetas `async_predict.*` (la misma
     estructura que `async_train.*`, más `job_id`, `data_path` y `mlflow_run_id` del modelo de
     usuario). El `GET` localiza el run por la etiqueta `async_predict.job_id`. Sobrevive a
     reinicios y lo ve cualquier réplica. Al terminar, el run se cierra `FINISHED`/`FAILED`.
     No se usa el `mlflow_run_id` del cuerpo: ese es el modelo reentrenado del usuario (se usa para
     predecir, igual que en síncrono) y no debe recibir artefactos de cada predicción.
   - **Resultado:** artefacto `async_predict/result.json.gz` (JSON del `PredictResponse` en gzip).
     Se lee **descargando la carpeta** `async_predict`, nunca el fichero suelto (fallo del proxy S3
     corregido en 14fe870). Tamaño medido con el vídeo de prueba (64 fotogramas): ver «Tamaño».
   - **Idempotente** por `job_id`: un `POST` con un `job_id` existente devuelve el trabajo (202).
   - **Concurrencia:** `PREDICT_JOB_MAX_CONCURRENT` (1) predicciones asíncronas por modelo y pod;
     más → `409` (el orquestador reintenta). Un vídeo ocupa la GPU entera.
   - **Latido** `PREDICT_JOB_HEARTBEAT_S` (60 s); sin latido durante `PREDICT_JOB_STALE_AFTER_S`
     (600 s) → `failed` / `PredictJobLost`. Duración máxima `PREDICT_JOB_MAX_DURATION_S`
     (43200 s = 12 h): el pod que lo lanzó detiene el hijo y cualquier `GET` lo da por
     `failed` / `PredictJobTimeout`. Prioridad del hijo `PREDICT_JOB_NICE` (10).
   - El estado es monótono: un estado terminal no cambia nunca.
2. **Orquestador:** para los modelos de `PREDICT_ASYNC_MODEL_KEYS` (`modelo-5`) y `mode=batch`, la
   tarea `predict` hace `POST /predict?wait=false&job_id=<task_id>` y libera el worker;
   `poll_predict_job` consulta `GET /predict/jobs/{job_id}` y se reprograma con
   `apply_async(countdown=…)` hasta estado terminal. El contrato hacia la plataforma
   (`GET /api/ml/tasks/{id}`) es **idéntico** al de la predicción síncrona: en éxito
   `response = normalize_vendor_response_to_contract("predict", result)`; en fallo `error_code =
   EXECUTION_ERROR` y `error_message = MODEL_HTTP_<http_status>: {"detail":"…"}`, lo que habría
   escrito el síncrono. Solo los fallos propios del modo asíncrono llevan código nuevo
   (`PREDICT_JOB_LOST`, `PREDICT_JOB_TIMEOUT`, `PREDICT_TIMEOUT`, `PREDICT_POLL_FAILED`,
   `PREDICT_JOB_NOT_FOUND`). `PREDICT_ASYNC_ENABLED=false` vuelve al síncrono sin desplegar.
3. **Plataforma:** al lanzar una predicción de vídeo de modelo-5 se muestra
   «La predicción del vídeo se procesa en segundo plano. Puede tardar; te avisaremos por correo y en
   la plataforma cuando termine.». La tarea sigue al orquestador sin el límite de 65 min (hasta
   `PREDICT_TRACKING_MAX_HOURS`, 14 h, desde el lanzamiento, tolerando 60 consultas fallidas
   seguidas), guarda `{ orch_task_id, launched_at }` en `task.result` y, si la plataforma se
   reinicia, la tarea reasignada retoma el seguimiento en vez de relanzar el vídeo. Al terminar,
   bien o mal, manda correo al usuario que la lanzó (en su idioma) y la tarea queda en
   éxito/fallo en la lista de tareas, que es el aviso en la plataforma (el entrenamiento no tiene
   otro aviso en la aplicación). El límite de tiempo de la organización también la cierra con correo.

## Contrato HTTP del servicio de modelos

`POST /models/<model-id>/predict?wait=false[&job_id=<id>]` — cuerpo = el `PredictRequest` del plugin.

| Respuesta | Cuándo |
|---|---|
| `202` + `PredictJob` | Trabajo creado, o ya existía con ese `job_id` (idempotente) |
| `200` + `PredictResponse` | `mode=inline`: se predice en síncrono |
| `400` | `job_id` con formato no válido, o de otro modelo |
| `409` | Ya hay `PREDICT_JOB_MAX_CONCURRENT` predicciones asíncronas en curso del modelo en este pod |
| `422` | El cuerpo no valida el esquema |
| `501` | El modelo no admite predicción asíncrona |
| `503` | MLflow no accesible (reintentable) |
| `500` | Error inesperado |

`GET /models/<model-id>/predict/jobs/{job_id}`: `200` + `PredictJob`, `404` si no existe para este
modelo, `501` si el modelo no admite predicción asíncrona, `503` si MLflow no responde, `500` si un
`succeeded` no tiene `result.json.gz` legible.

`PredictJob`: los mismos campos que `TrainJob` (`job_id`, `model_id`, `status`, `submitted_at`,
`updated_at`, `result`, `error`, `error_type`, `http_status`). `result` es exactamente el cuerpo que
habría devuelto `POST /predict` síncrono. `http_status` es el código que habría dado el síncrono:
`ArchiveLimitExceededError` → 413, `UserModelUnavailableError` y las `extra_predict_exceptions` del
modelo (p. ej. `InvalidVideoError`) → 422, resto → 500; `PredictJobLost`/`PredictJobTimeout` → 500.

## Tamaño del resultado

Medido con el vídeo de prueba (`outputs/a05/datos_prueba/predict_batch_video.mp4`, 64 fotogramas,
8 vacas, GPU local): 512 detecciones, **112 KB de JSON (~1,75 KB por fotograma, ~220 B por
detección), 23 KB en gzip**; 25 s de predicción (~0,4 s/fotograma) y ~31 s el trabajo asíncrono
completo (arranque del hijo + `load()` + vídeo). Extrapolado, un vídeo de 10 min a 25 fps
(15 000 fotogramas) son ~1,7 h de GPU, ~26 MB de JSON y ~5,5 MB en gzip. El resultado viaja entero
(servicio → orquestador (Redis) → plataforma) igual que en síncrono: la vía asíncrona no lo agrava,
solo deja de cortarlo. En MLflow se guarda comprimido.

## Despliegue

Orden: servicio de modelos → orquestador → plataforma. Cada paso es compatible con el anterior:
un servicio antiguo ignora `wait=false` y responde `200` en síncrono (el orquestador lo acepta como
resultado final); un orquestador antiguo nunca manda `wait=false`; una plataforma antigua sigue
con su límite de 65 min. Requiere que el pod de ml5 tenga GPU con memoria para **dos** copias del
modelo (la que sirve y la del proceso hijo) y acceso de escritura a MLflow.

## Fuera de alcance

Cancelación, progreso por fotograma, colas de varios vídeos en el servicio, guardar el resultado
fuera de Redis/MongoDB (afecta igual al síncrono).
