# Reentrenamiento asíncrono desde la plataforma — especificación

Fecha: 2026-10-09 · Estado: diseño acordado con el usuario · Alcance: solo implementación en código
(la infraestructura del proyecto no es nuestra: nada de Jobs de Kubernetes, GPU ni overlays).

## Problema

Hoy el entrenamiento es síncrono en toda la cadena:

- **Servicio de modelos:** `POST /models/<id>/train` ejecuta `plugin.train()` dentro de la petición
  HTTP y no responde hasta terminar (`app/infrastructure/http/router_factory.py`).
- **Orquestador** (`retech-lote2-xai-orquestador`): el worker Celery espera esa respuesta con
  `MODEL_HTTP_TIMEOUT_SECONDS` (1320 s en pre-ops). Con `visibility_timeout: 3600`, una tarea de más de
  1 h se reentrega y lanzaría un segundo entrenamiento.
- **Plataforma** (`retech-lote2-xai-plataforma`): `TrainTaskManager.pollUntilDone` sondea cada 1,5 s
  en memoria y abandona a los 30 min (`ORCH_TRAIN_POLLING_TIMEOUT`); si la plataforma se reinicia, el
  seguimiento se pierde.

Entrenamientos medidos (p. ej. ml46 con telemetría minutal) ya superan los 22 min.

## Diseño

El entrenamiento pasa a ser un **trabajo en segundo plano**, identificado por `job_id = mlflow_run_id`
(el run que ya pre-crea la plataforma).

1. **Servicio de modelos** (capa común, ningún plugin cambia su `train()`):
   - `POST /train?wait=false` → `202` con el estado del trabajo. Lanza `plugin.train()` en un
     **proceso aparte** (multiprocessing `spawn`), que construye su propia instancia del plugin
     (`plugin_class()` + `load()`), así no comparte CPU/GIL ni memoria con las predicciones.
   - `POST /train` sin `wait` o con `wait=true` → comportamiento síncrono actual, sin cambios
     (compatibilidad durante el despliegue escalonado de los tres repos).
   - `GET /train/{job_id}` → estado del trabajo.
   - Estado persistido en **etiquetas del run de MLflow** (prefijo `async_train.`) y el resultado en el
     artefacto `async_train/result.json` del mismo run: sobrevive a reinicios y lo ve cualquier réplica.
   - **Idempotente:** un `POST` con un `job_id` que ya tiene trabajo devuelve ese trabajo (202) y no
     lanza otro (protege de reentregas de Celery).
   - **Un entrenamiento a la vez por modelo y pod:** si hay uno vivo → `409`.
   - **Latido:** el proceso hijo actualiza `async_train.updated_at` cada `TRAIN_JOB_HEARTBEAT_S`
     (60 s). Si un trabajo `queued`/`running` lleva más de `TRAIN_JOB_STALE_AFTER_S` (600 s) sin
     latir, `GET` lo da por `failed` con `error_type = "TrainJobLost"` y lo persiste.
2. **Orquestador:** la tarea `train` hace `POST /train?wait=false` (segundos) y libera el worker; una
   tarea de seguimiento consulta `GET /train/{job_id}` y se reprograma con `self.retry(countdown=…)`
   hasta estado terminal. El contrato hacia la plataforma (`GET /api/ml/tasks/{id}`) no cambia.
3. **Plataforma:** el seguimiento deja de vivir en memoria: la tarea de entrenamiento se persiste como
   "entrenando" y un barrido periódico revisa las abiertas. Sin límite fijo de 30 min (el fallo por
   inactividad lo decide el latido del servicio de modelos). Al terminar: actualiza la tarea y envía
   correo al usuario (`AlertMailer`, nuevo `sendTrainingFinishedMail`).

Piloto extremo a extremo: **modelo-47** (único con entrenamiento habilitado en la plataforma; hoy lo
sirve el plugin m48).

## Contrato HTTP del servicio de modelos (lo consumen orquestador y plataforma)

`POST /models/<model-id>/train?wait=false` — cuerpo = el `TrainRequest` del plugin (igual que hoy;
`mlflow_run_id` obligatorio y no vacío).

| Respuesta | Cuándo |
|---|---|
| `202` + `TrainJob` | Trabajo creado, o ya existía para ese `job_id` (idempotente) |
| `400` | `mlflow_run_id` vacío, o el run no existe en MLflow (422 de FastAPI si el cuerpo no valida el esquema) |
| `409` | Ya hay un entrenamiento en curso de ese modelo en este pod |
| `503` | MLflow no accesible al registrar el trabajo (reintentable) |
| `500` | Error inesperado del servicio |

`GET /models/<model-id>/train/{job_id}`

| Respuesta | Cuándo |
|---|---|
| `200` + `TrainJob` | Existe el trabajo |
| `404` | No hay trabajo con ese `job_id` para este modelo |
| `503` | MLflow no accesible |

`TrainJob` (JSON):

```json
{
  "job_id": "<mlflow_run_id>",
  "model_id": "m47-dnsl-fallas-maquinaria-pasteurizado",
  "status": "queued | running | succeeded | failed",
  "submitted_at": "2026-10-09T10:00:00Z",
  "updated_at": "2026-10-09T10:05:00Z",
  "result": { "...": "TrainResponse del plugin, solo si succeeded" },
  "error": "mensaje, solo si failed",
  "error_type": "ValueError | FileNotFoundError | TrainingNotSupportedError | TrainJobLost | …"
}
```

Mapeo recomendado de `error_type` en el consumidor (equivale a los códigos del modo síncrono):
`TrainingNotSupportedError` → "no soportado" (antes 501); `FileNotFoundError`/`ValueError` → error de
datos (antes 400); `TrainJobLost` → proceso perdido, reintentable; resto → error interno (antes 500).

**Reintentos:** el estado de un trabajo es monótono (un estado terminal no cambia nunca) y el `POST` es
idempotente por `job_id`: reenviar el mismo `job_id` devuelve el trabajo existente, también si acabó en
`failed`. Para reintentar un entrenamiento fallido o perdido hay que crear un run nuevo en MLflow y usar
su `mlflow_run_id` (la plataforma ya crea un run por cada entrenamiento que lanza).

## Requisito previo por modelo

El proceso hijo se descarta al terminar, así que ya no importa que `train()` recargue el modelo en
memoria. Sí importa que **no sobrescriba el artefacto fijo en disco** (lo compartiría con el proceso
que sirve predicciones). Ver rama `fix/retrain-mlflow-only-persistence` (PR14).

## Fuera de alcance

Jobs de Kubernetes, GPU, recursos de pods, cancelación de entrenamientos, colas de varios
entrenamientos por modelo.

## Planes

1. `docs/superpowers/plans/2026-10-09-train-async-servicio-modelos.md` (este repo) — fija el contrato.
2. Orquestador — pendiente, se escribe tras cerrar el 1.
3. Plataforma — pendiente, se escribe tras cerrar el 2.
