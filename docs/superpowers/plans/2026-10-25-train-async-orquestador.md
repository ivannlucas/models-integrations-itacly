# Reentrenamiento asíncrono — orquestador · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que la tarea `train` del orquestador lance el entrenamiento con `POST /train?wait=false`, libere el worker en segundos y siga el trabajo con una tarea Celery que se reprograma hasta un estado terminal, sin cambiar lo que ve la plataforma en `GET /api/ml/tasks/{id}`.

**Architecture:** dos casos de uso nuevos en `app/application/use_cases/async_train.py` (enviar y consultar) sobre un método nuevo del puerto `ModelClient` que devuelve el código HTTP en vez de lanzar. El worker (`app/worker/tasks.py`) usa el de enviar en `execute_model_task` cuando `action == "train"` y una tarea nueva `poll_train_job` que llama al de consultar y se reprograma con `self.retry(countdown=…)`. Los modelos que no admiten asíncrono (501, el grupo de modelo base versionado) o servicios aún no actualizados (200) se resuelven en la misma llamada, como hoy.

**Tech Stack:** Python 3.10, FastAPI, Celery 5.4 (broker Redis/Valkey), `requests`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-reentrenamiento-asincrono.md` (en `models-integrations-itacly`). Contrato del servicio de modelos ya implementado en la rama `feature/async-train` de ese repo.

**Repo de trabajo:** `C:\Users\PC\Documents\retech-lote2-xai-orquestador`, rama nueva `feature/async-train` desde `develop` (`7475e68`), en la carpeta principal, sin worktrees. Los 42 ficheros que `git status` marca como modificados en `develop` solo difieren en CRLF: no se tocan. **Los commits los hace el usuario:** el paso final de cada tarea es dejar los ficheros listos, no commitear.

## Global Constraints

- El contrato hacia la plataforma no cambia: `TaskRun.status` ∈ `queued | running | success | failed`; en éxito `response` es la salida de `normalize_vendor_response_to_contract("train", <TrainResponse>)`.
- Un fallo se guarda con `error_message` en el formato que ya produce `HttpModelClient`: `MODEL_HTTP_<status>: <cuerpo>`, con cuerpo `{"detail": "<texto>"}`. La plataforma (`friendlyOrchFailureHint`, `isInvalidDatasetError` en `train-task-manager.ts`) saca de ahí el mensaje para el usuario.
- Modelos de modelo base versionado (ml13–19, 21, 23, 24) siguen síncronos: el servicio responde `501` a `?wait=false` y el orquestador repite la llamada sin `wait`.
- Ajustes nuevos en `app/settings.py`, con estos nombres y valores por defecto: `TRAIN_ASYNC_ENABLED: bool = True`, `TRAIN_ASYNC_POLL_INTERVAL_SECONDS: int = 30`, `TRAIN_ASYNC_STATUS_TIMEOUT_SECONDS: int = 30`, `TRAIN_ASYNC_MAX_POLL_ERRORS: int = 40`.
- Nada de infraestructura: no se tocan `docker-compose.yml`, Dockerfile ni overlays de pre-ops.
- Orden de despliegue: servicio de modelos → orquestador → plataforma. Hasta que llegue el plan de la plataforma, esta sigue cortando a los 30 min; el orquestador termina igual la tarea.
- Tests con `python3 -m pytest -p no:flask -q tests` (el `pytest-flask` global del WSL secuestra la fixture `client`).

## Review Focus

1. **Servicio de modelos sin actualizar** (ignora `wait`, entrena síncrono y responde 200): debe acabar en `success` con el contrato, no quedarse esperando un `job_id`. Test en Task 2.
2. **Pod del modelo reiniciado a mitad** (conexión rechazada / 503 durante minutos, luego `failed` con `TrainJobLost`): los errores transitorios no deben fallar la tarea; el `TrainJobLost` sí. Tests en Task 3 y Task 4.
3. **Corte de Redis justo al guardar el resultado final** de un seguimiento: se reutiliza `_save_with_retry`; no se pierde el resultado. Test en Task 4.
4. **Reentrega de `execute_model_task` por Celery** (worker caído entre recibir y terminar): el `POST` es idempotente por `job_id` en el servicio, así que como mucho hay dos cadenas de seguimiento escribiendo el mismo estado terminal. Se acepta; sin test.
5. **Worker reiniciado con seguimientos pendientes:** los mensajes con `countdown` no confirmados vuelven tras el `visibility_timeout` (3600 s), así que el seguimiento se retoma con hasta 1 h de retraso. Se acepta y se documenta en el README (Task 4); sin test.

---

### Task 1: `ModelClient.send` — llamada HTTP que devuelve el código

**Files:**
- Modify: `app/domain/ports/model_client.py`
- Modify: `app/infrastructure/clients/http_model_client.py`
- Test: `tests/test_http_model_client.py` (nuevo)

**Interfaces:**
- Produces (en `app/domain/ports/model_client.py`):
  ```python
  @dataclass(frozen=True)
  class ModelHttpResponse:
      status_code: int
      body: dict          # JSON parseado; {"raw": text} si no es JSON
      text: str           # cuerpo tal cual, para los mensajes MODEL_HTTP_<status>

      @property
      def ok(self) -> bool: ...   # 200 <= status_code < 300

  class ModelClient(ABC):
      def call(...) -> dict: ...  # sin cambios
      @abstractmethod
      def send(self, *, base_url: str, endpoint: str, method: str,
               headers: dict[str, str], body: dict[str, Any] | None = None,
               params: dict[str, str] | None = None, timeout: float) -> ModelHttpResponse: ...
  ```
  `send` no lanza por códigos HTTP; los `requests.RequestException` (conexión, timeout) sí se propagan.

- [ ] **Step 1: Escribir los tests** (monkeypatch de `requests.post`/`requests.get` en `app.infrastructure.clients.http_model_client` con un objeto falso que tenga `status_code`, `ok`, `text`, `json()`):
  - `test_send_post_returns_status_and_json`: POST con `params={"wait": "false"}` → `ModelHttpResponse(202, {"job_id": "r1"}, ...)`; el falso recibió `params={"wait": "false"}`, `json=<body>` y `timeout=<el pasado>`.
  - `test_send_does_not_raise_on_http_errors`: 501 → `status_code == 501`, `ok is False`, `text` intacto.
  - `test_send_non_json_body_is_wrapped_as_raw`: `json()` lanza → `body == {"raw": "<text>"}`.
  - `test_send_get_passes_params_and_no_body`: GET → el falso de `requests.get` recibe `params` y no `json`.
  - `test_call_still_raises_model_http_prefix`: `call` con un 400 lanza `RuntimeError` cuyo texto empieza por `"MODEL_HTTP_400: "`; con 200 devuelve el JSON. (Fija el comportamiento actual antes de reescribirlo sobre `send`.)

- [ ] **Step 2: Ejecutar** `python3 -m pytest -p no:flask -q tests/test_http_model_client.py` → fallan los de `send` (no existe) y pasa el de `call`.

- [ ] **Step 3: Implementar** `ModelHttpResponse` y el método abstracto en el puerto; `HttpModelClient.send` con `requests.get(url, params=..., timeout=...)` o `requests.post(url, params=..., json=body or {}, timeout=...)`, y reescribir `call` sobre `send` (`timeout=settings.MODEL_HTTP_TIMEOUT_SECONDS`, lanza `RuntimeError(f"MODEL_HTTP_{status}: {text[:2000]}")` si `not ok`, devuelve `body`). Conservar el comentario del GET sobre `mlflow_run_id` como querystring.

- [ ] **Step 4: Ejecutar** el fichero de tests y la suite → todo en verde.

- [ ] **Step 5: Dejar listo para commit** (lo hace el usuario): `feat(train): ModelClient.send devuelve el código HTTP`.

---

### Task 2: caso de uso de envío — `SubmitTrainUseCase`

**Files:**
- Modify: `app/application/use_cases/execute_task.py` (extraer la resolución del modelo)
- Create: `app/application/use_cases/async_train.py`
- Modify: `app/settings.py` (los 4 ajustes de Global Constraints)
- Modify: `app/interfaces/http/dependencies/container.py` (`self.submit_train_use_case`)
- Test: `tests/test_async_train_submit.py` (nuevo) y `tests/fakes.py` (nuevo: `FakeModelRepo`, `FakeModelClient` con cola de respuestas y registro de llamadas)

**Interfaces:**
- Consumes: `ModelClient.send`, `ModelHttpResponse` (Task 1); `normalize_vendor_response_to_contract`.
- Produces:
  ```python
  # execute_task.py
  @dataclass(frozen=True)
  class ResolvedModelCall:
      base_url: str
      endpoint: str
      headers: dict[str, str]
  def resolve_model_call(model_repo: ModelRepo, model_key: str, action: str,
                         extra_headers: dict | None) -> ResolvedModelCall
  # Lanza los mismos ValueError que hoy: MODEL_NOT_FOUND_OR_DISABLED, MODEL_MISSING_BASE_URL,
  # ACTION_NOT_SUPPORTED:<action>. ExecuteTaskUseCase.execute pasa a usarla.

  # async_train.py
  @dataclass(frozen=True)
  class TrainSubmission:
      job_id: str | None = None     # trabajo asíncrono creado → hay que seguirlo
      contract: dict | None = None  # el modelo ya terminó (síncrono) → resultado final

  class SubmitTrainUseCase:
      def __init__(self, model_repo: ModelRepo, model_client: ModelClient,
                   execute_task_use_case: ExecuteTaskUseCase) -> None
      def execute(self, *, model_key: str, payload: dict,
                  extra_headers: dict | None) -> TrainSubmission
  ```

Comportamiento de `execute`:

| Situación | Resultado |
|---|---|
| `settings.TRAIN_ASYNC_ENABLED` es `False` | `execute_task_use_case.execute(action="train", …)` → `TrainSubmission(contract=…)`, sin llamada con `wait` |
| `send(POST, params={"wait": "false"}, timeout=MODEL_HTTP_TIMEOUT_SECONDS)` → 202 | `TrainSubmission(job_id=body["job_id"])`; sin `job_id` en el cuerpo → `RuntimeError("MODEL_HTTP_202: <text>")` |
| → 200 (servicio que ignora `wait`) | `TrainSubmission(contract=normalize("train", body))` |
| → 501 (modelo base versionado) | repite sin `wait` con `execute_task_use_case` → `TrainSubmission(contract=…)` |
| → cualquier otro código | `RuntimeError(f"MODEL_HTTP_{status}: {text[:2000]}")` |

- [ ] **Step 1: Escribir los tests** con `FakeModelRepo` (un `ModelDefinition` con `base_url="http://m/models/x"`, endpoints por defecto) y `FakeModelClient`:
  - `test_202_returns_job_id`: `execute(...) == TrainSubmission(job_id="run-1")`; la única llamada fue `send` con `method="POST"`, `endpoint="/train"`, `params={"wait": "false"}`, `body=<payload>`.
  - `test_200_from_a_service_without_async_is_the_final_contract`: el cuerpo `{"mlflow_run_id": "run-1", "val_f1": 0.9}` → `contract["schema"] == "datagia.model_evaluation.v1"` y `contract["mlflow_run_id"] == "run-1"`; `job_id is None`.
  - `test_501_falls_back_to_a_synchronous_train`: primera respuesta 501, la llamada síncrona devuelve `{"mlflow_run_id": "", "base_version": {...}}` → `contract` con `base_version`; la segunda llamada no lleva `params`.
  - `test_other_errors_keep_the_model_http_format`: 400 con texto `{"detail":"CSV falta columnas"}` → `RuntimeError`, `str(exc) == 'MODEL_HTTP_400: {"detail":"CSV falta columnas"}'`.
  - `test_202_without_job_id_is_an_error`.
  - `test_disabled_flag_trains_synchronously` (monkeypatch `settings.TRAIN_ASYNC_ENABLED = False`): una sola llamada, sin `params`.
  - `test_unknown_model_raises_like_execute_task`: `ValueError("MODEL_NOT_FOUND_OR_DISABLED")`.

- [ ] **Step 2: Ejecutar** `python3 -m pytest -p no:flask -q tests/test_async_train_submit.py` → fallan por import.

- [ ] **Step 3: Implementar** `resolve_model_call` (y usarla en `ExecuteTaskUseCase.execute`), los ajustes en `Settings`, `TrainSubmission` y `SubmitTrainUseCase`; cablear `self.submit_train_use_case = SubmitTrainUseCase(self.model_repo, self.model_client, self.execute_task_use_case)` en `Container.__init__` después de `execute_task_use_case`.

- [ ] **Step 4: Ejecutar** los tests nuevos y la suite → verde.

- [ ] **Step 5: Dejar listo para commit:** `feat(train): enviar el entrenamiento con wait=false`.

---

### Task 3: caso de uso de consulta — `PollTrainJobUseCase`

**Files:**
- Modify: `app/application/use_cases/async_train.py`
- Modify: `app/interfaces/http/dependencies/container.py` (`self.poll_train_job_use_case`)
- Test: `tests/test_async_train_poll.py` (nuevo; reutiliza `tests/fakes.py`)

**Interfaces:**
- Consumes: `resolve_model_call`, `ModelClient.send` (Tasks 1–2).
- Produces:
  ```python
  class TrainPollTransientError(Exception):
      """El estado no se pudo leer ahora (conexión, timeout, 5xx); se reintenta."""

  @dataclass(frozen=True)
  class TrainJobPoll:
      state: Literal["pending", "succeeded", "failed"]
      contract: dict | None = None      # solo succeeded
      error_code: str | None = None     # solo failed
      error_message: str | None = None  # solo failed, formato MODEL_HTTP_<status>: {"detail": ...}

  class PollTrainJobUseCase:
      def __init__(self, model_repo: ModelRepo, model_client: ModelClient) -> None
      def execute(self, *, model_key: str, job_id: str,
                  extra_headers: dict | None) -> TrainJobPoll
  ```

Llamada: `send(method="GET", endpoint=f"{endpoint_de_train.rstrip('/')}/{quote(job_id, safe='')}", timeout=settings.TRAIN_ASYNC_STATUS_TIMEOUT_SECONDS)`.

| Respuesta | `TrainJobPoll` |
|---|---|
| `requests.RequestException`, o código 500/502/503/504 | lanza `TrainPollTransientError` |
| 404 | `failed`, `TRAIN_JOB_NOT_FOUND`, `MODEL_HTTP_404: <text>` |
| otro código no 2xx | `failed`, `TRAIN_POLL_FAILED`, `MODEL_HTTP_<status>: <text>` |
| 200, `status` `queued`/`running` | `pending` |
| 200, `succeeded` | `succeeded`, `contract = normalize("train", body["result"])` |
| 200, `failed` | `failed`, código y estado equivalente según `error_type` (tabla siguiente), `error_message = f"MODEL_HTTP_{status}: " + json.dumps({"detail": body["error"]}, ensure_ascii=False)` |

| `error_type` | `error_code` | estado equivalente |
|---|---|---|
| `TrainingNotSupportedError` | `TRAIN_NOT_SUPPORTED` | 501 |
| `FileNotFoundError`, `ValueError` | `TRAIN_DATA_ERROR` | 400 |
| `ArchiveLimitExceededError` | `TRAIN_DATA_ERROR` | 413 |
| `ModelPersistenceError` | `TRAIN_PERSISTENCE_ERROR` | 502 |
| `TrainJobLost` | `TRAIN_JOB_LOST` | 500 |
| cualquier otro | `TRAIN_FAILED` | 500 |

- [ ] **Step 1: Escribir los tests:**
  - `test_queued_and_running_are_pending` (parametrizado).
  - `test_succeeded_returns_the_contract`: `result = {"mlflow_run_id": "run-1", "test_f1": 0.8}` → `contract["mlflow_run_id"] == "run-1"`, `contract["schema"] == "datagia.model_evaluation.v1"`.
  - `test_failed_maps_error_type` (parametrizado con la tabla): `ValueError` → `("TRAIN_DATA_ERROR", 'MODEL_HTTP_400: {"detail": "CSV falta columnas"}')`; `TrainJobLost` → `TRAIN_JOB_LOST` y prefijo `MODEL_HTTP_500: `; `KeyError` → `TRAIN_FAILED`.
  - `test_non_ascii_detail_is_kept`: `"columna 'Tª'"` aparece tal cual en `error_message`.
  - `test_404_is_terminal`: `state == "failed"`, `error_code == "TRAIN_JOB_NOT_FOUND"`.
  - `test_5xx_and_connection_errors_are_transient` (parametrizado 500, 502, 503, 504 y `requests.ConnectionError`) → `pytest.raises(TrainPollTransientError)`.
  - `test_get_hits_train_job_path`: la llamada fue `GET` a `endpoint == "/train/run-1"`, `timeout == 30`, sin `body`.

- [ ] **Step 2: Ejecutar** `python3 -m pytest -p no:flask -q tests/test_async_train_poll.py` → fallan por import.

- [ ] **Step 3: Implementar** `TrainPollTransientError`, `TrainJobPoll` y `PollTrainJobUseCase` (la tabla de `error_type` como dict de módulo); cablear `self.poll_train_job_use_case = PollTrainJobUseCase(self.model_repo, self.model_client)` en el `Container`.

- [ ] **Step 4: Ejecutar** los tests nuevos y la suite → verde.

- [ ] **Step 5: Dejar listo para commit:** `feat(train): consultar el estado del entrenamiento asíncrono`.

---

### Task 4: worker — envío y seguimiento en Celery

**Files:**
- Modify: `app/worker/tasks.py`
- Modify: `README.md` (sección nueva "Entrenamiento asíncrono")
- Test: `tests/test_worker_train_async.py` (nuevo)

**Interfaces:**
- Consumes: `container.submit_train_use_case`, `container.poll_train_job_use_case`, `TrainSubmission`, `TrainJobPoll`, `TrainPollTransientError`, `settings.TRAIN_ASYNC_*`.
- Produces:
  ```python
  @celery_app.task(bind=True, name="poll_train_job", ignore_result=True, max_retries=None)
  def poll_train_job(self, task_id: str, model_key: str, job_id: str,
                     extra_headers: dict | None = None, poll_errors: int = 0) -> None
  ```

Comportamiento:
- `execute_model_task`, `action == "train"`: llama a `submit_train_use_case.execute(model_key, payload, extra_headers)` dentro del `try` actual (sus excepciones siguen el camino `EXECUTION_ERROR` de hoy).
  - Con `contract`: `set_success` con `_save_with_retry`, como hoy.
  - Con `job_id`: `poll_train_job.apply_async(kwargs={"task_id", "model_key", "job_id", "extra_headers"}, queue=payload["sector"], countdown=settings.TRAIN_ASYNC_POLL_INTERVAL_SECONDS)` dentro de `_save_with_retry`. Si no se consigue encolar: `set_failed(task_id, "TRAIN_POLL_SCHEDULE_FAILED", …)` con el `job_id` en el mensaje. La tarea se queda en `running`. Logar `task_id` y `job_id`.
- El resto de acciones no cambia.
- `poll_train_job`:
  - `pending` → `raise self.retry(kwargs={…, "poll_errors": 0}, countdown=INTERVAL)`.
  - `TrainPollTransientError` → `n = poll_errors + 1`. Si `n >= TRAIN_ASYNC_MAX_POLL_ERRORS`: `set_failed("TRAIN_POLL_FAILED", "<n> consultas seguidas sin respuesta del modelo (job_id=…): <último error>")`. Si no: `retry` con `poll_errors=n`.
  - `succeeded` → `set_success(task_id, contract)` con `_save_with_retry`, y el mismo `RESULT_SAVE_FAILED` de respaldo que `execute_model_task`.
  - `failed` → `set_failed(task_id, error_code, error_message)` con `_save_with_retry`.
  - Cualquier otra excepción → log + `set_failed("EXECUTION_ERROR", str(e)[:4000])`.
  - `self.retry` lanza `celery.exceptions.Retry`: que no quede dentro del `except Exception`.

Tests: `monkeypatch.setattr(tasks, "get_container", lambda: fake)`, con `fake = SimpleNamespace(task_repo=<repo en memoria>, submit_train_use_case=…, poll_train_job_use_case=…, execute_task_use_case=…)`. Ejecutar con `.apply(kwargs=…, task_id="t1")`: en modo eager, `retry` vuelve a ejecutar la tarea al momento con los kwargs nuevos (comprobado con Celery 5.4). El repo en memoria es `TaskRepoMemory` de `app/infrastructure/db/repositories/task_repo_memory.py` (cumple el puerto `TaskRepo`); crear la tarea con `create(TaskRun(task_id="t1", action="train", sector="lacteo", model_key="modelo-47"))` antes de ejecutar.

- [ ] **Step 1: Escribir los tests:**
  - `test_async_submit_schedules_poll_and_stays_running`: la submission con `job_id="run-1"` → la tarea queda `running`, `response == {}`; `apply_async` (monkeypatch en `tasks.poll_train_job`) recibe `queue="lacteo"`, `countdown=30` y `kwargs["job_id"] == "run-1"`.
  - `test_sync_submission_saves_success`: la submission con `contract` → `success` y `response == contract`; no se encola nada.
  - `test_submit_error_is_execution_error`: `RuntimeError('MODEL_HTTP_400: {"detail":"x"}')` → `failed`, `error_code == "EXECUTION_ERROR"`, mensaje intacto.
  - `test_predict_path_unchanged`: `action="predict"` → usa `execute_task_use_case`, no `submit_train_use_case`.
  - `test_poll_until_succeeded`: el poll devuelve `pending`, `pending` y `succeeded` → 3 llamadas; `success` con el contrato.
  - `test_poll_failed_saves_code_and_message`: el `TRAIN_DATA_ERROR` y su mensaje llegan tal cual al `TaskRun`.
  - `test_transient_errors_reset_after_a_good_read`: transitorio, transitorio, `pending`, transitorio, `succeeded`, con `TRAIN_ASYNC_MAX_POLL_ERRORS=3` → acaba en `success`.
  - `test_too_many_transient_errors_fail_the_task`: siempre transitorio con `TRAIN_ASYNC_MAX_POLL_ERRORS=3` → 3 llamadas, `failed`, `TRAIN_POLL_FAILED`, `"run-1"` en el mensaje.
  - `test_result_survives_a_redis_blip`: el `set_success` del repo falla una vez y luego funciona (monkeypatch de `tasks._SAVE_RETRY_DELAY_SECONDS = 0`) → `success`.

- [ ] **Step 2: Ejecutar** `python3 -m pytest -p no:flask -q tests/test_worker_train_async.py` → fallan.

- [ ] **Step 3: Implementar** según el comportamiento de arriba.

- [ ] **Step 4: README.** Sección "Entrenamiento asíncrono":
  - flujo `POST ?wait=false` → `poll_train_job` cada `TRAIN_ASYNC_POLL_INTERVAL_SECONDS`;
  - los 4 ajustes y `TRAIN_ASYNC_ENABLED=false` como vuelta atrás sin desplegar;
  - qué pasa con 200/501;
  - la tabla `error_type` → `error_code`;
  - que un worker reiniciado retoma los seguimientos tras el `visibility_timeout` (hasta 1 h);
  - que `MODEL_HTTP_TIMEOUT_SECONDS` sigue mandando en los modelos síncronos.

- [ ] **Step 5: Ejecutar** la suite completa → verde.

- [ ] **Step 6: Dejar listo para commit:** `feat(train): seguimiento asíncrono del entrenamiento en el worker`.

---

### Task 5: prueba contra el servicio de modelos real (local)

Sin código nuevo: verifica el contrato extremo a extremo antes de INT.

- [ ] **Step 1:** levantar en local el servicio de modelos de la rama `feature/async-train` solo con un modelo del grupo 2 que entrene rápido y otro del grupo 1, con un lanzador mínimo (ver memoria del entorno: `main.py` completo no arranca en el WSL), y un MLflow local (`mlflow server` sobre una carpeta del scratchpad).
- [ ] **Step 2:** llamar a `SubmitTrainUseCase` y `PollTrainJobUseCase` con `HttpModelClient` real desde un script en el scratchpad:
  - grupo 2 → 202, luego `pending` y por último `succeeded` con `mlflow_run_id`;
  - grupo 1 → 501 y luego síncrono con `base_version`;
  - run inexistente → `RuntimeError` con prefijo `MODEL_HTTP_400: `.
- [ ] **Step 3:** anotar el resultado en el README del orquestador o en la memoria; si algo no cuadra con la spec, parar y avisar.

---

## Ejecución (2026-10-25)

Hecho en `retech-lote2-xai-orquestador`, rama `feature/async-train`, sin commit. 55 tests en verde.
Prueba local de la Task 5 con el servicio de modelos real y sus plugins de prueba (`SlowTrainPlugin`,
`FailingTrainPlugin`, `TinyPromotingPlugin`) sobre un MLflow local: los seis casos según la spec.

Desviaciones tras la revisión independiente de la rama:

- `poll_train_job` no usa `self.retry`: si Redis falla al reprogramar, Celery 5.4 lanza `Reject` y la
  cadena se pierde con la tarea en `running`. Se reprograma con `apply_async(countdown=…)` dentro de
  `_save_with_retry` y la cola viaja en los kwargs (`queue`). Si no se consigue, `TRAIN_POLL_SCHEDULE_FAILED`.
- Si no se puede guardar el estado final, se vuelve a consultar a los 300 s en vez de abandonar.
- Un `status` que no es `queued`/`running`/`succeeded`/`failed` y un corte de Redis al leer el modelo
  cuentan como error transitorio (antes: `pending` infinito y `EXECUTION_ERROR`).
- Sin `mlflow_run_id` (la plataforma manda `""` si falla su MLflow) se entrena en síncrono.
- `tests/__init__.py` nuevo, para importar `tests.fakes`.
