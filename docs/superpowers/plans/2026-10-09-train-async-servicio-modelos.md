# Entrenamiento asíncrono en el servicio de modelos — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que `POST /models/<id>/train?wait=false` lance el entrenamiento en un proceso aparte y responda `202` al momento, con `GET /models/<id>/train/{job_id}` para consultar su estado, sin tocar el `train()` de ningún plugin.

**Architecture:** Capa común hexagonal. Dominio: `TrainJobRecord`/`TrainJobStatus` y dos puertos (`TrainJobStorePort`, `TrainExecutorPort`). Aplicación: `SubmitTrainJobUseCase` (idempotente, 409 si ocupado) y `GetTrainJobUseCase` (detecta trabajos perdidos por latido). Infraestructura: `MLflowTrainJobStore` (etiquetas `async_train.*` + artefacto `async_train/result.json` en el run `job_id = mlflow_run_id`) y `ProcessTrainExecutor` (multiprocessing `spawn`; el hijo hace `plugin_class()` + `load()` + `train()`). El router añade `?wait=false` y `GET /train/{job_id}`; sin `wait` el `/train` sigue siendo síncrono.

**Tech Stack:** Python 3.10, FastAPI 0.136, Pydantic v2, MLflow ≥ 2.19 (`MlflowClient.set_tag`, `log_text`, `download_artifacts`), `multiprocessing` (contexto `spawn`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-reentrenamiento-asincrono.md`

## Global Constraints

- `job_id` = `mlflow_run_id` del `TrainRequest`; en modo asíncrono es obligatorio y no vacío (`400` si falta).
- `POST /train` sin `wait` o con `wait=true` mantiene **exactamente** el comportamiento síncrono actual (códigos 200/400/501/500 incluidos).
- Prefijo de etiquetas MLflow: `async_train.`; artefacto de resultado: `async_train/result.json`.
- Variables de entorno: `TRAIN_JOB_HEARTBEAT_S` (por defecto `60`), `TRAIN_JOB_STALE_AFTER_S` (por defecto `600`).
- `error_type` de un trabajo perdido: `"TrainJobLost"`.
- Ningún plugin cambia (`app/plugins/**` no se toca).
- Nada de infraestructura (Jobs de K8s, GPU, overlays): fuera de alcance.
- Tests: `python3 -m pytest -p no:flask …` (el `pytest-flask` global secuestra el fixture `client` en el WSL local).
- flake8 del repo: `max-line-length = 120`, E501 ignorado.
- **No hacer commits**: el usuario hace los commits. Cada tarea termina dejando el árbol listo y los tests en verde.

## Review Focus

- Celery reentrega el mismo `POST` (mismo `job_id`) mientras entrena → no se lanza un segundo entrenamiento; se devuelve el trabajo existente (tests en Task 1 y Task 4).
- El pod se reinicia (o el hijo muere por OOM) a mitad de entrenamiento → `GET` acaba devolviendo `failed` / `TrainJobLost` en lugar de `running` para siempre (test en Task 1).
- `train()` devuelve métricas con tipos numpy (`np.float32`, `np.int64`, arrays) dentro de un `dict` → el resultado se guarda igualmente como JSON y el trabajo termina en `succeeded` (test en Task 3).
- Llega un segundo `POST` con otro `job_id` mientras hay un entrenamiento vivo en el pod → `409`, no dos entrenamientos compitiendo por CPU (tests en Task 1 y Task 4).
- MLflow no responde al consultar o registrar → `503` con mensaje claro, no un `500` genérico (test en Task 4).

## Estructura de ficheros

| Fichero | Acción | Responsabilidad |
|---|---|---|
| `app/domain/services/train_job.py` | Crear | `TrainJobStatus`, `TrainJobRecord` |
| `app/domain/ports/train_job_store_port.py` | Crear | Puerto de persistencia del estado |
| `app/domain/ports/train_executor_port.py` | Crear | Puerto del ejecutor en segundo plano |
| `app/domain/services/exceptions.py` | Modificar | `TrainJobBusyError` |
| `app/application/use_cases/train_model_use_case.py` | Modificar | Extraer `build_train_kwargs` (compartido sync/async) |
| `app/application/use_cases/train_job_use_cases.py` | Crear | `SubmitTrainJobUseCase`, `GetTrainJobUseCase` |
| `app/infrastructure/train_jobs/__init__.py` | Crear | Paquete |
| `app/infrastructure/train_jobs/mlflow_job_store.py` | Crear | `MLflowTrainJobStore` |
| `app/infrastructure/train_jobs/process_executor.py` | Crear | `run_train_job`, `ProcessTrainExecutor` |
| `app/application/dto/train_job_dto.py` | Crear | `TrainJobResponse` (JSON del contrato) |
| `app/infrastructure/http/router_factory.py` | Modificar | `?wait=false` y `GET /train/{job_id}` |
| `app/infrastructure/http/dependencies/container.py` | Modificar | Cablear store, ejecutor y casos de uso |
| `main.py` | Modificar | Pasar `model_id` al contenedor |
| `tests/unit/train_job_fakes.py` | Crear | Fakes importables (necesario para `spawn`) |
| `tests/unit/test_train_job_use_cases.py` | Crear | Tests de casos de uso |
| `tests/unit/test_mlflow_train_job_store.py` | Crear | Tests del store con cliente simulado |
| `tests/unit/test_process_train_executor.py` | Crear | Tests con procesos reales |
| `tests/unit/test_train_async_router.py` | Crear | Tests HTTP del contrato |
| `README.md` | Modificar | Documentar el contrato asíncrono |

---

### Task 1: Dominio y casos de uso

**Files:**
- Create: `app/domain/services/train_job.py`
- Create: `app/domain/ports/train_job_store_port.py`
- Create: `app/domain/ports/train_executor_port.py`
- Modify: `app/domain/services/exceptions.py` (añadir al final)
- Modify: `app/application/use_cases/train_model_use_case.py`
- Create: `app/application/use_cases/train_job_use_cases.py`
- Create: `tests/unit/train_job_fakes.py`
- Test: `tests/unit/test_train_job_use_cases.py`

**Interfaces:**
- Produces:
  - `TrainJobStatus(str, Enum)` con `QUEUED="queued"`, `RUNNING="running"`, `SUCCEEDED="succeeded"`, `FAILED="failed"` y propiedad `is_terminal -> bool`.
  - `TrainJobRecord` (dataclass frozen): `job_id: str, model_id: str, status: TrainJobStatus, submitted_at: float, updated_at: float, result: dict | None = None, error: str | None = None, error_type: str | None = None`. Tiempos en epoch segundos.
  - `TrainJobStorePort`: `get(job_id: str) -> TrainJobRecord | None`, `save(record: TrainJobRecord) -> None`, `heartbeat(job_id: str, at: float) -> None`.
  - `TrainExecutorPort`: `is_busy() -> bool`, `submit(*, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None`.
  - `TrainJobBusyError(Exception)`.
  - `build_train_kwargs(plugin, request) -> dict[str, Any]`.
  - `SubmitTrainJobUseCase(plugin, model_id: str, store, executor, clock=time.time)` con `execute(request) -> TrainJobRecord`.
  - `GetTrainJobUseCase(model_id: str, store, clock=time.time, stale_after_s: float | None = None)` con `execute(job_id: str) -> TrainJobRecord | None`.
  - Fakes en `tests/unit/train_job_fakes.py`: `InMemoryJobStore`, `FileJobStore`, `RecordingExecutor`, `EchoTrainPlugin`, `FailingTrainPlugin`, `SlowTrainPlugin`, `NumpyMetricsTrainPlugin`.

- [ ] **Step 1: Crear los fakes de test**

`tests/unit/train_job_fakes.py` (módulo importable: el contexto `spawn` de Task 3 necesita importar estas clases por nombre en el proceso hijo):

```python
"""Test doubles for async training (importable by name so `spawn` children can unpickle them)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
from pydantic import BaseModel

from app.domain.ports.train_executor_port import TrainExecutorPort
from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus


class InMemoryJobStore(TrainJobStorePort):
    """Single-process store."""

    def __init__(self) -> None:
        self.records: dict[str, TrainJobRecord] = {}

    def get(self, job_id: str) -> TrainJobRecord | None:
        return self.records.get(job_id)

    def save(self, record: TrainJobRecord) -> None:
        self.records[record.job_id] = record

    def heartbeat(self, job_id: str, at: float) -> None:
        record = self.records.get(job_id)
        if record is not None:
            self.records[job_id] = replace(record, updated_at=at)


class FileJobStore(TrainJobStorePort):
    """Picklable, cross-process store: one JSON file per job under ``root``."""

    def __init__(self, root: str) -> None:
        self.root = root

    def _path(self, job_id: str) -> str:
        return os.path.join(self.root, f"{job_id}.json")

    def get(self, job_id: str) -> TrainJobRecord | None:
        path = self._path(job_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["status"] = TrainJobStatus(data["status"])
        return TrainJobRecord(**data)

    def save(self, record: TrainJobRecord) -> None:
        data = asdict(record)
        data["status"] = record.status.value
        tmp = self._path(record.job_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self._path(record.job_id))

    def heartbeat(self, job_id: str, at: float) -> None:
        record = self.get(job_id)
        if record is not None:
            self.save(replace(record, updated_at=at))


class RecordingExecutor(TrainExecutorPort):
    """Records submissions without running anything; ``busy`` is set by the test."""

    def __init__(self, busy: bool = False, fail_on_submit: Exception | None = None) -> None:
        self.busy = busy
        self.fail_on_submit = fail_on_submit
        self.submitted: list[dict] = []

    def is_busy(self) -> bool:
        return self.busy

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        if self.fail_on_submit is not None:
            raise self.fail_on_submit
        self.submitted.append({"job_id": job_id, "plugin_class": plugin_class, "train_kwargs": train_kwargs})


class FakeTrainResponse(BaseModel):
    detail: str
    metrics: dict = {}
    mlflow_run_id: str = ""


class EchoTrainPlugin:
    """Minimal plugin: train() requires load() and reports the pid it ran in."""

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        if not self.loaded:
            raise RuntimeError("train() called before load()")
        return FakeTrainResponse(
            detail="ok", metrics={"data_path": data_path, "pid": os.getpid()}, mlflow_run_id=mlflow_run_id,
        )


class FailingTrainPlugin(EchoTrainPlugin):
    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        raise FileNotFoundError(f"no existe {data_path}")


class SlowTrainPlugin(EchoTrainPlugin):
    """Sleeps ``float(data_path)`` seconds before answering."""

    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        time.sleep(float(data_path))
        return super().train(data_path=data_path, mlflow_run_id=mlflow_run_id)


class NumpyMetricsTrainPlugin(EchoTrainPlugin):
    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        return FakeTrainResponse(
            detail="ok",
            metrics={"f1": np.float32(0.5), "n": np.int64(3), "cm": np.array([[1, 0], [0, 2]])},
            mlflow_run_id=mlflow_run_id,
        )


class SystemTrainPlugin(EchoTrainPlugin):
    """Declares ``system`` like ml40 does, to check kwargs forwarding."""

    def train(self, *, data_path: str, mlflow_run_id: str, system: str | None = None) -> FakeTrainResponse:
        return FakeTrainResponse(detail=f"system={system}", mlflow_run_id=mlflow_run_id)
```

- [ ] **Step 2: Escribir los tests que fallan**

`tests/unit/test_train_job_use_cases.py`:

```python
"""Async training use cases: submit (idempotent, busy, validation) and get (stale detection)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.application.use_cases.train_job_use_cases import GetTrainJobUseCase, SubmitTrainJobUseCase
from app.application.use_cases.train_model_use_case import build_train_kwargs
from app.domain.services.exceptions import TrainJobBusyError
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from tests.unit.train_job_fakes import EchoTrainPlugin, InMemoryJobStore, RecordingExecutor, SystemTrainPlugin

MODEL_ID = "fake-model"


def _request(run_id: str = "run-1", data_path: str = "s3://bucket/train.csv", **extra):
    return SimpleNamespace(mlflow_run_id=run_id, data_path=data_path, **extra)


def _submit(store=None, executor=None, plugin=None, now=1000.0):
    store = store if store is not None else InMemoryJobStore()
    executor = executor if executor is not None else RecordingExecutor()
    plugin = plugin if plugin is not None else EchoTrainPlugin()
    return SubmitTrainJobUseCase(plugin, MODEL_ID, store, executor, clock=lambda: now), store, executor


class TestBuildTrainKwargs:
    def test_basic_kwargs(self):
        assert build_train_kwargs(EchoTrainPlugin(), _request()) == {
            "data_path": "s3://bucket/train.csv", "mlflow_run_id": "run-1",
        }

    def test_forwards_system_only_when_declared(self):
        kwargs = build_train_kwargs(SystemTrainPlugin(), _request(system="aireado"))
        assert kwargs["system"] == "aireado"


class TestSubmit:
    def test_creates_queued_job_and_submits_plugin_class(self):
        use_case, store, executor = _submit()
        record = use_case.execute(_request())
        assert record == TrainJobRecord(
            job_id="run-1", model_id=MODEL_ID, status=TrainJobStatus.QUEUED, submitted_at=1000.0, updated_at=1000.0,
        )
        assert store.get("run-1") == record
        assert executor.submitted == [{
            "job_id": "run-1", "plugin_class": EchoTrainPlugin,
            "train_kwargs": {"data_path": "s3://bucket/train.csv", "mlflow_run_id": "run-1"},
        }]

    @pytest.mark.parametrize("run_id", ["", "   "])
    def test_empty_run_id_is_rejected(self, run_id):
        use_case, store, executor = _submit()
        with pytest.raises(ValueError, match="mlflow_run_id"):
            use_case.execute(_request(run_id=run_id))
        assert store.records == {} and executor.submitted == []

    def test_resubmitting_same_job_returns_existing_and_does_not_start_again(self):
        use_case, store, executor = _submit()
        first = use_case.execute(_request())
        store.save(replace(first, status=TrainJobStatus.RUNNING, updated_at=1010.0))
        executor.busy = True  # the first one is still alive: a redelivery must NOT get a 409
        again = use_case.execute(_request())
        assert again.status is TrainJobStatus.RUNNING
        assert len(executor.submitted) == 1

    def test_job_id_of_another_model_is_rejected(self):
        use_case, store, _ = _submit()
        store.save(TrainJobRecord("run-1", "other-model", TrainJobStatus.SUCCEEDED, 1.0, 2.0))
        with pytest.raises(ValueError, match="other-model"):
            use_case.execute(_request())

    def test_busy_executor_raises_and_stores_nothing(self):
        use_case, store, _ = _submit(executor=RecordingExecutor(busy=True))
        with pytest.raises(TrainJobBusyError):
            use_case.execute(_request(run_id="run-2"))
        assert store.get("run-2") is None

    def test_submit_failure_marks_job_failed(self):
        use_case, store, _ = _submit(executor=RecordingExecutor(fail_on_submit=OSError("no fork")))
        record = use_case.execute(_request())
        assert record.status is TrainJobStatus.FAILED
        assert record.error_type == "OSError"
        assert "no fork" in record.error
        assert store.get("run-1") == record


class TestGet:
    def _store_with(self, status, updated_at):
        store = InMemoryJobStore()
        store.save(TrainJobRecord("run-1", MODEL_ID, status, 1000.0, updated_at))
        return store

    def test_unknown_job_returns_none(self):
        assert GetTrainJobUseCase(MODEL_ID, InMemoryJobStore()).execute("nope") is None

    def test_job_of_another_model_returns_none(self):
        store = InMemoryJobStore()
        store.save(TrainJobRecord("run-1", "other-model", TrainJobStatus.RUNNING, 1.0, 1.0))
        assert GetTrainJobUseCase(MODEL_ID, store).execute("run-1") is None

    def test_fresh_running_job_is_returned_as_is(self):
        store = self._store_with(TrainJobStatus.RUNNING, updated_at=1500.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 1600.0, stale_after_s=600).execute("run-1")
        assert record.status is TrainJobStatus.RUNNING

    @pytest.mark.parametrize("status", [TrainJobStatus.QUEUED, TrainJobStatus.RUNNING])
    def test_silent_job_is_marked_lost_and_persisted(self, status):
        store = self._store_with(status, updated_at=1000.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 1601.0, stale_after_s=600).execute("run-1")
        assert record.status is TrainJobStatus.FAILED
        assert record.error_type == "TrainJobLost"
        assert store.get("run-1").status is TrainJobStatus.FAILED

    def test_terminal_job_is_never_marked_lost(self):
        store = self._store_with(TrainJobStatus.SUCCEEDED, updated_at=1000.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 99999.0, stale_after_s=600).execute("run-1")
        assert record.status is TrainJobStatus.SUCCEEDED

    def test_stale_window_defaults_to_env(self, monkeypatch):
        monkeypatch.setenv("TRAIN_JOB_STALE_AFTER_S", "10")
        store = self._store_with(TrainJobStatus.RUNNING, updated_at=1000.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 1011.0).execute("run-1")
        assert record.status is TrainJobStatus.FAILED
```

- [ ] **Step 3: Ejecutar y comprobar que falla**

Run: `python3 -m pytest -p no:flask tests/unit/test_train_job_use_cases.py -v`
Expected: FAIL en la colección con `ModuleNotFoundError: No module named 'app.domain.ports.train_executor_port'`.

- [ ] **Step 4: Implementar el dominio**

`app/domain/services/train_job.py`:

```python
"""State of an asynchronous training job (job_id = the MLflow run id the platform pre-creates)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainJobStatus(str, Enum):
    """Lifecycle: queued -> running -> succeeded | failed."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (TrainJobStatus.SUCCEEDED, TrainJobStatus.FAILED)


@dataclass(frozen=True)
class TrainJobRecord:
    """One training job. Times are epoch seconds; ``updated_at`` doubles as the heartbeat."""

    job_id: str
    model_id: str
    status: TrainJobStatus
    submitted_at: float
    updated_at: float
    result: dict | None = None
    error: str | None = None
    error_type: str | None = None
```

`app/domain/ports/train_job_store_port.py`:

```python
"""Port for persisting asynchronous training job state."""
from abc import ABC, abstractmethod

from app.domain.services.train_job import TrainJobRecord


class TrainJobStorePort(ABC):
    """Where job state lives. Must be shareable with (picklable for) the training child process."""

    @abstractmethod
    def get(self, job_id: str) -> TrainJobRecord | None:
        """Return the job, or None if no job exists for this id."""

    @abstractmethod
    def save(self, record: TrainJobRecord) -> None:
        """Create or overwrite the job."""

    @abstractmethod
    def heartbeat(self, job_id: str, at: float) -> None:
        """Bump ``updated_at`` only (never touches status)."""
```

`app/domain/ports/train_executor_port.py`:

```python
"""Port for running a training job in the background."""
from abc import ABC, abstractmethod

from app.domain.ports.train_job_store_port import TrainJobStorePort


class TrainExecutorPort(ABC):
    """Runs ``plugin_class().train(**train_kwargs)`` away from the request, reporting into ``store``."""

    @abstractmethod
    def is_busy(self) -> bool:
        """True while a previously submitted job is still running."""

    @abstractmethod
    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        """Start the job and return immediately."""
```

Añadir al final de `app/domain/services/exceptions.py`:

```python


class TrainJobBusyError(Exception):
    """Raised when an async training is requested while another one of the same model is still running (409)."""
```

- [ ] **Step 5: Extraer `build_train_kwargs`**

Sustituir el cuerpo de `app/application/use_cases/train_model_use_case.py` por:

```python
"""Generic train use case for model plugins."""
import inspect
from typing import Any
from app.domain.ports.model_plugin_port import ModelPluginPort
import logging

logger = logging.getLogger(__name__)


def build_train_kwargs(plugin: Any, request: Any) -> dict[str, Any]:
    """Map a TrainRequest to ``plugin.train`` kwargs (shared by the sync and the async paths)."""
    kwargs: dict[str, Any] = {
        "data_path": getattr(request, "data_path", ""),
        "mlflow_run_id": getattr(request, "mlflow_run_id", ""),
    }
    # Only ml40's train declares system (refrigeracion/aireado chosen by the user).
    if "system" in inspect.signature(plugin.train).parameters:
        kwargs["system"] = getattr(request, "system", None)
    return kwargs


class TrainModelUseCase:
    """Generic train use case."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self, request: Any) -> dict:
        """Executes the training process."""
        kwargs = build_train_kwargs(self._plugin, request)
        logger.info("Executing training, data_path=%s, mlflow_run_id=%s", kwargs["data_path"], kwargs["mlflow_run_id"])
        return self._plugin.train(**kwargs)
```

> Si la rama donde se implemente ya reenvía más campos en `TrainModelUseCase` (p. ej. `mode` del m47 en `fix/retrain-mlflow-only-persistence`), moverlos también a `build_train_kwargs`: es lo que garantiza que el modo asíncrono entrena exactamente igual que el síncrono.

- [ ] **Step 6: Implementar los casos de uso**

`app/application/use_cases/train_job_use_cases.py`:

```python
"""Async training use cases: POST /train?wait=false and GET /train/{job_id}."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace
from typing import Any, Callable

from app.application.use_cases.train_model_use_case import build_train_kwargs
from app.domain.ports.train_executor_port import TrainExecutorPort
from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.exceptions import TrainJobBusyError
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus

logger = logging.getLogger(__name__)

LOST_ERROR_TYPE = "TrainJobLost"


class SubmitTrainJobUseCase:
    """Register a job (job_id = mlflow_run_id) and start it in the background."""

    def __init__(
        self,
        plugin: Any,
        model_id: str,
        store: TrainJobStorePort,
        executor: TrainExecutorPort,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._plugin = plugin
        self._model_id = model_id
        self._store = store
        self._executor = executor
        self._clock = clock
        # FastAPI runs sync endpoints in a threadpool: check-then-submit must be atomic.
        self._lock = threading.Lock()

    def execute(self, request: Any) -> TrainJobRecord:
        job_id = (getattr(request, "mlflow_run_id", "") or "").strip()
        if not job_id:
            raise ValueError("mlflow_run_id es obligatorio en el entrenamiento asíncrono: es el job_id")
        with self._lock:
            existing = self._store.get(job_id)
            if existing is not None:
                if existing.model_id != self._model_id:
                    raise ValueError(f"El job_id {job_id} pertenece al modelo {existing.model_id}")
                # Idempotent: a redelivered request (e.g. Celery) never starts a second training.
                return existing
            if self._executor.is_busy():
                raise TrainJobBusyError(f"Ya hay un entrenamiento en curso del modelo {self._model_id}")
            now = self._clock()
            record = TrainJobRecord(
                job_id=job_id, model_id=self._model_id, status=TrainJobStatus.QUEUED,
                submitted_at=now, updated_at=now,
            )
            self._store.save(record)
            try:
                self._executor.submit(
                    job_id=job_id,
                    plugin_class=type(self._plugin),
                    train_kwargs=build_train_kwargs(self._plugin, request),
                    store=self._store,
                )
            except Exception as exc:
                logger.exception("Could not start training job %s for model '%s'", job_id, self._model_id)
                failed = replace(
                    record, status=TrainJobStatus.FAILED, updated_at=self._clock(),
                    error=f"No se pudo lanzar el proceso de entrenamiento: {exc}", error_type=type(exc).__name__,
                )
                self._store.save(failed)
                return failed
            logger.info("Training job %s queued for model '%s'", job_id, self._model_id)
            return record


class GetTrainJobUseCase:
    """Return a job; a non-terminal job that stopped heart-beating is reported (and persisted) as lost."""

    def __init__(
        self,
        model_id: str,
        store: TrainJobStorePort,
        clock: Callable[[], float] = time.time,
        stale_after_s: float | None = None,
    ) -> None:
        self._model_id = model_id
        self._store = store
        self._clock = clock
        self._stale_after_s = (
            stale_after_s if stale_after_s is not None else float(os.getenv("TRAIN_JOB_STALE_AFTER_S", "600"))
        )

    def execute(self, job_id: str) -> TrainJobRecord | None:
        record = self._store.get(job_id)
        if record is None or record.model_id != self._model_id:
            return None
        now = self._clock()
        if not record.status.is_terminal and now - record.updated_at > self._stale_after_s:
            record = replace(
                record, status=TrainJobStatus.FAILED, updated_at=now, error_type=LOST_ERROR_TYPE,
                error=(
                    f"El proceso de entrenamiento dejó de dar señales hace más de {int(self._stale_after_s)} s "
                    "(reinicio del pod o proceso terminado por el sistema)."
                ),
            )
            self._store.save(record)
            logger.warning("Training job %s for model '%s' marked as lost", job_id, self._model_id)
        return record
```

- [ ] **Step 7: Ejecutar y comprobar que pasa**

Run: `python3 -m pytest -p no:flask tests/unit/test_train_job_use_cases.py tests/unit/test_use_case_system_forwarding.py -v`
Expected: todos PASS (el segundo fichero confirma que el refactor de `build_train_kwargs` no rompe el reenvío de `system`).

- [ ] **Step 8: Dejar listo (sin commit)**

Run: `python3 -m flake8 app/domain app/application tests/unit/train_job_fakes.py tests/unit/test_train_job_use_cases.py`
Expected: sin salida. No hacer commit: lo hace el usuario.

---

### Task 2: Store en MLflow

**Files:**
- Create: `app/infrastructure/train_jobs/__init__.py` (vacío, una línea docstring)
- Create: `app/infrastructure/train_jobs/mlflow_job_store.py`
- Test: `tests/unit/test_mlflow_train_job_store.py`

**Interfaces:**
- Consumes: `TrainJobStorePort`, `TrainJobRecord`, `TrainJobStatus` (Task 1); `BaseMLflowTracker.TRACKING_URI` (`app/domain/services/mlflow_tracker.py`).
- Produces: `MLflowTrainJobStore(tracking_uri: str | None = None, client=None)`; constantes `TAG_PREFIX = "async_train."`, `RESULT_ARTIFACT = "async_train/result.json"`. Picklable (el cliente no viaja al hijo).

- [ ] **Step 1: Escribir los tests que fallan**

`tests/unit/test_mlflow_train_job_store.py`:

```python
"""MLflowTrainJobStore against a mocked MlflowClient."""
from __future__ import annotations

import json
import os
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from app.infrastructure.train_jobs.mlflow_job_store import RESULT_ARTIFACT, TAG_PREFIX, MLflowTrainJobStore


def _run(tags: dict):
    return SimpleNamespace(data=SimpleNamespace(tags=tags))


def _tags(status: str, **extra) -> dict:
    tags = {
        f"{TAG_PREFIX}status": status,
        f"{TAG_PREFIX}model_id": "m47",
        f"{TAG_PREFIX}submitted_at": "1000.0",
        f"{TAG_PREFIX}updated_at": "1010.5",
        "mlflow.runName": "otra-etiqueta",
    }
    tags.update({f"{TAG_PREFIX}{k}": v for k, v in extra.items()})
    return tags


class TestSave:
    def test_queued_sets_tags_with_status_last_and_no_artifact(self):
        client = MagicMock()
        MLflowTrainJobStore(client=client).save(TrainJobRecord("run-1", "m47", TrainJobStatus.QUEUED, 1000.0, 1000.0))
        client.log_text.assert_not_called()
        assert client.set_tag.call_args_list[-1] == call("run-1", f"{TAG_PREFIX}status", "queued")
        assert call("run-1", f"{TAG_PREFIX}model_id", "m47") in client.set_tag.call_args_list
        assert call("run-1", f"{TAG_PREFIX}submitted_at", "1000.0") in client.set_tag.call_args_list

    def test_succeeded_writes_result_before_status(self):
        client = MagicMock()
        record = TrainJobRecord("run-1", "m47", TrainJobStatus.SUCCEEDED, 1.0, 2.0, result={"detail": "ok", "f1": 0.9})
        MLflowTrainJobStore(client=client).save(record)
        names = [c[0] for c in client.mock_calls]
        assert names.index("log_text") < len(names) - 1
        assert client.mock_calls[-1] == call.set_tag("run-1", f"{TAG_PREFIX}status", "succeeded")
        client.log_text.assert_called_once_with("run-1", json.dumps({"detail": "ok", "f1": 0.9}, ensure_ascii=False), RESULT_ARTIFACT)

    def test_failed_truncates_long_error(self):
        client = MagicMock()
        record = TrainJobRecord("run-1", "m47", TrainJobStatus.FAILED, 1.0, 2.0, error="x" * 10000, error_type="ValueError")
        MLflowTrainJobStore(client=client).save(record)
        error_tag = next(c for c in client.set_tag.call_args_list if c.args[1] == f"{TAG_PREFIX}error")
        assert len(error_tag.args[2]) == 4000
        assert call("run-1", f"{TAG_PREFIX}error_type", "ValueError") in client.set_tag.call_args_list

    def test_heartbeat_only_touches_updated_at(self):
        client = MagicMock()
        MLflowTrainJobStore(client=client).heartbeat("run-1", 1234.5)
        client.set_tag.assert_called_once_with("run-1", f"{TAG_PREFIX}updated_at", "1234.5")


class TestGet:
    def test_missing_run_returns_none(self):
        client = MagicMock()
        client.get_run.side_effect = MlflowException("no run", error_code=RESOURCE_DOES_NOT_EXIST)
        assert MLflowTrainJobStore(client=client).get("run-x") is None

    def test_other_mlflow_errors_propagate(self):
        client = MagicMock()
        client.get_run.side_effect = MlflowException("boom", error_code=INTERNAL_ERROR)
        with pytest.raises(MlflowException):
            MLflowTrainJobStore(client=client).get("run-1")

    def test_run_without_job_tags_returns_none(self):
        client = MagicMock()
        client.get_run.return_value = _run({"mlflow.runName": "pre-creado por la plataforma"})
        assert MLflowTrainJobStore(client=client).get("run-1") is None

    def test_running_job_is_parsed_without_downloading(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("running"))
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert record == TrainJobRecord("run-1", "m47", TrainJobStatus.RUNNING, 1000.0, 1010.5)
        client.download_artifacts.assert_not_called()

    def test_failed_job_carries_error(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("failed", error="sin datos", error_type="ValueError"))
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert (record.status, record.error, record.error_type) == (TrainJobStatus.FAILED, "sin datos", "ValueError")

    def test_succeeded_job_loads_result_artifact(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("succeeded"))

        def fake_download(run_id, path, dst_path):
            local = os.path.join(dst_path, path)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "w", encoding="utf-8") as fh:
                json.dump({"detail": "ok"}, fh)
            return local

        client.download_artifacts.side_effect = fake_download
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert record.result == {"detail": "ok"}
        client.download_artifacts.assert_called_once()
        assert client.download_artifacts.call_args.args[:2] == ("run-1", RESULT_ARTIFACT)


def test_pickling_drops_the_client():
    store = MLflowTrainJobStore(tracking_uri="http://mlflow:5000", client=MagicMock())
    clone = pickle.loads(pickle.dumps(store))
    assert clone._client is None
    assert clone._tracking_uri == "http://mlflow:5000"
```

- [ ] **Step 2: Ejecutar y comprobar que falla**

Run: `python3 -m pytest -p no:flask tests/unit/test_mlflow_train_job_store.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.infrastructure.train_jobs'`.

- [ ] **Step 3: Implementar**

`app/infrastructure/train_jobs/__init__.py`:

```python
"""Asynchronous training: job state store and background executor."""
```

`app/infrastructure/train_jobs/mlflow_job_store.py`:

```python
"""Async training job state stored on the MLflow run itself (job_id = run_id).

Tags ``async_train.*`` hold status and timestamps (any replica can read them, they survive restarts);
the plugin's TrainResponse goes to the ``async_train/result.json`` artifact of the same run.
"""
from __future__ import annotations

import json
import tempfile

from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus

TAG_PREFIX = "async_train."
RESULT_ARTIFACT = "async_train/result.json"
_MAX_ERROR_CHARS = 4000  # well under MLflow's tag value limit


class MLflowTrainJobStore(TrainJobStorePort):
    """TrainJobStorePort backed by MLflow tags + one JSON artifact."""

    def __init__(self, tracking_uri: str | None = None, client=None) -> None:
        self._tracking_uri = tracking_uri or BaseMLflowTracker.TRACKING_URI
        self._client = client

    def __getstate__(self) -> dict:
        # The training child process builds its own client.
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    def get(self, job_id: str) -> TrainJobRecord | None:
        from mlflow.exceptions import MlflowException
        try:
            run = self.client.get_run(job_id)
        except MlflowException as exc:
            if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise
        tags = run.data.tags
        status = tags.get(f"{TAG_PREFIX}status")
        if not status:
            return None
        status = TrainJobStatus(status)
        return TrainJobRecord(
            job_id=job_id,
            model_id=tags.get(f"{TAG_PREFIX}model_id", ""),
            status=status,
            submitted_at=float(tags[f"{TAG_PREFIX}submitted_at"]),
            updated_at=float(tags[f"{TAG_PREFIX}updated_at"]),
            result=self._load_result(job_id) if status is TrainJobStatus.SUCCEEDED else None,
            error=tags.get(f"{TAG_PREFIX}error") or None,
            error_type=tags.get(f"{TAG_PREFIX}error_type") or None,
        )

    def save(self, record: TrainJobRecord) -> None:
        if record.result is not None:
            self.client.log_text(record.job_id, json.dumps(record.result, ensure_ascii=False), RESULT_ARTIFACT)
        tags = {
            "model_id": record.model_id,
            "submitted_at": str(record.submitted_at),
            "updated_at": str(record.updated_at),
        }
        if record.error:
            tags["error"] = record.error[:_MAX_ERROR_CHARS]
        if record.error_type:
            tags["error_type"] = record.error_type
        for key, value in tags.items():
            self.client.set_tag(record.job_id, f"{TAG_PREFIX}{key}", value)
        # Status last: a reader never sees "succeeded" before result.json exists.
        self.client.set_tag(record.job_id, f"{TAG_PREFIX}status", record.status.value)

    def heartbeat(self, job_id: str, at: float) -> None:
        self.client.set_tag(job_id, f"{TAG_PREFIX}updated_at", str(at))

    def _load_result(self, job_id: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            local = self.client.download_artifacts(job_id, RESULT_ARTIFACT, tmp)
            with open(local, encoding="utf-8") as fh:
                return json.load(fh)
```

- [ ] **Step 4: Ejecutar y comprobar que pasa**

Run: `python3 -m pytest -p no:flask tests/unit/test_mlflow_train_job_store.py -v`
Expected: todos PASS.

- [ ] **Step 5: Dejar listo (sin commit)**

Run: `python3 -m flake8 app/infrastructure/train_jobs tests/unit/test_mlflow_train_job_store.py`
Expected: sin salida.

---

### Task 3: Ejecutor en proceso aparte

**Files:**
- Create: `app/infrastructure/train_jobs/process_executor.py`
- Test: `tests/unit/test_process_train_executor.py`

**Interfaces:**
- Consumes: `TrainExecutorPort`, `TrainJobStorePort`, `TrainJobRecord`, `TrainJobStatus` (Task 1); fakes `FileJobStore`, `EchoTrainPlugin`, `FailingTrainPlugin`, `SlowTrainPlugin`, `NumpyMetricsTrainPlugin`.
- Produces:
  - `run_train_job(*, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort, heartbeat_s: float = HEARTBEAT_S) -> None` — función de módulo (picklable), usable también en el mismo proceso (Task 4 la usa así en tests).
  - `ProcessTrainExecutor(heartbeat_s: float = HEARTBEAT_S)` con `is_busy()` y `submit(...)`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/unit/test_process_train_executor.py`:

```python
"""ProcessTrainExecutor with real `spawn` child processes and a file-backed store."""
from __future__ import annotations

import os
import time

from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from app.infrastructure.train_jobs.process_executor import ProcessTrainExecutor, run_train_job
from tests.unit.train_job_fakes import (
    EchoTrainPlugin,
    FailingTrainPlugin,
    FileJobStore,
    NumpyMetricsTrainPlugin,
    SlowTrainPlugin,
)


def _queue(store, job_id):
    now = time.time()
    store.save(TrainJobRecord(job_id, "fake-model", TrainJobStatus.QUEUED, now, now))


def _wait(predicate, timeout=60.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {what}")


def _terminal(store, job_id):
    return _wait(lambda: (r := store.get(job_id)) is not None and r.status.is_terminal and r, what=f"{job_id} terminal")


def _submit(executor, store, job_id, plugin_class, data_path="s3://b/d.csv"):
    _queue(store, job_id)
    executor.submit(
        job_id=job_id, plugin_class=plugin_class,
        train_kwargs={"data_path": data_path, "mlflow_run_id": job_id}, store=store,
    )


def test_trains_in_a_child_process_and_stores_the_result(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-1", EchoTrainPlugin)
    record = _terminal(store, "run-1")
    assert record.status is TrainJobStatus.SUCCEEDED, record.error
    assert record.result["detail"] == "ok"
    assert record.result["mlflow_run_id"] == "run-1"
    assert record.result["metrics"]["data_path"] == "s3://b/d.csv"
    assert record.result["metrics"]["pid"] != os.getpid()


def test_failure_is_stored_with_its_type(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-2", FailingTrainPlugin, data_path="s3://b/missing.csv")
    record = _terminal(store, "run-2")
    assert record.status is TrainJobStatus.FAILED
    assert record.error_type == "FileNotFoundError"
    assert "s3://b/missing.csv" in record.error


def test_numpy_metrics_are_stored_as_plain_json(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-3", NumpyMetricsTrainPlugin)
    record = _terminal(store, "run-3")
    assert record.status is TrainJobStatus.SUCCEEDED, record.error
    assert record.result["metrics"] == {"f1": 0.5, "n": 3, "cm": [[1, 0], [0, 2]]}


def test_busy_while_alive_and_free_afterwards(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-4", SlowTrainPlugin, data_path="2")
    assert executor.is_busy()
    _terminal(store, "run-4")
    _wait(lambda: not executor.is_busy(), timeout=10, what="executor free")


def test_heartbeat_advances_updated_at_while_running(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-5", SlowTrainPlugin, data_path="3")
    first = _wait(lambda: (r := store.get("run-5")).status is TrainJobStatus.RUNNING and r, what="running")
    time.sleep(1.0)
    second = store.get("run-5")
    assert second.status is TrainJobStatus.RUNNING
    assert second.updated_at > first.updated_at
    assert _terminal(store, "run-5").status is TrainJobStatus.SUCCEEDED


def test_run_train_job_without_record_does_nothing(tmp_path):
    store = FileJobStore(str(tmp_path))
    run_train_job(job_id="ghost", plugin_class=EchoTrainPlugin, train_kwargs={}, store=store, heartbeat_s=60)
    assert store.get("ghost") is None
```

- [ ] **Step 2: Ejecutar y comprobar que falla**

Run: `python3 -m pytest -p no:flask tests/unit/test_process_train_executor.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.infrastructure.train_jobs.process_executor'`.

- [ ] **Step 3: Implementar**

`app/infrastructure/train_jobs/process_executor.py`:

```python
"""Run plugin training in a separate process (multiprocessing ``spawn``).

The child builds its own plugin (``plugin_class()`` + ``load()``) so training never shares the GIL,
memory or in-memory model with the process serving predictions, and is discarded when it ends.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import threading
import time
from dataclasses import replace
from typing import Any

from app.domain.ports.train_executor_port import TrainExecutorPort
from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.train_job import TrainJobStatus

logger = logging.getLogger(__name__)

HEARTBEAT_S = float(os.getenv("TRAIN_JOB_HEARTBEAT_S", "60"))


def _json_default(value: Any) -> Any:
    # numpy scalars/arrays (tolist/item) and datetimes inside free-form metric dicts.
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _to_jsonable(response: Any) -> dict:
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        raise TypeError(f"train() returned {type(response).__name__}; expected a pydantic model or a dict")
    return json.loads(json.dumps(data, default=_json_default))


def _heartbeat_loop(store: TrainJobStorePort, job_id: str, stop: threading.Event, every_s: float) -> None:
    while not stop.wait(every_s):
        try:
            store.heartbeat(job_id, time.time())
        except Exception:
            logger.warning("Heartbeat failed for training job %s", job_id, exc_info=True)


def run_train_job(
    *,
    job_id: str,
    plugin_class: type,
    train_kwargs: dict,
    store: TrainJobStorePort,
    heartbeat_s: float = HEARTBEAT_S,
) -> None:
    """Child-process entry point (also callable in-process). Never raises: the outcome goes to ``store``."""
    logging.basicConfig(level=logging.INFO)
    record = store.get(job_id)
    if record is None:
        logger.error("Training job %s not found in the store; aborting", job_id)
        return
    record = replace(record, status=TrainJobStatus.RUNNING, updated_at=time.time())
    store.save(record)

    stop = threading.Event()
    beater = threading.Thread(target=_heartbeat_loop, args=(store, job_id, stop, heartbeat_s), daemon=True)
    beater.start()
    try:
        plugin = plugin_class()
        plugin.load()
        result = _to_jsonable(plugin.train(**train_kwargs))
        final = replace(record, status=TrainJobStatus.SUCCEEDED, result=result, updated_at=time.time())
        logger.info("Training job %s succeeded", job_id)
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        final = replace(
            record, status=TrainJobStatus.FAILED, updated_at=time.time(),
            error=str(exc) or type(exc).__name__, error_type=type(exc).__name__,
        )
    finally:
        # Join before the final save so a late heartbeat can never overwrite the terminal state.
        stop.set()
        beater.join()
    store.save(final)


class ProcessTrainExecutor(TrainExecutorPort):
    """One child process per job; ``is_busy`` while any submitted child is alive."""

    def __init__(self, heartbeat_s: float = HEARTBEAT_S) -> None:
        self._ctx = mp.get_context("spawn")  # fork + torch/OpenMP threads can deadlock
        self._heartbeat_s = heartbeat_s
        self._procs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        with self._lock:
            # is_alive() also reaps finished children.
            self._procs = {job_id: p for job_id, p in self._procs.items() if p.is_alive()}
            return bool(self._procs)

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        proc = self._ctx.Process(
            target=run_train_job,
            kwargs={
                "job_id": job_id, "plugin_class": plugin_class, "train_kwargs": train_kwargs,
                "store": store, "heartbeat_s": self._heartbeat_s,
            },
            name=f"train-{job_id}",
            daemon=False,  # daemonic processes cannot have children (e.g. torch DataLoader workers)
        )
        proc.start()
        with self._lock:
            self._procs[job_id] = proc
        logger.info("Training job %s started in child process pid=%s", job_id, proc.pid)
```

- [ ] **Step 4: Ejecutar y comprobar que pasa**

Run: `python3 -m pytest -p no:flask tests/unit/test_process_train_executor.py -v`
Expected: 6 PASS (tarda unos segundos: cada test arranca un intérprete con `spawn`).

- [ ] **Step 5: Dejar listo (sin commit)**

Run: `python3 -m flake8 app/infrastructure/train_jobs tests/unit/test_process_train_executor.py`
Expected: sin salida.

---

### Task 4: Endpoints HTTP y cableado

**Files:**
- Create: `app/application/dto/train_job_dto.py`
- Modify: `app/infrastructure/http/router_factory.py` (endpoint `train`, líneas 100-118, y nuevo `GET`)
- Modify: `app/infrastructure/http/dependencies/container.py`
- Modify: `main.py:39`
- Modify: `tests/unit/train_job_fakes.py` (añadir `InlineExecutor`)
- Test: `tests/unit/test_train_async_router.py`

**Interfaces:**
- Consumes: `SubmitTrainJobUseCase`, `GetTrainJobUseCase`, `TrainJobBusyError` (Task 1); `MLflowTrainJobStore` (Task 2); `ProcessTrainExecutor`, `run_train_job` (Task 3).
- Produces:
  - `TrainJobResponse` (pydantic) con `from_record(record) -> TrainJobResponse`; JSON idéntico al `TrainJob` de la spec.
  - Atributos del contenedor: `submit_train_job_use_case`, `get_train_job_use_case`.
  - `ModelContainer(plugin, model_id: str = "")`.
  - HTTP: `POST /train?wait=false` → 202/400/409/503; `GET /train/{job_id}` → 200/404/503.

- [ ] **Step 1: Añadir `InlineExecutor` a los fakes**

Al final de `tests/unit/train_job_fakes.py`:

```python


class InlineExecutor(TrainExecutorPort):
    """Runs the job synchronously in the calling thread (HTTP tests)."""

    def __init__(self, busy: bool = False) -> None:
        self.busy = busy
        self.calls = 0

    def is_busy(self) -> bool:
        return self.busy

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        from app.infrastructure.train_jobs.process_executor import run_train_job

        self.calls += 1
        run_train_job(job_id=job_id, plugin_class=plugin_class, train_kwargs=train_kwargs, store=store, heartbeat_s=3600)
```

- [ ] **Step 2: Escribir los tests que fallan**

`tests/unit/test_train_async_router.py`:

```python
"""HTTP contract of async training: POST /train?wait=false and GET /train/{job_id}."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.application.use_cases.train_job_use_cases import GetTrainJobUseCase, SubmitTrainJobUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.infrastructure.http.router_factory import make_model_router
from tests.unit.train_job_fakes import EchoTrainPlugin, InlineExecutor, InMemoryJobStore

MODEL_ID = "fake-model"
BASE = f"/models/{MODEL_ID}"
BODY = {"data_path": "s3://bucket/train.csv", "mlflow_run_id": "run-1"}


class _PredictRequest(BaseModel):
    x: float = 0.0


class _PredictResponse(BaseModel):
    y: float = 0.0


class _BrokenStore(InMemoryJobStore):
    def get(self, job_id):
        raise ConnectionError("MLflow unreachable")


def _client(executor=None, store=None) -> tuple[TestClient, InMemoryJobStore, InlineExecutor]:
    store = store if store is not None else InMemoryJobStore()
    executor = executor if executor is not None else InlineExecutor()
    plugin = EchoTrainPlugin()
    plugin.load()
    container = SimpleNamespace(
        train_use_case=TrainModelUseCase(plugin),
        submit_train_job_use_case=SubmitTrainJobUseCase(plugin, MODEL_ID, store, executor),
        get_train_job_use_case=GetTrainJobUseCase(MODEL_ID, store),
    )
    app = FastAPI()
    app.state.containers = {MODEL_ID: container}
    app.state.load_errors = {}
    app.include_router(
        make_model_router(
            model_id=MODEL_ID, version="1.0.0",
            predict_request_type=_PredictRequest, predict_response_type=_PredictResponse,
            train_request_type=None, train_response_type=None,
        ),
        prefix=BASE,
    )
    return TestClient(app), store, executor


def test_async_post_returns_202_queued_then_get_returns_result():
    client, _, executor = _client()
    resp = client.post(f"{BASE}/train?wait=false", json=BODY)
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == "run-1"
    assert body["model_id"] == MODEL_ID
    assert body["status"] == "queued"
    assert body["submitted_at"].endswith("+00:00")
    assert executor.calls == 1

    got = client.get(f"{BASE}/train/run-1")
    assert got.status_code == 200
    assert got.json()["status"] == "succeeded"
    assert got.json()["result"]["detail"] == "ok"
    assert got.json()["error"] is None


def test_default_post_stays_synchronous():
    client, store, executor = _client()
    resp = client.post(f"{BASE}/train", json=BODY)
    assert resp.status_code == 200
    assert resp.json()["detail"] == "ok"
    assert store.records == {}
    assert executor.calls == 0


def test_wait_true_is_the_synchronous_path():
    client, store, _ = _client()
    assert client.post(f"{BASE}/train?wait=true", json=BODY).status_code == 200
    assert store.records == {}


def test_redelivered_post_is_idempotent():
    client, _, executor = _client()
    client.post(f"{BASE}/train?wait=false", json=BODY)
    again = client.post(f"{BASE}/train?wait=false", json=BODY)
    assert again.status_code == 202
    assert again.json()["status"] == "succeeded"
    assert executor.calls == 1


def test_empty_run_id_is_400():
    client, _, _ = _client()
    resp = client.post(f"{BASE}/train?wait=false", json={**BODY, "mlflow_run_id": ""})
    assert resp.status_code == 400
    assert "mlflow_run_id" in resp.json()["detail"]


def test_busy_model_is_409():
    client, _, _ = _client(executor=InlineExecutor(busy=True))
    resp = client.post(f"{BASE}/train?wait=false", json=BODY)
    assert resp.status_code == 409


def test_unknown_job_is_404():
    client, _, _ = _client()
    assert client.get(f"{BASE}/train/does-not-exist").status_code == 404


@pytest.mark.parametrize("method,path", [("post", "/train?wait=false"), ("get", "/train/run-1")])
def test_store_down_is_503(method, path):
    client, _, _ = _client(store=_BrokenStore())
    resp = getattr(client, method)(f"{BASE}{path}", **({"json": BODY} if method == "post" else {}))
    assert resp.status_code == 503
    assert "MLflow unreachable" in resp.json()["detail"]
```

- [ ] **Step 3: Ejecutar y comprobar que falla**

Run: `python3 -m pytest -p no:flask tests/unit/test_train_async_router.py -v`
Expected: FAIL — `test_async_post_returns_202…`, `test_redelivered_post_is_idempotent`, `test_empty_run_id_is_400`, `test_busy_model_is_409` y el caso POST de `test_store_down_is_503` reciben `200` (el parámetro `wait` aún no existe y se ignora); el caso GET de `test_store_down_is_503` recibe `404` (la ruta no existe). `test_unknown_job_is_404` y los dos tests síncronos ya pasan.

- [ ] **Step 4: Crear el DTO**

`app/application/dto/train_job_dto.py`:

```python
"""Response body of the async training endpoints (POST /train?wait=false, GET /train/{job_id})."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.services.train_job import TrainJobRecord


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


class TrainJobResponse(BaseModel):
    """State of an async training job. ``job_id`` is the MLflow run id sent in the TrainRequest."""

    job_id: str
    model_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    submitted_at: datetime
    updated_at: datetime
    result: dict | None = Field(default=None, description="TrainResponse del plugin, solo si succeeded")
    error: str | None = Field(default=None, description="Mensaje de error, solo si failed")
    error_type: str | None = Field(
        default=None,
        description="Clase del error (ValueError, FileNotFoundError, TrainingNotSupportedError, TrainJobLost…)",
    )

    @classmethod
    def from_record(cls, record: TrainJobRecord) -> "TrainJobResponse":
        return cls(
            job_id=record.job_id,
            model_id=record.model_id,
            status=record.status.value,
            submitted_at=_utc(record.submitted_at),
            updated_at=_utc(record.updated_at),
            result=record.result,
            error=record.error,
            error_type=record.error_type,
        )
```

- [ ] **Step 5: Modificar el router**

En `app/infrastructure/http/router_factory.py`, añadir a los imports:

```python
from app.application.dto.train_job_dto import TrainJobResponse
from app.domain.services.exceptions import TrainJobBusyError, TrainingNotSupportedError
```

(y quitar la línea antigua `from app.domain.services.exceptions import TrainingNotSupportedError`).

Sustituir el endpoint `train` completo por:

```python
    @router.post("/train", responses={202: {"model": TrainJobResponse}})
    def train(request: Request, body: _train_req, wait: bool = True) -> _train_resp:
        """Train the model. ``wait=false`` starts it in the background and returns 202 + the job state."""
        container = _get_container(request)
        if not wait:
            try:
                record = container.submit_train_job_use_case.execute(body)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except TrainJobBusyError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Could not register async training for model '%s'", model_id)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"No se pudo registrar el entrenamiento: {exc}",
                ) from exc
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=TrainJobResponse.from_record(record).model_dump(mode="json"),
            )
        try:
            return container.train_use_case.execute(body)
        except TrainingNotSupportedError as exc:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
            ) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during training for model '%s'", model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    @router.get("/train/{job_id}")
    def train_status(request: Request, job_id: str) -> TrainJobResponse:
        """State of an async training job started with POST /train?wait=false."""
        container = _get_container(request)
        try:
            record = container.get_train_job_use_case.execute(job_id)
        except Exception as exc:
            logger.exception("Could not read training job %s for model '%s'", job_id, model_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No se pudo consultar el entrenamiento: {exc}",
            ) from exc
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hay entrenamiento {job_id} para el modelo {model_id}",
            )
        return TrainJobResponse.from_record(record)
```

- [ ] **Step 6: Ejecutar los tests HTTP**

Run: `python3 -m pytest -p no:flask tests/unit/test_train_async_router.py -v`
Expected: todos PASS.

- [ ] **Step 7: Cablear el contenedor y `main.py`**

En `app/infrastructure/http/dependencies/container.py`, añadir imports:

```python
from app.application.use_cases.train_job_use_cases import GetTrainJobUseCase, SubmitTrainJobUseCase
from app.infrastructure.train_jobs.mlflow_job_store import MLflowTrainJobStore
from app.infrastructure.train_jobs.process_executor import ProcessTrainExecutor
```

y sustituir `__init__` por:

```python
    def __init__(self, plugin: ModelPluginPort, model_id: str = "") -> None:
        """Recibe el plugin concreto y crea los casos de uso."""
        self._plugin = plugin
        self._service = ModelRuntimeService(plugin)
        self.predict_use_case = PredictModelUseCase(plugin)
        self.stats_use_case = GetStatsUseCase(plugin)
        self.train_use_case = TrainModelUseCase(plugin)
        # Async training (POST /train?wait=false): state on the MLflow run, training in a child process.
        # Neither builds anything heavy here: the MLflow client is created lazily, no process starts yet.
        train_job_store = MLflowTrainJobStore()
        self.submit_train_job_use_case = SubmitTrainJobUseCase(
            plugin, model_id, train_job_store, ProcessTrainExecutor()
        )
        self.get_train_job_use_case = GetTrainJobUseCase(model_id, train_job_store)
```

En `main.py:39` cambiar:

```python
            container = ModelContainer(plugin=entry.plugin_class())
```

por:

```python
            container = ModelContainer(plugin=entry.plugin_class(), model_id=entry.model_id)
```

- [ ] **Step 8: Ejecutar toda la batería unitaria**

Run: `python3 -m pytest -p no:flask tests/unit -q`
Expected: todo PASS, incluidos `test_infrastructure.py` (usa `ModelContainer(plugin=…)` sin `model_id`) y los tests de `/train` existentes vía `conftest.py` (sin `wait` siguen síncronos). Si algún test previo ya fallaba en `main` antes de esta rama, comprobarlo con `git stash` y anotarlo; no se arregla aquí.

- [ ] **Step 9: Dejar listo (sin commit)**

Run: `python3 -m flake8 app main.py tests/unit/train_job_fakes.py tests/unit/test_train_async_router.py`
Expected: sin salida nueva respecto a `main` (comparar con `git stash; python3 -m flake8 app main.py; git stash pop` si aparece algo preexistente).

---

### Task 5: Documentación del contrato y prueba en un entorno con MLflow

**Files:**
- Modify: `README.md` (nueva sección "Entrenamiento asíncrono")

**Interfaces:**
- Consumes: el contrato de Task 4.
- Produces: la referencia que usarán los planes del orquestador y de la plataforma.

- [ ] **Step 1: Documentar en el README**

Añadir al `README.md`, tras la sección que describe los endpoints `/train`:

````markdown
## Entrenamiento asíncrono

`POST /models/<model-id>/train?wait=false` lanza el entrenamiento en un proceso aparte y responde `202`
con el estado del trabajo. El `job_id` es el `mlflow_run_id` del cuerpo (obligatorio). Sin `wait`, o con
`wait=true`, `/train` sigue siendo síncrono como hasta ahora.

```bash
curl -s -X POST "$BASE/models/<model-id>/train?wait=false" \
  -H 'Content-Type: application/json' \
  -d '{"data_path": "s3://<bucket>/<dataset>.csv", "mlflow_run_id": "<run_id>"}'
# 202 {"job_id": "<run_id>", "status": "queued", ...}

curl -s "$BASE/models/<model-id>/train/<run_id>"
# 200 {"status": "running" | "succeeded" | "failed", "result": {...}, "error": ..., "error_type": ...}
```

| Código | Significado |
|---|---|
| 202 | Trabajo creado o ya existente para ese `job_id` (reintentar el POST es seguro) |
| 400 | `mlflow_run_id` vacío |
| 409 | Ya hay otro entrenamiento de ese modelo en curso en el pod |
| 404 | (GET) No hay trabajo con ese `job_id` para este modelo |
| 503 | MLflow no accesible |

El estado vive en el run de MLflow (etiquetas `async_train.*` y artefacto `async_train/result.json`).
Si el proceso deja de dar señales más de `TRAIN_JOB_STALE_AFTER_S` segundos (600 por defecto; latido cada
`TRAIN_JOB_HEARTBEAT_S`, 60), el GET lo devuelve como `failed` con `error_type: "TrainJobLost"`.

Requisito por modelo: su `train()` no debe sobrescribir el artefacto fijo en disco (se comparte con el
proceso que sirve predicciones).
````

- [ ] **Step 2: Prueba manual en INT (con MLflow real)**

Esta prueba no se puede hacer en el WSL local (no arranca `main.py` completo ni hay MLflow). Desplegada la
imagen en INT, con un run pre-creado en MLflow y un CSV de entrenamiento del m48/m47 en S3:

```bash
BASE=https://<host-int>/models/m48-dnsl-fallas-maquinaria-pasteurizado
curl -s -o /dev/null -w '%{http_code}\n' -X POST "$BASE/train?wait=false" -H 'Content-Type: application/json' \
  -d '{"data_path":"s3://<bucket>/<csv>","mlflow_run_id":"<run_id>"}'        # esperado: 202
curl -s "$BASE/train/<run_id>"                                                # esperado: queued/running
curl -s -X POST "$BASE/train?wait=false" -H 'Content-Type: application/json' \
  -d '{"data_path":"s3://<bucket>/<csv>","mlflow_run_id":"<otro_run>"}'      # esperado: 409 mientras entrena
curl -s "$BASE/predict" ...                                                    # las predicciones siguen respondiendo
# al terminar:
curl -s "$BASE/train/<run_id>"                                                # esperado: succeeded + result
```

Comprobar en la UI de MLflow que el run tiene las etiquetas `async_train.*` y el artefacto
`async_train/result.json`, y que el resultado coincide con el que devolvía el `/train` síncrono.

- [ ] **Step 3: Dejar listo (sin commit)**

Revisar `git status` y confirmar que solo aparecen los ficheros de la tabla "Estructura de ficheros" más
la spec y este plan. No hacer commit: lo hace el usuario.

---

## Después de este plan

- **Plan 2 — orquestador** (`retech-lote2-xai-orquestador`): la tarea `train` hace `POST …/train?wait=false`
  y termina; tarea de seguimiento con `self.retry(countdown=…)` sobre `GET …/train/{job_id}`; mapeo de
  `error_type` según la spec; contrato hacia la plataforma sin cambios.
- **Plan 3 — plataforma** (`retech-lote2-xai-plataforma`): sustituir `pollUntilDone` en memoria por
  seguimiento persistente con barrido periódico, quitar el límite de 30 min y enviar
  `AlertMailer.sendTrainingFinishedMail` al terminar.
- **Repo de Bitbucket** (`itacyl-pan-modelsinference-svc`, rama `staging`): su `router_factory.py` es
  idéntico al de este repo; portar la capa cuando el usuario lo decida.
