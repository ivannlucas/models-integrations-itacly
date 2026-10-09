"""HTTP contract of async training: POST /train?wait=false and GET /train/{job_id}."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.application.use_cases.train_job_use_cases import GetTrainJobUseCase, SubmitTrainJobUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.services.exceptions import TrainJobRunNotFoundError, TrainJobStoreUnavailableError
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
        raise TrainJobStoreUnavailableError("MLflow unreachable")


class _MissingRunStore(InMemoryJobStore):
    def save(self, record):
        raise TrainJobRunNotFoundError(f"El run {record.job_id} no existe en MLflow")


class _BuggyStore(InMemoryJobStore):
    def get(self, job_id):
        raise RuntimeError("bug")


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
    assert body["submitted_at"].endswith(("Z", "+00:00"))  # UTC; pydantic v2 emits "Z"
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


def test_nonexistent_run_is_400_not_retryable_503():
    client, _, executor = _client(store=_MissingRunStore())
    resp = client.post(f"{BASE}/train?wait=false", json=BODY)
    assert resp.status_code == 400
    assert "run-1" in resp.json()["detail"]
    assert executor.calls == 0


@pytest.mark.parametrize("method,path", [("post", "/train?wait=false"), ("get", "/train/run-1")])
def test_programming_errors_are_500_not_503(method, path):
    client, _, _ = _client(store=_BuggyStore())
    resp = getattr(client, method)(f"{BASE}{path}", **({"json": BODY} if method == "post" else {}))
    assert resp.status_code == 500


def test_model_container_wires_async_training_lazily():
    from app.infrastructure.http.dependencies.container import ModelContainer
    from app.infrastructure.train_jobs.mlflow_job_store import MLflowTrainJobStore
    from app.infrastructure.train_jobs.process_executor import ProcessTrainExecutor

    container = ModelContainer(plugin=EchoTrainPlugin(), model_id="m47")
    submit, get = container.submit_train_job_use_case, container.get_train_job_use_case
    assert isinstance(submit, SubmitTrainJobUseCase) and isinstance(get, GetTrainJobUseCase)
    assert submit._model_id == get._model_id == "m47"
    assert isinstance(submit._executor, ProcessTrainExecutor) and not submit._executor.is_busy()
    assert isinstance(submit._store, MLflowTrainJobStore) and submit._store is get._store
    assert submit._store._client is None  # no MLflow connection until a job is submitted or read
