"""HTTP contract of async batch prediction: POST /predict?wait=false and GET /predict/jobs/{id}."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.use_cases.predict_job_use_cases import (
    GetPredictJobUseCase,
    SubmitPredictJobUseCase,
)
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.domain.services.exceptions import (
    TrainJobResultUnreadableError,
    TrainJobStoreUnavailableError,
)
from app.infrastructure.http.dependencies.container import ModelContainer
from app.infrastructure.http.router_factory import make_model_router
from app.infrastructure.predict_jobs.mlflow_job_store import MLflowPredictJobStore
from app.infrastructure.predict_jobs.process_executor import ProcessPredictExecutor
from tests.unit.predict_job_fakes import (
    EXTRA_PREDICT_EXCEPTIONS,
    PREDICT_ERRORS,
    EchoPredictPlugin,
    FakePredictRequest,
    FakePredictResponse,
    InlinePredictExecutor,
    InMemoryPredictJobStore,
    RecordingPredictExecutor,
    batch_body,
)

MODEL_ID = "fake-model"
BASE = f"/models/{MODEL_ID}"


def _client(store=None, executor=None, async_predict=True):
    plugin = EchoPredictPlugin()
    plugin.load()
    store = store if store is not None else InMemoryPredictJobStore()
    executor = executor if executor is not None else InlinePredictExecutor()
    container = SimpleNamespace(
        predict_use_case=PredictModelUseCase(plugin),
        submit_predict_job_use_case=SubmitPredictJobUseCase(
            plugin, MODEL_ID, store, executor,
            extra_predict_exceptions=EXTRA_PREDICT_EXCEPTIONS,
        ),
        get_predict_job_use_case=GetPredictJobUseCase(MODEL_ID, store),
    )
    app = FastAPI()
    app.state.containers = {MODEL_ID: container}
    app.include_router(make_model_router(
        model_id=MODEL_ID, version="1.0.0",
        predict_request_type=FakePredictRequest, predict_response_type=FakePredictResponse,
        train_request_type=None, train_response_type=None,
        extra_predict_exceptions=EXTRA_PREDICT_EXCEPTIONS, async_predict=async_predict,
    ), prefix=BASE)
    return TestClient(app), store


def test_async_result_equals_the_synchronous_response():
    client, _ = _client()
    body = batch_body(mlflow_run_id="user-run")
    sync = client.post(f"{BASE}/predict", json=body)
    assert sync.status_code == 200

    submitted = client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "t-1"}, json=body)
    assert submitted.status_code == 202
    assert submitted.json()["job_id"] == "t-1"

    job = client.get(f"{BASE}/predict/jobs/t-1")
    assert job.status_code == 200
    data = job.json()
    assert data["status"] == "succeeded"
    assert data["result"] == sync.json()
    assert data["result"]["predictions"][0]["run"] == "user-run"  # user model reached the plugin


def test_submit_is_idempotent_and_answers_202_again():
    client, _ = _client(executor=RecordingPredictExecutor())
    first = client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "t-1"},
                        json=batch_body())
    again = client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "t-1"},
                        json=batch_body())
    assert (first.status_code, again.status_code) == (202, 202)
    assert again.json()["status"] == "queued"


def test_model_without_async_prediction_answers_501_and_sync_still_works():
    client, _ = _client(async_predict=False)
    assert client.post(f"{BASE}/predict", params={"wait": "false"},
                       json=batch_body()).status_code == 501
    assert client.get(f"{BASE}/predict/jobs/t-1").status_code == 501
    assert client.post(f"{BASE}/predict", json=batch_body()).status_code == 200


def test_inline_with_wait_false_is_answered_synchronously():
    client, store = _client()
    res = client.post(f"{BASE}/predict", params={"wait": "false"}, json={"mode": "inline", "x": 2})
    assert res.status_code == 200
    assert res.json()["prediction"] == 4
    assert not store.records


def test_busy_is_409_and_bad_job_id_is_400():
    client, _ = _client(executor=RecordingPredictExecutor(busy=True))
    assert client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "t-1"},
                       json=batch_body()).status_code == 409
    client, _ = _client()
    assert client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "a b"},
                       json=batch_body()).status_code == 400


@pytest.mark.parametrize("key", sorted(PREDICT_ERRORS))
def test_failed_job_carries_the_synchronous_status(key):
    client, _ = _client()
    _, code = PREDICT_ERRORS[key]
    assert client.post(f"{BASE}/predict", json=batch_body(key)).status_code == code
    client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "t-1"}, json=batch_body(key))
    data = client.get(f"{BASE}/predict/jobs/t-1").json()
    assert (data["status"], data["http_status"]) == ("failed", code)
    assert data["result"] is None and data["error"]


def test_get_unknown_job_is_404():
    client, _ = _client()
    assert client.get(f"{BASE}/predict/jobs/nope").status_code == 404


class _BrokenStore(InMemoryPredictJobStore):
    def get(self, job_id):
        raise TrainJobStoreUnavailableError("MLflow unreachable")


class _UnreadableStore(InMemoryPredictJobStore):
    def get(self, job_id):
        raise TrainJobResultUnreadableError("result.json.gz es una carpeta")


def test_store_unavailable_is_503_and_unreadable_result_is_500():
    client, _ = _client(store=_BrokenStore())
    assert client.get(f"{BASE}/predict/jobs/t-1").status_code == 503
    assert client.post(f"{BASE}/predict", params={"wait": "false", "job_id": "t-1"},
                       json=batch_body()).status_code == 503
    client, _ = _client(store=_UnreadableStore())
    res = client.get(f"{BASE}/predict/jobs/t-1")
    assert res.status_code == 500 and "carpeta" in res.json()["detail"]


def test_container_wires_the_async_prediction_use_cases():
    container = ModelContainer(
        plugin=EchoPredictPlugin(), model_id="m5", extra_predict_exceptions=EXTRA_PREDICT_EXCEPTIONS
    )
    submit = container.submit_predict_job_use_case
    assert isinstance(submit, SubmitPredictJobUseCase)
    assert isinstance(container.get_predict_job_use_case, GetPredictJobUseCase)
    # pylint: disable=protected-access  # checks the wiring without starting anything
    assert isinstance(submit._store, MLflowPredictJobStore)
    assert isinstance(submit._executor, ProcessPredictExecutor)
    assert submit._extra == EXTRA_PREDICT_EXCEPTIONS


def test_registry_enables_async_prediction_only_for_ml5():
    from app.registry import REGISTRY  # pylint: disable=import-outside-toplevel
    assert [e.model_id for e in REGISTRY if e.async_predict] == ["ml5-meat-cow-behaviour"]
