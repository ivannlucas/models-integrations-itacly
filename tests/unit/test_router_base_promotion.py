"""HTTP surface of the versioned base: /train answers base_version, /rollback, /stats."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.infrastructure.http.dependencies.container import ModelContainer
from app.infrastructure.http.router_factory import make_model_router
from tests.conftest import TEST_REGISTRY
from tests.unit.tiny_promoting_plugin import (
    TinyPredictRequest,
    TinyPredictResponse,
    TinyPromotingPlugin,
    TinyTrainRequest,
    TinyTrainResponse,
)

PREFIX = "/models/tiny"


@pytest.fixture
def tiny_client():
    TinyPromotingPlugin.fail_load = False
    app = FastAPI()
    container = ModelContainer(TinyPromotingPlugin())
    container.init()
    app.state.containers = {"tiny": container}
    app.include_router(make_model_router(
        model_id="tiny", version="1", predict_request_type=TinyPredictRequest,
        predict_response_type=TinyPredictResponse, train_request_type=TinyTrainRequest,
        train_response_type=TinyTrainResponse,
    ), prefix=PREFIX)
    yield TestClient(app)
    TinyPromotingPlugin.fail_load = False


def test_train_without_run_answers_the_new_base_version(tiny_client):
    resp = tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    assert resp.status_code == 200
    assert resp.json()["base_version"]["version"] == "v1"
    assert tiny_client.get(f"{PREFIX}/stats").json()["base_version"]["version"] == "v1"


def test_rollback_endpoint(tiny_client):
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d2"})
    resp = tiny_client.post(f"{PREFIX}/rollback")
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_version"]["version"] == "v1" and body["previous_base_version"]["version"] == "v2"
    resp = tiny_client.post(f"{PREFIX}/rollback", json={"version": "v0"})
    assert resp.status_code == 200 and resp.json()["base_version"]["version"] == "v0"


def test_rollback_errors(tiny_client):
    assert tiny_client.post(f"{PREFIX}/rollback").status_code == 409            # v0, no parent
    assert tiny_client.post(f"{PREFIX}/rollback", json={"version": "v7"}).status_code == 404
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    TinyPromotingPlugin.fail_load = True
    resp = tiny_client.post(f"{PREFIX}/rollback")
    assert resp.status_code == 500 and "v0" in resp.json()["detail"]


def test_train_reload_failure_is_500(tiny_client):
    TinyPromotingPlugin.fail_load = True
    resp = tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    assert resp.status_code == 500 and "vuelve a v0" in resp.json()["detail"]


def test_models_without_versions_answer_501_on_rollback(client):
    entry = TEST_REGISTRY[0]
    resp = client.post(f"{entry.prefix}/rollback")
    assert resp.status_code == 501


def test_rollback_unexpected_error_is_500(tiny_client):
    container = tiny_client.app.state.containers["tiny"]
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})

    def boom(_version):
        raise OSError("s3 caído")

    container._base_store.set_active_version = boom  # pylint: disable=protected-access
    resp = tiny_client.post(f"{PREFIX}/rollback")
    assert resp.status_code == 500 and "s3 caído" in resp.json()["detail"]
