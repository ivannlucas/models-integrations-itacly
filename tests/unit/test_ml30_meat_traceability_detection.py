"""Endpoint tests for the ``ml30-meat-traceability-detection`` model."""
from app.domain.services.exceptions import DataContractError

PREFIX = "/models/ml30-meat-traceability-detection"

INLINE_PAYLOAD = {"mode": "inline", "sensor_temp_c": 7.5, "stage": "deboning"}


def test_health(client):
    """GET /health reports the model as loaded and ok."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml30-meat-traceability-detection"
    assert body["loaded"] is True


def test_stats(client):
    """GET /stats returns the model name."""
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml30-meat-traceability-detection"


def test_predict_inline_rejected(client):
    """Inline prediction removed for this model (product decision) — the /predict
    route only accepts PredictBatchRequest now, so "mode": "inline" fails Pydantic
    validation (422) instead of reaching the plugin. Batch is unaffected."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_predict_batch_data_contract_error_maps_to_422(client, fake_plugins):
    """DataContractError from the plugin maps to HTTP 422 with its message."""
    fake_plugins["ml30-meat-traceability-detection"].raise_on_batch = DataContractError(
        "El CSV no sigue el formato del modelo."
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/events.csv"})
    assert resp.status_code == 422
    assert "formato del modelo" in resp.json()["detail"]


def test_predict_batch(client):
    """A batch predict returns the model id."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/events.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml30-meat-traceability-detection"


def test_train(client):
    """POST /train returns metrics and the MLflow run id."""
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Training completed"
    assert isinstance(body["accuracy"], float)
    assert isinstance(body["n_train"], int)
    # la plataforma lee este campo para reconciliar el run de MLflow
    assert "mlflow_run_id" in body
