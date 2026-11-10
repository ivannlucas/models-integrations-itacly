"""Endpoint tests for the ``ml7-cereals-grain-pest-detection`` model."""
from app.domain.services.exceptions import InvalidImageError

PREFIX = "/models/ml7-cereals-grain-pest-detection"

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
INLINE_PAYLOAD = {"mode": "inline", "image_base64": _TINY_PNG_B64}


def test_health(client):
    """/health reports status ok, the model id and loaded=True."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml7-cereals-grain-pest-detection"
    assert body["loaded"] is True


def test_stats(client):
    """/stats reports the model id as model_name."""
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml7-cereals-grain-pest-detection"


def test_predict_inline(client):
    """Inline /predict returns a confidence in [0, 1] and the detections list."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml7-cereals-grain-pest-detection"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["total_detections"], int)
    assert isinstance(body["detections"], list)


def test_predict_batch(client):
    """Batch /predict returns 200 for the model id."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/grain.zip"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml7-cereals-grain-pest-detection"


def test_predict_inline_invalid_image_maps_to_422(client, fake_plugins):
    """InvalidImageError raised by the plugin maps to HTTP 422."""
    fake_plugins["ml7-cereals-grain-pest-detection"].raise_on_inline = InvalidImageError(
        "bad image"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_train(client):
    """/train returns 200 with the fine-tuning detail, map50 and n_images."""
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/dataset.yaml", "mlflow_run_id": "test-run-id"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Fine-tuning completado"
    assert isinstance(body["map50"], float)
    assert isinstance(body["n_images"], int)


def test_train_without_mlflow_run_id_rejected(client):
    """mlflow_run_id is required (no default) — a retrain must always be persisted to a
    specific MLflow run, never silently overwriting the fixed base artifact."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/dataset.yaml"})
    assert resp.status_code == 422
