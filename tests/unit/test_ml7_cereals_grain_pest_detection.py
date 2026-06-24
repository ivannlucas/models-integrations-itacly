"""Endpoint tests for the ``ml7-cereals-grain-pest-detection`` model."""
from app.domain.services.exceptions import InvalidImageError

PREFIX = "/models/ml7-cereals-grain-pest-detection"

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
INLINE_PAYLOAD = {"mode": "inline", "image_base64": _TINY_PNG_B64}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml7-cereals-grain-pest-detection"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml7-cereals-grain-pest-detection"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml7-cereals-grain-pest-detection"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["total_detections"], int)
    assert isinstance(body["detections"], list)


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/grain.zip"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml7-cereals-grain-pest-detection"


def test_predict_inline_invalid_image_maps_to_422(client, fake_plugins):
    fake_plugins["ml7-cereals-grain-pest-detection"].raise_on_inline = InvalidImageError("bad image")
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_train_returns_501(client):
    assert client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x", "mlflow_run_id": "test-run-id"}).status_code == 501
