"""Endpoint tests for the ``ml2-fungal-cnn-disease-detection`` model."""
from app.domain.services.exceptions import InvalidImageError

PREFIX = "/models/ml2-fungal-cnn-disease-detection"

# Minimal valid base64 PNG (1×1 pixel, red) — does not go through real inference
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

INLINE_PAYLOAD = {
    "mode": "inline",
    "image_base64": _TINY_PNG_B64,
}

_CLASSES = {"black_rot", "downy_mildew", "healthy", "powdery_mildew", "trunk_disease"}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml2-fungal-cnn-disease-detection"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml2-fungal-cnn-disease-detection"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml2-fungal-cnn-disease-detection"
    assert body["prediction"] in _CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == _CLASSES


def test_predict_inline_missing_field(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/leaves.zip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml2-fungal-cnn-disease-detection"
    assert isinstance(body["predictions"], list)


def test_predict_inline_invalid_image_maps_to_422(client, fake_plugins):
    fake_plugins["ml2-fungal-cnn-disease-detection"].raise_on_inline = InvalidImageError(
        "imagen no decodificable"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "imagen no decodificable" in resp.json()["detail"]


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/leaves_train.zip", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 501
