"""Endpoint tests for the ``cnn-fungal-detection`` model."""
from app.domain.services.exceptions import InvalidImageError

PREFIX = "/models/cnn-fungal-detection"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["model"] == "cnn-fungal-detection"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "cnn-fungal-detection"


def test_predict_inline(client, cnn_fungal_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=cnn_fungal_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cnn-fungal-detection"
    assert body["prediction"] == "healthy"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"healthy", "fungal"}


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/images"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cnn-fungal-detection"


def test_predict_invalid_image_maps_to_422(client, fake_plugins, cnn_fungal_inline_payload):
    fake_plugins["cnn-fungal-detection"].raise_on_inline = InvalidImageError("cannot decode")
    resp = client.post(f"{PREFIX}/predict", json=cnn_fungal_inline_payload)
    assert resp.status_code == 422


def test_predict_inline_missing_image_path(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 422


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train")
    assert resp.status_code == 501
