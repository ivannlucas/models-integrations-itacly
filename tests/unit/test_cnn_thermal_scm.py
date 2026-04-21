"""Endpoint tests for the ``cnn-thermal-scm`` model."""
from app.domain.services.exceptions import InvalidImageError

PREFIX = "/models/cnn-thermal-scm"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["model"] == "cnn-thermal-scm"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "cnn-thermal-scm"


def test_predict_inline(client, cnn_thermal_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=cnn_thermal_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cnn-thermal-scm"
    assert body["prediction"] == "Healthy"
    assert body["predicted_class_index"] == 0
    assert abs(body["probability_healthy"] + body["probability_scm"] - 1.0) < 1e-6


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/thermal"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cnn-thermal-scm"


def test_predict_invalid_image_maps_to_422(client, fake_plugins, cnn_thermal_inline_payload):
    fake_plugins["cnn-thermal-scm"].raise_on_inline = InvalidImageError("bad image")
    resp = client.post(f"{PREFIX}/predict", json=cnn_thermal_inline_payload)
    assert resp.status_code == 422


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train")
    assert resp.status_code == 501
