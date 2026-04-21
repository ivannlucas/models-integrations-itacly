"""Endpoint tests for the ``cow-behavior`` model."""
from app.domain.services.exceptions import (
    InsufficientFramesError,
    InvalidVideoError,
)

PREFIX = "/models/cow-behavior"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["model"] == "cow-behavior"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "cow-behavior"


def test_predict_inline(client, cow_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=cow_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cow-behavior"
    assert body["prediction"] == "walking"
    assert body["is_anomaly"] is False


def test_predict_inline_rejects_fewer_than_32_frames(client):
    payload = {"mode": "inline", "frames_base64": ["AAAA"] * 10}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={
            "mode": "batch",
            "data_path": "/tmp/cow.mp4",
            "detection_threshold": 0.5,
            "anomaly_threshold": 0.5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cow-behavior"


def test_predict_invalid_video_maps_to_422(client, fake_plugins):
    fake_plugins["cow-behavior"].raise_on_batch = InvalidVideoError("cannot open")
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/bad.mp4"},
    )
    assert resp.status_code == 422


def test_predict_insufficient_frames_maps_to_422(client, fake_plugins, cow_inline_payload):
    fake_plugins["cow-behavior"].raise_on_inline = InsufficientFramesError("need 32")
    resp = client.post(f"{PREFIX}/predict", json=cow_inline_payload)
    assert resp.status_code == 422


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train")
    assert resp.status_code == 501
