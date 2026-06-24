"""Endpoint tests for the ``ml5-meat-cow-behaviour`` model."""
from app.domain.services.exceptions import InsufficientFramesError, InvalidVideoError

PREFIX = "/models/ml5-meat-cow-behaviour"

# 32 base64 frames (content irrelevant — the FakePlugin never decodes them).
_FRAMES = ["dGVzdC1mcmFtZQ=="] * 32

INLINE_PAYLOAD = {
    "mode": "inline",
    "frames_base64": _FRAMES,
}

_BEHAVIORS = {"grazing", "walking", "drinking", "running", "grooming", "other", "none",
              "hidden", "resting-lying", "resting-standing", "ruminating-lying",
              "ruminating-standing"}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml5-meat-cow-behaviour"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml5-meat-cow-behaviour"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml5-meat-cow-behaviour"
    assert body["prediction"] in _BEHAVIORS
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["is_anomaly"], bool)
    assert isinstance(body["behavior_idx"], int)


def test_predict_inline_too_few_frames(client):
    payload = {"mode": "inline", "frames_base64": ["dGVzdA=="] * 5}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cows.mp4"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml5-meat-cow-behaviour"
    assert isinstance(body["predictions"], list)


def test_predict_batch_invalid_video_maps_to_422(client, fake_plugins):
    fake_plugins["ml5-meat-cow-behaviour"].raise_on_batch = InvalidVideoError(
        "Cannot open video file"
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/bad.mp4"})
    assert resp.status_code == 422
    assert "Cannot open video file" in resp.json()["detail"]


def test_predict_inline_insufficient_frames_maps_to_422(client, fake_plugins):
    fake_plugins["ml5-meat-cow-behaviour"].raise_on_inline = InsufficientFramesError(
        "not enough frames"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 501
