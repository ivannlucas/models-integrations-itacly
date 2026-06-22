"""Endpoint tests for the ``ml30-meat-traceability-detection`` model."""

PREFIX = "/models/ml30-meat-traceability-detection"

INLINE_PAYLOAD = {"mode": "inline", "sensor_temp_c": 7.5, "stage": "deboning"}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml30-meat-traceability-detection"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml30-meat-traceability-detection"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml30-meat-traceability-detection"
    assert body["pred_traceability_incident"] in (0, 1)
    assert 0.0 <= body["pred_score"] <= 1.0


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/events.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml30-meat-traceability-detection"


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Training completed"
    assert isinstance(body["accuracy"], float)
    assert isinstance(body["n_train"], int)
