"""Endpoint tests for the ``modelo10-lacteo`` model."""
PREFIX = "/models/modelo10-lacteo"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "model": "modelo10-lacteo",
        "version": "1.0.0",
        "loaded": True,
    }


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "modelo10-lacteo"


def test_predict_inline(client, lacteo_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=lacteo_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "modelo10-lacteo"
    assert body["prediction"] == "fly"
    assert body["confidence"] == 0.91
    assert body["vectors_count"] == 1
    assert len(body["detections"]) == 1


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "modelo10-lacteo"
    assert len(body["predictions"]) > 0


def test_train(client):
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/dataset.zip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Training completed successfully"
    assert body["metrics"]["train_samples"] == 100
    assert body["metrics"]["best_val_acc"] == 95.0
