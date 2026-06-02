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


def test_predict_inline_with_image_path(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "inline", "image_path": "/tmp/test.jpg"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "fly"


def test_predict_invalid_mode_returns_422(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "invalid"},
    )
    assert resp.status_code == 422


def test_predict_missing_mode_returns_422(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={},
    )
    assert resp.status_code == 422


def test_train_with_empty_data_path(client):
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": ""},
    )
    assert resp.status_code == 200


def test_stats_after_predict(client):
    client.post(
        f"{PREFIX}/predict",
        json={"mode": "inline", "image_base64": "dGVzdA=="},
    )
    body = client.get(f"{PREFIX}/stats").json()
    assert body["predict_count"] >= 0
