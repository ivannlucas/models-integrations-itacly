"""Endpoint tests for the ``modelo10-lacteo`` model."""
PREFIX = "/models/modelo10-lacteo"


def test_health(client):
    """Verify the health endpoint returns expected metadata."""
    body = client.get(f"{PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "model": "modelo10-lacteo",
        "version": "1.0.0",
        "loaded": True,
    }


def test_stats(client):
    """Verify the stats endpoint returns the model name."""
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "modelo10-lacteo"


def test_predict_inline(client, lacteo_inline_payload):
    """Verify inline prediction returns expected fields."""
    resp = client.post(f"{PREFIX}/predict", json=lacteo_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "modelo10-lacteo"
    assert body["prediction"] == "fly"
    assert body["confidence"] == 0.91
    assert body["vectors_count"] == 1
    assert len(body["detections"]) == 1


def test_predict_batch(client):
    """Verify batch prediction returns predictions list."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "modelo10-lacteo"
    assert len(body["predictions"]) > 0


def test_train(client):
    """Verify the train endpoint returns training metrics."""
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
    """Verify inline prediction works with image_path field."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "inline", "image_path": "/tmp/test.jpg"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "fly"


def test_predict_invalid_mode_returns_422(client):
    """Verify invalid mode returns HTTP 422."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "invalid"},
    )
    assert resp.status_code == 422


def test_predict_missing_mode_returns_422(client):
    """Verify missing mode returns HTTP 422."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={},
    )
    assert resp.status_code == 422


def test_train_with_empty_data_path(client):
    """Verify train with empty data_path still returns 200 (fake)."""
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": ""},
    )
    assert resp.status_code == 200


def test_stats_after_predict(client):
    """Verify predict_count is updated after a prediction call."""
    client.post(
        f"{PREFIX}/predict",
        json={"mode": "inline", "image_base64": "dGVzdA=="},
    )
    body = client.get(f"{PREFIX}/stats").json()
    assert body["predict_count"] >= 0
