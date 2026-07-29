"""Endpoint tests for ``ml23-lactic-market-price-forecast`` (GRU dairy price)."""

PREFIX = "/models/ml23-lactic-market-price-forecast"

INLINE_PAYLOAD = {
    "mode": "inline",
    "precio_lag_1": 0.9234,
    "precio_lag_3": 0.8734,
    "precio_lag_12": 0.9234,
    "current_price": 0.9187,
    "year": 2023.0,
    "mes": 1.0,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml23-lactic-market-price-forecast"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml23-lactic-market-price-forecast"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml23-lactic-market-price-forecast"
    assert "prediction" in body
    assert isinstance(body["features_used"], list)


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/dairy.csv"}
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml23-lactic-market-price-forecast"


def test_train_returns_501(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 501
