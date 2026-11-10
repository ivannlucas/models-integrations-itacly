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
    """/health reports status ok, the model id and loaded=True."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml23-lactic-market-price-forecast"
    assert body["loaded"] is True


def test_stats(client):
    """/stats reports the model id as model_name."""
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml23-lactic-market-price-forecast"


def test_predict_inline(client):
    """Inline /predict returns the prediction and the list of features used."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml23-lactic-market-price-forecast"
    assert "prediction" in body
    assert isinstance(body["features_used"], list)


def test_predict_batch(client):
    """Batch /predict returns 200 for the model id."""
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/dairy.csv"}
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml23-lactic-market-price-forecast"


def test_train_returns_200_with_metrics(client):
    """/train returns 200 with positive mae and n_train."""
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mae"] > 0
    assert body["n_train"] > 0


def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
