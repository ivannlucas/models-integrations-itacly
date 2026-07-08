"""Endpoint tests for ``ml17-meat-market-price-analysis`` (Ridge pork price)."""

PREFIX = "/models/ml17-meat-market-price-analysis"

INLINE_PAYLOAD = {
    "mode": "inline",
    "date": "2023-01-01",
    "target_price_pigmeat_class_e_es": 173.82,
    "eurostat_pigmeat_slaughter_tonnes_es": 381.88,
    "eurostat_pigmeat_slaughter_tonnes_eu": 1795.36,
    "cereal_feed_barley_price_monthly": 149.72,
    "cereal_feed_maize_price_monthly": 175.73,
    "mapa_porcino_otras_razas_price_monthly": 117.16,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml17-meat-market-price-analysis"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml17-meat-market-price-analysis"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml17-meat-market-price-analysis"
    assert "prediction" in body
    assert "y_pred" in body
    assert body["base_date"] == "2023-01-01"


def test_predict_inline_missing_required(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "date": "2023-01-01"})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/pork.csv"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml17-meat-market-price-analysis"
    assert body["line"] == "official_v1_4"


def test_train_returns_501(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 501
