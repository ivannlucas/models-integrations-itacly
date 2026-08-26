"""Endpoint tests for ``m21-cereal-price-spatial`` (ESP-CEREAL spatial cereal price)."""

PREFIX = "/models/m21-cereal-price-spatial"

INLINE_PAYLOAD = {
    "mode": "inline",
    "provincia": "Burgos",
    "cereal_predominante": "trigo",
    "date": "2024-01",
    "role": "comprador",
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "m21-cereal-price-spatial"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "m21-cereal-price-spatial"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "m21-cereal-price-spatial"
    assert "predictions" in body
    assert "H1" in body["predictions"]
    assert "H2" in body["predictions"]
    assert "H3" in body["predictions"]
    assert body["province"] == "Burgos"
    assert body["cereal"] == "trigo"


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cereal.csv", "month": "2024-01"}
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "m21-cereal-price-spatial"


def test_train_returns_501(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code in (200, 501)


def test_train_returns_metrics(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": ""}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert "mae_h1" in body
    assert "pearson_h1" in body
    assert "da_h1" in body
    assert "auc_h1" in body
    assert body["mae_h1"] is not None
