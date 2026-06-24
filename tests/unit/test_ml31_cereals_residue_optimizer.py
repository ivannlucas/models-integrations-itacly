"""Endpoint tests for the ``ml31-cereals-residue-optimizer`` model."""

PREFIX = "/models/ml31-cereals-residue-optimizer"

INLINE_PAYLOAD = {
    "mode": "inline",
    "Sup_Secano_ha": 100.0,
    "Sup_Regadio_ha": 20.0,
    "Lluvia_Primavera_mm": 180.0,
    "Sequia_Primavera": 1,
    "Cultivo": "Trigo",
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml31-cereals-residue-optimizer"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml31-cereals-residue-optimizer"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml31-cereals-residue-optimizer"
    assert isinstance(body["prediction"], (int, float))


def test_predict_inline_missing_field(client):
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "Cultivo"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cereal.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml31-cereals-residue-optimizer"


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Entrenamiento completado"
    assert isinstance(body["r2_test"], float)
