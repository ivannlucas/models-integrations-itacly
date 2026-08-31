"""Endpoint tests for ``ml16-meat-raw-material-price-alert``."""
from app.domain.services.exceptions import InsufficientRowsError

PREFIX = "/models/ml16-meat-raw-material-price-alert"
MODEL_ID = "ml16-meat-raw-material-price-alert"

_ROW = {
    "fecha": "2020-07-01",
    "month": 7,
    "indice_animales": 338.25,
    "indice_insumos": 191.79,
    "precip_total": 14.03,
    "precip_max": 3.34,
    "wet_days": 2.11,
    "wash_days": 0.26,
    "animales_afectados": 2,
}

# 10 meses = warmup(6) + lookback(3) + 1, el mínimo histórico que exige el modelo.
INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 10}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == MODEL_ID


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["target_animales_pred"] in (0, 1)
    assert body["target_insumos_pred"] in (0, 1)
    assert 0.0 <= body["target_animales_proba"] <= 1.0
    assert 0.0 <= body["target_insumos_proba"] <= 1.0


def test_predict_inline_too_few_rows_rejected(client):
    """rows below the DTO min_length (10 = warmup 6 + lookback 3 + 1) must fail Pydantic validation."""
    payload = {"mode": "inline", "rows": [_ROW] * 5}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_inline_insufficient_rows_maps_to_422(client, fake_plugins):
    """InsufficientRowsError raised by the plugin (e.g. all rows drop out after dropna on lag
    features) must map to HTTP 422, not 500."""
    fake_plugins[MODEL_ID].raise_on_inline = InsufficientRowsError(
        "Histórico insuficiente: se necesitan al menos 10 meses."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "10 meses" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/dataset_base.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"])
    assert body["predictions"][0]["target_insumos_pred"] in (0, 1)


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/dataset_base.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_train_rows"] > 0
    assert body["n_test_rows"] > 0
    assert 0.0 <= body["target_animales_f1"] <= 1.0
    assert 0.0 <= body["target_insumos_f1"] <= 1.0
