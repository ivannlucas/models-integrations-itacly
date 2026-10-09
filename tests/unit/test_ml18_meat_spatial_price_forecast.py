"""Endpoint tests for ``ml18-meat-spatial-price-forecast`` (GRU, wiring only).

Correctness against the real artifact (golden_cases) is validated by the `verification` skill,
not here — these tests use FakePlugin and only check wiring: request/response schemas, the
min_length DTO guard and InsufficientRowsError -> 422 mapping.
"""
from app.domain.services.exceptions import InsufficientRowsError

PREFIX = "/models/ml18-meat-spatial-price-forecast"
MODEL_ID = "ml18-meat-spatial-price-forecast"

_ROW = {
    "Fecha": "2021-01-01",
    "CCAA": "ANDALUCIA",
    "Producto": "CARNE POLLO",
    "Poblacion": 8202220.0,
    "RentaHogar": 25248.0,
    "CONSUMO X CAPITA": 1.5,
    "PENETRACION (%)": 50.0,
    "PRECIO MEDIO KG": 4.5,
}

# Minimo operativo real: lookback=12 meses consecutivos por combinacion CCAA-Producto.
INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 12}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"])
    assert "predicted_price" in body["predictions"][0]


def test_predict_inline_too_few_rows_rejected(client):
    """rows below the DTO min_length (12 = lookback) must fail Pydantic validation."""
    payload = {"mode": "inline", "rows": [_ROW] * 11}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_inline_insufficient_rows_maps_to_422(client, fake_plugins):
    """InsufficientRowsError raised by the plugin (no (CCAA, Producto) group reaches the
    12-month lookback) must map to HTTP 422, not 500."""
    fake_plugins[MODEL_ID].raise_on_inline = InsufficientRowsError(
        "No se generaron secuencias: cada combinación CCAA-Producto necesita al menos 12 meses de histórico real consecutivos."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "al menos 12 meses" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a18_panel.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"])


def test_train_returns_200_with_metrics(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 200
    body = resp.json()
    assert {"test_mape_pct", "test_r2", "val_mape_pct", "n_train", "epochs_run"} <= body.keys()


def test_train_without_mlflow_run_id_returns_422(client):
    assert client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"}).status_code == 422
