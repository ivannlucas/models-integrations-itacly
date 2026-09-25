"""Endpoint tests for ``ml14-wine-phyto-price-forecast`` (LSTM, wiring only).

Correctness against the real artifact (golden_cases) is validated by the `verification` skill,
not here — these tests use FakePlugin and only check wiring: request/response schemas, the
min_length=24 DTO guard and InsufficientDataError -> 422 mapping.
"""
from app.domain.services.exceptions import InsufficientDataError

PREFIX = "/models/ml14-wine-phyto-price-forecast"
MODEL_ID = "ml14-wine-phyto-price-forecast"

_ROW = {
    "date": "2025-10-26",
    "PROTECCION_FITO": 125.035925,
    "CARBURANTES": 133.106987,
    "COPPER_EUR_TON": 9463.383125,
    "GAS_EUR_MMBTU": 9.356627,
    "COIL_EUR_BARRIL": 54.387474,
    "DEXUSEU": 1.1624,
}

# Mínimo operativo real: feature_lookback(12) + seq_len(12) = 24 filas semanales.
INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 24}


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
    assert body["model_used"] == "LSTM"
    assert body["horizon_weeks"] == 16
    assert isinstance(body["predicted_price"], float)
    assert isinstance(body["current_price"], float)
    assert isinstance(body["drift_baseline"], float)


def test_predict_inline_too_few_rows_rejected(client):
    """rows below the DTO min_length (24 = feature_lookback 12 + seq_len 12) must fail Pydantic validation."""
    payload = {"mode": "inline", "rows": [_ROW] * 23}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_inline_insufficient_data_maps_to_422(client, fake_plugins):
    """InsufficientDataError raised by the plugin (invalid/duplicate/non-weekly dates, nulls,
    or too few rows surviving feature derivation) must map to HTTP 422, not 500."""
    fake_plugins[MODEL_ID].raise_on_inline = InsufficientDataError(
        "Historial insuficiente para inferencia: registros_recibidos=23, registros_minimos_requeridos=24."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "registros_minimos_requeridos=24" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a14_history.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"])
    assert body["predictions"][0]["model_used"] == "LSTM"


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 501
