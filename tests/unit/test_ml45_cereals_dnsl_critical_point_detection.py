"""Endpoint tests for ``ml45-cereals-dnsl-critical-point-detection``."""
from app.domain.services.exceptions import InsufficientWindowHistoryError

PREFIX = "/models/ml45-cereals-dnsl-critical-point-detection"
MODEL_ID = "ml45-cereals-dnsl-critical-point-detection"

_SENSOR_ROW = {
    "plenum_temp": [70.0] * 240,
    "exhaust_air_temp": [40.0] * 240,
    "exhaust_air_humidity": [60.0] * 240,
    "static_pressure": [10.0] * 240,
    "burner_power": [55.0] * 240,
    "fan_speed": [1200.0] * 240,
    "discharge_frequency": [30.0] * 240,
    "grain_moisture_in": [20.0] * 240,
    "ambient_temp": [15.0] * 240,
    "ambient_humidity": [50.0] * 240,
    "setpoint_temp": [75.0] * 240,
    "timestamp": [f"2029-01-01 00:{i:02d}:00" if i < 60 else f"2029-01-01 0{i // 60}:{i % 60:02d}:00" for i in range(240)],
    "cycle_id": 1,
}

INLINE_PAYLOAD = {"mode": "inline", **_SENSOR_ROW}


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
    assert body["predicted_anomaly_class"] in (0, 1)
    assert 0.0 <= body["anomaly_probability"] <= 1.0
    assert body["Estado interpretativo"] in ("Normal", "Vigilancia", "Criticidad detectada")


def test_predict_inline_missing_sensor_rejected(client):
    """Missing a required sensor array (and no data_path) must fail Pydantic validation."""
    payload = dict(INLINE_PAYLOAD)
    del payload["plenum_temp"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/fake.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_anomaly_class"] == 0


def test_predict_batch_insufficient_window_history_maps_to_422(client, fake_plugins):
    """InsufficientWindowHistoryError raised by the plugin must map to HTTP 422."""
    fake_plugins[MODEL_ID].raise_on_batch = InsufficientWindowHistoryError(
        "No se generaron ventanas: se necesitan al menos 240 filas consecutivas."
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/short.csv"})
    assert resp.status_code == 422


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert body["n_windows"] > 0
