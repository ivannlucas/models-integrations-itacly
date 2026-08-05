"""Endpoint tests for ``ml43-cereals-dnsl-anomaly-fault-detection`` (CU43+CU44)."""
from app.domain.services.exceptions import InsufficientSensorWindowError

PREFIX = "/models/ml43-cereals-dnsl-anomaly-fault-detection"
MODEL_ID = "ml43-cereals-dnsl-anomaly-fault-detection"

INLINE_PAYLOAD = {
    "mode": "inline",
    "temp_zona1": 180.0,
    "temp_zona2": 178.0,
    "temp_zona3": 182.0,
    "temp_salida_gases": 95.0,
    "presion_camara": 2.0,
    "presion_ventilacion": 10.0,
    "potencia_kw": 45.0,
    "flujo_gas": 12.0,
    "humedad_relativa": 40.0,
    "temp_ambiente": 22.0,
    "setpoint_temp": 180.0,
    "posicion_valvula": 55.0,
    "velocidad_ventilador": 900.0,
}


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
    assert body["predicted_anomaly_label"] in ("Fallo", "No Fallo")
    assert 0.0 <= body["anomaly_probability"] <= 1.0


def test_predict_inline_missing_sensor_rejected(client):
    """Missing a required sensor field must fail Pydantic validation (422)."""
    payload = dict(INLINE_PAYLOAD)
    del payload["temp_zona1"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/fake_cycles.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_anomaly_class"] == 0


def test_predict_batch_insufficient_sensor_window_maps_to_422(client, fake_plugins):
    """InsufficientSensorWindowError raised by the plugin must map to HTTP 422."""
    fake_plugins[MODEL_ID].raise_on_batch = InsufficientSensorWindowError(
        "CSV falta columnas de sensor requeridas: ['temp_zona1']."
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/bad.csv"})
    assert resp.status_code == 422


def test_train(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/train_cycles.csv", "mlflow_run_id": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert body["n_windows_total"] > 0
