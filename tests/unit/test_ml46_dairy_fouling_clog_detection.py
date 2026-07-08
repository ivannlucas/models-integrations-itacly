"""Endpoint tests for ``ml46-dairy-fouling-clog-detection``."""

PREFIX = "/models/ml46-dairy-fouling-clog-detection"

_ROW = {
    "timestamp": "2026-01-11T06:01:00+00:00",
    "asset_id": "asset_00",
    "flow_kg_s": 6.6,
    "pressure_in_kPa": 250.0,
    "pressure_out_kPa": 163.0,
    "dP_kPa": 87.0,
    "Th_in_C": 90.0,
    "Tc_in_C": 55.0,
    "Th_out_C": 85.0,
    "Tc_out_C": 59.0,
    "Twall_C": 70.0,
    "vibration_mm_s": 1.8,
    "flow_sp_kg_s": 6.6,
    "Th_sp_C": 90.0,
    "Tc_sp_C": 55.0,
    "protein_g_L_nominal": 32.0,
    "fat_g_L_nominal": 38.0,
    "solids_g_L_nominal": 125.0,
    "Ca_mM_nominal": 30.0,
    "PO4_mM_nominal": 20.0,
    "pH_nominal": 6.6,
    "phase": "production",
    "maintenance_active": 0,
    "asset_family": "robust_phe",
    "milk_type": "high_solids",
    "last_maintenance_type": "none",
}

INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 120}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml46-dairy-fouling-clog-detection"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml46-dairy-fouling-clog-detection"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml46-dairy-fouling-clog-detection"
    assert body["pred_stage_name"] == "stable"
    assert isinstance(body["pred_severity"], float)
    assert isinstance(body["is_alert"], bool)


def test_predict_inline_insufficient_history_rejected(client):
    """rows below SEQ_LEN (120) must fail Pydantic validation (min_length)."""
    payload = {"mode": "inline", "rows": [_ROW] * 5}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/telemetry.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml46-dairy-fouling-clog-detection"
    assert isinstance(body["predictions"], list)
    assert isinstance(body["alerts"], list)


def test_predict_insufficient_telemetry_history_maps_to_422(client, fake_plugins):
    from app.domain.services.exceptions import InsufficientTelemetryHistoryError

    fake_plugins["ml46-dairy-fouling-clog-detection"].raise_on_inline = InsufficientTelemetryHistoryError(
        "Se requieren al menos 120 filas de telemetría."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert isinstance(body["stage_accuracy"], float)
    assert isinstance(body["n_windows"], int)
