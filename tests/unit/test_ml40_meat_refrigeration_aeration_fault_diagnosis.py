"""Endpoint tests for ``ml40-meat-refrigeration-aeration-fault-diagnosis``."""
from app.domain.services.exceptions import (
    InsufficientCycleHistoryError,
    UnknownDiagnosisSystemError,
)

PREFIX = "/models/ml40-meat-refrigeration-aeration-fault-diagnosis"
MODEL_ID = "ml40-meat-refrigeration-aeration-fault-diagnosis"

_AIREADO_ROW = {
    "run_id": 0,
    "time_min": 0,
    "Kg_embutido": 1150.0,
    "T_amb": 22.0,
    "T_set": 14.0,
    "N_fan_Hz": 40.0,
    "RH_cab": 75.0,
    "T_cab": 15.0,
    "T_evap_sat": -4.5,
    "P_comp_W": 1300.0,
}

INLINE_PAYLOAD = {"mode": "inline", "rows": [_AIREADO_ROW] * 100}


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
    assert body["prediction_name"] == "NORMAL"
    assert isinstance(body["prediction"], int)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_health"] in ("ESTABLE", "DEGRADADO")


def test_predict_inline_too_few_rows_rejected(client):
    """rows below the DTO min_length (60) must fail Pydantic validation."""
    payload = {"mode": "inline", "rows": [_AIREADO_ROW] * 10}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_inline_insufficient_cycle_history_maps_to_422(client, fake_plugins):
    """InsufficientCycleHistoryError raised by the plugin must map to HTTP 422."""
    fake_plugins[MODEL_ID].raise_on_inline = InsufficientCycleHistoryError(
        "El sistema refrigeracion requiere al menos 100 minutos de histórico por ciclo."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "100 minutos" in resp.json()["detail"]


def test_predict_inline_unknown_system_maps_to_422(client, fake_plugins):
    """UnknownDiagnosisSystemError raised by the plugin must map to HTTP 422."""
    fake_plugins[MODEL_ID].raise_on_inline = UnknownDiagnosisSystemError(
        "Las columnas de entrada no corresponden a ningún subsistema conocido."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/refrigeracion.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["system"] in ("refrigeracion", "aireado")
    assert body["n_runs"] == len(body["predictions"])
    assert body["predictions"][0]["prediction_name"] == "NORMAL"
    assert body["model_health"] in ("ESTABLE", "DEGRADADO")


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/dataset_aireado.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["system"] == "aireado"
    assert body["n_runs_train"] > 0
    assert 0.0 <= body["f1_macro"] <= 1.0
