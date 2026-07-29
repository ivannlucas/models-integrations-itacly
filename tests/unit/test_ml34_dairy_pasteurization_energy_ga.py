"""Endpoint tests for ``ml34-dairy-pasteurization-energy-ga``."""
from app.domain.services.exceptions import ThermalSafetyViolationError

PREFIX = "/models/ml34-dairy-pasteurization-energy-ga"

INLINE_PAYLOAD = {
    "mode": "inline",
    "T_in_leche": 6.78,
    "F_flow": 4675.09,
    "T_servicio": 79.73,
    "t_ciclo": 80.0,
    "Delta_P": 0.481,
}

# Optimize travels as an inline request differentiated by model_key="optimize"
# (the transport contract only exposes inline/batch; the plugin dispatches to the
# GA branch internally). F_flow/T_servicio are omitted — the GA chooses them.
OPTIMIZE_PAYLOAD = {
    "mode": "inline",
    "model_key": "optimize",
    "T_in_leche": 6.78,
    "Delta_P": 0.481,
    "t_ciclo": 80.0,
    "seed": 1,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml34-dairy-pasteurization-energy-ga"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml34-dairy-pasteurization-energy-ga"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml34-dairy-pasteurization-energy-ga"
    assert isinstance(body["E_consumo_pred"], (int, float))
    assert isinstance(body["T_out_pred"], (int, float))


def test_predict_inline_missing_required_field(client):
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "F_flow"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize(client):
    resp = client.post(f"{PREFIX}/predict", json=OPTIMIZE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml34-dairy-pasteurization-energy-ga"
    assert isinstance(body["IA_F_flow"], (int, float))
    assert isinstance(body["IA_T_servicio"], (int, float))
    assert body["IA_factible"] is True
    assert body["seed"] == 1


def test_predict_optimize_default_seed(client):
    payload = {k: v for k, v in OPTIMIZE_PAYLOAD.items() if k != "seed"}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["seed"] == 1


def test_predict_optimize_missing_scenario_field(client):
    payload = {k: v for k, v in OPTIMIZE_PAYLOAD.items() if k != "Delta_P"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize_thermal_violation_maps_to_422(client, fake_plugins):
    """ThermalSafetyViolationError raised by the plugin must map to HTTP 422."""
    plugin = fake_plugins["ml34-dairy-pasteurization-energy-ga"]
    plugin.raise_on_inline = ThermalSafetyViolationError(
        "El GA no encontró solución factible para el escenario."
    )
    resp = client.post(f"{PREFIX}/predict", json=OPTIMIZE_PAYLOAD)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/pasteurizacion.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml34-dairy-pasteurization-energy-ga"
    assert body["predictions"][0]["row"] == 0


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Fine-tuning completado"
    assert isinstance(body["rmse_E_consumo"], float)
    assert isinstance(body["r2_E_consumo"], float)
    assert isinstance(body["mae_T_out_leche"], float)
    assert isinstance(body["n_samples"], int)
    assert isinstance(body["epochs_executed"], int)
