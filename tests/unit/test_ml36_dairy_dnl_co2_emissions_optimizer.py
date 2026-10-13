"""Endpoint tests for ``ml36-dairy-dnl-co2-emissions-optimizer``."""
from app.domain.services.exceptions import ThermalSafetyViolationError

MODEL = "ml36-dairy-dnl-co2-emissions-optimizer"
PREFIX = f"/models/{MODEL}"

CONTEXT = {"F_milk": 4995.54, "T_in": 4.12, "Fat_perc": 3.3, "Viscosity": 1.99, "t_ciclo": 5}

INLINE_PAYLOAD = {
    "mode": "inline",
    **CONTEXT,
    "T_serv": 78.34,
    "Delta_P": 0.742,
    "Regeneration_perc": 91.54,
}

# Optimize travels as an inline request differentiated by model_key="optimize";
# the 3 controls are omitted — the GA chooses them.
OPTIMIZE_PAYLOAD = {"mode": "inline", "model_key": "optimize", **CONTEXT}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL
    assert isinstance(body["T_out_pred"], (int, float))
    assert isinstance(body["CO2_emissions_pred"], (int, float))


def test_predict_inline_missing_context_field(client):
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "F_milk"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_inline_missing_control_field(client):
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "Delta_P"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize(client):
    resp = client.post(f"{PREFIX}/predict", json=OPTIMIZE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL
    assert body["recommended_factible"] is True
    assert body["recommended_T_out_pred"] >= 72.5
    assert body["decision_mode"] == "hybrid"
    assert body["current_CO2_pred"] is None
    assert body["co2_saving_vs_current"] is None


def test_predict_optimize_with_current_operation(client):
    payload = {**OPTIMIZE_PAYLOAD, "T_serv": 79.5, "Delta_P": 0.75, "Regeneration_perc": 91.88}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 200


def test_predict_optimize_partial_current_operation_rejected(client):
    payload = {**OPTIMIZE_PAYLOAD, "T_serv": 79.5}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize_invalid_decision_mode(client):
    payload = {**OPTIMIZE_PAYLOAD, "decision_mode": "turbo"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize_missing_context_field(client):
    payload = {k: v for k, v in OPTIMIZE_PAYLOAD.items() if k != "Viscosity"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize_thermal_violation_maps_to_422(client, fake_plugins):
    """ThermalSafetyViolationError raised by the plugin must map to HTTP 422."""
    plugin = fake_plugins[MODEL]
    plugin.raise_on_inline = ThermalSafetyViolationError(
        "La recomendación no cumple la restricción térmica."
    )
    assert client.post(f"{PREFIX}/predict", json=OPTIMIZE_PAYLOAD).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/pasteurizacion.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL
    assert body["predictions"][0]["row"] == 0
    assert "CO2_emissions_pred" in body["predictions"][0]


def test_predict_batch_optimize(client):
    """mode=batch + model_key=optimize must reach the GA branch, not the MLP estimate."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "model_key": "optimize", "data_path": "/tmp/escenarios.csv"},
    )
    assert resp.status_code == 200
    row = resp.json()["predictions"][0]
    assert "recommended_T_serv" in row
    assert "CO2_emissions_pred" not in row
    # Context must survive into the row — the batch XAI context builder needs it.
    assert row["F_milk"] == 4995.54
    assert row["T_in"] == 4.12


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Fine-tuning completado"
    assert isinstance(body["mae_co2"], float)
    assert isinstance(body["r2_t_out"], float)
    assert isinstance(body["n_samples"], int)
    assert isinstance(body["epochs_executed"], int)
