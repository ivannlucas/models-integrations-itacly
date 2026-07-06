"""Endpoint tests for ``ml35-dairy-ann-cleaning-cost``."""

PREFIX = "/models/ml35-dairy-ann-cleaning-cost"

INLINE_PAYLOAD = {
    "mode": "inline",
    "temp_entrada_leche": 4.5,
    "temp_ambiente": 20.0,
    "temp_setpoint_leche": 76.0,
    "temp_proceso_leche": 76.0,
    "temp_agua_servicio": 86.0,
    "flujo_leche_lh": 3500.0,
    "horas_desde_limpieza": 10.0,
    "presion_diferencial_bar": 1.2,
}

INLINE_DERIVED_PAYLOAD = {
    "mode": "inline",
    "temp_entrada_leche": 4.5,
    "temp_ambiente": 20.0,
    "temp_setpoint_leche": 76.0,
    # temp_proceso_leche and temp_agua_servicio omitted — should be derived
    "flujo_leche_lh": 3500.0,
    "horas_desde_limpieza": 10.0,
    "presion_diferencial_bar": 1.2,
}

OPTIMIZE_PAYLOAD = {
    "mode": "optimize",
    "temp_entrada_leche": 4.5,
    "temp_ambiente": 20.0,
    "horas_desde_limpieza": 10.0,
    "presion_diferencial_bar": 1.2,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml35-dairy-ann-cleaning-cost"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml35-dairy-ann-cleaning-cost"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml35-dairy-ann-cleaning-cost"
    assert isinstance(body["consumo_agua_l"], (int, float))
    assert isinstance(body["pu_logrado"], (int, float))


def test_predict_inline_derived_fields(client):
    """temp_proceso_leche and temp_agua_servicio are optional and derived server-side."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_DERIVED_PAYLOAD)
    assert resp.status_code == 200
    assert "consumo_agua_l" in resp.json()


def test_predict_inline_missing_required_field(client):
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "flujo_leche_lh"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize(client):
    resp = client.post(f"{PREFIX}/predict", json=OPTIMIZE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml35-dairy-ann-cleaning-cost"
    # optimize response fields
    assert isinstance(body.get("opt_temp_leche") or body.get("consumo_agua_l"), (int, float))


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/dairy.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml35-dairy-ann-cleaning-cost"


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Fine-tuning completado"
    assert isinstance(body["mae"], float)
    assert isinstance(body["r2"], float)
    assert isinstance(body["n_samples"], int)
