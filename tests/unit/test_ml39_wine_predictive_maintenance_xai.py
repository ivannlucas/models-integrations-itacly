"""Endpoint tests for ``ml39-wine-predictive-maintenance-xai`` (wiring only).

Correctness of Integrated Gradients/SHAP against the real artifacts (a39 is the XAI companion
to ml37, same model/dataset/run_id — see inbox/a39/manifest.yaml) is validated by the
`verification` skill and by tests/unit/test_ml39_inference.py (real code, not FakePlugin) — this
file only checks wiring: request/response schemas and the DataContractError -> 422 mapping.
"""
from app.domain.services.exceptions import DataContractError

PREFIX = "/models/ml39-wine-predictive-maintenance-xai"
MODEL_ID = "ml39-wine-predictive-maintenance-xai"

_PHYSICAL_SENSORS = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1",
)


def _reading(t: float) -> dict:
    row = {"t": t}
    row.update({name: 1.0 + t for name in _PHYSICAL_SENSORS})
    return row


INLINE_PAYLOAD = {
    "mode": "inline",
    "readings": [_reading(0.0), _reading(0.1), _reading(0.2)],
}


def test_health(client):
    """GET /health reports the model as ok and loaded."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    """GET /stats returns the model name."""
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline_without_shap(client):
    """Inline /predict returns IG importance per head; shap_importance is null by default."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    for target in ("cooler", "valve", "pump", "accumulator"):
        assert target in body["predicted_class"]
        assert target in body["ig_importance"]
    assert body["shap_importance"] is None


def test_predict_inline_with_shap_flag_accepted(client):
    """include_shap=true is a valid request field (FakePlugin ignores it — real SHAP behaviour
    is covered by test_ml39_inference.py against the real plugin)."""
    payload = {**INLINE_PAYLOAD, "include_shap": True}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200


def test_predict_inline_missing_readings_field_returns_422(client):
    """Inline /predict without the required 'readings' field fails validation with 422."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 422


def test_predict_inline_data_contract_error_maps_to_422(client, fake_plugins):
    """A broken sensor (missing/non-finite) raises DataContractError, mapped to 422."""
    fake_plugins[MODEL_ID].raise_on_inline = DataContractError(
        "El sensor 'PS1' contiene valores no numéricos, NaN o Inf"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "PS1" in resp.json()["detail"]


def test_predict_batch(client):
    """Batch /predict returns one explanation row per cycle_id."""
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a39_batch.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_rows"] == len(body["predictions"])


def test_predict_batch_data_contract_error_maps_to_422(client, fake_plugins):
    """A malformed long-format table (missing columns) raises DataContractError -> 422."""
    fake_plugins[MODEL_ID].raise_on_batch = DataContractError(
        "Faltan columnas obligatorias en el fichero de entrada: ['PS1']"
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/bad.csv"})
    assert resp.status_code == 422


def test_train_returns_501(client):
    """training.supported=false: a39 explains ml37's model, it does not train one."""
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/a39_train.csv", "mlflow_run_id": "any"},
    )
    assert resp.status_code == 501
