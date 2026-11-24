"""Endpoint tests for ``ml37-wine-press-predictive-maintenance`` (wiring only).

Correctness against the real artifact (LSTM multitask + neurosymbolic rule fusion, 20 golden
cases from the real UCI hydraulic-proxy test split) is validated by the `verification` skill,
not here — these tests use FakePlugin and only check wiring: request/response schemas and the
DataContractError -> 422 mapping. The real explainability code (deliverable a39, absorbed into
this plugin) is tested in test_ml37_xai.py.
"""
from app.domain.services.exceptions import DataContractError

PREFIX = "/models/ml37-wine-press-predictive-maintenance"
MODEL_ID = "ml37-wine-press-predictive-maintenance"

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


def test_predict_inline(client):
    """Inline /predict returns per-head predictions, risk, confidence level and alerts."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    for target in ("cooler", "valve", "pump", "accumulator"):
        assert target in body["predictions"]
        assert 0.0 <= body["risk"][target] <= 1.0
        assert body["confidence_level"][target] in ("green", "yellow", "red")
    assert isinstance(body["alerts"], list)
    assert isinstance(body["rule_evidence"], list)


def test_predict_inline_includes_ig_explanation_by_default(client):
    """include_xai defaults to true inline: the a39 IG explanation comes with the prediction."""
    body = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD).json()
    for target in ("cooler", "valve", "pump", "accumulator"):
        assert set(body["ig_importance"][target]) <= set(_PHYSICAL_SENSORS)
    assert body["shap_importance"] is None
    assert body["xai_error"] is None


def test_predict_inline_without_xai_has_no_explanation(client):
    """include_xai=false returns the bare prediction (no IG cost)."""
    body = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_xai": False}).json()
    assert body["ig_importance"] is None
    assert body["predictions"]["cooler"] == 3


def test_predict_inline_with_shap(client):
    """include_shap=true adds the SHAP importance on top of IG."""
    body = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_shap": True}).json()
    assert body["shap_importance"] is not None
    assert body["ig_importance"] is not None


def test_xai_defaults_on_in_inline_and_batch_requests():
    """include_xai defaults to true in both request modes; SHAP stays opt-in."""
    # pylint: disable-next=import-outside-toplevel
    from app.plugins.ml37_wine_press_predictive_maintenance.predict_dto import (
        PredictBatchRequest,
        PredictInlineRequest,
    )
    batch = PredictBatchRequest(data_path="/tmp/a37_batch.csv")
    inline = PredictInlineRequest(readings=[])
    assert batch.include_xai is True and inline.include_xai is True
    assert batch.include_shap is False and inline.include_shap is False


def test_predict_shap_without_xai_returns_422(client):
    """include_shap requires include_xai, inline and batch (same rule as ml48)."""
    inline = client.post(
        f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_xai": False, "include_shap": True},
    )
    batch = client.post(f"{PREFIX}/predict", json={
        "mode": "batch", "data_path": "/tmp/a37_batch.csv", "include_xai": False,
        "include_shap": True,
    })
    assert inline.status_code == 422
    assert batch.status_code == 422


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
    """Batch /predict returns one prediction row per cycle_id."""
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a37_batch.csv"},
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


def test_train_returns_200_with_metrics(client):
    """training.supported=true: refit del LSTM multitarea con el mismo procedimiento original."""
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/a37_train.csv", "mlflow_run_id": "test-run-id"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["accuracy"]["cooler"] <= 1.0
    assert body["n_train"] > 0 and body["n_test"] > 0


def test_train_without_mlflow_run_id_returns_422(client):
    """mlflow_run_id is required — the retrained model only ever lives in that MLflow run."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/a37_train.csv"})
    assert resp.status_code == 422
