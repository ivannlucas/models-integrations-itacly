"""Endpoint tests for ``ml19-cereals-cost-forecast`` (wiring only).

Correctness against the real artifact (golden_cases) is validated by the `verification` skill,
not here — these tests use FakePlugin and only check wiring: request/response schemas, the
'date' format guard and DataContractError -> 422 mapping.
"""
from app.domain.services.exceptions import DataContractError

PREFIX = "/models/ml19-cereals-cost-forecast"
MODEL_ID = "ml19-cereals-cost-forecast"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline_no_date_uses_latest(client):
    """Without 'date', the service should still answer (latest row of the bundled dataset)."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert len(body["horizons"]) >= 1
    assert body["horizons"][0]["reg"]["signal_str"] in ("LONG", "SHORT", "FLAT")


def test_predict_inline_with_date(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "date": "2020-01"})
    assert resp.status_code == 200
    assert resp.json()["date_label"] == "2020-01"


def test_predict_inline_rejects_malformed_date(client):
    """'date' must match YYYY-MM — this is a Pydantic pattern guard, not a plugin call."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "date": "2020/01"})
    assert resp.status_code == 422


def test_predict_inline_unavailable_date_maps_to_422(client, fake_plugins):
    """DataContractError raised by the plugin (date outside the bundled reference dataset)
    must map to HTTP 422, not 500."""
    fake_plugins[MODEL_ID].raise_on_inline = DataContractError(
        "Fecha '1999-01' no disponible en el dataset de referencia empaquetado."
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "date": "1999-01"})
    assert resp.status_code == 422
    assert "no disponible" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a19_dates.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_rows"] == len(body["predictions"])


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 501
