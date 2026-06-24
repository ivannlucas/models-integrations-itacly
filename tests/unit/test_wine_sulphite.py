"""Endpoint tests for the ``wine-sulphite`` model."""
from app.domain.services.exceptions import NoValidSimulationPointError

PREFIX = "/models/wine-sulphite"


def test_health(client):
    """Verify the health endpoint returns expected metadata."""
    body = client.get(f"{PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "model": "wine-sulphite",
        "version": "1.2.0",
        "loaded": True,
    }


def test_stats(client):
    """Verify the stats endpoint returns the model name."""
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "wine-sulphite"


def test_predict_inline(client, wine_so2_inline_payload):
    """Verify inline prediction returns expected SO2 recommendation fields."""
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-sulphite"
    assert body["intervention_recommended"] is True
    assert body["recommended_free_so2"] == 32.0
    assert body["recommended_molecular_so2"] >= 0.6


def test_predict_inline_missing_required_field(client, wine_so2_inline_payload):
    """Verify missing required fields return HTTP 422."""
    payload = {**wine_so2_inline_payload}
    del payload["pH"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    """Verify batch prediction returns predictions list."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/wine.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-sulphite"
    assert body["predictions"][0]["intervention_recommended"] is True


def test_predict_inline_with_mlflow_run_id(client, wine_so2_inline_payload):
    """Verify mlflow_run_id in predict request is accepted (ignored by fake)."""
    payload = {**wine_so2_inline_payload, "mlflow_run_id": "mlflow-run-123"}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "wine-sulphite"


def test_predict_batch_with_mlflow_run_id(client):
    """Verify mlflow_run_id in batch predict is accepted."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/wine.csv", "mlflow_run_id": "mlflow-run-123"},
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "wine-sulphite"


def test_stats_with_mlflow_run_id(client):
    """Verify stats endpoint accepts mlflow_run_id query parameter."""
    body = client.get(f"{PREFIX}/stats", params={"mlflow_run_id": "mlflow-run-123"}).json()
    assert body["model_name"] == "wine-sulphite"


def test_predict_no_valid_simulation_point_maps_to_422(
    client, fake_plugins, wine_so2_inline_payload
):
    """Verify NoValidSimulationPointError maps to HTTP 422."""
    fake_plugins["wine-sulphite"].raise_on_inline = NoValidSimulationPointError(
        "no feasible SO2 dose"
    )
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 422
    assert "no feasible" in resp.json()["detail"]


def test_train_without_mlflow_run_id_returns_422(client):
    """Verify train without mlflow_run_id returns HTTP 422 (required field)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine.csv"})
    assert resp.status_code == 422


def test_train_returns_501(client):
    """Verify train returns HTTP 501 when no training backend is wired for wine-sulphite."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 501


def test_predict_unexpected_error_maps_to_500(
    client, fake_plugins, wine_so2_inline_payload
):
    """Verify unexpected RuntimeError maps to HTTP 500."""
    fake_plugins["wine-sulphite"].raise_on_inline = RuntimeError("unexpected failure")
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 500


def test_predict_invalid_mode_returns_422(client):
    """Verify invalid mode returns HTTP 422."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "invalid"},
    )
    assert resp.status_code == 422


def test_predict_missing_mode_returns_422(client):
    """Verify missing mode returns HTTP 422."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={},
    )
    assert resp.status_code == 422
