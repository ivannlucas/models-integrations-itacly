"""Endpoint tests for the ``wine-sulphite`` model."""
from unittest.mock import MagicMock

from app.domain.services.exceptions import NoValidSimulationPointError

PREFIX = "/models/ml25_wine_sulphites"


def test_health(client):
    """Health endpoint returns status ok with correct model id, version, and loaded flag."""
    body = client.get(f"{PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "model": "wine-sulphite",
        "version": "1.2.0",
        "loaded": True,
    }


def test_stats(client):
    """Stats endpoint returns a response with the correct model_name field."""
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "wine-sulphite"


def test_predict_inline(client, wine_so2_inline_payload):
    """Inline predict returns 200 with expected SO2 recommendation fields."""
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-sulphite"
    assert body["intervention_recommended"] is True
    assert body["recommended_free_so2"] == 32.0
    assert body["recommended_molecular_so2"] >= 0.6


def test_predict_inline_missing_required_field(client, wine_so2_inline_payload):
    """Omitting a required field produces a 422 Unprocessable Entity response."""
    payload = {**wine_so2_inline_payload}
    del payload["pH"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    """Batch predict returns 200 with a predictions list containing intervention flags."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/wine.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-sulphite"
    assert body["predictions"][0]["intervention_recommended"] is True


def test_predict_no_valid_simulation_point_maps_to_422(
    client, fake_plugins, wine_so2_inline_payload
):
    """NoValidSimulationPointError raised by the plugin is mapped to HTTP 422."""
    fake_plugins["wine-sulphite"].raise_on_inline = NoValidSimulationPointError(
        "no feasible SO2 dose"
    )
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 422
    assert "no feasible" in resp.json()["detail"]


def test_train_returns_501(client):
    """Train endpoint returns 501 Not Implemented when the plugin raises TrainingNotSupportedError."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine.csv"})
    assert resp.status_code == 501


def test_predict_inline_unexpected_exception_maps_to_500(client, fake_plugins, wine_so2_inline_payload):
    """An unexpected runtime exception from the plugin is mapped to HTTP 500."""
    fake_plugins["wine-sulphite"].raise_on_inline = RuntimeError("unexpected crash")
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 500


def test_train_unexpected_exception_maps_to_500(client, fake_plugins):
    """An unexpected runtime exception during training is mapped to HTTP 500."""
    fake_plugins["wine-sulphite"].train = MagicMock(side_effect=RuntimeError("train crash"))
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine.csv"})
    assert resp.status_code == 500


def test_predict_batch_missing_data_path_returns_422(client):
    """Batch predict request without the required data_path field returns 422."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch"})
    assert resp.status_code == 422
