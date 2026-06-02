"""Endpoint tests for the ``wine-sulphite`` model."""
from app.domain.services.exceptions import NoValidSimulationPointError

PREFIX = "/models/ml25_wine_sulphites"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "model": "wine-sulphite",
        "version": "1.2.0",
        "loaded": True,
    }


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "wine-sulphite"


def test_predict_inline(client, wine_so2_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-sulphite"
    assert body["intervention_recommended"] is True
    assert body["recommended_free_so2"] == 32.0
    assert body["recommended_molecular_so2"] >= 0.6


def test_predict_inline_missing_required_field(client, wine_so2_inline_payload):
    payload = {**wine_so2_inline_payload}
    del payload["pH"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
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
    fake_plugins["wine-sulphite"].raise_on_inline = NoValidSimulationPointError(
        "no feasible SO2 dose"
    )
    resp = client.post(f"{PREFIX}/predict", json=wine_so2_inline_payload)
    assert resp.status_code == 422
    assert "no feasible" in resp.json()["detail"]


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine.csv"})
    assert resp.status_code == 501
