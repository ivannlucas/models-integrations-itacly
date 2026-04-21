"""Endpoint tests for the ``cereal-price-forecast`` model."""
from app.domain.services.exceptions import UnsupportedProductError

PREFIX = "/models/cereal-price-forecast"


def test_health(client):
    resp = client.get(f"{PREFIX}/health")
    assert resp.status_code == 200
    assert resp.json()["model"] == "cereal-price-forecast"
    assert resp.json()["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "cereal-price-forecast"
    assert body["predict_count"] == 0


def test_predict_inline(client, cereal_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=cereal_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cereal-price-forecast"
    assert body["product_name"] == "Milling wheat"
    assert body["prediction"] == 245.7


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/cereal.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "cereal-price-forecast"
    assert len(body["predictions"]) == 2


def test_predict_unsupported_product_maps_to_422(
    client, fake_plugins, cereal_inline_payload
):
    fake_plugins["cereal-price-forecast"].raise_on_inline = UnsupportedProductError(
        "Unknown product 'Rice'"
    )
    resp = client.post(f"{PREFIX}/predict", json=cereal_inline_payload)
    assert resp.status_code == 422
    assert "Rice" in resp.json()["detail"]


def test_predict_inline_missing_product_name(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 422


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train")
    assert resp.status_code == 501
