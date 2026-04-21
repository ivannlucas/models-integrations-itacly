"""Endpoint tests for the ``wine-price-fluctuation`` model."""
from app.domain.services.exceptions import InsufficientDataError

PREFIX = "/models/wine-price-fluctuation"


def test_health(client):
    resp = client.get(f"{PREFIX}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "ok",
        "model": "wine-price-fluctuation",
        "version": "1.0.0",
        "loaded": True,
    }


def test_stats(client):
    resp = client.get(f"{PREFIX}/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "wine-price-fluctuation"
    assert body["predict_count"] == 0
    assert body["last_predict_at"] is None


def test_predict_inline(client, wine_pf_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=wine_pf_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-price-fluctuation"
    assert body["prediction"] == 1
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_type"] == "xgboost"


def test_predict_inline_rejects_fewer_than_22_records(client):
    payload = {
        "mode": "inline",
        "records": [
            {"campaign": "2023/2024", "week": w, "price_red": 40.0}
            for w in range(1, 10)
        ],
    }
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/wine_prices.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "wine-price-fluctuation"
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 2


def test_predict_domain_exception_maps_to_422(client, fake_plugins, wine_pf_inline_payload):
    fake_plugins["wine-price-fluctuation"].raise_on_inline = InsufficientDataError("too short")
    resp = client.post(f"{PREFIX}/predict", json=wine_pf_inline_payload)
    assert resp.status_code == 422
    assert "too short" in resp.json()["detail"]


def test_predict_unknown_exception_maps_to_500(client, fake_plugins, wine_pf_inline_payload):
    fake_plugins["wine-price-fluctuation"].raise_on_inline = RuntimeError("boom")
    resp = client.post(f"{PREFIX}/predict", json=wine_pf_inline_payload)
    assert resp.status_code == 500


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train")
    assert resp.status_code == 501


def test_stats_increments_after_predict(client, wine_pf_inline_payload):
    client.post(f"{PREFIX}/predict", json=wine_pf_inline_payload)
    body = client.get(f"{PREFIX}/stats").json()
    assert body["predict_count"] == 1
    assert body["last_predict_at"] is not None
