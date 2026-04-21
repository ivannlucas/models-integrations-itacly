"""Endpoint tests for the ``meat-price-forecast`` model."""
from app.domain.services.exceptions import InsufficientRowsError

PREFIX = "/models/meat-price-forecast"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "model": "meat-price-forecast",
        "version": "1.0.0",
        "loaded": True,
    }


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "meat-price-forecast"


def test_predict_inline(client, meat_inline_payload):
    resp = client.post(f"{PREFIX}/predict", json=meat_inline_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "meat-price-forecast"
    assert set(body["prediction"].keys()) == {"bovino", "porcino", "ovino", "ave", "carne"}
    assert body["rows_used"] == 4


def test_predict_inline_rejects_fewer_than_4_rows(client):
    payload = {
        "mode": "inline",
        "rows": [
            {
                "date": "2024-01-01",
                "bovino": 130.0,
                "porcino": 128.0,
                "ovino": 135.0,
                "ave": 120.0,
                "carne": 129.0,
            }
        ],
    }
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/meat.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "meat-price-forecast"


def test_predict_insufficient_rows_maps_to_422(client, fake_plugins, meat_inline_payload):
    fake_plugins["meat-price-forecast"].raise_on_inline = InsufficientRowsError(
        "not enough rows after dropna"
    )
    resp = client.post(f"{PREFIX}/predict", json=meat_inline_payload)
    assert resp.status_code == 422


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train")
    assert resp.status_code == 501
