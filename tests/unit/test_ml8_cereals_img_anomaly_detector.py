"""Endpoint tests for the ``ml8-cereals-img-anomaly-detector`` model."""
from app.domain.services.exceptions import InvalidImageError

PREFIX = "/models/ml8-cereals-img-anomaly-detector"

# Minimal valid base64 PNG (1×1 pixel, red) — does not go through real inference
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

INLINE_PAYLOAD = {
    "mode": "inline",
    "image_base64": _TINY_PNG_B64,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml8-cereals-img-anomaly-detector"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml8-cereals-img-anomaly-detector"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml8-cereals-img-anomaly-detector"
    assert body["categoria"] in ("sano", "hongos", "insectos", "otros")
    assert body["cereal"] in ("trigo", "maiz", "arroz", "sorgo")
    assert 0.0 <= body["confianza_categoria"] <= 1.0
    assert 0.0 <= body["confianza_cereal"] <= 1.0
    assert set(body["probabilidades_categoria"].keys()) == {"sano", "hongos", "insectos", "otros"}
    assert set(body["probabilidades_cereal"].keys()) == {"trigo", "maiz", "arroz", "sorgo"}


def test_predict_inline_missing_field(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/cereales.zip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml8-cereals-img-anomaly-detector"
    assert isinstance(body["predictions"], list)


def test_predict_inline_invalid_image_maps_to_422(client, fake_plugins):
    fake_plugins["ml8-cereals-img-anomaly-detector"].raise_on_inline = InvalidImageError(
        "imagen no decodificable"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "imagen no decodificable" in resp.json()["detail"]


def test_train(client):
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/cereales_train.zip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"] == "Entrenamiento completado"
    assert isinstance(body["train_samples"], int)
    assert isinstance(body["best_val_acc_cat"], float)
    assert isinstance(body["best_val_acc_cer"], float)
