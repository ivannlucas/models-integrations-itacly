"""Endpoint tests for ``ml15-wine-ipi-price-forecast``."""
from app.domain.services.exceptions import MissingRequiredFeatureError

PREFIX = "/models/ml15-wine-ipi-price-forecast"
MODEL_ID = "ml15-wine-ipi-price-forecast"

# caso_001 from inbox/a15/manifest.yaml golden_cases (verified against the real artifact).
INLINE_PAYLOAD = {
    "mode": "inline",
    "ipi_national_current": 123.85, "ipi_national_lag_1": 123.72, "ipi_national_lag_2": 123.87,
    "ipi_national_lag_3": 124.06, "ipi_national_lag_4": 124.06, "ipi_national_lag_5": 124.45,
    "ipi_national_lag_6": 124.19, "chem_sector_lag_11": 153.7234453479238,
    "copper_lag_14": 132.92181735608338, "eur_usd_lag_17": 95.6270726013976,
    "oil_brent_lag_12": 183.46434257741848, "usa_lag_1": 110.272157564906,
    "date": "2025-01-01",
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == MODEL_ID


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["horizon"] == 6
    assert isinstance(body["y_pred"], float)


def test_predict_inline_missing_feature_maps_to_422(client, fake_plugins):
    """MissingRequiredFeatureError raised by the plugin must map to HTTP 422, not 500."""
    fake_plugins[MODEL_ID].raise_on_inline = MissingRequiredFeatureError(
        "Faltan variables obligatorias del modelo: ['copper_lag_14']"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "copper_lag_14" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a15_panel.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"])


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/a15_train.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_train_rows"] > 0
    assert body["n_test_rows"] > 0
