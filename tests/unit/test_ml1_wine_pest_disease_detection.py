"""Endpoint tests for ``ml1-wine-pest-disease-detection`` (wiring only).

Correctness against the real artifact (golden_cases: YOLO + 3-backbone Keras ensemble + RF
tabular fallback) is validated by the `verification` skill, not here — these tests use
FakePlugin and only check wiring: request/response schemas, the image/no-image branch and the
DataContractError/InvalidImageError -> 422 mapping.
"""
from app.domain.services.exceptions import DataContractError, InvalidImageError

PREFIX = "/models/ml1-wine-pest-disease-detection"
MODEL_ID = "ml1-wine-pest-disease-detection"

# Minimal valid base64 PNG (1x1 pixel, red) — does not go through real inference
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

_SENSORS = {
    "Mes": 6, "Temp_Max_Media_5d_C": 28.5, "Temp_Min_Media_5d_C": 14.2,
    "Hum_Rel_Max_5d_Pct": 85.0, "Hum_Rel_Min_5d_Pct": 40.0, "Hum_Suelo_Pct": 22.0,
    "Lluvia_Acum_5d_mm": 3.5, "Horas_Humedad_Foliar": 6.0, "GDD_Acumulado": 850.0,
}

INLINE_PAYLOAD_WITH_IMAGE = {
    "mode": "inline",
    "image_base64": _TINY_PNG_B64,
    "sensors": _SENSORS,
}

INLINE_PAYLOAD_NO_IMAGE = {
    "mode": "inline",
    "sensors": _SENSORS,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline_with_image_uses_hybrid_ensemble(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD_WITH_IMAGE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["metodo"] == "HIBRIDO_ENSAMBLE"
    assert 0.0 <= body["confianza"] <= 1.0
    assert 0.0 <= body["riesgo"] <= 1.0


def test_predict_inline_without_image_uses_tabular_fallback(client):
    """No imagen -> modo de contingencia: solo riesgo via RandomForest, sin clase."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD_NO_IMAGE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["metodo"] == "RIESGO_TABULAR_EXPERT"
    assert body["enfermedad"] == "IMAGEN_NO_DISPONIBLE"
    assert body["confianza"] == 0.0


def test_predict_inline_missing_sensors_returns_422(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "image_base64": _TINY_PNG_B64})
    assert resp.status_code == 422


def test_predict_inline_invalid_image_maps_to_422(client, fake_plugins):
    fake_plugins[MODEL_ID].raise_on_inline = InvalidImageError("imagen no decodificable")
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD_WITH_IMAGE)
    assert resp.status_code == 422
    assert "imagen no decodificable" in resp.json()["detail"]


def test_predict_inline_sensor_mismatch_maps_to_422(client, fake_plugins):
    """validate_and_align_sensors raises DataContractError on missing/unknown sensor columns."""
    fake_plugins[MODEL_ID].raise_on_inline = DataContractError(
        "Faltan columnas de sensores obligatorias: ['GDD_Acumulado']"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD_NO_IMAGE)
    assert resp.status_code == 422
    assert "Faltan columnas" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/a1_batch.zip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_rows"] == len(body["predictions"])


def test_predict_batch_bad_archive_maps_to_422(client, fake_plugins):
    """safe_extract_zip's ValueError is wrapped by the plugin as DataContractError (422)."""
    fake_plugins[MODEL_ID].raise_on_batch = DataContractError(
        "El ZIP de entrada no contiene un CSV valido"
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/bad.zip"})
    assert resp.status_code == 422


def test_train_returns_200_with_metrics(client):
    """training.supported=true, pero solo para los expertos RandomForest tabulares (ver
    constants.py/manifest.yaml training.scope_note) — el YOLO + ensemble Keras no se reentrena."""
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/a1_sensores_train.csv", "mlflow_run_id": "test-run-id"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["accuracy"] <= 1.0
    assert body["n_train"] > 0 and body["n_test"] > 0


def test_train_without_mlflow_run_id_returns_422(client):
    """mlflow_run_id is required — every retrain must be persisted to a specific MLflow run."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/a1_sensores_train.csv"})
    assert resp.status_code == 422
