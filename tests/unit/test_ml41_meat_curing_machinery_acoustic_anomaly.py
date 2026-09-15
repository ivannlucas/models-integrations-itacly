"""Endpoint tests for the ``ml41-meat-curing-machinery-acoustic-anomaly`` model."""
from app.domain.services.exceptions import InvalidAudioError, UnsupportedMachineConfigurationError

PREFIX = "/models/ml41-meat-curing-machinery-acoustic-anomaly"

# A few bytes of base64 — the FakePlugin never runs real inference on this, only
# wiring/schema validation is exercised here.
_TINY_AUDIO_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="

INLINE_PAYLOAD = {
    "mode": "inline",
    "audio_base64": _TINY_AUDIO_B64,
    "machine": "fan",
    "machine_id": "id_00",
    "snr": "0_dB",
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml41-meat-curing-machinery-acoustic-anomaly"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml41-meat-curing-machinery-acoustic-anomaly"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml41-meat-curing-machinery-acoustic-anomaly"
    assert body["machine"] == "fan"
    assert body["machine_id"] == "id_00"
    assert body["snr"] == "0_dB"
    assert body["predicted_label"] in (0, 1)
    assert isinstance(body["maha_score"], float)
    assert isinstance(body["mse_score"], float)
    assert isinstance(body["threshold_used"], float)


def test_predict_inline_missing_field(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "audio_base64": _TINY_AUDIO_B64})
    assert resp.status_code == 422


def test_predict_inline_invalid_machine_rejected_by_schema(client):
    payload = {**INLINE_PAYLOAD, "machine": "not-a-machine"}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_inline_unsupported_combination_maps_to_422(client, fake_plugins):
    fake_plugins["ml41-meat-curing-machinery-acoustic-anomaly"].raise_on_inline = (
        UnsupportedMachineConfigurationError("combinación no soportada")
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "combinación no soportada" in resp.json()["detail"]


def test_predict_inline_invalid_audio_maps_to_422(client, fake_plugins):
    fake_plugins["ml41-meat-curing-machinery-acoustic-anomaly"].raise_on_inline = InvalidAudioError(
        "audio no decodificable"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "audio no decodificable" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "/tmp/ml41_batch.zip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml41-meat-curing-machinery-acoustic-anomaly"
    assert isinstance(body["predictions"], list)


def test_train(client):
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/ml41_train.zip", "mlflow_run_id": "test-run-id"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "combinación" in body["detail"]
    assert isinstance(body["per_combination"], list)
    assert body["per_combination"][0]["machine"] == "fan"
