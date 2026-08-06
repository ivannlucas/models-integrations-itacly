"""Endpoint tests for ``ml9-cereals-infestation-sequence-classifier``.

Wiring only (routing, schema validation, exception mapping) against the FakePlugin. Numerical
correctness is verified against the manifest golden_cases in outputs/a09/verification_report.md.
"""

PREFIX = "/models/ml9-cereals-infestation-sequence-classifier"

_ROW = {
    "sample_id": "S_0_0061",
    "timestamp": "2026-01-03T00:00:00",
    "co2_ppm": 434.46344,
    "temp_c": 18.640623,
    "ambient_rh_pct": 54.155678,
    "humidity_grain_pct": 11.904089,
}

# The delivered pipeline needs 48 consecutive hourly observations per series (window_size=48).
INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 48}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml9-cereals-infestation-sequence-classifier"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml9-cereals-infestation-sequence-classifier"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml9-cereals-infestation-sequence-classifier"
    assert body["pred_label"] == "sano"
    assert body["pred_class"] == 0
    assert isinstance(body["confidence"], float)
    assert body["low_confidence"] is False
    assert body["n_rows_used"] == 48


def test_predict_inline_threshold_flags_low_confidence(client):
    """`threshold` never changes the class — it only flags insufficient confidence."""
    payload = dict(INLINE_PAYLOAD, threshold=0.999)
    body = client.post(f"{PREFIX}/predict", json=payload).json()
    assert body["pred_class"] == 0
    assert body["low_confidence"] is True


def test_predict_inline_below_window_size_rejected(client):
    """rows below WINDOW_SIZE (48) must fail Pydantic validation (min_length)."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": [_ROW] * 10})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/serie.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml9-cereals-infestation-sequence-classifier"
    assert body["n_windows"] == 37
    assert body["n_series"] == 1
    assert isinstance(body["predictions"], list)
    assert set(body["class_distribution"]) == {"sano", "insectos", "moho_critico"}


def test_predict_insufficient_sequence_history_maps_to_422(client, fake_plugins):
    from app.domain.services.exceptions import InsufficientSequenceHistoryError

    fake_plugins["ml9-cereals-infestation-sequence-classifier"].raise_on_inline = (
        InsufficientSequenceHistoryError("Se requieren al menos 48 observaciones horarias.")
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_predict_batch_unexpected_error_maps_to_500(client, fake_plugins):
    """An error that is not a declared domain exception must not be masked as 422."""
    fake_plugins["ml9-cereals-infestation-sequence-classifier"].raise_on_batch = RuntimeError("boom")
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/serie.csv"})
    assert resp.status_code == 500


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    # manifest.training.metrics_returned
    for metric in ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "log_loss"):
        assert isinstance(body[metric], float)
    assert isinstance(body["n_windows_test"], int)
    assert body["artifact_path"].endswith("user_final_winner.pt")


def test_train_bad_csv_maps_to_400(client, fake_plugins):
    """A ValueError from train() (missing columns) must surface as 400, not 500."""
    plugin = fake_plugins["ml9-cereals-infestation-sequence-classifier"]

    def _boom(_plugin, *, data_path):
        raise ValueError("El CSV de entrenamiento no trae las columnas requeridas: ['target']")

    plugin._train_factory = _boom  # pylint: disable=protected-access
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/bad.csv", "mlflow_run_id": ""})
    assert resp.status_code == 400
