PREFIX = "/models/ml47-dairy-dnsl-pasteurization-fault-detection"

INLINE_PAYLOAD = {
    "mode": "inline",
    "PS1": [100.0] * 60,
    "PS3": [50.0] * 60,
    "EPS1": [500.0] * 60,
    "FS1": [20.0] * 60,
    "TS1": [55.0] * 60,
    "TS2": [50.0] * 60,
    "VS1": [0.5] * 60,
    "Time_Segundos": [round(i * 0.1, 1) for i in range(60)],
    "Cycle_ID": 1,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml47-dairy-dnsl-pasteurization-fault-detection"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml47-dairy-dnsl-pasteurization-fault-detection"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml47-dairy-dnsl-pasteurization-fault-detection"
    assert body["Enfriador_Fouling"] in (0, 1, 2)
    assert body["Valvula_Switch"] in (0, 1, 2)
    assert body["Bomba_Leakage"] in (0, 1, 2)
    assert body["Acumulador_Gas"] in (0, 1, 2)
    assert 0.0 <= body["Confianza_Fouling"] <= 1.0


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cycle_data.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml47-dairy-dnsl-pasteurization-fault-detection"
    assert len(body["predictions"]) == 1


def test_train_defaults_to_fine_tune(client, fake_plugins):
    calls = []
    plugin = fake_plugins["ml47-dairy-dnsl-pasteurization-fault-detection"]
    real_train = plugin.train

    def spy(**kwargs):
        calls.append(kwargs.get("mode"))
        return real_train(data_path=kwargs["data_path"], mlflow_run_id=kwargs["mlflow_run_id"])

    spy.__signature__ = __import__("inspect").signature(lambda *, data_path, mlflow_run_id, mode="fine_tune": None)
    plugin.train = spy
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/plant.csv", "mlflow_run_id": "run-1"})
    assert resp.status_code == 200
    assert calls == ["fine_tune"]
    body = resp.json()
    assert {"exact_match", "accuracy", "f1_macro", "n_train", "n_val", "n_test", "mlflow_run_id"} <= set(body)


def test_train_accepts_full_mode(client):
    resp = client.post(f"{PREFIX}/train",
                       json={"data_path": "/tmp/plant.csv", "mlflow_run_id": "run-1", "mode": "full"})
    assert resp.status_code == 200


def test_train_rejects_unknown_mode(client):
    resp = client.post(f"{PREFIX}/train",
                       json={"data_path": "/tmp/plant.csv", "mlflow_run_id": "run-1", "mode": "partial"})
    assert resp.status_code == 422


def test_train_without_mlflow_run_id_returns_422(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/plant.csv"})
    assert resp.status_code == 422
