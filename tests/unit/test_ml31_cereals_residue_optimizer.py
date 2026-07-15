"""Endpoint tests for the ``ml31-cereals-residue-optimizer`` (LP) model."""

PREFIX = "/models/ml31-cereals-residue-optimizer"

OPTIMIZE_PAYLOAD = {
    "mode": "optimize",
    "optimization_mode": "minimize_residue",
    "surface_tolerance_pct": 25.0,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml31-cereals-residue-optimizer"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml31-cereals-residue-optimizer"


def test_predict_optimize(client):
    resp = client.post(f"{PREFIX}/predict", json=OPTIMIZE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml31-cereals-residue-optimizer"
    assert body["solver_status"] == "OPTIMAL"
    assert body["optimization_mode"] == "minimize_residue"
    assert isinstance(body["crop_allocation"], dict)
    assert "total_residue_t" in body


def test_predict_optimize_invalid_mode(client):
    payload = {"mode": "optimize", "optimization_mode": "not_a_mode"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/scenarios.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml31-cereals-residue-optimizer"


def test_train_returns_501(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/dataset.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 501
