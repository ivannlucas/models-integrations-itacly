"""Endpoint (wiring) tests for the ``ml31-cereals-residue-optimizer`` v2.0 LP model.

Correctness against the audited scenarios lives in the verification skill
(golden dataset); these tests only exercise the router/DTO wiring via FakePlugin.
"""

PREFIX = "/models/ml31-cereals-residue-optimizer"

OPTIMIZE_PAYLOAD = {
    "mode": "optimize",
    "reference_year": 2023,
    "optimization_mode": "minimize_residue",
}

PARETO_PAYLOAD = {
    "mode": "pareto",
    "reference_year": 2023,
    "num_points": 20,
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
    assert isinstance(body["total_residue_t"], (int, float))
    assert isinstance(body["residue_reduction_pct"], (int, float))
    assert isinstance(body["crop_allocation"], dict)


def test_predict_pareto(client):
    resp = client.post(f"{PREFIX}/predict", json=PARETO_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml31-cereals-residue-optimizer"
    assert isinstance(body["pareto_points"], list)
    assert isinstance(body["bounds"], dict)


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/scenarios.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml31-cereals-residue-optimizer"
    assert isinstance(body["predictions"], list)


def test_train_returns_501(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": ""})
    assert resp.status_code == 501
