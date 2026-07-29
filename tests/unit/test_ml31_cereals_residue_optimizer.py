"""Endpoint tests for the ``ml31-cereals-residue-optimizer`` (LP) model."""

PREFIX = "/models/ml31-cereals-residue-optimizer"

OPTIMIZE_PAYLOAD = {
    "mode": "inline",
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
    payload = {"mode": "inline", "optimization_mode": "not_a_mode"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_optimize_null_expected_spring_rain_mm(client):
    """Real platform payload sends expected_spring_rain_mm: null when the form
    field is left blank (see fermentation-optimization.js's modelo-31 inline
    payload) — must fall back to the field's default (130.0), not 422."""
    payload = {
        "mode": "inline",
        "reference_year": 2026,
        "optimization_mode": "minimize_residue",
        "surface_tolerance_pct": 25,
        "climate_factor": 1,
        "expected_spring_rain_mm": None,
    }
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["solver_status"] == "OPTIMAL"


def test_predict_pareto_request_accepted(client):
    """model_key="pareto" is a valid inline sub-mode (routing itself is covered by
    the plugin's own unit tests; this fake-plugin harness only verifies the DTO
    accepts the shape and the endpoint returns 200)."""
    payload = {"mode": "inline", "model_key": "pareto", "num_points": 5}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml31-cereals-residue-optimizer"


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/scenarios.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml31-cereals-residue-optimizer"


def test_train_returns_501(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/dataset.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 501
