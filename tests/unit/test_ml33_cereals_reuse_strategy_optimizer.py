"""Endpoint tests for the ``ml33-cereals-reuse-strategy-optimizer`` (MILP) model."""

PREFIX = "/models/ml33-cereals-reuse-strategy-optimizer"

ONE_LOT = {
    "generated_volume_tons": 12.5,
    "moisture_pct": 19.5,
    "subproduct_type": "Husk",
    "season": "Rainy",
}


def test_health(client):
    """GET /health reports the model as loaded and ok."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml33-cereals-reuse-strategy-optimizer"
    assert body["loaded"] is True


def test_stats(client):
    """GET /stats returns the model name."""
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml33-cereals-reuse-strategy-optimizer"


def test_predict_inline_single_lot(client):
    """An inline predict for one lot returns its assignment and the distribution."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "lots": [ONE_LOT]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml33-cereals-reuse-strategy-optimizer"
    assert len(body["results"]) == 1
    assert body["results"][0]["ai_assignment_source"] == "exact_min_emissions"
    assert "distribution" in body


def test_predict_inline_multiple_lots_and_capacity_override(client):
    """Several lots with a capacity override return one result per lot."""
    payload = {
        "mode": "inline",
        "lots": [ONE_LOT, {**ONE_LOT, "subproduct_type": "Bran", "season": "Dry"}],
        "lots_per_day": 2,
        "animal_feed_capacity": 5.0,
    }
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_predict_inline_requires_at_least_one_lot(client):
    """An inline predict with no lots is rejected with 422."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "lots": []})
    assert resp.status_code == 422


def test_predict_batch(client):
    """A batch predict returns the model id."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/lots.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml33-cereals-reuse-strategy-optimizer"


def test_train_returns_501(client):
    """POST /train is not implemented for this model (501)."""
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/dataset.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 501
