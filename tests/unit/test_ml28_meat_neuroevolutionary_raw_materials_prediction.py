"""Endpoint tests for ``ml28-meat-neuroevolutionary-raw-materials-prediction``."""

PREFIX = "/models/ml28-meat-neuroevolutionary-raw-materials-prediction"
MODEL_ID = "ml28-meat-neuroevolutionary-raw-materials-prediction"

INLINE_ROW = {
    "mode": "inline",
    "date": "2025-01-26",
    "raw_material_id": "RM_BEEF_TRIM_A",
    "destination_profile": "cooked_standard_line",
    "current_inventory_tons": 36.0,
    "expected_requirement_tons": 24.0,
    "lead_time_days": 6.0,
    "safety_coverage_days": 11.0,
    "expected_yield_rate": 0.88,
    "expected_waste_rate": 0.02,
    "unit_purchase_cost": 3.88,
    "shelf_life_days": 28,
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
    resp = client.post(f"{PREFIX}/predict", json=INLINE_ROW)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["recommended_action"] in ("BUY", "DO_NOT_BUY")
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    # Gating invariant: order_quantity_tons must be 0.0 whenever the trigger is off.
    if body["purchase_trigger_flag"] == 0:
        assert body["order_quantity_tons"] == 0.0


def test_predict_inline_missing_field_rejected(client):
    payload = dict(INLINE_ROW)
    del payload["current_inventory_tons"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_inline_negative_inventory_rejected(client):
    payload = dict(INLINE_ROW)
    payload["current_inventory_tons"] = -5.0
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/fake.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert len(body["predictions"]) == 1
    assert body["summary"]["stockout_guardrail_pass"] is True


def test_train_returns_501(client):
    """training.supported=false — /train must map to HTTP 501, not silently succeed."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": ""})
    assert resp.status_code == 501
