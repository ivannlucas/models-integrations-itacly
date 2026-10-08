"""Endpoint tests for ``ml21-cereals-price-spatial`` (ESP-CEREAL spatial cereal price)."""

PREFIX = "/models/ml21-cereals-price-spatial"

INLINE_PAYLOAD = {
    "mode": "inline",
    "provincia": "Burgos",
    "cereal_predominante": "trigo",
    "date": "2024-01",
    "role": "comprador",
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml21-cereals-price-spatial"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml21-cereals-price-spatial"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml21-cereals-price-spatial"
    assert "predictions" in body
    assert "H1" in body["predictions"]
    assert "H2" in body["predictions"]
    assert "H3" in body["predictions"]
    assert body["province"] == "Burgos"
    assert body["cereal"] == "trigo"


def test_predict_batch(client):
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cereal.csv", "month": "2024-01"}
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "ml21-cereals-price-spatial"


def test_train_returns_501(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code in (200, 501)


def test_train_returns_metrics(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert "mae_h1" in body
    assert "pearson_h1" in body
    assert "da_h1" in body
    assert "auc_h1" in body
    assert body["mae_h1"] is not None


def test_stats_fetches_mlflow_metrics_when_run_id_given():
    """Regression: stats(mlflow_run_id=...) accepted the parameter but never used it —
    every retrain's /stats always reported this plugin's own static/served metrics, no
    matter which trained run was asked about, same class of bug found and fixed in
    ml43_cereals_dnsl_anomaly_fault_detection/ml3_wine_disease_pest_forecast (see plugin.py::stats)."""
    from unittest.mock import patch

    from app.plugins.ml21_cereals_price_spatial.plugin import Ml21CerealsPriceSpatialPlugin

    plugin = Ml21CerealsPriceSpatialPlugin()
    with patch(
        "app.plugins.ml21_cereals_price_spatial.plugin.BaseMLflowTracker"
    ) as mock_tracker_cls:
        mock_tracker_cls.return_value.get_metrics.return_value = {"H1_reg_MAE": 12.5}
        mock_tracker_cls.return_value.get_params.return_value = {}
        resp = plugin.stats(mlflow_run_id="some-run-id")

    assert resp.metrics["H1_reg_MAE"] == 12.5
    assert resp.metrics["mlflow"]["metrics"]["H1_reg_MAE"] == 12.5


def test_stats_overwrites_legacy_per_horizon_keys_with_the_real_run_values():
    """Regression: train() actually logs mae_h{h}/pearson_h{h}/da_h{h}/auc_h{h} — not
    the legacy H{h}_reg_MAE/H{h}_reg_Pearson/H{h}_clf_DA/H{h}_clf_AUC keys the test
    above mocked. The flattening loop only ADDS the real flat keys; it never touches
    the legacy ones (built from self._metadata), so every retrain's /stats kept
    showing the served model's fixed per-horizon numbers under those exact tiles."""
    from unittest.mock import patch

    from app.plugins.ml21_cereals_price_spatial.plugin import Ml21CerealsPriceSpatialPlugin

    plugin = Ml21CerealsPriceSpatialPlugin()
    with patch(
        "app.plugins.ml21_cereals_price_spatial.plugin.BaseMLflowTracker"
    ) as mock_tracker_cls:
        mock_tracker_cls.return_value.get_metrics.return_value = {
            "mae_h1": 11.2, "pearson_h1": 0.62, "da_h1": 0.58, "auc_h1": 0.71,
        }
        mock_tracker_cls.return_value.get_params.return_value = {}
        resp = plugin.stats(mlflow_run_id="some-run-id")

    assert resp.metrics["H1_reg_MAE"] == 11.2
    assert resp.metrics["H1_reg_Pearson"] == 0.62
    assert resp.metrics["H1_clf_DA"] == 0.58
    assert resp.metrics["H1_clf_AUC"] == 0.71
