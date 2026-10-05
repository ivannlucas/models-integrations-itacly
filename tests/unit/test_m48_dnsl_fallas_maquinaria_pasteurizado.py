"""Tests de wiring (FakePlugin) y de la lógica XAI de m48. La correctitud contra el golden dataset
se valida en el skill verification (inbox/a48/manifest.yaml)."""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

PREFIX = "/models/m48-dnsl-fallas-maquinaria-pasteurizado"
MODEL_ID = "m48-dnsl-fallas-maquinaria-pasteurizado"

INLINE_PAYLOAD = {
    "mode": "inline",
    "PS1": [100.0] * 60, "PS3": [50.0] * 60, "EPS1": [500.0] * 60, "FS1": [20.0] * 60,
    "TS1": [55.0] * 60, "TS2": [50.0] * 60, "VS1": [0.5] * 60,
    "Time_Segundos": [round(i * 0.1, 1) for i in range(60)],
    "Cycle_ID": 1,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok" and body["model"] == MODEL_ID and body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline_includes_local_xai_by_default(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["Enfriador_Fouling"] in (0, 1, 2)
    assert [r["componente"] for r in body["xai"]] == ["Fouling", "Válvula", "Bomba", "Acumulador"]
    assert all(r["riesgo_incluye_shap"] is False and r["sensor_mas_relevante"] is None for r in body["xai"])


def test_predict_inline_with_shap_and_without_xai(client):
    body = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_shap": True}).json()
    assert body["xai"][0]["riesgo_incluye_shap"] is True and body["xai"][0]["sensor_mas_relevante"] == "PS3"
    assert client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_xai": False}).json()["xai"] is None


def test_shap_requires_xai(client):
    resp = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_xai": False, "include_shap": True})
    assert resp.status_code == 422
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "x.csv", "include_shap": True})
    assert resp.status_code == 422


def test_inline_requires_sensors_or_path(client):
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "VS1"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cycle_data.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID and len(body["predictions"]) == 1 and body["xai"] is None


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/t.csv", "mlflow_run_id": "run"})
    assert resp.status_code == 200
    assert {"exact_match", "accuracy", "f1_macro", "recall_macro", "n_train", "n_test"} <= set(resp.json())


# ── Lógica XAI real (sin artefactos: red con pesos aleatorios) ──────────────────

torch = pytest.importorskip("torch")


def _random_model():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.model_loader import CNN_Pasteurizer
    torch.manual_seed(0)
    return CNN_Pasteurizer(n_sensors=28, n_classes=3, dropout_prob=0.2).eval()


def test_find_temporal_peaks():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.xai import find_temporal_peaks
    cam = np.array([0.1, 0.8, 0.9, 0.2, 0.75, 0.75])
    assert find_temporal_peaks(cam) == [(1, 2, pytest.approx(0.85)), (4, 5, pytest.approx(0.75))]


def test_risk_scores_and_shap_bonus():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.xai import compute_risk_scores
    risk = compute_risk_scores([2, 1, 0, 0], [0.9, 0.8, 1.0, 1.0])
    assert risk["Fouling"] == {"score": 0.9, "level": "CRÍTICO"}
    assert risk["Válvula"] == {"score": 0.4, "level": "NORMAL"}
    boosted = compute_risk_scores([1, 0, 0, 0], [0.9, 1, 1, 1], {"Fouling": {"TS1": 0.3, "TS2": 0.2, "FS1": 0.1}})
    assert boosted["Fouling"]["score"] == pytest.approx(0.45 * 1.6)


def test_recommendations_sorted_by_urgency():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.xai import generate_maintenance_recommendations
    recs = generate_maintenance_recommendations([0, 2, 1, 0], [1.0, 0.9, 0.8, 1.0])
    assert [r["urgencia"] for r in recs] == ["Alta", "Media", "Ninguna", "Ninguna"]
    assert recs[0]["componente"] == "Válvula" and recs[0]["intervalo_ciclos"] == 0
    assert recs[-1]["intervalo_ciclos"] is None


def test_gradcam_and_local_report_without_shap():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado import xai
    model = _random_model()
    x = torch.randn(1, 28, 600)
    results = xai.compute_gradcam_all_heads(model, x)
    assert len(results) == 4 and all(r["cam"].shape == (600,) and 0.0 <= r["cam"].max() <= 1.0 for r in results)
    rows = xai.build_local_report([0, 1, 2, 0], [0.9, 0.8, 0.7, 0.6], results, None, cycle_id=5, include_cam=True)
    assert [r["componente"] for r in rows] == ["Fouling", "Válvula", "Bomba", "Acumulador"]
    assert all(r["sensor_mas_relevante"] is None and r["riesgo_incluye_shap"] is False for r in rows)
    assert len(rows[0]["gradcam_cam"]) == 600 and rows[0]["cycle_id"] == 5


def test_ccp_without_shap_is_gradcam_only():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado import xai
    model = _random_model()
    X = torch.randn(6, 28, 600)
    ccps, summary = xai.detect_critical_control_points(model, X, [f"c{i}" for i in range(28)], cycle_ids=range(6), n_samples=4)
    assert summary["n_cycles_analyzed"] == 4 and summary["shap_applied"] is False and summary["shap"] == []
    assert {c["componente"] for c in ccps} == {"Fouling", "Válvula", "Bomba", "Acumulador"}
    assert all(c["sensor_critico"] is None and c["severidad"] in ("NORMAL", "WARNING", "CRÍTICO") for c in ccps)


def test_train_dto_rejects_missing_columns(tmp_path):
    import pandas as pd
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.trainer import _load_training_csv
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Cycle_ID": [1], "PS1": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        _load_training_csv(str(path))


def test_predict_dto_batch_defaults():
    from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.predict_dto import PredictBatchRequest
    req = PredictBatchRequest(data_path="x.csv")
    assert req.include_xai is False and req.include_shap is False and req.n_samples == 50
    with pytest.raises(ValidationError):
        PredictBatchRequest(data_path="x.csv", n_samples=0)
