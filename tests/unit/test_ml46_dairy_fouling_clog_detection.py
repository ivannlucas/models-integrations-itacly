"""Endpoint tests for ``ml46-dairy-fouling-clog-detection``."""
import pandas as pd
import pytest

PREFIX = "/models/ml46-dairy-fouling-clog-detection"

_ROW = {
    "timestamp": "2026-01-11T06:01:00+00:00",
    "asset_id": "asset_00",
    "flow_kg_s": 6.6,
    "pressure_in_kPa": 250.0,
    "pressure_out_kPa": 163.0,
    "dP_kPa": 87.0,
    "Th_in_C": 90.0,
    "Tc_in_C": 55.0,
    "Th_out_C": 85.0,
    "Tc_out_C": 59.0,
    "Twall_C": 70.0,
    "vibration_mm_s": 1.8,
    "flow_sp_kg_s": 6.6,
    "Th_sp_C": 90.0,
    "Tc_sp_C": 55.0,
    "protein_g_L_nominal": 32.0,
    "fat_g_L_nominal": 38.0,
    "solids_g_L_nominal": 125.0,
    "Ca_mM_nominal": 30.0,
    "PO4_mM_nominal": 20.0,
    "pH_nominal": 6.6,
    "phase": "production",
    "maintenance_active": 0,
    "asset_family": "robust_phe",
    "milk_type": "high_solids",
    "last_maintenance_type": "none",
}

INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 120}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml46-dairy-fouling-clog-detection"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml46-dairy-fouling-clog-detection"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml46-dairy-fouling-clog-detection"
    assert body["pred_stage_name"] == "stable"
    assert isinstance(body["pred_severity"], float)
    assert isinstance(body["is_alert"], bool)


def test_predict_inline_insufficient_history_rejected(client):
    """rows below SEQ_LEN (120) must fail Pydantic validation (min_length)."""
    payload = {"mode": "inline", "rows": [_ROW] * 5}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/telemetry.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml46-dairy-fouling-clog-detection"
    assert isinstance(body["predictions"], list)
    assert isinstance(body["alerts"], list)


def test_predict_insufficient_telemetry_history_maps_to_422(client, fake_plugins):
    from app.domain.services.exceptions import InsufficientTelemetryHistoryError

    fake_plugins["ml46-dairy-fouling-clog-detection"].raise_on_inline = InsufficientTelemetryHistoryError(
        "Se requieren al menos 120 filas de telemetría."
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_validate_raw_columns_rejects_missing_columns():
    """Missing required raw telemetry columns must be rejected up front (v2 input_contract parity),
    instead of silently degrading into NaN-filled features downstream."""
    from app.plugins.ml46_dairy_fouling_clog_detection.preprocessing import validate_raw_columns

    df = pd.DataFrame([{"timestamp": "2026-01-01T00:00:00Z", "asset_id": "asset_00"}])
    with pytest.raises(ValueError, match="CSV falta columnas requeridas"):
        validate_raw_columns(df)


def test_validate_raw_columns_accepts_full_contract():
    from app.plugins.ml46_dairy_fouling_clog_detection.constants import RAW_FIXED_COLUMNS
    from app.plugins.ml46_dairy_fouling_clog_detection.preprocessing import validate_raw_columns

    df = pd.DataFrame([{c: 0 for c in RAW_FIXED_COLUMNS}])
    validate_raw_columns(df)  # must not raise


def test_consolidate_alerts_groups_continuous_event_into_one_episode():
    """A fault that stays actionable for hours must yield ONE alert episode, not a fresh
    alert every cooldown window — the v2 episode-based consolidation."""
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.evaluation import consolidate_alerts

    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = [
        {"asset_id": "asset_00", "alert_type": "fouling", "timestamp": base + pd.Timedelta(minutes=5 * i), "actionable_alert": 1}
        for i in range(30)  # spans 145 min at 5-min steps — well beyond a 60-min cooldown
    ]
    alerts = consolidate_alerts(pd.DataFrame(rows), cooldown_min=60)
    assert len(alerts) == 1
    assert alerts.iloc[0]["n_signals"] == 30
    assert alerts.iloc[0]["alert_episode_id"] == "AE000001"


def test_consolidate_alerts_splits_episodes_separated_by_gap():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.evaluation import consolidate_alerts

    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = [
        {"asset_id": "asset_00", "alert_type": "fouling", "timestamp": base, "actionable_alert": 1},
        {"asset_id": "asset_00", "alert_type": "fouling", "timestamp": base + pd.Timedelta(minutes=200), "actionable_alert": 1},
    ]
    alerts = consolidate_alerts(pd.DataFrame(rows), cooldown_min=60)
    assert len(alerts) == 2


def test_get_expected_architecture_returns_scenario_contract():
    from app.plugins.ml46_dairy_fouling_clog_detection.model_loader import get_expected_architecture

    manifest = {
        "artifact_contract": {
            "scenario_contracts": {"no_clock": {"architecture": {"n_features": 76, "channels": 64}}},
        },
    }
    assert get_expected_architecture(manifest, "no_clock") == {"n_features": 76, "channels": 64}


def test_get_expected_architecture_missing_manifest_returns_none():
    from app.plugins.ml46_dairy_fouling_clog_detection.model_loader import get_expected_architecture

    assert get_expected_architecture({}, "no_clock") is None


def _feature_artifacts(**overrides):
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import FeatureArtifacts

    defaults = dict(
        numeric_feature_names=["flow_kg_s"],
        medians={"flow_kg_s": 5.0},
        iqrs={"flow_kg_s": 1.0},
        train_asset_baselines={},
        global_baseline={},
        predicate_thresholds={},
        stage_class_weights=[1.0, 1.0, 1.0],
        foul_pos_weight=1.0,
        clog_pos_weight=1.0,
        actionable_foul_pos_weight=1.0,
        full_feature_names=["flow_kg_s"],
        no_clock_feature_names=["flow_kg_s"],
    )
    defaults.update(overrides)
    return FeatureArtifacts(**defaults)


def test_validate_feature_artifacts_rejects_missing_medians():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.artifact_validation import validate_feature_artifacts

    artifacts = _feature_artifacts(medians={})
    with pytest.raises(ValueError, match="Missing medians"):
        validate_feature_artifacts(artifacts, "no_clock", ["flow_kg_s"])


def test_validate_feature_artifacts_accepts_consistent_bundle():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.artifact_validation import validate_feature_artifacts

    report = validate_feature_artifacts(_feature_artifacts(), "no_clock", ["flow_kg_s"])
    assert report["ok"] is True


def test_validate_policy_artifact_rejects_incomplete_policy():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.artifact_validation import validate_policy_artifact

    with pytest.raises(ValueError, match="incomplete"):
        validate_policy_artifact({"clog_prob_thr": 0.5}, "no_clock")


def test_validate_checkpoint_compatibility_rejects_wrong_feature_count():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.model_arch import (
        PredictiveTCN,
        validate_checkpoint_compatibility,
    )

    model = PredictiveTCN(n_features=76, channels=64, dilations=(1, 2, 4, 8, 16), dropout=0.15)
    wrong_model = PredictiveTCN(n_features=50, channels=64, dilations=(1, 2, 4, 8, 16), dropout=0.15)
    with pytest.raises(ValueError, match="compatibility check failed"):
        validate_checkpoint_compatibility(model, wrong_model.state_dict())


def test_validate_checkpoint_compatibility_accepts_matching_state_dict():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.model_arch import (
        PredictiveTCN,
        validate_checkpoint_compatibility,
    )

    model = PredictiveTCN(n_features=76, channels=64, dilations=(1, 2, 4, 8, 16), dropout=0.15)
    report = validate_checkpoint_compatibility(model, model.state_dict())
    assert report["ok"] is True


def test_validate_checkpoint_compatibility_rejects_architecture_contract_mismatch():
    from app.plugins.ml46_dairy_fouling_clog_detection._vendor.model_arch import (
        PredictiveTCN,
        validate_checkpoint_compatibility,
    )

    model = PredictiveTCN(n_features=76, channels=64, dilations=(1, 2, 4, 8, 16), dropout=0.15)
    architecture_contract = {"architecture": {"channels": 32}}  # wrong channels on purpose
    with pytest.raises(ValueError, match="Architecture contract mismatch"):
        validate_checkpoint_compatibility(model, model.state_dict(), architecture_contract=architecture_contract)


def test_stats_uses_dynamic_dataset_and_test_metrics():
    """stats() must reflect whatever checkpoint/manifest is actually loaded, not a stale literal."""
    from app.plugins.ml46_dairy_fouling_clog_detection.plugin import Ml46DairyFoulingClogDetectionPlugin

    plugin = Ml46DairyFoulingClogDetectionPlugin()
    plugin._feature_artifacts = _feature_artifacts(
        dataset_fingerprint={"telemetry": {"n_assets": 100, "rows": 2288800}},
        created_from_train_assets=[f"asset_{i:02d}" for i in range(60)],
    )
    plugin._manifest = {"summary": {"test_window_metrics": {"stage_accuracy": 0.9775}}}
    stats = plugin.stats()
    assert stats.metrics["n_total_assets"] == 100
    assert stats.metrics["n_telemetry_rows_total"] == 2288800
    assert stats.metrics["n_train_assets"] == 60
    assert stats.metrics["stage_accuracy"] == 0.9775
    assert "100" in stats.description


def test_stats_handles_missing_manifest_gracefully():
    """No load() yet (or an artifact bundle without the new fields) must not crash stats()."""
    from app.plugins.ml46_dairy_fouling_clog_detection.plugin import Ml46DairyFoulingClogDetectionPlugin

    plugin = Ml46DairyFoulingClogDetectionPlugin()
    stats = plugin.stats()
    assert stats.metrics["n_total_assets"] is None
    assert stats.metrics["stage_accuracy"] is None


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert isinstance(body["stage_accuracy"], float)
    assert isinstance(body["n_windows"], int)
