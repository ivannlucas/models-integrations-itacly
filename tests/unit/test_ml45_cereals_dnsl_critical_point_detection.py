"""Endpoint tests for ``ml45-cereals-dnsl-critical-point-detection``."""
from app.domain.services.exceptions import InsufficientWindowHistoryError

PREFIX = "/models/ml45-cereals-dnsl-critical-point-detection"
MODEL_ID = "ml45-cereals-dnsl-critical-point-detection"

_SENSOR_ROW = {
    "plenum_temp": [70.0] * 240,
    "exhaust_air_temp": [40.0] * 240,
    "exhaust_air_humidity": [60.0] * 240,
    "static_pressure": [10.0] * 240,
    "burner_power": [55.0] * 240,
    "fan_speed": [1200.0] * 240,
    "discharge_frequency": [30.0] * 240,
    "grain_moisture_in": [20.0] * 240,
    "ambient_temp": [15.0] * 240,
    "ambient_humidity": [50.0] * 240,
    "setpoint_temp": [75.0] * 240,
    "timestamp": [f"2029-01-01 00:{i:02d}:00" if i < 60 else f"2029-01-01 0{i // 60}:{i % 60:02d}:00" for i in range(240)],
    "cycle_id": 1,
}

INLINE_PAYLOAD = {"mode": "inline", **_SENSOR_ROW}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == MODEL_ID


def test_stats_without_mlflow_run_id_exposes_only_the_unified_metric_keys():
    """Modelo 43-45 audit, metrics unification: the base/served-model metrics dict must
    contain exactly the same unified key set as ml43_cereals_dnsl_anomaly_fault_detection's own stats() (no
    legacy selected_threshold/bare fallo_f1-only naming)."""
    from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import (
        Ml45CerealsDnslCriticalPointDetectionPlugin,
    )

    resp = Ml45CerealsDnslCriticalPointDetectionPlugin().stats()

    assert set(resp.metrics.keys()) == {
        "accuracy", "fallo_auc", "fallo_precision", "fallo_recall",
        "macro_f1", "macro_recall", "decision_threshold",
    }


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    # predicted_anomaly_class/Probabilidad de anomalia/Umbral de detección de anomalias/
    # Margen respecto al umbral were dropped from the contract (modelo 43-44-45 audit,
    # point 2): duplicates of, or trivially derivable from, the fields kept below.
    assert body["predicted_anomaly_label"] in ("Fallo", "No Fallo")
    assert "predicted_anomaly_class" not in body
    assert "Probabilidad de anomalia" not in body
    assert "Umbral de detección de anomalias" not in body
    assert "Margen respecto al umbral" not in body
    assert 0.0 <= body["anomaly_probability"] <= 1.0
    assert body["Estado interpretativo"] in ("Normal", "Vigilancia", "Criticidad detectada")


def test_predict_inline_missing_sensor_rejected(client):
    """Missing a required sensor array (and no data_path) must fail Pydantic validation."""
    payload = dict(INLINE_PAYLOAD)
    del payload["plenum_temp"]
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/fake.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_anomaly_label"] == "No Fallo"
    assert "predicted_anomaly_class" not in body["predictions"][0]


def test_predict_batch_insufficient_window_history_maps_to_422(client, fake_plugins):
    """InsufficientWindowHistoryError raised by the plugin must map to HTTP 422."""
    fake_plugins[MODEL_ID].raise_on_batch = InsufficientWindowHistoryError(
        "No se generaron ventanas: se necesitan al menos 240 filas consecutivas."
    )
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/short.csv"})
    assert resp.status_code == 422


def test_train(client):
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": "test-run-id"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert body["n_windows_train"] > 0
    # n_windows_test/mlflow_run_id (modelo 43-44-45 audit, point 6): metrics must come from a
    # held-out split, and the run they were uploaded to must be echoed back — otherwise
    # predict/stats have no way to ever use the retrained model.
    assert body["n_windows_test"] > 0
    assert body["mlflow_run_id"]


def test_train_requires_mlflow_run_id(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv"})
    assert resp.status_code == 422


# ── Feedback (modelo 43-44-45 audit, points 2/3/6): dedup + artifact validation ──────────

class _FakeExplainerReport:
    """Stand-in for DNFLExplainer.explain() — explain_window() only reads these two keys."""

    def __init__(self, anomaly_probability: float, final_report: dict):
        self._result = {
            "prediction": {"anomaly_probability": anomaly_probability},
            "final_report": final_report,
        }

    def explain(self, **_kwargs):
        return self._result


def test_explain_window_drops_duplicate_and_derivable_fields():
    """Point 2: predicted_anomaly_class (dup of predicted_anomaly_label), 'Probabilidad de
    anomalia' (dup of anomaly_probability), 'Umbral de detección de anomalias' (dup of
    decision_threshold) and 'Margen respecto al umbral' (trivially derivable from the two
    kept fields) must not appear in the row explain_window() builds — 'Estado
    interpretativo'/'Evidencia'/'Recomendacion' (real, non-duplicate content) must survive
    untouched."""
    from app.plugins.ml45_cereals_dnsl_critical_point_detection.postprocessing import explain_window

    explainer = _FakeExplainerReport(
        anomaly_probability=0.61,
        final_report={
            "Estado interpretativo": "Criticidad detectada",
            "Evidencia": "PCC térmico-humedad tardío detectado.",
            "Probabilidad de anomalia": 0.61,
            "Umbral de detección de anomalias": 0.73,
            "Margen respecto al umbral": -0.12,
            "Recomendacion": "Detener el secadero y revisar humedad de grano.",
        },
    )
    row = explain_window(explainer, x_window=None, s_stats=None, xai_background="not-none", threshold=0.73)

    assert row["predicted_anomaly_label"] == "No Fallo"  # 0.61 < 0.73
    assert row["anomaly_probability"] == 0.61
    assert row["decision_threshold"] == 0.73
    assert "predicted_anomaly_class" not in row
    assert "Probabilidad de anomalia" not in row
    assert "Umbral de detección de anomalias" not in row
    assert "Margen respecto al umbral" not in row
    assert row["Estado interpretativo"] == "Criticidad detectada"
    assert row["Evidencia"] == "PCC térmico-humedad tardío detectado."
    assert row["Recomendacion"] == "Detener el secadero y revisar humedad de grano."


def _build_synthetic_dryer_csv(cycle_labels, rows_per_cycle=240):
    """Builds a multi-cycle synthetic dryer CSV — each cycle exactly rows_per_cycle
    rows (one window per cycle at SEQUENCE_LENGTH=240), labeled NORMAL or a real fault
    name from config.yaml's fault_types. FAULT cycles get all 11 sensors shifted by
    +50 so the two classes are trivially separable."""
    import numpy as np
    import pandas as pd

    from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import SENSOR_COLUMNS

    rng = np.random.default_rng(7)
    frames = []
    for cycle_idx, label in enumerate(cycle_labels):
        values = {
            col: 20.0 + 3.0 * i + rng.normal(0, 0.1, rows_per_cycle)
            for i, col in enumerate(SENSOR_COLUMNS)
        }
        if label != "NORMAL":
            for col in SENSOR_COLUMNS:
                values[col] += 50.0
        frame = pd.DataFrame(values)
        frame["cycle_id"] = f"cycle_{cycle_idx}"
        frame["timestamp"] = pd.date_range("2029-01-01", periods=rows_per_cycle, freq="min")
        frame["fault_name"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _build_ml45_model_cfg():
    """model: section from a45-dnsl-cereals-deteccion-puntos-criticos/config/config.yaml,
    plus the input/sequence/stats dims train() derives from the data — matches what a
    real served checkpoint's model_cfg would contain."""
    from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
        DEFAULT_THRESHOLD, SENSOR_COLUMNS, SEQUENCE_LENGTH, STATS_CREATION,
    )

    return {
        "lstm": {
            "hidden_size": 64, "num_layers": 1, "dropout": 0.25,
            "bidirectional": False, "embedding_dim": 32,
        },
        "fuzzy": {
            "n_mf": 3, "n_rules": 48, "train_membership_params": True,
            "train_rule_params": True, "temperature": 0.7, "t_norm": "product",
            "normalize_rules": True, "init_alpha": "sparse", "use_log_bias": False,
            "lambda_anomaly": 0.80,
        },
        "input_features": len(SENSOR_COLUMNS),
        "sequence_length": SEQUENCE_LENGTH,
        "n_stats_features": len(STATS_CREATION) * len(SENSOR_COLUMNS),
        "training_kwargs": {"threshold": DEFAULT_THRESHOLD},
    }


class TestTrainFaithfulPipeline:
    """Regression for modelo 43-44-45 audit, Fase 5: train() previously fine-tuned the
    served checkpoint's weights for a fixed 30 epochs and never recalibrated
    decision_threshold — every retrained model's checkpoint kept propagating the base
    model's fixed threshold via model_cfg["training_kwargs"]["threshold"], the exact
    key download_user_model_from_mlflow() reads on predict/stats. Now train() runs the
    real from-scratch pipeline (split-by-cycle-id, DNFLoss, Adam+CosineAnnealingLR,
    warmup+early-stopping, 101-point threshold search) and writes the calibrated
    threshold back into that same key before saving the checkpoint."""

    def _train_once(self, tmp_path, cycle_labels, *, epoch_limit=None):
        from unittest.mock import patch

        import numpy as np
        import torch

        from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import (
            Ml45CerealsDnslCriticalPointDetectionPlugin,
        )

        df = _build_synthetic_dryer_csv(cycle_labels)
        csv_path = tmp_path / "dryer_train.csv"
        df.to_csv(csv_path, index=False)

        plugin = Ml45CerealsDnslCriticalPointDetectionPlugin()
        plugin._model = object()  # only needs to be non-None for _require_loaded()
        plugin._model_cfg = _build_ml45_model_cfg()

        saved_checkpoints = []
        real_torch_save = torch.save

        def _capture_torch_save(obj, path, *a, **kw):
            if isinstance(obj, dict) and "model_cfg" in obj:
                saved_checkpoints.append(obj)
            return real_torch_save(obj, path, *a, **kw)

        np.random.seed(0)
        torch.manual_seed(0)

        with patch(
            "app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin.upload_artifacts_to_mlflow",
            return_value="fake-m45-run-id",
        ), patch(
            "app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin.torch.save",
            side_effect=_capture_torch_save,
        ):
            if epoch_limit is not None:
                with patch(
                    "app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin.TRAIN_NUM_EPOCHS",
                    epoch_limit,
                ):
                    resp = plugin.train(data_path=str(csv_path), mlflow_run_id="")
            else:
                resp = plugin.train(data_path=str(csv_path), mlflow_run_id="")

        return resp, saved_checkpoints

    def test_split_is_by_cycle_id_not_row_count(self, tmp_path):
        """10 cycles -> train=cycles[0:6], val=cycles[6:8], test=cycles[8:10] per
        split_train_val_test_by_id's formula with val_pct=13.3/test_pct=20.0 — one
        window per cycle (240 rows each), so n_train/n_val/n_test mirror cycle counts."""
        cycle_labels = ["NORMAL"] * 10
        resp, _ = self._train_once(tmp_path, cycle_labels, epoch_limit=2)

        assert resp.n_windows_train == 6
        assert resp.n_windows_val == 2
        assert resp.n_windows_test == 2
        assert resp.n_windows_total == 10

    def test_decision_threshold_is_calibrated_and_written_into_the_checkpoint(self, tmp_path):
        """The whole point of this rewrite: model_cfg["training_kwargs"]["threshold"]
        in the SAVED checkpoint must be the newly calibrated value, not the base
        model's original DEFAULT_THRESHOLD — that key is exactly what
        download_user_model_from_mlflow() reads back for predict/stats."""
        cycle_labels = ["NORMAL"] * 6 + ["FILTER_CLOGGED", "NORMAL"] + ["NORMAL"] * 2
        resp, saved_checkpoints = self._train_once(tmp_path, cycle_labels, epoch_limit=2)

        assert 0.0 <= resp.decision_threshold <= 1.0
        assert len(saved_checkpoints) == 1
        saved_threshold = saved_checkpoints[0]["model_cfg"]["training_kwargs"]["threshold"]
        assert saved_threshold == resp.decision_threshold


def test_stats_overwrites_legacy_keys_with_the_real_run_values():
    """Regression: stats(mlflow_run_id=...) used to flatten mlflow_metrics onto
    base.metrics under train()'s OLD test_*-prefixed key names, which never shared a
    name with the base/fixed-model reference keys (accuracy, macro_f1, fallo_auc,
    fallo_precision, fallo_recall, selected_threshold) — so those tiles kept showing
    the same hardcoded numbers for every retrain. Now train() logs metrics under the
    exact unified key names TEST_METRICS itself uses, and stats() no longer needs a
    separate legacy-key alias map — it overwrites base.metrics directly (modelo 43-45
    audit, metrics unification)."""
    from unittest.mock import patch

    from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import (
        Ml45CerealsDnslCriticalPointDetectionPlugin,
    )

    plugin = Ml45CerealsDnslCriticalPointDetectionPlugin()
    with patch(
        "app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin.BaseMLflowTracker"
    ) as mock_tracker_cls:
        mock_tracker_cls.return_value.get_metrics.return_value = {
            "accuracy": 0.8619, "macro_f1": 0.8527,
            "fallo_auc": 0.9284, "fallo_precision": 0.8797, "fallo_recall": 0.7609,
            "macro_recall": 0.83, "decision_threshold": 0.61,
            "n_windows_train": 5336, "n_windows_val": 1064, "n_windows_test": 1600,
        }
        mock_tracker_cls.return_value.get_params.return_value = {}
        resp = plugin.stats(mlflow_run_id="some-run-id")

    assert resp.metrics["accuracy"] == 0.8619
    assert resp.metrics["macro_f1"] == 0.8527
    assert resp.metrics["fallo_auc"] == 0.9284
    assert resp.metrics["fallo_precision"] == 0.8797
    assert resp.metrics["fallo_recall"] == 0.7609
    assert resp.metrics["macro_recall"] == 0.83
    assert resp.metrics["decision_threshold"] == 0.61
    assert resp.metrics["n_windows_train"] == 5336
    assert resp.metrics["n_windows_val"] == 1064
    assert resp.metrics["n_windows_test"] == 1600
    assert "selected_threshold" not in resp.metrics
    assert "mlflow" not in resp.metrics


class TestValidateSavedArtifact:
    """Point 6 (0-byte artifacts): unit coverage for the helper itself, mirroring
    ml43_cereals_dnsl_anomaly_fault_detection's plugin.py::_validate_saved_artifact tests — same helper, copied
    into this plugin's train() to catch a 0-byte/corrupt save before it ever reaches
    MLflow (and, from there, the platform's model listing)."""

    def test_raises_on_missing_file(self, tmp_path):
        from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import _validate_saved_artifact

        import pytest
        with pytest.raises(ValueError, match="vacío"):
            _validate_saved_artifact(tmp_path / "missing.pt", loader=lambda p: None)

    def test_raises_on_zero_byte_file(self, tmp_path):
        from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import _validate_saved_artifact

        import pytest
        path = tmp_path / "empty.pkl"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="vacío"):
            _validate_saved_artifact(path, loader=lambda p: open(p, "rb").read())

    def test_raises_on_a_nonempty_but_corrupt_file(self, tmp_path):
        from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import _validate_saved_artifact

        import pytest
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not actually a pickle stream")

        def _loader(p):
            raise ValueError("bad magic number")

        with pytest.raises(ValueError, match="corrupto"):
            _validate_saved_artifact(path, loader=_loader)

    def test_passes_for_a_real_readable_artifact(self, tmp_path):
        from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import _validate_saved_artifact

        path = tmp_path / "ok.bin"
        path.write_bytes(b"not empty")
        _validate_saved_artifact(path, loader=lambda p: p.read_bytes())  # does not raise
