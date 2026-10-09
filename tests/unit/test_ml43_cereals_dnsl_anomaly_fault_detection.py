import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import SENSOR_COLUMNS, SEQ_LENGTH, SOLAPAMIENTO_BETA

PREFIX = "/models/ml43-cereals-dnsl-anomaly-fault-detection"

INLINE_PAYLOAD = {
    "mode": "inline",
    "temp_zona1": 180.0,
    "temp_zona2": 178.0,
    "temp_zona3": 182.0,
    "temp_salida_gases": 95.0,
    "presion_camara": 2.0,
    "presion_ventilacion": 10.0,
    "potencia_kw": 45.0,
    "flujo_gas": 12.0,
    "humedad_relativa": 40.0,
    "temp_ambiente": 22.0,
    "setpoint_temp": 180.0,
    "posicion_valvula": 55.0,
    "velocidad_ventilador": 900.0,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml43-cereals-dnsl-anomaly-fault-detection"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "ml43-cereals-dnsl-anomaly-fault-detection"


def test_stats_without_mlflow_run_id_exposes_only_the_unified_metric_keys():
    """Modelo 43-45 audit, metrics unification: the base/served-model metrics dict must
    contain exactly the unified key set (no legacy anomaly_* keys), and outputs must no
    longer list the removed xai_result/xai_error fields (replaced by Estado del sistema/
    Subsistemas con alteraciones/Variables con alteraciones/Acciones correctivas
    recomendadas)."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

    resp = Ml43CerealsDnslAnomalyFaultDetectionPlugin().stats()

    assert set(resp.metrics.keys()) == {
        "accuracy", "fallo_auc", "fallo_precision", "fallo_recall",
        "macro_f1", "macro_recall", "decision_threshold",
    }
    output_names = {o.name for o in resp.outputs}
    assert "xai_result" not in output_names
    assert "xai_error" not in output_names
    assert "Estado del sistema" in output_names
    assert "Acciones correctivas recomendadas" in output_names


def test_avg_latency_ms_is_computed_from_recorded_predictions():
    """Fase 4 (modelo 43-44 audit): perf_counter() was measured but only logged —
    stats() hardcoded avg_latency_ms=None. Tests _record()/stats() directly (no real
    model artifacts needed) since predict_batch just forwards its measured latency
    into _record(), already exercised end-to-end by test_predict_batch above."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

    plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
    assert plugin.stats().runtime_stats.avg_latency_ms is None  # no predictions yet

    plugin._record(120.0)
    plugin._record(80.0)

    stats = plugin.stats()
    assert stats.runtime_stats.total_predictions == 2
    assert stats.runtime_stats.avg_latency_ms == 100.0


def test_predict_inline_not_supported(client):
    # ml43-cereals-dnsl-anomaly-fault-detection requires a real 180-row window: inline (single-snapshot)
    # prediction was retired. PredictRequest no longer has an "inline" mode, so
    # this body fails schema validation before it ever reaches the plugin.
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cereal_cycle.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml43-cereals-dnsl-anomaly-fault-detection"
    assert len(body["predictions"]) == 1


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/cereal_train.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 200
    body = resp.json()
    assert "detail" in body
    assert "macro_f1" in body


# ── Fase 1 (modelo 43-44 audit): train/val/test split + checkpoint selection ──
#
# The tests above use the router's FakePlugin double (see conftest.py's FAKE_FACTORIES),
# which never runs real training code. The regression test below instantiates the real
# Ml43CerealsDnslAnomalyFaultDetectionPlugin directly and calls train() on synthetic data, bypassing HTTP/
# FakePlugin entirely — same pattern as test_ml10_dairy_disease_vector_detection_unit.py's direct-plugin tests.

def _build_synthetic_cereal_csv(n_windows: int, corrupt_test_fold: bool) -> pd.DataFrame:
    """Builds a single-cycle synthetic sensor CSV yielding exactly ``n_windows`` windows
    of SEQ_LENGTH rows each, with the same 50% overlap (SOLAPAMIENTO_BETA) the real
    windowing code uses — see _vendor/preprocess.py::create_sequences, step =
    seq_length * (1 - solapamiento_beta).

    Rows [0, n_rows/2) are labeled NORMAL, rows [n_rows/2, n_rows) are labeled as a
    fault ("VALVULA_OBSTRUIDA", not in NORMAL_TOKENS).

    When corrupt_test_fold=True, the tail rows are overwritten with unrelated random
    sensor values and relabeled NORMAL — but ONLY from the first row that is *exclusive*
    to the window at the 85th-percentile boundary onward (i.e. never touching any row
    shared with an earlier window). Because windows overlap by (seq_length - step) rows,
    a window's first `step` rows are shared with the previous window and its last
    `seq_length - step` rows are exclusive to it. Starting corruption at
    `window(ceil(n_windows*0.85)).start + step` guarantees every window strictly before
    the 85th percentile (i.e. every window a correct 70/15/15 split would ever use for
    train or validation) is completely untouched by the corruption, while the corrupted
    region still falls entirely inside the "last 20%" that the CURRENT (buggy) 80/20
    split reuses for checkpoint selection.
    """
    step = int(SEQ_LENGTH * (1 - SOLAPAMIENTO_BETA))
    n_rows = SEQ_LENGTH + step * (n_windows - 1)

    rng = np.random.default_rng(1234)
    values = {
        col: 20.0 + 3.0 * i + rng.normal(0, 0.1, n_rows)
        for i, col in enumerate(SENSOR_COLUMNS)
    }

    half = n_rows // 2
    fault_name = ["NORMAL"] * half + ["VALVULA_OBSTRUIDA"] * (n_rows - half)
    for col in SENSOR_COLUMNS:
        values[col][half:] += 50.0  # clearly separable "fault" signal

    if corrupt_test_fold:
        test_start_window = int(np.ceil(n_windows * 0.85))
        corrupt_row_start = test_start_window * step + step
        assert corrupt_row_start < n_rows, "n_windows too small for a clean corruption region"
        n_corrupt = n_rows - corrupt_row_start
        for col in SENSOR_COLUMNS:
            values[col][corrupt_row_start:] = rng.normal(500.0, 5.0, n_corrupt)
        for i in range(corrupt_row_start, n_rows):
            fault_name[i] = "NORMAL"

    df = pd.DataFrame(values)
    df["cycle_id"] = "cycle_1"
    df["timestamp"] = pd.date_range("2026-01-01", periods=n_rows, freq="min")
    df["fault_name"] = fault_name
    return df


class TestTrainCheckpointSelectionUsesOnlyValidation:
    """Regression test for the modelo 43-44 audit (Fase 1): the checkpoint selected
    during training must never be influenced by the held-out test set.

    Method: train twice with train+validation data that is byte-identical between
    both runs, differing ONLY in the tail rows that fall exclusively inside the
    held-out test fold (see _build_synthetic_cereal_csv). With a correct train/val/
    test separation, the checkpoint-selection metric (val_f1) must come out
    identical in both runs, since checkpoint selection never looks at that region.
    Before the fix, val_f1 was actually computed on that same region (the old 80/20
    "test" split doubled as the checkpoint-selection set), so corrupting it changes
    val_f1 — proving the leak.
    """

    N_WINDOWS = 20  # → n_train=14, n_val=3, n_test=3 with the 70/15/15 split

    def _train_once(self, tmp_path, corrupt_test_fold: bool):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        df = _build_synthetic_cereal_csv(self.N_WINDOWS, corrupt_test_fold)
        csv_path = tmp_path / f"cereal_train_{'corrupt' if corrupt_test_fold else 'clean'}.csv"
        df.to_csv(csv_path, index=False)

        np.random.seed(0)
        torch.manual_seed(0)
        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.upload_artifacts_to_mlflow",
            return_value="fake-run-id",
        ):
            return plugin.train(data_path=str(csv_path), mlflow_run_id="")

    def test_checkpoint_selection_unaffected_by_test_fold_corruption(self, tmp_path):
        resp_clean = self._train_once(tmp_path, corrupt_test_fold=False)
        resp_corrupt = self._train_once(tmp_path, corrupt_test_fold=True)

        # val_f1 (the metric checkpoint selection actually optimizes on) is an internal
        # training-time value, not part of the public TrainResponse contract (modelo 43-45
        # audit, metrics unification) — decision_threshold is likewise calibrated purely
        # from the VALIDATION split (101-point grid search), so it's an equally valid,
        # still-public stand-in to confirm checkpoint selection is unaffected by test-fold
        # corruption.
        assert resp_clean.decision_threshold == resp_corrupt.decision_threshold, (
            "El umbral calibrado en validación (decision_threshold) cambió al corromper "
            "únicamente las ventanas del fold de test — el test set está contaminando "
            "la selección del checkpoint (fuga de datos). decision_threshold clean="
            f"{resp_clean.decision_threshold} corrupt={resp_corrupt.decision_threshold}"
        )

    def test_reports_real_macro_metrics_on_test_split(self, tmp_path):
        """Fase 2 (modelo 43-44 audit): macro_f1/macro_recall must be computed
        from the real held-out test split — not hardcoded, not copied from val/train."""
        resp = self._train_once(tmp_path, corrupt_test_fold=False)

        for field in ("macro_f1", "macro_recall"):
            value = getattr(resp, field)
            assert isinstance(value, float), f"{field} debe ser un float, no {type(value)}"
            assert 0.0 <= value <= 1.0, f"{field}={value} fuera de rango [0, 1]"

        # Not the feedback's unsourced reference values — those must never be hardcoded.
        assert resp.macro_f1 != 0.9498
        assert resp.macro_recall != 0.9585

    def test_returns_the_mlflow_run_id_artifacts_were_uploaded_to(self, tmp_path):
        """Fase 3, Bug A (modelo 43-44 audit): TrainResponse must return the real
        run_id used, so the platform can persist it even if its own pre-created run
        failed and the backend had to start a new one — see mlflow_utils.py."""
        resp = self._train_once(tmp_path, corrupt_test_fold=False)
        assert resp.mlflow_run_id == "fake-run-id"

    def test_decision_threshold_is_returned_and_logged_to_mlflow(self, tmp_path):
        """Regression: previously /stats always reported the same hardcoded
        DECISION_THRESHOLD=0.41 for every retrain, because train() never computed nor
        logged a per-run threshold to MLflow — see plugin.py train()/stats(). This CSV
        (N_WINDOWS=20, single cycle) has too few distinct cycle_id values (1) for the
        real repo's split-by-cycle-id, so split_train_val_test_by_id() falls back to a
        row-count split — the point of this test is only that decision_threshold is
        actually returned AND present in the metrics dict handed to MLflow, so
        stats(mlflow_run_id=...)'s existing metric-flattening loop has something to
        surface, regardless of which split path produced it."""
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        df = _build_synthetic_cereal_csv(self.N_WINDOWS, corrupt_test_fold=False)
        csv_path = tmp_path / "cereal_train_threshold.csv"
        df.to_csv(csv_path, index=False)

        np.random.seed(0)
        torch.manual_seed(0)
        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.upload_artifacts_to_mlflow",
            return_value="fake-run-id",
        ) as mock_upload:
            resp = plugin.train(data_path=str(csv_path), mlflow_run_id="")

        assert 0.0 <= resp.decision_threshold <= 1.0
        logged_metrics = mock_upload.call_args.kwargs["metrics"]
        assert logged_metrics["decision_threshold"] == resp.decision_threshold

    def test_test_accuracy_auc_precision_recall_are_computed_and_logged(self, tmp_path):
        """Regression: train() computed macro_f1/macro_recall but never accuracy/
        fallo_auc/fallo_precision/fallo_recall — TEST_METRICS's own keys had no real
        per-run counterpart to overwrite from at all, so stats() could never surface
        real values under those tiles no matter what else got fixed."""
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        df = _build_synthetic_cereal_csv(self.N_WINDOWS, corrupt_test_fold=False)
        csv_path = tmp_path / "cereal_train_test_metrics.csv"
        df.to_csv(csv_path, index=False)

        np.random.seed(0)
        torch.manual_seed(0)
        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.upload_artifacts_to_mlflow",
            return_value="fake-run-id",
        ) as mock_upload:
            resp = plugin.train(data_path=str(csv_path), mlflow_run_id="")

        for field in ("accuracy", "fallo_precision", "fallo_recall"):
            value = getattr(resp, field)
            assert 0.0 <= value <= 1.0, f"{field}={value} out of [0, 1]"
        # fallo_auc may legitimately be NaN when the test split has only one class —
        # not asserted here beyond being present as a float field on the response.

        logged_metrics = mock_upload.call_args.kwargs["metrics"]
        assert logged_metrics["accuracy"] == resp.accuracy
        assert logged_metrics["fallo_precision"] == resp.fallo_precision
        assert logged_metrics["fallo_recall"] == resp.fallo_recall

    def test_decision_threshold_calibration_on_a_mixed_class_validation_fold(self, tmp_path):
        """The single-cycle synthetic CSV above falls back to a row-count split (too
        few cycle_ids) — see test above. This builds 10 minimal one-window cycles
        (SEQ_LENGTH rows each) so the real split_train_val_test_by_id() takes its
        cycle-id path: with TRAIN_EXTERNAL_VAL_PCT=13.3/TEST_PCT=20.0 and 10 cycles,
        train=cycles[0:6], val=cycles[6:8], test=cycles[8:10]. Labeling cycle 6 FAULT
        and cycle 7 NORMAL makes the val fold genuinely mixed-class, actually
        exercising optimize_threshold_by_f1's 101-point grid search (not a degenerate
        single-class sweep)."""
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        cycle_labels = ["NORMAL"] * 6 + ["VALVULA_OBSTRUIDA", "NORMAL"] + ["NORMAL"] * 2

        rng = np.random.default_rng(99)
        frames = []
        for cycle_idx, label in enumerate(cycle_labels):
            values = {
                col: 20.0 + 3.0 * i + rng.normal(0, 0.1, SEQ_LENGTH)
                for i, col in enumerate(SENSOR_COLUMNS)
            }
            if label != "NORMAL":
                for col in SENSOR_COLUMNS:
                    values[col] += 50.0
            frame = pd.DataFrame(values)
            frame["cycle_id"] = f"cycle_{cycle_idx}"
            frame["timestamp"] = pd.date_range("2026-01-01", periods=SEQ_LENGTH, freq="min")
            frame["fault_name"] = label
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)

        csv_path = tmp_path / "cereal_train_mixed_val.csv"
        df.to_csv(csv_path, index=False)

        np.random.seed(0)
        torch.manual_seed(0)
        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.upload_artifacts_to_mlflow",
            return_value="fake-run-id",
        ) as mock_upload:
            resp = plugin.train(data_path=str(csv_path), mlflow_run_id="")

        assert 0.0 <= resp.decision_threshold <= 1.0
        assert resp.n_windows_val == 2  # cycles 6-7, one window each
        logged_metrics = mock_upload.call_args.kwargs["metrics"]
        assert logged_metrics["decision_threshold"] == resp.decision_threshold

    def test_train_aborts_with_a_clear_error_when_the_model_artifact_is_corrupted(self, tmp_path):
        """Feedback 3 (modelo 43-44, 09/09/2026), point 8: previously investigated but left
        unimplemented — torch.save has no return value to check, so a write interrupted
        partway (disk full, pod killed) can leave a 0-byte file that used to get uploaded
        to MLflow as-is, reported as 'Entrenamiento completado'. Simulates that by making
        torch.save write an empty file, and asserts training aborts before ever attempting
        the MLflow upload."""
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        df = _build_synthetic_cereal_csv(self.N_WINDOWS, corrupt_test_fold=False)
        csv_path = tmp_path / "cereal_train_corrupt_artifact.csv"
        df.to_csv(csv_path, index=False)

        def _write_empty_file(_checkpoint, path, *_a, **_kw):
            Path(path).write_bytes(b"")

        np.random.seed(0)
        torch.manual_seed(0)
        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        with patch("app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.torch.save", side_effect=_write_empty_file), \
                patch("app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.upload_artifacts_to_mlflow") as mock_upload:
            with pytest.raises(ValueError, match="best_dnf_model.pt"):
                plugin.train(data_path=str(csv_path), mlflow_run_id="")
            mock_upload.assert_not_called()


class TestValidateSavedArtifact:
    """Feedback 3 (modelo 43-44 audit), point 8: unit coverage for the helper itself,
    independent of a full training run (see also
    TestTrainCheckpointSelectionUsesOnlyValidation.test_train_aborts_with_a_clear_error_
    when_the_model_artifact_is_corrupted for the end-to-end wiring)."""

    def test_raises_on_missing_file(self, tmp_path):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import _validate_saved_artifact

        with pytest.raises(ValueError, match="vacío"):
            _validate_saved_artifact(tmp_path / "missing.pt", loader=lambda p: None)

    def test_raises_on_zero_byte_file(self, tmp_path):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import _validate_saved_artifact

        path = tmp_path / "empty.pkl"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="vacío"):
            _validate_saved_artifact(path, loader=lambda p: pickle.load(open(p, "rb")))

    def test_raises_on_a_nonempty_but_corrupt_file(self, tmp_path):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import _validate_saved_artifact

        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not actually a pickle stream")
        with pytest.raises(ValueError, match="corrupto"):
            _validate_saved_artifact(path, loader=lambda p: pickle.load(open(p, "rb")))

    def test_passes_for_a_real_readable_artifact(self, tmp_path):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import _validate_saved_artifact

        path = tmp_path / "ok.pkl"
        with open(path, "wb") as f:
            pickle.dump({"a": 1}, f)
        _validate_saved_artifact(path, loader=lambda p: pickle.load(open(p, "rb")))  # does not raise


# ── Feedback 3 (modelo 43-44, 09/09/2026): labels + XAI-for-every-window ──────────────

class _FakeExplainer:
    """Stand-in for DNFLExplainer — records how many windows it was asked to explain."""

    def __init__(self, final_report: dict):
        self._final_report = final_report
        self.calls = 0

    def explain(self, *, x_window, s_stats, background_windows, anomaly_threshold):
        self.calls += 1
        return {"final_report": self._final_report}


def _dummy_batch_inputs(n_windows: int):
    x_arr = np.zeros((n_windows, SEQ_LENGTH, len(SENSOR_COLUMNS)), dtype=np.float32)
    stats_scaled = np.zeros((n_windows, 1), dtype=np.float32)
    return x_arr, stats_scaled


def test_format_batch_predictions_uses_spanish_labels():
    """Point 1: predicted_anomaly_label must be 'Fallo'/'No Fallo' — mirrors
    ml45_cereals_dnsl_critical_point_detection/postprocessing.py, which already returns
    these Spanish labels for the same binary decision."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection import postprocessing

    x_arr, stats_scaled = _dummy_batch_inputs(2)
    scores = np.array([0.1, 0.9], dtype=np.float32)  # below / above threshold=0.5

    predictions = postprocessing.format_batch_predictions(
        x_arr, scores, cycle_ids=None, threshold=0.5,
        explainer=None, xai_background=None,
        X_scaled=x_arr, stats_scaled=stats_scaled,
    )
    assert predictions[0]["predicted_anomaly_label"] == "No Fallo"
    assert predictions[1]["predicted_anomaly_label"] == "Fallo"


def test_format_batch_predictions_runs_xai_for_every_window_not_only_failures():
    """Point 3: XAI must run for 'No Fallo' rows too. Previously gated behind
    `if is_anomaly`, so every non-anomalous row's xai_result was None by construction —
    not because the explanation failed, but because run_xai was never even called."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection import postprocessing

    final_report = {
        "Estado_del_sistema": {"Estado_interpretativo": "Normal"},
        "Mensaje_operativo": ["No se requieren acciones correctivas."],
    }
    explainer = _FakeExplainer(final_report)
    x_arr, stats_scaled = _dummy_batch_inputs(2)
    scores = np.array([0.1, 0.9], dtype=np.float32)  # No Fallo / Fallo

    predictions = postprocessing.format_batch_predictions(
        x_arr, scores, cycle_ids=None, threshold=0.5,
        explainer=explainer, xai_background=np.zeros((4, SEQ_LENGTH, len(SENSOR_COLUMNS)), dtype=np.float32),
        X_scaled=x_arr, stats_scaled=stats_scaled,
    )
    assert explainer.calls == 2, "run_xai must run for both the No Fallo and the Fallo window"
    assert predictions[0]["xai_result"] == final_report
    assert predictions[1]["xai_result"] == final_report
    assert predictions[0]["xai_error"] is None
    assert predictions[1]["xai_error"] is None


# ── Feedback 3 (modelo 43-44, 10/09/2026): _build_final_report's real branching ───────
#
# The tests above (test_format_batch_predictions_runs_xai_for_every_window_not_only_
# failures) prove run_xai() is called for every window, but do so with a fake explainer
# that returns a fixed final_report — they never exercise DNFLExplainer's own
# _build_final_report(), which is what actually decides, per Estado_interpretativo,
# whether the row gets Mensaje_operativo or Bloques_principales/Variables_clave/
# Acciones_sugeridas. _build_final_report doesn't read `self` at all (pure dict-in,
# dict-out), so it's called directly here without constructing a real DNFLExplainer
# (which would need a trained model) — this is a plain unit test, no model/training runs.

def _action_report(state_label: str, *, blocks=None, actions=None) -> dict:
    return {
        "prediction": {"ensemble": {"probability": 0.05}, "decision_threshold": 0.5},
        "state_label": state_label,
        "ranked_blocks": blocks or [],
        "suggested_actions": actions or [],
    }


def test_build_final_report_normal_has_only_mensaje_operativo():
    """No Fallo + Normal: Estado del sistema = Normal, Mensaje_operativo present,
    Bloques_principales/Variables_clave/Acciones_sugeridas absent entirely (not empty
    lists) — see plataforma's deriveModelo43XaiColumns, which relies on this absence to
    show 'No aplica'."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.xai.explainer import DNFLExplainer

    report = _action_report(
        "normal",
        blocks=[{"block": "termico", "support_variables": ["temp_zona1"]}],
        actions=[{"action": "should be ignored for Normal"}],
    )
    final_report = DNFLExplainer._build_final_report(None, action_report=report)

    assert final_report["Estado_del_sistema"]["Estado_interpretativo"] == "Normal"
    assert final_report["Mensaje_operativo"] == [
        "No se requieren acciones correctivas.",
        "Mantener monitorización ordinaria.",
    ]
    assert "Bloques_principales" not in final_report
    assert "Variables_clave" not in final_report
    assert "Acciones_sugeridas" not in final_report


def test_build_final_report_normal_with_signals_has_blocks_variables_actions():
    """No Fallo + Normal con señales: unlike plain Normal, this state does NOT take the
    Mensaje_operativo shortcut — it gets the same Bloques_principales/Variables_clave/
    Acciones_sugeridas treatment as the anomaly states."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.xai.explainer import DNFLExplainer

    report = _action_report(
        "normal_with_signals",
        blocks=[{"block": "ventilacion_presion", "support_variables": ["presion_camara", "presion_ventilacion"]}],
        actions=[{"action": "Inspeccionar ventilación y estado del actuador o ventilador."}],
    )
    final_report = DNFLExplainer._build_final_report(None, action_report=report)

    assert final_report["Estado_del_sistema"]["Estado_interpretativo"] == "Normal con señales"
    assert "Mensaje_operativo" not in final_report
    assert final_report["Bloques_principales"] == ["ventilacion_presion"]
    assert final_report["Variables_clave"] == ["presion_camara", "presion_ventilacion"]
    assert final_report["Acciones_sugeridas"] == ["Inspeccionar ventilación y estado del actuador o ventilador."]


def test_build_final_report_unconfirmed_alert_keeps_blocks_variables_actions():
    """Fallo + Alerta no confirmada: regression guard — must keep the same
    Bloques_principales/Variables_clave/Acciones_sugeridas shape it already had."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.xai.explainer import DNFLExplainer

    report = _action_report(
        "unconfirmed_alert",
        blocks=[{"block": "ventilacion_presion", "support_variables": ["presion_ventilacion"]}],
        actions=[{"action": "Revisar filtro de ventilación."}],
    )
    final_report = DNFLExplainer._build_final_report(None, action_report=report)

    assert final_report["Estado_del_sistema"]["Estado_interpretativo"] == "Alerta no confirmada"
    assert "Mensaje_operativo" not in final_report
    assert final_report["Bloques_principales"] == ["ventilacion_presion"]
    assert final_report["Variables_clave"] == ["presion_ventilacion"]
    assert final_report["Acciones_sugeridas"] == ["Revisar filtro de ventilación."]


def test_build_final_report_confirmed_anomaly_keeps_blocks_variables_actions():
    """Fallo + Anomalía confirmada: regression guard — must keep the same
    Bloques_principales/Variables_clave/Acciones_sugeridas shape it already had."""
    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.xai.explainer import DNFLExplainer

    report = _action_report(
        "confirmed_anomaly",
        blocks=[{"block": "termico", "support_variables": ["temp_zona1", "temp_zona2"]},
                {"block": "presion", "support_variables": ["presion_camara"]}],
        actions=[{"action": "Detener el horno."}, {"action": "Inspeccionar sensores de temperatura."}],
    )
    final_report = DNFLExplainer._build_final_report(None, action_report=report)

    assert final_report["Estado_del_sistema"]["Estado_interpretativo"] == "Anomalía confirmada"
    assert "Mensaje_operativo" not in final_report
    assert final_report["Bloques_principales"] == ["termico", "presion"]
    assert final_report["Variables_clave"] == ["temp_zona1", "temp_zona2", "presion_camara"]
    assert final_report["Acciones_sugeridas"] == ["Detener el horno.", "Inspeccionar sensores de temperatura."]


class TestPredictUsesCalibratedThresholdFromTrainedRun:
    """Regression: predict_batch used to classify every window as Fallo/No Fallo with
    the served model's fixed self._threshold (0.41), even when a genuinely different,
    user-trained model (different weights) was loaded via mlflow_run_id — the response's
    decision_threshold stayed 0.41 no matter which run was used. See
    plugin.py::_load_model_for_predict / mlflow_utils.py::get_calibrated_threshold."""

    def test_load_model_for_predict_uses_the_runs_calibrated_threshold(self):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        fake_bundle = (object(), {}, object(), object(), None, object(), "/tmp/fake")
        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.download_user_model_from_mlflow",
            return_value=fake_bundle,
        ), patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.get_calibrated_threshold",
            return_value=0.7676,
        ) as mock_get_threshold:
            ctx = plugin._load_model_for_predict("some-run-id")

        assert ctx["threshold"] == 0.7676
        mock_get_threshold.assert_called_once_with("some-run-id", default=plugin._threshold)

    def test_served_ctx_uses_the_instance_threshold(self):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

        plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
        assert plugin._served_ctx()["threshold"] == plugin._threshold


def test_stats_overwrites_legacy_anomaly_keys_with_the_real_run_values():
    """Regression: stats(mlflow_run_id=...) used to flatten mlflow_metrics onto
    base.metrics under train()'s OLD test_*-prefixed key names, which never shared a
    name with TEST_METRICS's own keys (anomaly_acc, anomaly_f1, ...) — only
    decision_threshold happened to match by coincidence, so every retrain's /stats kept
    showing the served model's fixed reference numbers under the other tiles. Now
    train() logs metrics under the exact unified key names TEST_METRICS itself uses
    (modelo 43-45 audit), so stats() can overwrite base.metrics directly, key by key,
    with no alias map needed."""
    from unittest.mock import patch

    from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import Ml43CerealsDnslAnomalyFaultDetectionPlugin

    plugin = Ml43CerealsDnslAnomalyFaultDetectionPlugin()
    with patch(
        "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin.BaseMLflowTracker"
    ) as mock_tracker_cls:
        mock_tracker_cls.return_value.get_metrics.return_value = {
            "accuracy": 0.91, "fallo_auc": 0.93,
            "fallo_precision": 0.85, "fallo_recall": 0.79,
            "macro_f1": 0.82, "macro_recall": 0.80, "decision_threshold": 0.55,
            "n_windows_train": 700, "n_windows_val": 150, "n_windows_test": 150,
        }
        mock_tracker_cls.return_value.get_params.return_value = {}
        resp = plugin.stats(mlflow_run_id="some-run-id")

    assert resp.metrics["accuracy"] == 0.91
    assert resp.metrics["fallo_auc"] == 0.93
    assert resp.metrics["fallo_precision"] == 0.85
    assert resp.metrics["fallo_recall"] == 0.79
    assert resp.metrics["macro_f1"] == 0.82
    assert resp.metrics["macro_recall"] == 0.80
    assert resp.metrics["decision_threshold"] == 0.55
    assert resp.metrics["n_windows_train"] == 700
    assert resp.metrics["n_windows_val"] == 150
    assert resp.metrics["n_windows_test"] == 150
    # No leftover legacy anomaly_* keys, and no unfiltered debug dump of the whole
    # MLflow metrics/params payload.
    assert "anomaly_acc" not in resp.metrics
    assert "mlflow" not in resp.metrics


class TestGetCalibratedThreshold:
    """Unit coverage for the mlflow_utils helper itself, independent of the plugin
    wiring tested above."""

    def test_returns_the_logged_metric_when_present(self):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils import get_calibrated_threshold

        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils.BaseMLflowTracker"
        ) as mock_tracker_cls:
            mock_tracker_cls.return_value.get_metrics.return_value = {"decision_threshold": 0.63, "test_f1": 0.8}
            assert get_calibrated_threshold("run-1", default=0.41) == 0.63

    def test_falls_back_to_default_when_metric_missing(self):
        """Runs trained before this metric existed never logged decision_threshold."""
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils import get_calibrated_threshold

        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils.BaseMLflowTracker"
        ) as mock_tracker_cls:
            mock_tracker_cls.return_value.get_metrics.return_value = {"test_f1": 0.8}
            assert get_calibrated_threshold("run-1", default=0.41) == 0.41

    def test_falls_back_to_default_on_mlflow_error(self):
        from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils import get_calibrated_threshold

        with patch(
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils.BaseMLflowTracker"
        ) as mock_tracker_cls:
            mock_tracker_cls.return_value.get_metrics.side_effect = RuntimeError("MLflow unreachable")
            assert get_calibrated_threshold("run-1", default=0.41) == 0.41
