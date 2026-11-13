"""ml46 with real-world CSVs: empty cells, bad timestamps, no valid window.

Reproduced with the real dataset (head of s3://…/datasets/modelo-46/train.csv) before the fix:
* train with an empty maintenance_active on production rows -> "cannot convert float NaN to
  integer" (evaluation.py, after fine-tuning);
* train with maintenance_type/fault_type/maintenance_planned empty for a whole maintenance
  period -> IndexError (mode() of an all-empty slice, data.py);
* predict with no valid window (phase in Spanish, < 120 rows) -> KeyError 'asset_id' (HTTP 500);
* predict with a bad timestamp or a missing column -> ValueError (HTTP 500, English message).
On the real data without blanks the predictions are identical before and after (max diff 0.0).
"""
# pylint: disable=protected-access,redefined-outer-name  # private helpers; pytest fixtures
import dataclasses

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import (
    InsufficientTelemetryHistoryError,
    InvalidTelemetryInputError,
)
from app.plugins.ml46_dairy_fouling_clog_detection import plugin as ml46
from app.plugins.ml46_dairy_fouling_clog_detection import preprocessing

# Medians of the production rows of the real dataset (asset_01).
_TYPICAL = {
    "flow_kg_s": 5.229, "pressure_in_kPa": 300.137, "pressure_out_kPa": 149.44,
    "dP_kPa": 150.228, "Th_in_C": 82.785, "Tc_in_C": 46.531, "Th_out_C": 80.215,
    "Tc_out_C": 50.335, "Twall_C": 81.614, "vibration_mm_s": 2.126, "flow_sp_kg_s": 5.276,
    "Th_sp_C": 82.852, "Tc_sp_C": 46.631, "protein_g_L_nominal": 30.691,
    "fat_g_L_nominal": 38.054, "solids_g_L_nominal": 121.545, "Ca_mM_nominal": 19.264,
    "PO4_mM_nominal": 6.462, "pH_nominal": 6.688, "ambient_T_C": 21.735,
    "ambient_RH_pct": 55.034, "batch_thermal_history_factor": 0.985,
}


def _telemetry(rows: int = 420, maintenance=(200, 230)) -> pd.DataFrame:
    """One asset, one row per minute, production with one maintenance period."""
    rng = np.random.default_rng(0)
    data = {k: v * (1 + 0.01 * rng.standard_normal(rows)) for k, v in _TYPICAL.items()}
    frame = pd.DataFrame(data)
    frame.insert(0, "timestamp", pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
                 .strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    frame.insert(1, "asset_id", "phe_line_1")
    frame["asset_family"] = "standard_phe"
    frame["milk_type"] = "whole"
    frame["phase"] = "production"
    frame["maintenance_active"] = 0
    frame["maintenance_type"] = "none"
    frame["fault_type"] = "none"
    frame["maintenance_planned"] = 0
    start, end = maintenance
    frame.loc[start:end, ["phase", "maintenance_active", "maintenance_type", "fault_type"]] = [
        "maintenance", 1, "unclog", "clogging"]
    frame["Rf_m2K_W"] = np.linspace(0.0001, 0.0009, rows)
    return frame


@pytest.fixture(scope="module")
def plugin():
    """The real plugin (local artifacts), fine-tuning for a single epoch."""
    instance = ml46.Ml46DairyFoulingClogDetectionPlugin()
    instance.load()
    instance._train_cfg = dataclasses.replace(instance._train_cfg, epochs=1)
    return instance


class _Tracker:
    run_id = "test"

    def log_params(self, *_a, **_k):
        """No-op."""

    def log_metrics(self, *_a, **_k):
        """No-op."""

    def log_metric(self, *_a, **_k):
        """No-op."""

    def upload_artifacts(self, *_a, **_k):
        """No-op."""


def _train(plugin, frame, tmp_path, monkeypatch):
    monkeypatch.setattr(ml46, "require_train_tracker", lambda run_id: _Tracker())
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return plugin.train(data_path=str(path), mlflow_run_id="test")


def test_normalize_only_fills_blanks():
    """A file without blanks is returned unchanged (same predictions as before the fix)."""
    frame = _telemetry()
    pd.testing.assert_frame_equal(preprocessing.normalize_raw_telemetry(frame), frame,
                                  check_dtype=False)


def test_normalize_fills_flags_and_maintenance_types():
    """Empty flags -> 0; empty type/cause -> other inside a maintenance, none outside."""
    frame = _telemetry()
    frame["maintenance_active"] = frame["maintenance_active"].astype(float)
    frame.loc[[5, 210], "maintenance_active"] = np.nan
    frame.loc[[10, 205], ["maintenance_type", "fault_type"]] = np.nan
    out = preprocessing.normalize_raw_telemetry(frame)
    assert out.loc[5, "maintenance_active"] == 0 and out.loc[210, "maintenance_active"] == 0
    assert out.loc[10, "maintenance_type"] == "none" and out.loc[205, "maintenance_type"] == "other"
    assert out.loc[205, "fault_type"] == "other"


def test_bad_timestamp_names_the_row_and_the_format():
    """The vendored message was 'Some telemetry timestamps could not be parsed.'"""
    frame = _telemetry()
    frame.loc[10, "timestamp"] = "31/01/2026 10:00"
    with pytest.raises(InvalidTelemetryInputError, match=r"fila 12: '31/01/2026 10:00'.*ISO 8601"):
        preprocessing.normalize_raw_telemetry(frame)


def test_train_with_empty_maintenance_cells(plugin, tmp_path, monkeypatch):
    """Empty maintenance_active in production + a maintenance period with no type/cause/planned."""
    frame = _telemetry()
    frame["maintenance_active"] = frame["maintenance_active"].astype(float)
    frame["maintenance_planned"] = frame["maintenance_planned"].astype(float)
    frame.loc[[154, 299, 404], "maintenance_active"] = np.nan  # rows that end a window (stride 5)
    frame.loc[200:230, ["maintenance_type", "fault_type", "maintenance_planned"]] = np.nan
    response = _train(plugin, frame, tmp_path, monkeypatch)
    assert response.n_windows > 0


def test_train_without_valid_windows_explains_why(plugin, tmp_path, monkeypatch):
    """phase in Spanish: the message names the values found and the expected English ones."""
    frame = _telemetry()
    frame["phase"] = frame["phase"].replace({"production": "producción"})
    with pytest.raises(ValueError, match=r"producción.*ninguno es 'production'"):
        _train(plugin, frame, tmp_path, monkeypatch)


@pytest.mark.parametrize("change, error, message", [
    (lambda f: f.assign(phase=f["phase"].replace({"production": "producción"})),
     InsufficientTelemetryHistoryError, "ninguno es 'production'"),
    (lambda f: f.iloc[:100], InsufficientTelemetryHistoryError, "máximo 100"),
    (lambda f: f.drop(columns=["Th_in_C"]), InvalidTelemetryInputError, "Th_in_C"),
])
def test_predict_batch_input_errors_are_explained(plugin, tmp_path, change, error, message):
    """Each used to be an HTTP 500 (KeyError 'asset_id' or a bare ValueError)."""
    path = tmp_path / "predict.csv"
    change(_telemetry().drop(columns=["Rf_m2K_W"])).to_csv(path, index=False)
    with pytest.raises(error, match=message):
        plugin.predict_batch(data_path=str(path))


def test_input_errors_are_422_in_predict():
    """Both ml46 input errors are registered as client errors of /predict."""
    # pylint: disable-next=import-outside-toplevel
    from app.registry import REGISTRY

    entry = next(e for e in REGISTRY if e.model_id == "ml46-dairy-fouling-clog-detection")
    assert {InsufficientTelemetryHistoryError, InvalidTelemetryInputError} <= set(
        entry.extra_predict_exceptions)


def test_stats_types_numeric_optional_columns(plugin):
    """QA saw ambient_T_C, ambient_RH_pct and batch_thermal_history_factor typed as text."""
    types = {f.name: f.type for f in plugin.stats().inputs}
    for column in ("ambient_T_C", "ambient_RH_pct", "batch_thermal_history_factor"):
        assert types[column] == "float"
    assert types["maintenance_active"] == "int" and types["phase"] == "string"
