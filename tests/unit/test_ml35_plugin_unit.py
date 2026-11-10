"""Unit tests for the real ml35 plugin logic (no served artifacts needed).

A tiny seeded ANN plus scalers fitted on synthetic rows stand in for the S3 artifacts —
these tests check the feature vector the ANN receives, not the model's accuracy.
"""
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.domain.services import mlflow_artifacts
from app.plugins.ml35_dairy_ann_cleaning_cost import plugin as ml35
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import FEATURES, TARGET_COL
from app.plugins.ml35_dairy_ann_cleaning_cost.model_loader import PasteurizationANN

# Manual values from the DataGia SHAP bug report, with both optional fields left empty.
REPORTED_INLINE = {
    "temp_entrada_leche": 4.5,
    "temp_ambiente": 22.0,
    "temp_setpoint_leche": 72.0,
    "temp_proceso_leche": None,
    "temp_agua_servicio": None,
    "flujo_leche_lh": 2510.0,
    "horas_desde_limpieza": 10.0,
    "presion_diferencial_bar": 1.0,
}


def _synthetic_rows(n: int = 64) -> pd.DataFrame:
    """n plausible operating rows (8 ANN features + target), seeded."""
    rng = np.random.default_rng(0)
    setpoint = rng.uniform(72, 78, n)
    return pd.DataFrame({
        "temp_entrada_leche": rng.uniform(2.5, 6, n),
        "temp_ambiente": rng.uniform(12, 32, n),
        "temp_setpoint_leche": setpoint,
        "temp_proceso_leche": setpoint,
        "temp_agua_servicio": setpoint + 10,
        "flujo_leche_lh": rng.uniform(2500, 5000, n),
        "horas_desde_limpieza": rng.uniform(0, 20, n),
        "presion_diferencial_bar": rng.uniform(0.3, 1.6, n),
        TARGET_COL: rng.uniform(15000, 35000, n),
    })


@pytest.fixture(name="plugin")
def fixture_plugin(monkeypatch):
    """ml35 loaded through load() with a tiny seeded ANN and scalers fitted on synthetic rows."""
    torch.manual_seed(0)
    rows = _synthetic_rows()
    served = (
        PasteurizationANN(len(FEATURES)).eval(),
        StandardScaler().fit(rows[FEATURES]),
        StandardScaler().fit(rows[[TARGET_COL]]),
    )
    monkeypatch.setattr(ml35, "load_artifacts", lambda: served)
    p = ml35.Ml35DairyAnnCleaningCostPlugin()
    p.load()
    return p


def test_inline_xai_values_are_the_ann_input_after_defaults(plugin):
    """Inline xai_feature_values is the exact 8-feature vector the ANN received."""
    resp = plugin.predict_inline(features=dict(REPORTED_INLINE))

    xai = resp.xai_feature_values
    assert list(xai) == FEATURES
    assert xai["temp_proceso_leche"] == 72.0
    assert xai["temp_agua_servicio"] == 82.0
    # The consumo returned is exactly the ANN output for that same vector, given in full.
    assert resp.consumo_agua_l == plugin.predict_inline(features=dict(xai)).consumo_agua_l


def test_inline_keeps_optional_values_when_given(plugin):
    """Given optional temperatures are used as is, not replaced by the defaults."""
    features = dict(REPORTED_INLINE, temp_proceso_leche=73.0, temp_agua_servicio=90.0)
    xai = plugin.predict_inline(features=features).xai_feature_values
    assert xai["temp_proceso_leche"] == 73.0
    assert xai["temp_agua_servicio"] == 90.0


def test_batch_blank_optional_cells_are_filled_not_nan(plugin, tmp_path):
    """Blank optional CSV cells get the same defaults as inline, never NaN."""
    csv = tmp_path / "batch.csv"
    df = _synthetic_rows(3)[FEATURES]
    df.loc[0, ["temp_proceso_leche", "temp_agua_servicio"]] = np.nan
    df.to_csv(csv, index=False)

    row = plugin.predict_batch(data_path=str(csv)).predictions[0]

    assert math.isfinite(row["consumo_agua_l"])
    assert row["xai_feature_values"]["temp_proceso_leche"] == df.loc[0, "temp_setpoint_leche"]
    setpoint = df.loc[0, "temp_setpoint_leche"]
    assert row["xai_feature_values"]["temp_agua_servicio"] == setpoint + 10


def test_batch_rejects_rows_below_pu_min(plugin, tmp_path):
    """A row below the PU safety rule is flagged as rejected, not counted as an error."""
    csv = tmp_path / "batch.csv"
    df = _synthetic_rows(2)[FEATURES]
    # 72.02 °C at 4900 L/h gives PU ≈ 12, like the 11 rejected rows in the X_test.csv report.
    unsafe = ["temp_setpoint_leche", "temp_proceso_leche", "flujo_leche_lh"]
    df.loc[1, unsafe] = [72.02, 72.02, 4900.0]
    df.to_csv(csv, index=False)

    preds = plugin.predict_batch(data_path=str(csv)).predictions

    assert "consumo_agua_l" in preds[0]
    assert preds[0]["rejected_by_safety_rule"] == 0
    assert "consumo_agua_l" not in preds[1]
    assert preds[1]["rejected_by_safety_rule"] == 1
    assert "PU=" in preds[1]["error"]


def test_batch_other_errors_are_not_safety_rejections(plugin, tmp_path):
    """Any other row failure is a plain error row, without the safety-rejection flag."""
    csv = tmp_path / "batch.csv"
    df = _synthetic_rows(1)[FEATURES]
    df["flujo_leche_lh"] = df["flujo_leche_lh"].astype(object)
    df.loc[0, "flujo_leche_lh"] = "n/a"
    df.to_csv(csv, index=False)

    row = plugin.predict_batch(data_path=str(csv)).predictions[0]

    assert "consumo_agua_l" not in row
    assert "flujo_leche_lh" in row["error"]
    assert "rejected_by_safety_rule" not in row


def test_train_fills_blank_optional_cells(plugin, tmp_path, monkeypatch):
    """train() fills blank optional cells too, so the metrics are finite."""
    csv = tmp_path / "train.csv"
    df = _synthetic_rows()
    df.loc[:5, ["temp_proceso_leche", "temp_agua_servicio"]] = np.nan
    df.to_csv(csv, index=False)
    # The retrained model is only persisted in MLflow: a mocked tracker accepts the upload.
    monkeypatch.setattr(mlflow_artifacts, "BaseMLflowTracker", MagicMock())

    resp = plugin.train(data_path=str(csv), mlflow_run_id="run-1")

    assert math.isfinite(resp.mae)
    assert math.isfinite(resp.r2)


# Same figures as the platform's model card (fermentation-optimization.js, modelo-35), which
# come from the a35 repo's models/metrics/test_metrics.txt and val_metrics.txt.
CARD_METRICS = {
    "test_r2": 0.9932, "test_mae": 341.58, "test_rmse": 417.75, "test_mae_relative_pct": 1.39,
    "val_r2": 0.9935,
}


def test_stats_metrics_match_the_model_card(plugin):
    """/stats reports the served model's test/validation figures of the model card."""
    metrics = plugin.stats().metrics
    for key, value in CARD_METRICS.items():
        assert metrics[key] == value, key
    assert metrics["test_samples"] == 750


def test_stats_describe_every_ann_input_and_prediction_output(plugin):
    """/stats lists the 8 ANN inputs and only the outputs the platform can produce."""
    stats = plugin.stats()
    assert [f.name for f in stats.inputs] == FEATURES
    # Only what the platform lets the user run: no GA "optimize" outputs.
    assert [f.name for f in stats.outputs] == [
        "consumo_agua_l", "pu_logrado", "rejected_by_safety_rule",
    ]


def test_stats_for_a_retrain_show_its_own_metrics_not_the_served_model(plugin, monkeypatch):
    """With a retraining run, /stats shows that run's training metrics, not the served ones."""

    class FakeTracker:
        """BaseMLflowTracker stand-in for a finished fine-tuning run."""

        def __init__(self, run_id):
            self.run_id = run_id

        def get_metrics(self):
            """Metrics train() logs for a fine-tuning run."""
            return {"mae": 120.5, "r2": 0.998, "n_samples": 5000}

        def get_params(self):
            """Params train() logs for a fine-tuning run."""
            return {"epochs": 100}

    monkeypatch.setattr(mlflow_artifacts, "BaseMLflowTracker", FakeTracker)
    metrics = plugin.stats(mlflow_run_id="run-1").metrics

    assert metrics["train_r2"] == 0.998
    assert metrics["train_mae"] == 120.5
    assert metrics["n_samples"] == 5000
    assert not [k for k in metrics if k.startswith(("test_", "val_"))]
