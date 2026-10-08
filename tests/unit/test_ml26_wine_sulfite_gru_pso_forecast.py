"""Tests for ``ml26-wine-sulfite-gru-pso-forecast``.

Endpoint tests use the FakePlugin wiring from conftest (no artifacts). The unit tests below exercise
the real plugin helpers (preprocessing, postprocessing, restricted unpickler, fine-tuning) on small
toy inputs — wiring only; numeric correctness against the golden dataset lives in
outputs/a26/verification_report.md.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import InsufficientSequenceHistoryError
from app.plugins.ml26_wine_sulfite_gru_pso_forecast import preprocessing
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import WINDOW
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.exceptions import InvalidWineryInputError

PREFIX = "/models/ml26-wine-sulfite-gru-pso-forecast"
MODEL_ID = "ml26-wine-sulfite-gru-pso-forecast"

FEATURES = [
    "volume_l", "ambient_temp_c", "stage_progress", "hours_since_last_lab", "last_so2_addition_mg_l",
    "hours_since_last_addition", "temperature_c", "dissolved_oxygen_mg_l", "density_g_ml", "co2_g_l",
    "ph_lab", "free_sulfite_lab", "total_sulfite_lab", "lab_sample", "wine_type_red", "wine_type_rose",
    "wine_type_white", "stage_fermentation", "stage_must", "stage_pre_bottling", "stage_stabilization",
    "stage_storage",
]
LOT = {"lot_id": "LOT-T1", "wine_type": "red", "volume_l": 8000.0, "ambient_temp_c": 16.0}


def _readings(n: int, lot_id: str = "LOT-T1", with_progress: bool = True) -> list[dict]:
    rows = []
    for i in range(n):
        stage = "must" if i < 10 else "fermentation"
        row = {
            "lot_id": lot_id,
            "timestamp": f"2026-01-01T00:00:00Z+{2 * i}h",
            "timestamp_index": i,
            "stage": stage,
            "temperature_c": 18.0 + 0.05 * i,
            "dissolved_oxygen_mg_l": 1.2,
            "density_g_ml": 1.08 - 0.001 * i,
            "co2_g_l": 0.3,
        }
        if i % 4 == 0:
            row.update({"ph_lab": 3.4, "free_sulfite_lab": 20.0 - 0.1 * i, "total_sulfite_lab": 60.0})
        if i == 5:
            row["dose_mg_l"] = 4.0
        if with_progress:
            row["stage_progress"] = (i / 9) if i < 10 else (i - 10) / 60
        rows.append(row)
    return rows


# ── Endpoint wiring (FakePlugin) ─────────────────────────────────────────────

def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == MODEL_ID


def test_predict_inline_operational(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "lot": LOT, "readings": _readings(30)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["risk_band"] in {"bajo", "medio", "alto"}
    assert 0.0 <= body["underprotection_risk_72h"] <= 1.0


def test_predict_inline_without_stage_progress_rejected(client):
    """stage_progress is mandatory on every reading (manifest KI-01) -> Pydantic 422."""
    payload = {"mode": "inline", "lot": LOT, "readings": _readings(30, with_progress=False)}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_inline_stage_progress_out_of_range_rejected(client):
    readings = _readings(30)
    readings[-1]["stage_progress"] = 1.5
    payload = {"mode": "inline", "lot": LOT, "readings": readings}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_inline_feature_window(client):
    payload = {"mode": "inline", "feature_window": [[0.0] * 22] * 24}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 200


def test_predict_inline_short_history_rejected(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "lot": LOT, "readings": _readings(10)})
    assert resp.status_code == 422


def test_predict_inline_both_modes_rejected(client):
    payload = {"mode": "inline", "lot": LOT, "readings": _readings(30), "feature_window": [[0.0] * 22] * 24}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_inline_unknown_wine_type_rejected(client):
    payload = {"mode": "inline", "lot": {**LOT, "wine_type": "sparkling"}, "readings": _readings(30)}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/lecturas.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_lots"] == 1
    assert body["predictions"][0]["lot_id"] == "LOT-00013"


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/sequential.csv", "mlflow_run_id": "run-1"})
    assert resp.status_code == 200
    assert "val_overall_rmse" in resp.json()


@pytest.mark.parametrize("exc", [
    InsufficientSequenceHistoryError("historial insuficiente"),
    InvalidWineryInputError("lot_id huérfano"),
])
def test_domain_errors_map_to_422(client, fake_plugins, exc):
    fake_plugins[MODEL_ID].raise_on_inline = exc
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "lot": LOT, "readings": _readings(30)})
    assert resp.status_code == 422


# ── Preprocessing (real helpers, no artifacts) ───────────────────────────────

def _frames(n=30, **kw):
    return preprocessing.frames_from_inline({"lot": LOT, "readings": _readings(n, **kw)})


def test_prepare_windows_shape_and_metadata():
    lots, readings = _frames(30)
    raw = preprocessing.build_raw_frame(lots, readings)
    x, meta, last = preprocessing.prepare_windows(raw, FEATURES)
    assert x.shape == (1, WINDOW, 22)
    assert meta.iloc[0]["lot_id"] == "LOT-T1"
    assert meta.iloc[0]["timestamp_index"] == 29
    assert list(last.columns) == FEATURES
    assert np.isfinite(x).all()


def test_stage_progress_is_used_as_provided():
    lots, readings = _frames(30)
    raw = preprocessing.build_raw_frame(lots, readings)
    x, _, _ = preprocessing.prepare_windows(raw, FEATURES)
    assert x[0, -1, FEATURES.index("stage_progress")] == pytest.approx(19 / 60)


def test_missing_stage_progress_column_rejected():
    lots, readings = _frames(30, with_progress=False)
    with pytest.raises(InvalidWineryInputError, match="stage_progress"):
        preprocessing.build_raw_frame(lots, readings)


def test_partial_stage_progress_rejected():
    lots, readings = _frames(30)
    readings.loc[5, "stage_progress"] = np.nan
    with pytest.raises(InvalidWineryInputError, match="stage_progress es obligatorio"):
        preprocessing.build_raw_frame(lots, readings)


def test_stage_progress_out_of_range_rejected():
    lots, readings = _frames(30)
    readings.loc[5, "stage_progress"] = -0.1
    with pytest.raises(InvalidWineryInputError, match=r"\[0, 1\]"):
        preprocessing.build_raw_frame(lots, readings)


def test_lab_values_carried_forward():
    lots, readings = _frames(30)
    raw = preprocessing.build_raw_frame(lots, readings)
    x, _, _ = preprocessing.prepare_windows(raw, FEATURES)
    # last lab at index 28 -> 1 step (2 h) before the prediction instant
    assert x[0, -1, FEATURES.index("hours_since_last_lab")] == pytest.approx(2.0)
    assert x[0, -1, FEATURES.index("free_sulfite_lab")] == pytest.approx(20.0 - 2.8)


def test_short_history_raises():
    lots, readings = _frames(20)
    raw = preprocessing.build_raw_frame(lots, readings)
    with pytest.raises(InsufficientSequenceHistoryError):
        preprocessing.prepare_windows(raw, FEATURES)


def test_orphan_lot_rejected():
    lots = pd.DataFrame([LOT])
    readings = pd.DataFrame(_readings(30, lot_id="LOT-OTHER"))
    with pytest.raises(InvalidWineryInputError):
        preprocessing.build_raw_frame(lots, readings)


def test_unknown_stage_rejected():
    lots, readings = _frames(30)
    readings.loc[3, "stage"] = "aging"
    with pytest.raises(InvalidWineryInputError):
        preprocessing.build_raw_frame(lots, readings)


def test_batch_csv_split_and_inconsistent_lot():
    df = pd.DataFrame(_readings(30)).assign(wine_type="red", volume_l=8000.0, ambient_temp_c=16.0)
    lots, readings = preprocessing.frames_from_batch_csv(df)
    assert len(lots) == 1 and "wine_type" not in readings.columns
    df.loc[0, "volume_l"] = 1.0
    with pytest.raises(InvalidWineryInputError):
        preprocessing.frames_from_batch_csv(df)


def test_feature_window_validation():
    assert preprocessing.validate_feature_window([[0.0] * 22] * 24, 22).shape == (1, 24, 22)
    with pytest.raises(InvalidWineryInputError):
        preprocessing.validate_feature_window([[0.0] * 21] * 24, 22)


# ── Model helpers (torch) ────────────────────────────────────────────────────

def _tiny_loaded_model():
    torch = pytest.importorskip("torch")
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.gru_model import build_gru_from_config
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import LoadedModel

    torch.manual_seed(0)
    cfg = {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0, "head_dropout": 0.0, "bidirectional": True,
           "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4, "epochs": 2, "patience": 2,
           "lr_patience": 1, "lr_decay_factor": 0.5, "gradient_clip": 1.0, "input_dim": 22}
    net = build_gru_from_config(cfg, input_dim=22, output_dim=2).eval()
    return LoadedModel(network=net, feature_names=FEATURES,
                       target_names=["future_free_sulfite_72h", "underprotection_risk_72h"],
                       mean=np.zeros((1, 22), np.float32), std=np.ones((1, 22), np.float32),
                       y_mean=np.array([25.0, 0.5], np.float32), y_std=np.array([7.0, 0.26], np.float32),
                       config=cfg, source="fixed")


def test_predict_windows_clips_risk_and_rounds():
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.postprocessing import predict_windows, risk_band

    model = _tiny_loaded_model()
    model.y_std[1] = 1e6  # force out-of-range raw risk -> must be clipped to [0, 1]
    preds = predict_windows(model, np.random.default_rng(0).normal(size=(3, 24, 22)).astype(np.float32))
    assert preds.shape == (3, 2)
    assert ((preds[:, 1] >= 0.0) & (preds[:, 1] <= 1.0)).all()
    assert np.allclose(preds, np.round(preds, 6))
    assert [risk_band(v) for v in (0.1, 0.33, 0.65, 0.66)] == ["bajo", "medio", "medio", "alto"]


def test_user_model_roundtrip(tmp_path):
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import load_user_model, save_user_model
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.postprocessing import predict_windows

    model = _tiny_loaded_model()
    save_user_model(model, tmp_path)
    again = load_user_model(tmp_path, source="mlflow:test")
    x = np.random.default_rng(1).normal(size=(2, 24, 22)).astype(np.float32)
    assert np.array_equal(predict_windows(model, x), predict_windows(again, x))
    assert again.source == "mlflow:test"


def test_restricted_unpickler_rejects_foreign_globals(tmp_path):
    pytest.importorskip("torch")
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.sequence_bundle import load_sequence_bundle

    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(pd.Timestamp("2026-01-01")))
    with pytest.raises(pickle.UnpicklingError):
        load_sequence_bundle(path)


def _sequential_training_frame(n_lots: int = 8, steps: int = 70) -> pd.DataFrame:
    rows = []
    for lot in range(n_lots):
        for i in range(steps):
            rows.append({
                "lot_id": f"L{lot:02d}", "timestamp_index": i, "wine_type": ["red", "rose", "white"][lot % 3],
                "volume_l": 5000.0 + 100 * lot, "ambient_temp_c": 15.0, "stage": "fermentation" if i < 40 else "storage",
                "temperature_c": 18.0, "dissolved_oxygen_mg_l": 1.0 + 0.01 * i, "density_g_ml": 1.05, "co2_g_l": 0.4,
                "ph_lab": 3.4, "free_sulfite_lab": 25.0 - 0.1 * i, "total_sulfite_lab": 70.0,
                "actual_dose_mg_l": 3.0 if i % 20 == 0 else 0.0, "free_sulfite_mg_l": 25.0 - 0.1 * i + 0.2 * lot,
                "stage_progress": (i / 39) if i < 40 else (i - 40) / (steps - 41),
            })
    return pd.DataFrame(rows)


def test_fine_tune_does_not_mutate_served_model():
    torch = pytest.importorskip("torch")
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import training

    base = _tiny_loaded_model()
    before = {k: v.clone() for k, v in base.network.state_dict().items()}
    data = training.prepare_training_data(_sequential_training_frame(), base)
    assert data["train"][0].shape[1:] == (24, 22)
    result = training.fine_tune(base, data, base.config)
    assert result.epochs_run >= 1 and result.n_lots["train"] == 5
    assert set(result.val_report) >= {"future_free_sulfite_72h", "underprotection_risk_72h", "overall_rmse"}
    for key, value in base.network.state_dict().items():
        assert torch.equal(value, before[key])


def test_fine_tune_requires_validation_lots():
    pytest.importorskip("torch")
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import training

    base = _tiny_loaded_model()
    data = training.prepare_training_data(_sequential_training_frame(n_lots=3), base)
    with pytest.raises(ValueError):
        training.fine_tune(base, data, base.config)


# ── Real plugin class with an injected tiny model (no artifacts, MLflow mocked) ─

def _real_plugin():
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.plugin import Ml26WineSulfiteGruPsoForecastPlugin

    plugin = Ml26WineSulfiteGruPsoForecastPlugin()
    plugin._model = _tiny_loaded_model()  # pylint: disable=protected-access
    return plugin


def test_real_plugin_inline_operational_and_feature_window():
    plugin = _real_plugin()
    assert plugin.is_loaded()
    resp = plugin.predict_inline(features={"lot": LOT, "readings": _readings(30), "feature_window": None})
    assert resp.lot_id == "LOT-T1" and resp.timestamp_index == 29
    assert resp.model_source == "fixed" and len(resp.xai_feature_values) == 22
    resp_fw = plugin.predict_inline(features={"lot": None, "readings": None, "feature_window": [[0.1] * 22] * 24})
    assert resp_fw.lot_id is None
    assert plugin.stats().runtime_stats.total_predictions == 2


def test_real_plugin_batch(tmp_path):
    plugin = _real_plugin()
    rows = _readings(30) + _readings(26, lot_id="LOT-T2")
    df = pd.DataFrame(rows)
    df["wine_type"] = np.where(df["lot_id"] == "LOT-T1", "red", "white")
    df["volume_l"], df["ambient_temp_c"] = 8000.0, 16.0
    path = tmp_path / "lecturas.csv"
    df.to_csv(path, index=False)
    resp = plugin.predict_batch(data_path=str(path))
    assert resp.n_lots == 2
    assert {p.lot_id for p in resp.predictions} == {"LOT-T1", "LOT-T2"}


def test_real_plugin_stats_contract():
    stats = _real_plugin().stats()
    names = {f.name for f in stats.inputs}
    assert {"lot.wine_type", "readings[].stage", "readings[].stage_progress"} <= names
    assert stats.metrics["future_free_sulfite_72h_mae"] == pytest.approx(1.5082)
    assert "synthetic_data_warning" in stats.metrics


def test_real_plugin_train_uploads_only_to_mlflow(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import model_loader
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import plugin as plugin_mod

    local_before = set(model_loader._store.local_dir.rglob("*"))  # pylint: disable=protected-access
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    csv = tmp_path / "sequential.csv"
    _sequential_training_frame().to_csv(csv, index=False)
    plugin = _real_plugin()
    before = {k: v.clone() for k, v in plugin._model.network.state_dict().items()}  # pylint: disable=protected-access
    resp = plugin.train(data_path=str(csv), mlflow_run_id="run123")
    assert resp.n_lots_train == 5 and resp.n_windows_val > 0
    tracker.upload_artifacts.assert_called_once()
    assert resp.mlflow_run_id == "run123"
    assert set(model_loader._store.local_dir.rglob("*")) == local_before  # nothing written locally
    assert resp.upload_warning is None
    for key, value in plugin._model.network.state_dict().items():  # pylint: disable=protected-access
        assert (value == before[key]).all()  # served model untouched


def test_real_plugin_train_requires_stage_progress(tmp_path):
    csv = tmp_path / "no_sp.csv"
    frame = _sequential_training_frame()
    frame.loc[3, "stage_progress"] = np.nan
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="stage_progress"):
        _real_plugin().train(data_path=str(csv), mlflow_run_id="run-1")


def test_real_plugin_train_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    _sequential_training_frame().drop(columns=["free_sulfite_mg_l"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="free_sulfite_mg_l"):
        _real_plugin().train(data_path=str(csv), mlflow_run_id="run-1")


def test_real_plugin_uses_and_cleans_mlflow_model(tmp_path, monkeypatch):
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import plugin as plugin_mod
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import save_user_model

    user_dir = tmp_path / "mlflow_tmp"
    user_model = _tiny_loaded_model()
    save_user_model(user_model, user_dir / "model")
    user_model.source = "mlflow:abc"
    monkeypatch.setattr(plugin_mod, "download_user_model_from_mlflow", lambda run_id: (user_model, str(user_dir)))
    resp = _real_plugin().predict_inline(
        features={"lot": LOT, "readings": _readings(30), "feature_window": None}, mlflow_run_id="abc",
    )
    assert resp.model_source == "mlflow:abc"
    assert not user_dir.exists()  # temp dir removed in finally


def test_mlflow_download_without_artifacts_is_a_422_not_the_base_model(monkeypatch):
    from unittest.mock import MagicMock

    from app.domain.services.exceptions import UserModelUnavailableError
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import mlflow_utils

    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker",
                        MagicMock(return_value=MagicMock(download_artifacts=MagicMock(return_value=""))))
    with pytest.raises(UserModelUnavailableError, match="missing"):
        mlflow_utils.download_user_model_from_mlflow("missing")


def test_real_plugin_train_fails_when_mlflow_upload_fails(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from app.domain.services.exceptions import ModelPersistenceError
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import plugin as plugin_mod

    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    csv = tmp_path / "sequential.csv"
    _sequential_training_frame().to_csv(csv, index=False)
    with pytest.raises(ModelPersistenceError, match="mlflow caído"):
        _real_plugin().train(data_path=str(csv), mlflow_run_id="run-1")


def test_real_plugin_unloaded_raises():
    from app.domain.services.exceptions import ModelNotLoadedError
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast.plugin import Ml26WineSulfiteGruPsoForecastPlugin

    with pytest.raises(ModelNotLoadedError):
        Ml26WineSulfiteGruPsoForecastPlugin().predict_inline(features={"feature_window": [[0.0] * 22] * 24})
