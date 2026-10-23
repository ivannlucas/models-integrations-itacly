"""Tests for ml48 /train — the real trainer code, not the FakePlugin wiring.

inbox/a48/codigo ships the same training code as a47 (src/training/, src/fine_tuning/,
preprocess.py), so ml48's trainer is the ml47 port: ``full`` = ``main.py train`` and
``fine_tune`` = ``main.py fine_tune``. On top of ml47, the retrained model carries its own SHAP
background. The synthetic cycles below are small but shaped like the delivered CSVs.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.plugins.ml48_dairy_dnsl_critical_point_detection import constants, trainer
from app.plugins.ml48_dairy_dnsl_critical_point_detection.constants import SENSOR_COLUMNS
from app.plugins.ml48_dairy_dnsl_critical_point_detection.model_loader import CNN_Pasteurizer

TARGETS = ["Target_Fouling", "Target_Valvula", "Target_Bomba", "Target_Acumulador"]


def _cycles(n_cycles=20, steps=12, target_names=TARGETS, seed=0):
    """Labelled cycles whose sensor level depends on the label, so the label is learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(n_cycles):
        label = cid % 3
        for t in range(steps):
            row = {"Cycle_ID": cid, "Time_Segundos": round(t / 10, 1)}
            row.update({c: 50.0 + 10 * label + t + rng.normal(0, 0.1) for c in SENSOR_COLUMNS})
            row.update({name: label for name in target_names})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "plant_cycles.csv"
    _cycles().to_csv(path, index=False)
    return str(path)


# ── Datos ─────────────────────────────────────────────────────────────────────

def test_each_cycle_keeps_its_own_label():
    df = trainer.engineer_cycle_features(_cycles(n_cycles=3))
    cols = trainer.feature_columns(df)
    _, y = trainer.build_tensors(df, [0, 1, 2], cols, StandardScaler().fit(df[cols]))
    assert y.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]


@pytest.mark.parametrize("names", [TARGETS, ["Fouling", "Valvula", "Bomba", "Acumulador"]])
def test_training_csv_accepts_original_and_legacy_target_names(tmp_path, names):
    path = tmp_path / "cycles.csv"
    _cycles(n_cycles=2, target_names=names).to_csv(path, index=False)
    assert list(trainer.load_training_frame(str(path))[TARGETS].iloc[-1]) == [1, 1, 1, 1]


@pytest.mark.parametrize("bad,match", [("no_targets", "Target_Fouling"), ("no_sensor", "PS3"), ("label_3", "0, 1 o 2")])
def test_invalid_training_csv_is_a_clear_error(tmp_path, bad, match):
    df = _cycles(n_cycles=2)
    if bad == "no_targets":
        df = df.drop(columns=TARGETS)
    elif bad == "no_sensor":
        df = df.drop(columns=["PS3"])
    else:
        df.loc[0, "Target_Bomba"] = 3
    path = tmp_path / "cycles.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        trainer.load_training_frame(str(path))


def test_split_is_by_cycle_deterministic_and_70_15_15():
    first, second = trainer.split_cycle_ids(np.arange(100)), trainer.split_cycle_ids(np.arange(100))
    assert [list(a) for a in first] == [list(b) for b in second]
    assert tuple(len(part) for part in first) == (70, 15, 15)


def test_too_few_cycles_is_a_clear_error():
    with pytest.raises(ValueError, match="ciclos"):
        trainer.split_cycle_ids(np.arange(3))


def test_tensors_follow_ascending_cycle_id_like_the_original():
    df = trainer.engineer_cycle_features(_cycles(n_cycles=3))
    cols = trainer.feature_columns(df)
    _, y = trainer.build_tensors(df, [2, 0, 1], cols, StandardScaler().fit(df[cols]))
    assert y[:, 0].tolist() == [0, 1, 2]


def test_augmentation_noise_is_the_originals_global_numpy_stream():
    # preprocess.py draws np.random.normal per sensor after seed_everything(42); default_rng differs.
    df = _cycles(n_cycles=6)
    np.random.seed(42)
    augmented, ids = trainer._augment_train_cycles(df, np.array([4, 1, 3]))  # pylint: disable=protected-access
    assert list(ids[3:]) == [6, 7, 8]  # copies of 1, 3, 4, ascending like the original + 50000

    np.random.seed(42)
    source = df[df["Cycle_ID"].isin([1, 3, 4])]
    expected_ps1 = source["PS1"].to_numpy() + np.random.normal(0, source["PS1"].std() * 0.2, size=len(source))
    assert np.allclose(augmented[augmented["Cycle_ID"] >= 6]["PS1"].to_numpy(), expected_ps1)


def test_digital_twin_follows_the_inference_switch(csv_path, monkeypatch):
    monkeypatch.setitem(constants.TRAIN_HYPERPARAMS, "epochs", 1)
    monkeypatch.setattr(trainer, "_device", lambda: torch.device("cpu"))
    off = trainer.train_full(csv_path, apply_twin=False)
    on = trainer.train_full(csv_path, apply_twin=True)
    ts1 = off.feature_cols.index("TS1")
    assert off.scaler.mean_[ts1] == pytest.approx(off.ts1_mean_train, abs=1.0)  # real temperatures
    assert on.scaler.mean_[ts1] == pytest.approx(65.0, abs=1.0)                 # shifted to 65 °C


def test_metrics_follow_the_reported_formulas():
    y_true = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 2, 0]])
    y_pred = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 2, 1]])
    metrics = trainer.compute_test_metrics(y_true, y_pred)
    assert metrics["exact_match"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx((1 + 1 + 1 + 0.75) / 4)
    assert metrics["precision_macro"] == pytest.approx((9 + 1 + 0.5 + 1) / 12)
    assert metrics["recall_macro"] == pytest.approx((9 + 0.5 + 1 + 1) / 12)
    assert metrics["f1_macro"] == pytest.approx((9 + 2 / 3 + 2 / 3 + 1) / 12)


def test_hyperparameters_are_the_delivered_config_yaml():
    hp = constants.TRAIN_HYPERPARAMS
    assert (hp["learning_rate"], hp["dropout_rate"], hp["max_lambda"]) == (
        0.0022243234786004373, 0.20219689010649033, 3.7585293696964697)
    assert (hp["epochs"], hp["warmup_epochs"], hp["ramp_up_epochs"], hp["patience"], hp["batch_size"]) == (
        300, 10, 80, 15, 32)
    assert (hp["noise_level"], hp["test_size_1"], hp["test_size_2"], hp["random_state"]) == (0.2, 0.30, 0.50, 42)
    assert constants.FINE_TUNE_HYPERPARAMS == {"epochs": 50, "patience": 7, "lr_divisor": 10, "batch_size": 32}


# ── Procedimientos ───────────────────────────────────────────────────────────

def _base_artifacts():
    df = trainer.engineer_cycle_features(_cycles())
    cols = trainer.feature_columns(df)
    scaler = StandardScaler().fit(df[cols])
    torch.manual_seed(0)
    model = CNN_Pasteurizer(n_sensors=len(cols), n_classes=3,
                            dropout_prob=constants.TRAIN_HYPERPARAMS["dropout_rate"])
    return model, scaler, cols


def test_fine_tune_only_updates_the_four_heads(csv_path, monkeypatch):
    monkeypatch.setitem(constants.FINE_TUNE_HYPERPARAMS, "epochs", 2)
    base, scaler, cols = _base_artifacts()
    before = {k: v.clone() for k, v in base.state_dict().items()}

    result = trainer.fine_tune(csv_path, base_model=base, scaler=scaler, feature_cols=cols)

    after = dict(result.model.named_parameters())
    for name in after:
        if name.startswith("features."):
            assert torch.equal(after[name], before[name]), f"backbone weight changed: {name}"
    assert any(not torch.equal(after[n], before[n]) for n in after if n.startswith("head_"))
    assert all(torch.equal(base.state_dict()[k], v) for k, v in before.items()), "base model mutated"
    assert result.scaler is scaler and result.feature_cols == cols


def test_full_training_returns_hold_out_metrics_and_its_shap_background(csv_path, monkeypatch):
    monkeypatch.setitem(constants.TRAIN_HYPERPARAMS, "epochs", 2)
    result = trainer.train_full(csv_path)
    assert (result.metrics["n_train"], result.metrics["n_val"], result.metrics["n_test"]) == (14, 3, 3)
    _, _, test_ids = trainer.split_cycle_ids(np.arange(20))
    assert result.shap_background.shape == (3, 28, 600)
    assert result.shap_background_cycle_ids == sorted(int(c) for c in test_ids)


# ── Cadena completa: train() del plugin → MLflow → predict con ese run ───────

@pytest.fixture
def plugin_with_mlflow(monkeypatch, tmp_path):
    """Real plugin + real mlflow_utils; only the MLflow server is replaced by a folder per run."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection import mlflow_utils, plugin

    store = tmp_path / "mlflow"

    class FolderTracker:
        TRACKING_URI = "file://test"

        def __init__(self, run_id=""):
            self.run_id = run_id

        def connect(self, run_id):
            self.run_id = run_id

        def log_metrics(self, metrics, step=0):
            pass

        def set_tags(self, tags):
            pass

        def upload_artifacts(self, local_dir, artifact_path=""):
            shutil.copytree(local_dir, store / self.run_id / artifact_path, dirs_exist_ok=True)

        def download_artifacts(self, dest_dir, artifact_path=""):
            source = store / self.run_id / artifact_path
            if not source.exists():
                return ""
            return str(shutil.copytree(source, Path(dest_dir) / artifact_path))

    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", FolderTracker)
    base, scaler, cols = _base_artifacts()
    monkeypatch.setattr(plugin, "load_artifacts_from_dir", lambda path: (base, scaler, cols, 45.0))
    monkeypatch.setattr(plugin, "load_shap_background", lambda path: None)
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    instance = plugin.Ml48DairyDnslCriticalPointDetectionPlugin()
    instance.load()
    return instance, base, store


@pytest.mark.parametrize("mode", ["fine_tune", "full"])
def test_train_registers_a_new_model_in_mlflow_and_predict_uses_it(plugin_with_mlflow, csv_path, monkeypatch, mode):
    instance, base, store = plugin_with_mlflow
    monkeypatch.setitem(constants.FINE_TUNE_HYPERPARAMS, "epochs", 2)
    monkeypatch.setitem(constants.TRAIN_HYPERPARAMS, "epochs", 2)
    base_before = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}

    response = instance.train(data_path=csv_path, mlflow_run_id="run-ft", mode=mode)

    assert response.mode == mode and response.mlflow_run_id == "run-ft"
    uploaded = sorted(p.name for p in (store / "run-ft" / "model").iterdir())
    assert uploaded == sorted([constants.MODEL_FILENAME, constants.SCALER_FILENAME,
                               constants.FEATURE_COLUMNS_FILENAME, constants.TS1_MEAN_FILENAME,
                               constants.SHAP_BACKGROUND_FILENAME, constants.SHAP_BACKGROUND_META_FILENAME])
    meta = json.loads((store / "run-ft" / "model" / constants.SHAP_BACKGROUND_META_FILENAME).read_text())
    assert len(meta["cycle_ids"]) == response.n_test

    ctx = instance._context("run-ft")  # pylint: disable=protected-access
    try:
        registered = ctx["model"].state_dict()
        assert any(not torch.equal(registered[k].cpu(), base_before[k]) for k in base_before if k.startswith("head_"))
        assert ctx["shap_background"].shape == (response.n_test, 28, 600)
    finally:
        shutil.rmtree(ctx["temp_dir"], ignore_errors=True)
    served = instance._model.state_dict()  # pylint: disable=protected-access
    assert all(torch.equal(served[k].cpu(), v) for k, v in base_before.items())


def test_train_request_defaults_to_fine_tune_and_accepts_full():
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.train_dto import TrainRequest

    assert TrainRequest(data_path="x.csv", mlflow_run_id="run-1").mode == "fine_tune"
    assert TrainRequest(data_path="x.csv", mlflow_run_id="run-1", mode="full").mode == "full"
    with pytest.raises(ValueError):
        TrainRequest(data_path="x.csv", mlflow_run_id="run-1", mode="whatever")


def test_train_request_maps_the_platform_transport_mode_to_fine_tune():
    """The platform always sends mode="batch" (train-task-manager.ts) and the orchestrator forwards it."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.train_dto import TrainRequest

    for transport in ("batch", "inline"):
        assert TrainRequest(data_path="x.csv", mlflow_run_id="run-1", mode=transport).mode == "fine_tune"
