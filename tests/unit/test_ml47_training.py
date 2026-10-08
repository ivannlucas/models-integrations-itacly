"""Tests for ml47 /train — the real trainer code, not the FakePlugin wiring.

Ports of the AI team's two procedures (inbox/a47/codigo):
``full`` = src/training/trainer.py + src/data_processing/preprocess.py (``main.py train``) and
``fine_tune`` = src/fine_tuning/fine_tuner.py (``main.py fine_tune``). The synthetic cycles below
are small but shaped like the delivered CSVs (Cycle_ID, Time_Segundos, 7 sensors, Target_*).
"""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection import constants, trainer
from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.constants import SENSOR_COLUMNS
from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.model_loader import CNN_Pasteurizer

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


# ── Datos: etiquetas, columnas y features por ciclo ──────────────────────────

def test_each_cycle_keeps_its_own_label():
    df = trainer.engineer_cycle_features(_cycles(n_cycles=3))
    cols = trainer.feature_columns(df)
    scaler = StandardScaler().fit(df[cols])
    _, y = trainer.build_tensors(df, [0, 1, 2], cols, scaler)
    assert y.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]


def test_rolling_features_never_mix_cycles():
    df = trainer.engineer_cycle_features(_cycles(n_cycles=3))
    first_rows = df[df["Cycle_ID"] == 1].sort_values("Time_Segundos")
    ps1 = first_rows["PS1"].to_numpy()
    assert first_rows["PS1_rmean"].iloc[1] == pytest.approx(ps1[:2].mean())
    assert first_rows["PS1_lag1"].iloc[0] == pytest.approx(ps1[0])  # bfill inside the cycle


def test_feature_column_order_matches_the_delivered_artifact():
    cols = trainer.feature_columns(trainer.engineer_cycle_features(_cycles(n_cycles=1)))
    expected = (SENSOR_COLUMNS + [f"{c}_rmean" for c in SENSOR_COLUMNS]
                + [f"{c}_rstd" for c in SENSOR_COLUMNS] + [f"{c}_lag1" for c in SENSOR_COLUMNS])
    assert cols == expected


@pytest.mark.parametrize("names", [TARGETS, ["Fouling", "Valvula", "Bomba", "Acumulador"]])
def test_training_csv_accepts_original_and_legacy_target_names(tmp_path, names):
    path = tmp_path / "cycles.csv"
    _cycles(n_cycles=2, target_names=names).to_csv(path, index=False)
    assert list(trainer.load_training_frame(str(path))[TARGETS].iloc[-1]) == [1, 1, 1, 1]


def test_training_csv_without_targets_is_rejected(tmp_path):
    path = tmp_path / "cycles.csv"
    _cycles(n_cycles=2).drop(columns=TARGETS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Target_Fouling"):
        trainer.load_training_frame(str(path))


def test_split_is_by_cycle_deterministic_and_70_15_15():
    ids = np.arange(100)
    first, second = trainer.split_cycle_ids(ids), trainer.split_cycle_ids(ids)
    assert [list(a) for a in first] == [list(b) for b in second]
    train, val, test = first
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)


def test_too_few_cycles_is_a_clear_error():
    with pytest.raises(ValueError, match="ciclos"):
        trainer.split_cycle_ids(np.arange(3))


def test_tensors_follow_ascending_cycle_id_like_the_original():
    # The original builds tensors from a Cycle_ID-sorted frame; the split order is shuffled.
    df = trainer.engineer_cycle_features(_cycles(n_cycles=3))
    cols = trainer.feature_columns(df)
    _, y = trainer.build_tensors(df, [2, 0, 1], cols, StandardScaler().fit(df[cols]))
    assert y[:, 0].tolist() == [0, 1, 2]


def test_augmented_copies_keep_the_order_of_their_source_cycles():
    df = _cycles(n_cycles=6)
    _, ids = trainer._augment_train_cycles(df, np.array([4, 1, 3]))
    assert list(ids[3:]) == [6, 7, 8]  # copies of 1, 3, 4 in that order


# ── Métricas: mismas fórmulas que la memoria (Tabla 6) ───────────────────────

def test_metrics_follow_the_reported_formulas():
    y_true = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 2, 0]])
    y_pred = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 2, 1]])
    metrics = trainer.compute_test_metrics(y_true, y_pred)
    assert metrics["exact_match"] == pytest.approx(0.75)                  # 3/4 cycles fully right
    assert metrics["accuracy"] == pytest.approx((1 + 1 + 1 + 0.75) / 4)   # mean per component
    # Macro = mean over the 12 component x class values. Only component 4 has errors:
    # class 0 P=1 R=0.5 F1=2/3, class 1 P=0.5 R=1 F1=2/3, class 2 perfect; the other 9 are 1.
    assert metrics["precision_macro"] == pytest.approx((9 + 1 + 0.5 + 1) / 12)
    assert metrics["recall_macro"] == pytest.approx((9 + 0.5 + 1 + 1) / 12)
    assert metrics["f1_macro"] == pytest.approx((9 + 2 / 3 + 2 / 3 + 1) / 12)


# ── Hiperparámetros: los del config.yaml entregado, nunca inventados ─────────

def test_full_training_uses_the_original_optuna_hyperparameters():
    assert constants.TRAIN_HYPERPARAMS == {
        "learning_rate": 0.0022243234786004373, "dropout_rate": 0.20219689010649033,
        "max_lambda": 3.7585293696964697, "epochs": 300, "warmup_epochs": 10,
        "ramp_up_epochs": 80, "patience": 15, "batch_size": 32,
    }
    assert (constants.NOISE_LEVEL, constants.SPLIT_TEST_SIZE_1, constants.SPLIT_TEST_SIZE_2,
            constants.RANDOM_STATE) == (0.2, 0.30, 0.50, 42)


def test_fine_tuning_uses_the_original_calibration_settings():
    assert constants.FINE_TUNE_HYPERPARAMS == {
        "epochs": 50, "patience": 7, "lr_divisor": 10, "batch_size": 32,
    }


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

    # Backbone *parameters* stay frozen. Its BatchNorm running stats (buffers) do move, exactly as
    # in the original: fine_tuner.py calls model.train() on the whole network.
    after = dict(result.model.named_parameters())
    for name in after:
        if name.startswith("features."):
            assert torch.equal(after[name], before[name]), f"backbone weight changed: {name}"
    assert any(not torch.equal(after[n], before[n]) for n in after if n.startswith("head_"))
    assert all(torch.equal(base.state_dict()[k], v) for k, v in before.items()), "base model mutated"
    assert result.scaler is scaler and result.feature_cols == cols
    assert {"exact_match", "accuracy", "f1_macro", "n_train", "n_val", "n_test"} <= set(result.metrics)


def test_full_training_returns_a_new_model_and_hold_out_metrics(csv_path, monkeypatch):
    monkeypatch.setitem(constants.TRAIN_HYPERPARAMS, "epochs", 2)
    result = trainer.train_full(csv_path)
    assert result.feature_cols == trainer.feature_columns(trainer.engineer_cycle_features(_cycles(1)))
    assert (result.metrics["n_train"], result.metrics["n_val"], result.metrics["n_test"]) == (14, 3, 3)
    assert 0.0 <= result.metrics["exact_match"] <= 1.0


def test_full_training_is_reproducible(csv_path, monkeypatch):
    """Same split, seeds and augmentation every run, so the same model *behaviour*.

    Weights are not compared: CPU conv backward differs at float32 rounding level between runs,
    and for the conv biases that sit right before a BatchNorm (true gradient = 0, BN cancels
    them) Adam normalises that rounding noise into lr-sized steps. Those biases have no effect
    on the outputs, which is what this test checks.
    """
    monkeypatch.setitem(constants.TRAIN_HYPERPARAMS, "epochs", 2)
    # On a GPU host cuDNN kernels are non-deterministic (and the trainer deliberately avoids the
    # process-global torch.use_deterministic_algorithms), so two runs differ by ~3e-4. Pin CPU so
    # the test checks the seeding, not the host's hardware.
    monkeypatch.setattr(trainer, "_device", lambda: torch.device("cpu"))
    first, second = trainer.train_full(csv_path), trainer.train_full(csv_path)
    df = trainer.engineer_cycle_features(trainer.load_training_frame(csv_path))
    x, _ = trainer.build_tensors(df, sorted(df["Cycle_ID"].unique()), first.feature_cols, first.scaler)
    first.model.eval()
    second.model.eval()
    with torch.no_grad():
        for a, b in zip(first.model(x), second.model(x)):
            assert torch.allclose(a, b, atol=1e-4)
    assert first.metrics["exact_match"] == second.metrics["exact_match"]


def test_train_request_defaults_to_fine_tune_and_accepts_full():
    from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.train_dto import TrainRequest

    assert TrainRequest(data_path="x.csv", mlflow_run_id="run-1").mode == "fine_tune"
    assert TrainRequest(data_path="x.csv", mlflow_run_id="run-1", mode="full").mode == "full"
    with pytest.raises(ValueError):
        TrainRequest(data_path="x.csv", mlflow_run_id="run-1", mode="whatever")


def test_use_case_forwards_mode_to_the_plugin():
    from app.application.use_cases.train_model_use_case import TrainModelUseCase
    from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.train_dto import TrainRequest

    calls = []

    class Plugin:
        def train(self, *, data_path, mlflow_run_id, mode="fine_tune"):
            calls.append(mode)

    TrainModelUseCase(Plugin()).execute(TrainRequest(data_path="x.csv", mlflow_run_id="run-1", mode="full"))
    TrainModelUseCase(Plugin()).execute(TrainRequest(data_path="x.csv", mlflow_run_id="run-1"))
    assert calls == ["full", "fine_tune"]


# ── Cadena completa: train() del plugin → MLflow → predict con ese run ───────

@pytest.fixture
def plugin_with_mlflow(monkeypatch, tmp_path):
    """Real plugin + real mlflow_utils; only the MLflow server is replaced by a folder per run."""
    import shutil
    from pathlib import Path

    from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection import mlflow_utils, plugin

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
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    instance = plugin.Ml47DairyDnslPasteurizationFaultDetectionPlugin()
    instance.load()
    return instance, base, store, mlflow_utils


@pytest.mark.parametrize("mode", ["fine_tune", "full"])
def test_train_registers_a_new_model_in_mlflow_and_predict_uses_it(plugin_with_mlflow, csv_path, monkeypatch, mode):
    instance, base, store, mlflow_utils = plugin_with_mlflow
    monkeypatch.setitem(constants.FINE_TUNE_HYPERPARAMS, "epochs", 2)
    monkeypatch.setitem(constants.TRAIN_HYPERPARAMS, "epochs", 2)
    base_before = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}

    response = instance.train(data_path=csv_path, mlflow_run_id="run-ft", mode=mode)

    assert response.mode == mode and response.mlflow_run_id == "run-ft"
    uploaded = sorted(p.name for p in (store / "run-ft" / "model").iterdir())
    assert uploaded == sorted([constants.MODEL_FILENAME, constants.SCALER_FILENAME,
                               constants.FEATURE_COLUMNS_FILENAME, constants.TS1_MEAN_FILENAME])

    # What predict(mlflow_run_id="run-ft") loads is the retrained model, not the served one…
    ctx = instance._load_model_for_predict("run-ft")  # pylint: disable=protected-access
    registered = ctx["model"].state_dict()
    assert any(not torch.equal(registered[k].cpu(), base_before[k]) for k in base_before if k.startswith("head_"))
    # …while the served model, used without mlflow_run_id, is untouched.
    assert instance._load_model_for_predict("") is None  # pylint: disable=protected-access
    served = instance._model.state_dict()  # pylint: disable=protected-access
    assert all(torch.equal(served[k].cpu(), v) for k, v in base_before.items())


# ── Inferencia y carga: fidelidad con predictor.py ───────────────────────────

def test_batch_csv_in_raw_bench_format_uses_time_column(tmp_path):
    # data/raw/hydraulic_raw.csv (what main.py predict reads) has "Time" at 100 Hz, no Time_Segundos.
    from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.preprocessing import build_dataframe_from_csv
    raw = _cycles(n_cycles=2, steps=40).rename(columns={"Time_Segundos": "Time"})
    raw["Time"] = raw.groupby("Cycle_ID").cumcount() * 0.01
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    x_df, cycle_ids = build_dataframe_from_csv(str(path), ts1_mean_train=45.0, apply_digital_twin_flag=False)
    assert cycle_ids.value_counts().tolist() == [5, 5]  # 0.00–0.39 s → round(1) → 0.0…0.4
    assert "PS1_rmean" in x_df.columns


def test_loaded_model_uses_the_original_dropout(tmp_path):
    import joblib
    from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.model_loader import load_artifacts_from_dir
    cols = trainer.feature_columns(trainer.engineer_cycle_features(_cycles(n_cycles=1)))
    torch.save(CNN_Pasteurizer(n_sensors=len(cols)).state_dict(), tmp_path / constants.MODEL_FILENAME)
    joblib.dump(StandardScaler().fit(np.zeros((2, len(cols)))), tmp_path / constants.SCALER_FILENAME)
    joblib.dump(cols, tmp_path / constants.FEATURE_COLUMNS_FILENAME)
    joblib.dump(45.0, tmp_path / constants.TS1_MEAN_FILENAME)
    model, *_ = load_artifacts_from_dir(tmp_path)
    # fine_tune trains dropout_final from this model: it must be config.yaml's 0.2022, not 0.5.
    assert model.dropout_final.p == constants.TRAIN_HYPERPARAMS["dropout_rate"]
