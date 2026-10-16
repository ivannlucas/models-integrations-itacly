"""Tests for ml36's real retraining code (not the FakePlugin wiring).

training.py ports the AI team's split_normalization.py + trainer.py. On the delivered
data/processed/df_pasteurizacion_master.csv the split and the scalers equal the delivered ones
and the retrained model matches final_test_metrics.json; that test needs inbox/a36/codigo (not
committed) and is skipped without it. The rest runs on small synthetic cycles.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import ModelPersistenceError
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer import plugin as plugin_mod
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer import training
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.constants import (
    FEATURE_ORDER, MODEL_CONFIG_FILENAME, MODEL_FILENAME, SCALER_X_FILENAME, SCALER_Y_FILENAME, TARGETS,
    TRAIN_ARCHITECTURE, TRAIN_SEED, TRAIN_SEED_OFFSET,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.preprocessing import add_derived_features

CODE = Path(__file__).resolve().parents[2] / "inbox" / "a36" / "codigo"


def _cycles(n_cycles=12, steps=20, seed=0, with_cycle_id=True) -> pd.DataFrame:
    """Cycles shaped like df_pasteurizacion_master.csv: t_ciclo restarts at 5 every cycle."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(n_cycles):
        for t in range(steps):
            t_serv, regen = rng.uniform(75, 84), rng.uniform(87, 93)
            row = {"T_serv": t_serv, "F_milk": rng.uniform(4500, 5800), "T_in": rng.uniform(3.5, 7.5),
                   "Fat_perc": rng.uniform(3.2, 4.2), "Viscosity": rng.uniform(1.9, 2.2), "Regeneration_perc": regen,
                   "Delta_P": rng.uniform(0.6, 1.6), "t_ciclo": 5.0 * (t + 1)}
            row["T_out"] = t_serv - 1.3 + rng.normal(0, 0.02)
            row["CO2_emissions"] = 20 + 0.8 * (t_serv - 78) - 0.9 * (regen - 90) + rng.normal(0, 0.3)
            if with_cycle_id:
                row["cycle_id"] = cid
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_keeps_whole_cycles_and_is_70_15_15_by_cycle():
    df = training.prepare_frame(_cycles(n_cycles=20))
    train, val, test = training.split_frame(df)
    ids = [set(part["cycle_id"]) for part in (train, val, test)]
    assert [len(i) for i in ids] == [14, 3, 3]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_cycles_are_rebuilt_from_t_ciclo_resets_without_cycle_id():
    df = _cycles(n_cycles=4, steps=3, with_cycle_id=False)
    assert training.detect_cycle_ids(df).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_split_is_balanced_by_season_when_a_month_column_exists():
    df = training.prepare_frame(_cycles(n_cycles=20))
    df["month"] = np.where(df["cycle_id"] < 10, 1, 7)  # 10 winter cycles + 10 summer cycles
    train, val, test = training.split_frame(df)
    for part, expected in ((train, 7), (val, 1), (test, 2)):
        months = part.groupby("cycle_id")["month"].first()
        assert (months == 1).sum() == expected and (months == 7).sum() == expected


def test_derived_features_are_completed_with_the_original_formulas():
    df = _cycles(n_cycles=3)
    out = training.prepare_frame(df)
    assert np.allclose(out[FEATURE_ORDER].to_numpy(), add_derived_features(df)[FEATURE_ORDER].to_numpy())


@pytest.mark.parametrize("bad,match", [("no_target", "CO2_emissions"), ("two_cycles", "al menos 3 ciclos")])
def test_invalid_training_csv_is_a_clear_error(bad, match):
    df = _cycles(n_cycles=2) if bad == "two_cycles" else _cycles().drop(columns=["CO2_emissions"])
    with pytest.raises(ValueError, match=match):
        training.train_model(df)


def test_train_model_fits_new_scalers_and_a_new_mlp_with_the_original_config(monkeypatch):
    monkeypatch.setattr(training, "TRAIN_EPOCHS", 3)
    result = training.train_model(_cycles())
    train, _, _ = training.split_frame(training.prepare_frame(_cycles()))
    assert np.allclose(result["scaler_x"].mean_, train[FEATURE_ORDER].mean().to_numpy())
    assert np.allclose(result["scaler_y"].data_max_, train[TARGETS].max().to_numpy())
    assert {k: result["config"][k] for k in TRAIN_ARCHITECTURE} == TRAIN_ARCHITECTURE
    assert result["epochs_run"] == 3 and set(result["metrics"]) == set(TARGETS)
    assert TRAIN_SEED + TRAIN_SEED_OFFSET == 501


def test_training_is_reproducible(monkeypatch):
    monkeypatch.setattr(training, "TRAIN_EPOCHS", 3)
    first, second = training.train_model(_cycles()), training.train_model(_cycles())
    for key, value in first["model"].state_dict().items():
        assert np.array_equal(value.numpy(), second["model"].state_dict()[key].numpy())


def test_retraining_reproduces_the_delivered_split_scalers_and_metrics():
    master_csv = CODE / "data" / "processed" / "df_pasteurizacion_master.csv"
    if not master_csv.exists():
        pytest.skip("inbox/a36/codigo not present")
    master = pd.read_csv(master_csv)
    train, val, test = training.split_frame(training.prepare_frame(master))
    for name, part in (("train", train), ("val", val), ("test", test)):
        delivered = pd.read_csv(CODE / "data" / "splits" / f"{name}_raw.csv")
        assert np.allclose(part[FEATURE_ORDER + TARGETS].to_numpy(), delivered[FEATURE_ORDER + TARGETS].to_numpy(),
                           rtol=0, atol=1e-12)

    result = training.train_model(master)
    scaler_x = joblib.load(CODE / "models" / "artifacts" / SCALER_X_FILENAME)
    scaler_y = joblib.load(CODE / "models" / "artifacts" / SCALER_Y_FILENAME)
    assert np.allclose(result["scaler_x"].mean_, scaler_x.mean_, rtol=0, atol=1e-12)
    assert np.array_equal(result["scaler_y"].data_max_, scaler_y.data_max_)
    history = json.loads((CODE / "models" / "metrics" / "final_training_history.json").read_text())
    assert result["epochs_run"] == len(history["val_loss"])
    expected = json.loads((CODE / "models" / "metrics" / "final_test_metrics.json").read_text())
    for target in TARGETS:
        for key in ("mae", "r2"):  # equal up to float32 differences across torch/BLAS builds
            assert result["metrics"][target][key] == pytest.approx(expected[target][key], rel=1e-3)


# ── plugin.train: MLflow only ──────────────────────────────────────────────────

@pytest.fixture
def instance():
    return plugin_mod.Ml36DairyDnlCo2EmissionsOptimizerPlugin()


def test_train_uploads_the_new_model_and_its_scalers_to_mlflow(instance, monkeypatch, tmp_path):
    monkeypatch.setattr(training, "TRAIN_EPOCHS", 2)
    csv = tmp_path / "cycles.csv"
    _cycles().to_csv(csv, index=False)
    store = tmp_path / "mlflow"
    tracker = MagicMock()
    tracker.upload_artifacts.side_effect = lambda d, artifact_path: shutil.copytree(d, store / artifact_path)
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))

    resp = instance.train(data_path=str(csv), mlflow_run_id="run-1")
    assert (resp.n_train, resp.n_val, resp.n_test, resp.epochs_executed) == (160, 40, 40, 2)  # 12 cycles -> 8/2/2
    assert sorted(p.name for p in (store / "model").iterdir()) == sorted(
        [MODEL_FILENAME, MODEL_CONFIG_FILENAME, SCALER_X_FILENAME, SCALER_Y_FILENAME])
    uploaded_scaler = joblib.load(store / "model" / SCALER_X_FILENAME)
    assert list(uploaded_scaler.feature_names_in_) == FEATURE_ORDER


def test_train_fails_when_mlflow_upload_fails(instance, monkeypatch, tmp_path):
    monkeypatch.setattr(training, "TRAIN_EPOCHS", 1)
    csv = tmp_path / "cycles.csv"
    _cycles().to_csv(csv, index=False)
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    with pytest.raises(ModelPersistenceError, match="mlflow caído"):
        instance.train(data_path=str(csv), mlflow_run_id="run-1")
