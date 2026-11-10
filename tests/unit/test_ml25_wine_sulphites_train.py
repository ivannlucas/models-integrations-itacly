"""Unit tests for Ml25WineSulphitesPlugin.train() and model_loader helpers."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.plugins.ml25_wine_sulphites.mlflow_utils import download_user_predictor_from_mlflow
from app.plugins.ml25_wine_sulphites.model_loader import get_artifacts_dir
from app.plugins.ml25_wine_sulphites.plugin import Ml25WineSulphitesPlugin


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_wine_csv(folder: Path) -> Path:
    """Minimal wine-quality CSV (30 rows) sufficient for RF training."""
    rng = np.random.default_rng(42)
    n = 30
    free_so2 = rng.uniform(10, 50, n)
    df = pd.DataFrame({
        "fixed acidity": rng.uniform(6, 9, n),
        "volatile acidity": rng.uniform(0.2, 0.8, n),
        "citric acid": rng.uniform(0, 0.5, n),
        "residual sugar": rng.uniform(1, 10, n),
        "chlorides": rng.uniform(0.05, 0.1, n),
        "density": rng.uniform(0.99, 1.00, n),
        "pH": rng.uniform(3.0, 3.8, n),
        "sulphates": rng.uniform(0.4, 0.8, n),
        "alcohol": rng.uniform(9, 13, n),
        "free sulfur dioxide": free_so2,
        "total sulfur dioxide": free_so2 + rng.uniform(20, 100, n),
        "quality": rng.integers(4, 9, n).astype(float),
    })
    path = folder / "wine.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(name="wine_csv")
def fixture_wine_csv(tmp_path: Path) -> Path:
    """Minimal wine-quality CSV (30 rows) sufficient for RF training."""
    return _write_wine_csv(tmp_path)


def _train(csv_path: Path, artifacts_dir: Path) -> dict:
    """Run Ml25WineSulphitesPlugin.train(). mlflow_run_id is mandatory (every retrain must be
    persisted to a specific MLflow run) — use a dummy id since these tests aren't about that."""
    plugin = Ml25WineSulphitesPlugin()
    plugin.load = MagicMock()
    with patch(
        "app.plugins.ml25_wine_sulphites.model_loader.get_artifacts_dir",
        return_value=artifacts_dir,
    ), patch("app.domain.services.mlflow_artifacts.BaseMLflowTracker"):
        return plugin.train(data_path=str(csv_path), mlflow_run_id="test-run-id")


# ── mlflow_run_id en la respuesta ─────────────────────────────────────────────

def test_train_echoes_mlflow_run_id(wine_csv, tmp_path):
    """La respuesta devuelve el run de MLflow al que se subieron los artefactos.

    La plataforma lo toma como autoritativo (train-task-manager.ts,
    confirmedRunId): sin el, si el run que pre-creo ella fallo, el run que de
    verdad tiene el modelo queda huerfano y predict sigue resolviendo al
    modelo servido en vez de al reentrenado.
    """
    plugin = Ml25WineSulphitesPlugin()
    plugin.load = MagicMock()
    with patch(
        "app.plugins.ml25_wine_sulphites.model_loader.get_artifacts_dir",
        return_value=tmp_path,
    ), patch("app.domain.services.mlflow_artifacts.BaseMLflowTracker") as tracker_cls:
        result = plugin.train(data_path=str(wine_csv), mlflow_run_id="run-abc123")

    assert result.mlflow_run_id == "run-abc123"
    tracker_cls.assert_called_once_with("run-abc123")


# ── contrato de columnas del CSV de entrenamiento ─────────────────────────────

@pytest.fixture(name="wine_csv_underscores")
def fixture_wine_csv_underscores(wine_csv: Path, tmp_path: Path) -> Path:
    """El mismo CSV con los nombres de columna que documenta la plataforma.

    El esquema que la UI le ensena al usuario para modelo-25 usa guiones bajos
    (fixed_acidity, free_sulfur_dioxide...), mientras que los artefactos y las
    rutas de prediccion usan la convencion con espacios de FEATURES_QUAL.
    """
    df = pd.read_csv(wine_csv)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    path = tmp_path / "wine_underscores.csv"
    df.to_csv(path, index=False)
    return path


def test_train_accepts_platform_column_names(wine_csv_underscores, tmp_path):
    """train() acepta el CSV con los nombres que documenta la plataforma."""
    result = _train(wine_csv_underscores, tmp_path)
    assert result.mae_quality >= 0
    assert result.n_train + result.n_test == 30


def test_train_accepts_canonical_column_names(trained):
    """train() sigue aceptando la convencion con espacios de los artefactos."""
    assert trained.result.mae_quality >= 0


def test_train_missing_target_raises_value_error(wine_csv, tmp_path):
    """Sin la columna 'quality' el error es un ValueError que la nombra (400, no 500)."""
    df = pd.read_csv(wine_csv).drop(columns=["quality"])
    path = tmp_path / "sin_target.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="quality"):
        _train(path, tmp_path)


def test_train_missing_feature_raises_value_error(wine_csv, tmp_path):
    """Si falta una feature, el error la nombra en vez de reventar con KeyError."""
    df = pd.read_csv(wine_csv).drop(columns=["alcohol"])
    path = tmp_path / "sin_feature.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="alcohol"):
        _train(path, tmp_path)


# ── train() return value ───────────────────────────────────────────────────────

def test_train_returns_expected_keys(trained):
    """train() result contains all documented top-level keys."""
    result = trained.result
    assert result.detail == "Training completed"
    for key in ("mae_quality", "mae_bound_so2", "n_train", "n_test", "training_time_s"):
        assert hasattr(result, key), f"missing key: {key}"


def test_train_split_sums_to_dataset_size(trained):
    """n_train + n_test equals the total number of rows in the input CSV."""
    result = trained.result
    assert result.n_train + result.n_test == 30


def test_train_mae_values_are_non_negative(trained):
    """MAE values for quality and bound SO2 models are non-negative after training."""
    result = trained.result
    assert result.mae_quality >= 0
    assert result.mae_bound_so2 >= 0


# ── Artifact files: the retrained model lives only in its MLflow run ──────────

def _train_capturing_upload(csv_path: Path, artifacts_dir: Path):
    """Run train() with MLflow mocked; return (plugin.load mock, uploaded file contents)."""
    plugin = Ml25WineSulphitesPlugin()
    plugin.load = MagicMock()
    uploaded: dict[str, bytes] = {}

    def _capture(local_dir, **_kwargs):
        for f in Path(local_dir).iterdir():
            uploaded[f.name] = f.read_bytes()

    with patch(
        "app.plugins.ml25_wine_sulphites.model_loader.get_artifacts_dir",
        return_value=artifacts_dir,
    ), patch("app.domain.services.mlflow_artifacts.BaseMLflowTracker") as tracker_cls:
        tracker_cls.return_value.upload_artifacts.side_effect = _capture
        result = plugin.train(data_path=str(csv_path), mlflow_run_id="run-abc123")
    return plugin.load, uploaded, result


@pytest.fixture(name="trained", scope="module")
def fixture_trained(tmp_path_factory) -> SimpleNamespace:
    """One real train() on the canonical CSV, shared by the tests that only inspect its outcome.

    Training the two RFs takes ~6 s; these tests read the same response, uploaded files and
    base folder, so they share a single run instead of retraining each time.
    """
    folder = tmp_path_factory.mktemp("ml25_train")
    base_dir = folder / "base"
    base_dir.mkdir()
    load_mock, uploaded, result = _train_capturing_upload(_write_wine_csv(folder), base_dir)
    return SimpleNamespace(load=load_mock, uploaded=uploaded, result=result, base_dir=base_dir)


def test_train_never_writes_to_base_artifacts_dir(trained):
    """A retrain must not overwrite the served base model's local artifacts."""
    assert not list(trained.base_dir.iterdir())


def test_train_uploads_models_and_metadata_to_mlflow(trained):
    """The retrained RFs and metadata.json (with metrics) go to the MLflow run."""
    assert {"quality_rf.pkl", "bound_rf.pkl", "metadata.json"} <= set(trained.uploaded)
    metadata = json.loads(trained.uploaded["metadata.json"])
    assert "mae_mean" in metadata["metrics"]["quality_cv"]
    assert "bound_cv" in metadata["metrics"]
    assert trained.result.upload_warning is None


def test_train_does_not_reload_base_model(trained):
    """The served base model is not reloaded or replaced after a retrain."""
    trained.load.assert_not_called()


def test_retrained_model_round_trips_through_mlflow(trained):
    """What train() uploads to the run is what predict loads back from it.

    train() writes the RFs with joblib.dump; the loader used pickle.load, which cannot read
    them ("UnpicklingError: STACK_GLOBAL requires str"), so every prediction with a retrained
    modelo-25 failed. The other tests mock the whole download and never caught it.
    """
    uploaded = trained.uploaded

    def _download(dest_dir, artifact_path):
        model_dir = Path(dest_dir) / artifact_path
        model_dir.mkdir(parents=True)
        for name, content in uploaded.items():
            (model_dir / name).write_bytes(content)
        return str(model_dir)

    with patch("app.domain.services.mlflow_artifacts.BaseMLflowTracker") as tracker_cls:
        tracker_cls.return_value.download_artifacts.side_effect = _download
        model_qual, model_bound, metadata, temp_dir = download_user_predictor_from_mlflow(
            "run-abc123"
        )

    try:
        for model in (model_qual, model_bound):
            assert model.predict(np.zeros((1, model.n_features_in_))).shape == (1,)
        assert "quality_cv" in metadata["metrics"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── model_loader helpers ──────────────────────────────────────────────────────

def test_get_artifacts_dir_points_to_ml25_wine_sulphites():
    """get_artifacts_dir() returns a path whose final component is 'ml25_wine_sulphites'."""
    assert get_artifacts_dir().name == "ml25_wine_sulphites"


# ── Training duration ─────────────────────────────────────────────────────────

def test_train_time_is_non_negative(trained):
    """training_time_s is a non-negative float."""
    assert trained.result.training_time_s >= 0.0
