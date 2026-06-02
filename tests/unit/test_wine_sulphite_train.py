"""Unit tests for WineSulphitePlugin.train() and model_loader helpers."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.plugins.ml25_wine_sulphites.plugin import WineSulphitePlugin


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def wine_csv(tmp_path: Path) -> Path:
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
    path = tmp_path / "wine.csv"
    df.to_csv(path, index=False)
    return path


def _train(csv_path: Path, artifacts_dir: Path) -> dict:
    """Run WineSulphitePlugin.train() with S3 upload and reload mocked out."""
    plugin = WineSulphitePlugin()
    plugin.load = MagicMock()
    with (
        patch(
            "app.plugins.ml25_wine_sulphites.model_loader.get_artifacts_dir",
            return_value=artifacts_dir,
        ),
        patch("app.plugins.ml25_wine_sulphites.model_loader.upload_artifact"),
    ):
        return plugin.train(data_path=str(csv_path))


# ── train() return value ───────────────────────────────────────────────────────

def test_train_returns_expected_keys(wine_csv, tmp_path):
    result = _train(wine_csv, tmp_path)
    assert result["detail"] == "Training completed"
    for key in ("mae_quality", "mae_bound_so2", "n_train", "n_test", "training_time_s"):
        assert key in result, f"missing key: {key}"


def test_train_split_sums_to_dataset_size(wine_csv, tmp_path):
    result = _train(wine_csv, tmp_path)
    assert result["n_train"] + result["n_test"] == 30


def test_train_mae_values_are_non_negative(wine_csv, tmp_path):
    result = _train(wine_csv, tmp_path)
    assert result["mae_quality"] >= 0
    assert result["mae_bound_so2"] >= 0


# ── Artifact files ─────────────────────────────────────────────────────────────

def test_train_writes_pkl_files(wine_csv, tmp_path):
    _train(wine_csv, tmp_path)
    assert (tmp_path / "quality_rf.pkl").exists()
    assert (tmp_path / "bound_rf.pkl").exists()


def test_train_writes_metadata_json(wine_csv, tmp_path):
    _train(wine_csv, tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert "metrics" in metadata
    assert "quality_cv" in metadata["metrics"]
    assert "bound_cv" in metadata["metrics"]
    assert "mae_mean" in metadata["metrics"]["quality_cv"]


# ── S3 upload ─────────────────────────────────────────────────────────────────

def test_train_uploads_all_three_artifacts(wine_csv, tmp_path):
    uploaded: list[str] = []
    plugin = WineSulphitePlugin()
    plugin.load = MagicMock()
    with (
        patch(
            "app.plugins.ml25_wine_sulphites.model_loader.get_artifacts_dir",
            return_value=tmp_path,
        ),
        patch(
            "app.plugins.ml25_wine_sulphites.model_loader.upload_artifact",
            side_effect=uploaded.append,
        ),
    ):
        plugin.train(data_path=str(wine_csv))

    assert set(uploaded) == {"quality_rf.pkl", "bound_rf.pkl", "metadata.json"}


# ── Hot reload ────────────────────────────────────────────────────────────────

def test_train_calls_load_after_saving(wine_csv, tmp_path):
    plugin = WineSulphitePlugin()
    mock_load = MagicMock()
    plugin.load = mock_load

    with (
        patch(
            "app.plugins.ml25_wine_sulphites.model_loader.get_artifacts_dir",
            return_value=tmp_path,
        ),
        patch("app.plugins.ml25_wine_sulphites.model_loader.upload_artifact"),
    ):
        plugin.train(data_path=str(wine_csv))

    mock_load.assert_called_once()


# ── model_loader helpers ──────────────────────────────────────────────────────

def test_get_artifacts_dir_points_to_wine_sulphite():
    from app.plugins.ml25_wine_sulphites.model_loader import get_artifacts_dir
    assert get_artifacts_dir().name == "wine_sulphite"


def test_upload_artifact_delegates_to_store():
    from app.plugins.ml25_wine_sulphites.model_loader import upload_artifact
    mock_store = MagicMock()
    with patch("app.plugins.ml25_wine_sulphites.model_loader._store", mock_store):
        upload_artifact("model.pkl")
    mock_store.upload.assert_called_once_with("model.pkl")
