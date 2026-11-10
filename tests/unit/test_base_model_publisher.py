"""persist_retrained_bundle: MLflow (optional, failure = warning) + publish as the new base."""
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.domain.services.exceptions import ModelPersistenceError
from app.infrastructure import artifact_store, base_model_publisher
from app.infrastructure.base_model_publisher import persist_retrained_bundle


@pytest.fixture(name="bundle")
def fixture_bundle(tmp_path):
    """Local bundle folder with a single model.pkl; returns its path as str."""
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "model.pkl").write_bytes(b"new")
    return str(d)


def test_the_test_store_never_points_at_the_repo_artifacts(isolated_base_store):
    """In tests the base store lives in the isolated folder, never in the repo artifacts."""
    store = base_model_publisher._store_for("m")  # pylint: disable=protected-access
    assert artifact_store.ARTIFACTS_ROOT not in store.local_dir.parents
    assert store.local_dir == isolated_base_store / "m"


def test_publishes_without_mlflow_run(bundle, isolated_base_store):
    """Without a run the bundle is published as v1 and MLflow is not called."""
    upload = MagicMock()
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv",
                                      metrics={"mae": 2.0}, upload_to_mlflow=upload)
    upload.assert_not_called()
    assert result.upload_warning is None
    assert result.base_version["version"] == "v1" and result.base_version["mlflow_run_id"] is None
    assert result.base_version["data_path"] == "d.csv"
    assert (isolated_base_store / "m" / "versions" / "v1" / "model.pkl").read_bytes() == b"new"


def test_with_run_uploads_to_mlflow_and_records_the_run(bundle):
    """With a run the bundle is uploaded and the run id recorded in the version."""
    upload = MagicMock()
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={},
                                      mlflow_run_id="run-1", upload_to_mlflow=upload)
    upload.assert_called_once_with()
    assert result.base_version["mlflow_run_id"] == "run-1" and result.upload_warning is None


def test_mlflow_failure_is_a_warning_and_the_base_is_still_published(bundle):
    """An MLflow failure becomes a warning and the base is still published."""
    upload = MagicMock(side_effect=ConnectionError("mlflow caído"))
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={},
                                      mlflow_run_id="run-1", upload_to_mlflow=upload)
    assert "mlflow caído" in result.upload_warning and "run-1" in result.upload_warning
    assert result.base_version["version"] == "v1"


def test_publish_failure_is_a_model_persistence_error(bundle, monkeypatch):
    """A failure publishing the base raises ModelPersistenceError."""
    store = MagicMock(publish_version=MagicMock(side_effect=OSError("s3 caído")))
    monkeypatch.setattr(base_model_publisher, "_store_for", lambda folder: store)
    with pytest.raises(ModelPersistenceError, match="s3 caído"):
        persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={})


def test_non_finite_metrics_become_null(bundle, isolated_base_store):
    """NaN/inf metrics are stored as null and non-scalar metrics are dropped."""
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv",
                                      metrics={"auc_h1": float("nan"), "mae": 1.0,
                                               "inf": float("inf"), "n": 3, "label": "x",
                                               "nested": {"a": 1}})
    assert result.base_version["metrics"] == {
        "auc_h1": None, "mae": 1.0, "inf": None, "n": 3, "label": "x",
    }
    raw = (isolated_base_store / "m" / "versions" / "v1" / "version.json").read_text()
    assert "NaN" not in raw and json.loads(raw)["metrics"]["auc_h1"] is None


def test_numpy_scalar_metrics_are_kept(bundle):
    """numpy scalar metrics are converted to plain numbers."""
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv",
                                      metrics={"f": np.float32(1.5), "i": np.int64(4),
                                               "n": np.float64("nan")})
    assert result.base_version["metrics"] == {"f": 1.5, "i": 4, "n": None}


def test_blank_run_id_counts_as_no_run(bundle):
    """A blank run id is treated as no run."""
    upload = MagicMock()
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={},
                                      mlflow_run_id="   ", upload_to_mlflow=upload)
    upload.assert_not_called()
    assert result.base_version["mlflow_run_id"] is None and result.upload_warning is None
