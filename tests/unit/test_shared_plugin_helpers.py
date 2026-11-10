"""Shared plugin helpers: mlflow_artifacts, PluginRuntimeMixin, batch_rows, torch_device, DTOs."""
# pylint: disable=protected-access  # the mixin's API is the plugins' private helpers
import base64
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from app.application.dto import plugin_dto
from app.application.dto.stats_dto import RuntimeStats, StatsResponse
from app.domain.services import image_cam, mlflow_artifacts, torch_device
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.domain.services.xai_values import xai_feature_values


class _FakeTracker:
    """BaseMLflowTracker stand-in recording uploads and serving a fixed download folder."""

    uploads: list = []
    download_result = ""
    fail_upload = False
    fail_read = False

    def __init__(self, run_id=""):
        self.run_id = run_id

    def download_artifacts(self, dest_dir, artifact_path=""):
        """Copy nothing; return the configured result ('' = no artifacts, else dest_dir)."""
        _ = artifact_path
        if self.download_result:
            Path(dest_dir, "model.bin").write_text("weights", encoding="utf-8")
            return dest_dir
        return ""

    def upload_artifacts(self, local_dir, artifact_path=""):
        """Record the uploaded files, or fail if configured to."""
        if self.fail_upload:
            raise RuntimeError("s3 down")
        _FakeTracker.uploads.append((sorted(os.listdir(local_dir)), artifact_path))

    def get_params(self):
        """Return fixed params (or fail if configured to)."""
        if self.fail_read:
            raise RuntimeError("mlflow down")
        return {"epochs": "3"}

    def get_metrics(self):
        """Return fixed metrics."""
        return {"mae": 1.5}


@pytest.fixture(name="tracker", autouse=True)
def fixture_tracker(monkeypatch):
    """Patch the shared module's tracker and track the temp dirs it creates."""
    _FakeTracker.uploads = []
    _FakeTracker.download_result = ""
    _FakeTracker.fail_upload = False
    _FakeTracker.fail_read = False
    monkeypatch.setattr(mlflow_artifacts, "BaseMLflowTracker", _FakeTracker)
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)
    _FakeTracker.created = created
    return _FakeTracker


def _leaked(tracker):
    return [p for p in tracker.created if Path(p).exists()]


def test_download_returns_loaded_model_and_temp_dir(tracker):
    """With artifacts, the loader's result comes back with the temp dir the caller must remove."""
    tracker.download_result = "yes"
    loaded, tmp = mlflow_artifacts.download_run_artifacts(
        "run-1", lambda path: Path(path, "model.bin").read_text(encoding="utf-8"), prefix="t_"
    )
    assert loaded == "weights"
    assert Path(tmp).name.startswith("t_")
    with mlflow_artifacts.removing_dir(tmp):
        assert Path(tmp).exists()
    assert not Path(tmp).exists()


@pytest.mark.parametrize("has_artifacts", [False, True], ids=["no_artifacts", "loader_none"])
def test_download_returns_none_and_cleans_up(tracker, has_artifacts):
    """No artifacts, or a loader that finds an incomplete model → None and no temp dir left."""
    tracker.download_result = "yes" if has_artifacts else ""
    assert mlflow_artifacts.download_run_artifacts("run-1", lambda _p: None, prefix="t_") is None
    assert tracker.created and not _leaked(tracker)


def test_download_cleans_up_when_loader_raises(tracker):
    """A loader error propagates unchanged and the temp dir is removed."""
    tracker.download_result = "yes"

    def broken(_path):
        raise KeyError("model_state_dict")

    with pytest.raises(KeyError):
        mlflow_artifacts.download_run_artifacts("run-1", broken, prefix="t_")
    assert not _leaked(tracker)


def test_removing_dir_removes_on_error_and_ignores_none(tmp_path):
    """removing_dir deletes the folder even if the block fails; None is a no-op."""
    target = tmp_path / "user_model"
    target.mkdir()
    with pytest.raises(ValueError):
        with mlflow_artifacts.removing_dir(str(target)):
            raise ValueError("predict failed")
    assert not target.exists()
    with mlflow_artifacts.removing_dir(None):
        pass


def test_require_train_tracker():
    """Without a run the retrain cannot be saved; with one the tracker is returned."""
    with pytest.raises(ModelPersistenceError, match="Sin run de MLflow"):
        mlflow_artifacts.require_train_tracker("")
    assert mlflow_artifacts.require_train_tracker("run-1").run_id == "run-1"


def test_upload_model_writes_uploads_and_cleans(tracker):
    """The writer's files are uploaded under artifact_path and the temp dir is removed."""
    mlflow_artifacts.upload_model_to_mlflow(
        tracker("run-1"),
        lambda d: Path(d, "model.pt").write_text("x", encoding="utf-8"),
        prefix="u_",
        artifact_path="classifier",
    )
    assert tracker.uploads == [(["model.pt"], "classifier")]
    assert not _leaked(tracker)


@pytest.mark.parametrize("where", ["write", "upload"])
def test_upload_failure_is_a_persistence_error(tracker, where):
    """Any failure writing or uploading → ModelPersistenceError with the plugin's subject."""
    tracker.fail_upload = where == "upload"

    def write(_d):
        if where == "write":
            raise OSError("disk full")

    with pytest.raises(ModelPersistenceError) as excinfo:
        mlflow_artifacts.upload_model_to_mlflow(
            tracker("run-1"), write, prefix="u_", subject="El ensemble"
        )
    assert str(excinfo.value).startswith(
        "El ensemble reentrenado no se ha podido guardar en MLflow: "
    )
    assert not _leaked(tracker)


def test_upload_dir_maps_failure(tracker, tmp_path):
    """upload_dir_to_mlflow uploads an existing folder and maps failures, keeping the folder."""
    (tmp_path / "bundle.pkl").write_text("x", encoding="utf-8")
    mlflow_artifacts.upload_dir_to_mlflow(tracker("run-1"), str(tmp_path))
    assert tracker.uploads == [(["bundle.pkl"], "model")]
    tracker.fail_upload = True
    with pytest.raises(ModelPersistenceError, match="El modelo reentrenado no se ha podido"):
        mlflow_artifacts.upload_dir_to_mlflow(tracker("run-1"), str(tmp_path))
    assert tmp_path.exists()


def _stats():
    return StatsResponse(
        model_name="m", version="1", description="d", task_type="t", framework="f",
        inputs=[], outputs=[], metrics={"r2": 0.9},
        runtime_stats=RuntimeStats(total_predictions=0),
    )


def test_add_mlflow_run_to_stats(tracker):
    """The run's params/metrics go under metrics['mlflow']; MLflow errors are swallowed."""
    assert "mlflow" not in mlflow_artifacts.add_mlflow_run_to_stats(_stats(), "").metrics
    enriched = mlflow_artifacts.add_mlflow_run_to_stats(_stats(), "run-1")
    assert enriched.metrics == {
        "r2": 0.9, "mlflow": {"params": {"epochs": "3"}, "metrics": {"mae": 1.5}},
    }
    tracker.fail_read = True
    assert mlflow_artifacts.add_mlflow_run_to_stats(_stats(), "run-1").metrics == {"r2": 0.9}


class _Plugin(PluginRuntimeMixin):
    """Minimal plugin using the mixin."""

    def __init__(self):
        self.model = None

    def is_loaded(self):
        """Loaded once a model is set."""
        return self.model is not None


def test_plugin_runtime_mixin():
    """_require_loaded follows is_loaded(); _record counts per instance."""
    plugin, other = _Plugin(), _Plugin()
    with pytest.raises(ModelNotLoadedError, match="El modelo no está cargado."):
        plugin._require_loaded()
    plugin.model = object()
    plugin._require_loaded()
    plugin._record()
    plugin._record()
    assert plugin._runtime_stats() == RuntimeStats(
        total_predictions=2, avg_latency_ms=None
    )
    assert plugin.predict_count == 2 and plugin.last_predict_at
    assert other.predict_count == 0 and other.last_predict_at is None


def test_row_error_capture_appends_error_row_and_goes_on():
    """A failing row becomes {**error_row, "error": msg}; good rows are untouched."""
    rows = []
    for idx, value in enumerate([1, 0, 2]):
        with row_error_capture(rows, {"row": idx}, what=f"fila {idx}"):
            rows.append({"row": idx, "inv": 1 / value})
    assert rows == [
        {"row": 0, "inv": 1.0},
        {"row": 1, "error": "division by zero"},
        {"row": 2, "inv": 0.5},
    ]


def test_safe_torch_device(monkeypatch):
    """CPU without CUDA or when the CUDA probe fails; CUDA only when the probe runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert torch_device.safe_torch_device() == torch.device("cpu")

    def broken_cuda(self, *args, **kwargs):
        _ = self, args, kwargs
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.nn.Module, "cuda", broken_cuda)
    assert torch_device.safe_torch_device() == torch.device("cpu")


def test_plugin_dto_bases():
    """The bases carry the shared leading fields, in order, with the shared defaults."""
    assert list(plugin_dto.BatchRequestBase.model_fields) == ["mode", "data_path", "mlflow_run_id"]
    assert list(plugin_dto.InlineRequestBase.model_fields) == [
        "mode", "model_key", "threshold", "mlflow_run_id",
    ]
    response = plugin_dto.BatchResponseBase(model_id="m", predictions=[{"a": 1}])
    assert response.model_dump() == {
        "model_id": "m", "predictions": [{"a": 1}], "output_path": None,
    }
    with pytest.raises(ValidationError):
        plugin_dto.TrainRequestBase(data_path="x.csv", mlflow_run_id="  ")


def test_latency_tracking():
    """_record_latency counts the prediction; _avg_latency_ms rounds as asked."""
    plugin = _Plugin()
    assert plugin._avg_latency_ms(1) is None
    plugin._record_latency(10.0)
    plugin._record_latency(5.25)
    assert plugin.predict_count == 2
    assert plugin._avg_latency_ms() == 7.625
    assert plugin._avg_latency_ms(1) == 7.6
    assert _Plugin()._avg_latency_ms() is None


def test_xai_feature_values():
    """Numeric, non-NaN values of the requested columns, from a dict or a Series."""
    row = {"a": "1.5", "b": None, "c": float("nan"), "d": "x", "e": 3, "f": "nan"}
    assert xai_feature_values(row, ["a", "b", "c", "d", "e", "f", "missing"]) == {
        "a": 1.5, "e": 3.0,
    }
    assert xai_feature_values(pd.Series(row), ["a", "e"]) == {"a": 1.5, "e": 3.0}
    assert not xai_feature_values({}, ["a"])


def test_compute_and_encode_cam():
    """CAM overlay is a JPEG data URI, downscaled to at most MAX_HEATMAP_SIZE."""
    feature_map = torch.rand(1, 4, 7, 7)
    weight = torch.rand(3, 4)
    uri = image_cam.compute_and_encode_cam(feature_map, weight, 1, Image.new("RGB", (1280, 640)))
    assert uri.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.size == (image_cam.MAX_HEATMAP_SIZE, 320)
    cam = image_cam.class_activation_map(feature_map, weight, 1)
    assert cam.shape == (7, 7) and 0.0 <= cam.min() and cam.max() <= 1.0
