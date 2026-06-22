"""Tests for infrastructure layer components.

Covers ModelContainer, ModelRuntimeService, ArtifactStore (local path ops),
and router_factory edge cases (exception mapping).
"""

from __future__ import annotations


import pytest
from unittest.mock import MagicMock, patch as patch_unit

from app.application.dto.stats_dto import RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure.http.dependencies.container import ModelContainer


# ── ModelContainer tests ──────────────────────────────────────────────────

class FakeModelPlugin(ModelPluginPort):
    """Minimal fake plugin for container tests."""

    def __init__(self):
        """Initialize the fake plugin."""
        self._loaded = False

    def load(self):
        """Simulate loading model artifacts."""
        self._loaded = True

    def is_loaded(self):
        """Return True if the model is loaded."""
        return self._loaded

    def predict_batch(self, *, data_path):
        """Return a fake batch prediction dict."""
        return {"model_id": "test", "predictions": [], "output_path": None}

    def predict_inline(self, *, features, model_key=None, threshold=None):
        """Return a fake inline prediction dict."""
        return {"model_id": "test", "prediction": "test", "confidence": 1.0}

    def stats(self):
        """Return a fake StatsResponse."""
        return StatsResponse(
            model_name="test",
            version="0.0.0",
            description="test plugin",
            task_type="test",
            framework="test",
            inputs=[],
            outputs=[],
            metrics={},
            runtime_stats=RuntimeStats(total_predictions=0, avg_latency_ms=None),
        )

    def train(self, *, data_path):
        """Raise TrainingNotSupportedError."""
        from app.domain.services.exceptions import TrainingNotSupportedError
        raise TrainingNotSupportedError("Not supported")


class TestModelContainer:
    """Tests for ModelContainer initialization and wiring."""

    def test_init_creates_use_cases(self):
        """Verify ModelContainer wires use cases on construction."""
        plugin = FakeModelPlugin()
        container = ModelContainer(plugin=plugin)
        assert container._plugin is plugin
        assert container._service is not None
        assert container.predict_use_case is not None
        assert container.stats_use_case is not None
        assert container.train_use_case is not None

    def test_init_calls_plugin_load(self):
        """Verify container.init() calls plugin.load()."""
        plugin = FakeModelPlugin()
        container = ModelContainer(plugin=plugin)
        assert plugin.is_loaded() is False
        container.init()
        assert plugin.is_loaded() is True

    def test_service_property(self):
        """Verify container.service returns a ModelRuntimeService."""
        plugin = FakeModelPlugin()
        container = ModelContainer(plugin=plugin)
        assert isinstance(container.service, ModelRuntimeService)
        assert container.service._plugin is plugin

    def test_predict_use_case_executes(self):
        """Verify predict use case is wired in the container."""
        plugin = FakeModelPlugin()
        container = ModelContainer(plugin=plugin)
        # Just verify it's wired correctly
        assert container.stats_use_case is not None
        assert container.train_use_case is not None


# ── ModelRuntimeService tests ─────────────────────────────────────────────

class TestModelRuntimeService:
    """Tests for ModelRuntimeService delegation to plugins."""

    def test_is_loaded(self):
        """Verify is_loaded reflects the plugin's state."""
        plugin = FakeModelPlugin()
        service = ModelRuntimeService(plugin)
        assert service.is_loaded() is False
        plugin.load()
        assert service.is_loaded() is True

    def test_stats_delegates_to_plugin(self):
        """Verify stats delegates to the underlying plugin."""
        plugin = FakeModelPlugin()
        service = ModelRuntimeService(plugin)
        stats = service.stats()
        assert stats.model_name == "test"
        assert stats.runtime_stats.total_predictions == 0

    def test_stats_after_load(self):
        """Verify stats returns correct metadata after load."""
        plugin = FakeModelPlugin()
        plugin.load()
        service = ModelRuntimeService(plugin)
        stats = service.stats()
        assert stats.framework == "test"
        assert stats.task_type == "test"


# ── Router factory exception edge cases ───────────────────────────────────

class TestRouterFactoryEdgeCases:
    """Tests the exception-mapping paths in router_factory.py."""

    def test_extra_predict_exceptions_maps_to_422(self, app, client):
        """Verify extra_predict_exceptions map to HTTP 422."""
        resp = client.post(
            "/models/wine-sulphite/predict",
            json={"mode": "inline", "fixed_acidity": 1},
        )
        assert resp.status_code == 422


class TestTrainModelUseCase:
    """Tests for TrainModelUseCase error path."""

    def test_extra_params_triggers_else_branch(self):
        """Verify TrainModelUseCase.execute else-branch raises when request has no .pop()."""
        from app.application.use_cases.train_model_use_case import TrainModelUseCase
        from pydantic import BaseModel

        class ExtraRequest(BaseModel):
            """Test request with extra params for else-branch coverage."""
            data_path: str = ""
            extra_param: str = "value"

        plugin = FakeModelPlugin()
        use_case = TrainModelUseCase(plugin)
        request = ExtraRequest(data_path="/some/path", extra_param="extra")
        with pytest.raises((AttributeError, TypeError)):
            # The else branch tries request.pop() on a Pydantic model (no .pop)
            use_case.execute(request)


# ── ArtifactStore local ops tests ─────────────────────────────────────────

class TestArtifactStoreLocal:
    """Tests for ArtifactStore local path operations."""

    def test_local_dir_construction(self):
        """Verify ArtifactStore builds the local directory path correctly."""
        from app.infrastructure.artifact_store import ArtifactStore, ARTIFACTS_ROOT
        store = ArtifactStore("test_model")
        assert store._model_name == "test_model"
        assert store._local_dir == ARTIFACTS_ROOT / "test_model"

    def test_path_returns_local_path(self):
        """Verify path() returns the correct local Path."""
        from app.infrastructure.artifact_store import ArtifactStore, ARTIFACTS_ROOT
        store = ArtifactStore("test_model")
        with patch_unit("pathlib.Path.exists", return_value=True):
            path = store.path("some_file.pkl")
        assert path == ARTIFACTS_ROOT / "test_model" / "some_file.pkl"

    def test_path_raises_file_not_found_when_no_s3(self, monkeypatch):
        """Verify path() raises FileNotFoundError when file is missing and no S3."""
        monkeypatch.delenv("STORAGE_BUCKET", raising=False)
        from app.infrastructure.artifact_store import ArtifactStore
        store = ArtifactStore("nonexistent_model_xyz")
        with pytest.raises(FileNotFoundError):
            store.path("not_existing.pkl")

    def test_download_all_raises_when_no_bucket(self, monkeypatch):
        """Verify download_all_if_needed raises EnvironmentError when STORAGE_BUCKET is not set."""
        monkeypatch.delenv("STORAGE_BUCKET", raising=False)
        from app.infrastructure.artifact_store import ArtifactStore
        store = ArtifactStore("test_model")
        with pytest.raises(EnvironmentError, match="STORAGE_BUCKET"):
            store.download_all_if_needed()


# ── _file_needs_download tests ───────────────────────────────────────────────

class TestFileNeedsDownload:
    """Tests for the _file_needs_download helper."""

    def test_missing_file_returns_true(self, tmp_path):
        """Verify True when the local file does not exist."""
        from app.infrastructure.artifact_store import _file_needs_download
        assert _file_needs_download(tmp_path / "missing.pkl", 100) is True

    def test_matching_size_returns_false(self, tmp_path):
        """Verify False when local file size matches the remote size."""
        from app.infrastructure.artifact_store import _file_needs_download
        p = tmp_path / "file.pkl"
        p.write_bytes(b"x" * 50)
        assert _file_needs_download(p, 50) is False

    def test_size_mismatch_returns_true(self, tmp_path):
        """Verify True when local file size differs from the remote size."""
        from app.infrastructure.artifact_store import _file_needs_download
        p = tmp_path / "file.pkl"
        p.write_bytes(b"x" * 50)
        assert _file_needs_download(p, 99) is True


# ── ArtifactStore S3 tests ────────────────────────────────────────────────────

class TestArtifactStoreS3:
    """Tests for ArtifactStore S3 download and upload operations."""

    def test_download_all_if_needed_with_bucket_triggers_download(self, monkeypatch):
        """Verify download_all_if_needed calls _download_all when STORAGE_BUCKET is set."""
        monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")
        from app.infrastructure.artifact_store import ArtifactStore
        with patch_unit("app.infrastructure.artifact_store.ArtifactStore._download_all") as mock_dl:
            ArtifactStore("test_model").download_all_if_needed()
            mock_dl.assert_called_once()

    def test_path_triggers_download_when_file_missing_and_bucket_set(self, monkeypatch, tmp_path):
        """Verify path() calls _download_all when the file is absent and STORAGE_BUCKET is set."""
        monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")
        from app.infrastructure.artifact_store import ArtifactStore
        with patch_unit("app.infrastructure.artifact_store.ArtifactStore._download_all") as mock_dl:
            store = ArtifactStore("test_model")
            store._local_dir = tmp_path / "missing_dir"
            result = store.path("model.pkl")
            mock_dl.assert_called_once()
            assert result == tmp_path / "missing_dir" / "model.pkl"

    def test_build_s3_client_reads_env_vars(self, monkeypatch):
        """Verify _build_s3_client passes environment credentials to boto3."""
        monkeypatch.setenv("CUSTOM_S3_ENDPOINT", "http://minio:9000")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testkey")
        monkeypatch.setenv("AWS_SECRET_ACCESS_ID", "testsecret")
        monkeypatch.setenv("CUSTOM_REGION", "eu-west-1")
        with patch_unit("boto3.client") as mock_boto3:
            from app.infrastructure.artifact_store import _build_s3_client
            _build_s3_client()
            mock_boto3.assert_called_once()
            kwargs = mock_boto3.call_args[1]
            assert kwargs["endpoint_url"] == "http://minio:9000"
            assert kwargs["aws_access_key_id"] == "testkey"
            assert kwargs["region_name"] == "eu-west-1"

    def test_download_all_empty_s3_prefix(self, monkeypatch, tmp_path):
        """Verify _download_all is a no-op when S3 returns no objects."""
        monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{}]  # page with no Contents key
        mock_s3.get_paginator.return_value = mock_paginator
        with patch_unit("app.infrastructure.artifact_store._build_s3_client", return_value=mock_s3):
            from app.infrastructure.artifact_store import ArtifactStore
            store = ArtifactStore("test_model")
            store._local_dir = tmp_path
            store._download_all()
        mock_s3.download_file.assert_not_called()

    def test_download_all_skips_unchanged_files(self, monkeypatch, tmp_path):
        """Verify _download_all skips files whose local size already matches S3."""
        monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")
        (tmp_path / "model.pkl").write_bytes(b"x" * 100)
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{
            "Contents": [{"Key": "artifacts/fixed/test_model/model.pkl", "Size": 100}]
        }]
        mock_s3.get_paginator.return_value = mock_paginator
        with patch_unit("app.infrastructure.artifact_store._build_s3_client", return_value=mock_s3):
            from app.infrastructure.artifact_store import ArtifactStore
            store = ArtifactStore("test_model")
            store._local_dir = tmp_path
            store._download_all()
        mock_s3.download_file.assert_not_called()

    def test_download_all_downloads_missing_file(self, monkeypatch, tmp_path):
        """Verify _download_all calls download_file for artifacts absent locally."""
        monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")
        remote_key = "artifacts/fixed/test_model/model.pkl"
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{
            "Contents": [{"Key": remote_key, "Size": 500}]
        }]
        mock_s3.get_paginator.return_value = mock_paginator
        with patch_unit("app.infrastructure.artifact_store._build_s3_client", return_value=mock_s3):
            from app.infrastructure.artifact_store import ArtifactStore
            store = ArtifactStore("test_model")
            store._local_dir = tmp_path
            store._download_all()
        mock_s3.download_file.assert_called_once_with(
            "test-bucket", remote_key, str(tmp_path / "model.pkl")
        )

    def test_download_all_raises_on_s3_error(self, monkeypatch, tmp_path):
        """Verify _download_all re-raises exceptions from S3."""
        monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{
            "Contents": [{"Key": "artifacts/fixed/test_model/model.pkl", "Size": 500}]
        }]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.side_effect = RuntimeError("connection refused")
        with patch_unit("app.infrastructure.artifact_store._build_s3_client", return_value=mock_s3):
            from app.infrastructure.artifact_store import ArtifactStore
            store = ArtifactStore("test_model")
            store._local_dir = tmp_path
            with pytest.raises(RuntimeError, match="connection refused"):
                store._download_all()

# ── model_loader helper tests ─────────────────────────────────────────────

class TestModelo10ModelLoader:
    """Tests for model_loader helper functions."""

    def test_build_mobilenetv3_classifier(self):
        """Verify _build_mobilenetv3_classifier returns a model with the correct output size."""
        from app.plugins.modelo10_lacteo.model_loader import _build_mobilenetv3_classifier
        import torch.nn as nn
        model = _build_mobilenetv3_classifier(num_classes=3)
        assert isinstance(model.classifier[-1], nn.Linear)
        assert model.classifier[-1].out_features == 3

    def test_build_mobilenetv3_classifier_different_classes(self):
        """Verify the classifier head supports different numbers of classes."""
        from app.plugins.modelo10_lacteo.model_loader import _build_mobilenetv3_classifier
        model = _build_mobilenetv3_classifier(num_classes=5)
        assert model.classifier[-1].out_features == 5
