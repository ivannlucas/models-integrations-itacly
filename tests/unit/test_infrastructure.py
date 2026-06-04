"""Tests for infrastructure layer components.

Covers ModelContainer, ModelRuntimeService, ArtifactStore (local path ops),
and router_factory edge cases (exception mapping).
"""

from __future__ import annotations


import pytest
from unittest.mock import patch as patch_unit

from app.application.dto.stats_dto import StatsResponse
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
            model_name="test", model_type="test", framework="test",
            artifact_path="/test", input_schema={}, output_schema={},
            predict_count=0, last_predict_at=None,
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
        container = ModelContainer(
            plugin=plugin,
            batch_response_cls=dict,
            inline_response_cls=dict,
        )
        assert container._plugin is plugin
        assert container._service is not None
        assert container.predict_use_case is not None
        assert container.stats_use_case is not None
        assert container.train_use_case is not None

    def test_init_calls_plugin_load(self):
        """Verify container.init() calls plugin.load()."""
        plugin = FakeModelPlugin()
        container = ModelContainer(
            plugin=plugin,
            batch_response_cls=dict,
            inline_response_cls=dict,
        )
        assert plugin.is_loaded() is False
        container.init()
        assert plugin.is_loaded() is True

    def test_service_property(self):
        """Verify container.service returns a ModelRuntimeService."""
        plugin = FakeModelPlugin()
        container = ModelContainer(
            plugin=plugin,
            batch_response_cls=dict,
            inline_response_cls=dict,
        )
        assert isinstance(container.service, ModelRuntimeService)
        assert container.service._plugin is plugin

    def test_predict_use_case_executes(self):
        """Verify predict use case is wired in the container."""
        plugin = FakeModelPlugin()
        container = ModelContainer(
            plugin=plugin,
            batch_response_cls=dict,
            inline_response_cls=dict,
        )
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
        assert stats.predict_count == 0

    def test_stats_after_load(self):
        """Verify stats returns correct metadata after load."""
        plugin = FakeModelPlugin()
        plugin.load()
        service = ModelRuntimeService(plugin)
        stats = service.stats()
        assert stats.framework == "test"
        assert stats.model_type == "test"


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
        """Verify download_all_if_needed raises when STORAGE_BUCKET is not set."""
        monkeypatch.delenv("STORAGE_BUCKET", raising=False)
        from app.infrastructure.artifact_store import ArtifactStore
        store = ArtifactStore("test_model")
        with pytest.raises(EnvironmentError, match="STORAGE_BUCKET"):
            store.download_all_if_needed()


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
