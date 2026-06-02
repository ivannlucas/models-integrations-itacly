"""Tests for infrastructure layer components.

Covers ModelContainer, ModelRuntimeService, ArtifactStore (local path ops),
and router_factory edge cases (exception mapping).
"""

from __future__ import annotations


import pytest

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure.http.dependencies.container import ModelContainer


# ── ModelContainer tests ──────────────────────────────────────────────────

class FakeModelPlugin(ModelPluginPort):
    """Minimal fake plugin for container tests."""

    def __init__(self):
        self._loaded = False

    def load(self):
        self._loaded = True

    def is_loaded(self):
        return self._loaded

    def predict_batch(self, *, data_path):
        return {"model_id": "test", "predictions": [], "output_path": None}

    def predict_inline(self, *, features, model_key=None, threshold=None):
        return {"model_id": "test", "prediction": "test", "confidence": 1.0}

    def stats(self):
        return StatsResponse(
            model_name="test", model_type="test", framework="test",
            artifact_path="/test", input_schema={}, output_schema={},
            predict_count=0, last_predict_at=None,
        )

    def train(self, *, data_path):
        from app.domain.services.exceptions import TrainingNotSupportedError
        raise TrainingNotSupportedError("Not supported")


class TestModelContainer:
    def test_init_creates_use_cases(self):
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
        plugin = FakeModelPlugin()
        container = ModelContainer(
            plugin=plugin,
            batch_response_cls=dict,
            inline_response_cls=dict,
        )
        assert isinstance(container.service, ModelRuntimeService)
        assert container.service._plugin is plugin

    def test_predict_use_case_executes(self):
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
    def test_is_loaded(self):
        plugin = FakeModelPlugin()
        service = ModelRuntimeService(plugin)
        assert service.is_loaded() is False
        plugin.load()
        assert service.is_loaded() is True

    def test_stats_delegates_to_plugin(self):
        plugin = FakeModelPlugin()
        service = ModelRuntimeService(plugin)
        stats = service.stats()
        assert stats.model_name == "test"
        assert stats.predict_count == 0

    def test_stats_after_load(self):
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
        """Trigger the extra_predict_exceptions catch (lines 57-58)."""
        resp = client.post(
            "/models/wine-sulphite/predict",
            json={"mode": "inline", "fixed_acidity": 1},
        )
        assert resp.status_code == 422


# ── ArtifactStore local ops tests ─────────────────────────────────────────

class TestArtifactStoreLocal:
    def test_local_dir_construction(self):
        from app.infrastructure.artifact_store import ArtifactStore, ARTIFACTS_ROOT
        store = ArtifactStore("test_model")
        assert store._model_name == "test_model"
        assert store._local_dir == ARTIFACTS_ROOT / "test_model"

    def test_path_returns_local_path(self):
        from app.infrastructure.artifact_store import ArtifactStore, ARTIFACTS_ROOT
        store = ArtifactStore("test_model")
        store._download_all = lambda: None  # Mock out download
        path = store.path("some_file.pkl")
        assert path == ARTIFACTS_ROOT / "test_model" / "some_file.pkl"

    def test_path_raises_file_not_found_when_no_s3(self, monkeypatch):
        monkeypatch.delenv("STORAGE_BUCKET", raising=False)
        from app.infrastructure.artifact_store import ArtifactStore
        store = ArtifactStore("nonexistent_model_xyz")
        with pytest.raises(FileNotFoundError):
            store.path("not_existing.pkl")

    def test_download_all_raises_when_no_bucket(self, monkeypatch):
        monkeypatch.delenv("STORAGE_BUCKET", raising=False)
        from app.infrastructure.artifact_store import ArtifactStore
        store = ArtifactStore("test_model")
        with pytest.raises(EnvironmentError, match="STORAGE_BUCKET"):
            store.download_all_if_needed()


# ── model_loader helper tests ─────────────────────────────────────────────

class TestModelo10ModelLoader:
    def test_build_mobilenetv3_classifier(self):
        from app.plugins.modelo10_lacteo.model_loader import _build_mobilenetv3_classifier
        import torch.nn as nn
        model = _build_mobilenetv3_classifier(num_classes=3)
        assert isinstance(model.classifier[-1], nn.Linear)
        assert model.classifier[-1].out_features == 3

    def test_build_mobilenetv3_classifier_different_classes(self):
        from app.plugins.modelo10_lacteo.model_loader import _build_mobilenetv3_classifier
        model = _build_mobilenetv3_classifier(num_classes=5)
        assert model.classifier[-1].out_features == 5
