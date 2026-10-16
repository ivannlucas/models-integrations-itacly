"""ModelContainer promotes, reloads and rolls back the base of plugins with BASE_ARTIFACT_FOLDER."""
import pytest

from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.services.exceptions import (
    BaseReloadError,
    BaseVersionNotFoundError,
    NoPreviousBaseVersionError,
)
from app.infrastructure import base_model_publisher
from app.infrastructure.http.dependencies.container import ModelContainer
from tests.conftest import FakePlugin
from tests.unit.tiny_promoting_plugin import (
    TINY_FOLDER,
    TinyPredictRequest,
    TinyPromotingPlugin,
    TinyTrainRequest,
)


@pytest.fixture(autouse=True)
def _reset_tiny_flags():
    TinyPromotingPlugin.fail_load = False
    TinyPromotingPlugin.fail_after_publish = False
    TinyPromotingPlugin.after_publish_hook = None
    yield
    TinyPromotingPlugin.fail_load = False
    TinyPromotingPlugin.fail_after_publish = False
    TinyPromotingPlugin.after_publish_hook = None


@pytest.fixture
def container():
    c = ModelContainer(TinyPromotingPlugin())
    c.init()
    return c


def _store():
    return base_model_publisher._store_for(TINY_FOLDER)  # pylint: disable=protected-access


def _served(c) -> str:
    return c.predict_use_case.execute(TinyPredictRequest()).value


def test_init_reads_the_pointer_and_starts_at_v0(container):
    assert container.promotes_base
    assert container.base_version()["version"] == "v0"
    assert _served(container) == "base"


def test_train_publishes_and_serves_the_new_base_with_a_fresh_instance(container):
    old_plugin = container._plugin  # pylint: disable=protected-access
    resp = container.train_use_case.execute(TinyTrainRequest(data_path="datos-1"))
    assert resp.base_version.version == "v1"
    assert container._plugin is not old_plugin  # pylint: disable=protected-access
    assert old_plugin.value == "base"            # in-flight requests keep their instance
    assert _served(container) == "datos-1"
    assert container.base_version()["version"] == "v1"
    assert container.stats_use_case.execute().base_version.version == "v1"


def test_two_trains_chain_versions(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    resp = container.train_use_case.execute(TinyTrainRequest(data_path="b"))
    assert resp.base_version.version == "v2" and resp.base_version.parent == "v1"
    assert _served(container) == "b"


def test_reload_failure_restores_the_pointer_and_keeps_serving(container):
    TinyPromotingPlugin.fail_load = True
    with pytest.raises(BaseReloadError, match="v1"):
        container.train_use_case.execute(TinyTrainRequest(data_path="roto"))
    assert _store().active_version() == "v0"
    assert _served(container) == "base"


def test_train_failing_after_publish_restores_pointer(container):
    TinyPromotingPlugin.fail_after_publish = True
    with pytest.raises(RuntimeError, match="fallo tras publicar"):
        container.train_use_case.execute(TinyTrainRequest(data_path="x"))
    assert _store().active_version() == "v0"
    assert _served(container) == "base"


def test_rollback_without_version_goes_to_the_parent(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    container.train_use_case.execute(TinyTrainRequest(data_path="b"))
    result = container.rollback()
    assert result["base_version"]["version"] == "v1"
    assert result["previous_base_version"]["version"] == "v2"
    assert _served(container) == "a"
    # the next train inherits from the restored version
    resp = container.train_use_case.execute(TinyTrainRequest(data_path="c"))
    assert resp.base_version.version == "v3" and resp.base_version.parent == "v1"


def test_rollback_to_an_explicit_version_and_back_to_v0(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    container.train_use_case.execute(TinyTrainRequest(data_path="b"))
    assert container.rollback("v0")["base_version"]["version"] == "v0"
    assert _served(container) == "base"


def test_rollback_from_v0_has_no_parent(container):
    with pytest.raises(NoPreviousBaseVersionError):
        container.rollback()


def test_rollback_to_an_unknown_version(container):
    with pytest.raises(BaseVersionNotFoundError, match="v9"):
        container.rollback("v9")
    assert _store().active_version() == "v0"


def test_rollback_to_a_version_that_does_not_load_keeps_current(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    TinyPromotingPlugin.fail_load = True
    with pytest.raises(BaseReloadError):
        container.rollback("v0")
    assert _store().active_version() == "v1"
    assert _served(container) == "a"


def test_plugins_without_base_folder_are_unchanged():
    plugin = FakePlugin(model_id="x", inline_factory=lambda *a, **k: None,
                        batch_factory=lambda *a, **k: None)
    c = ModelContainer(plugin)
    assert not c.promotes_base
    assert isinstance(c.train_use_case, TrainModelUseCase)
    c.init()
    assert c.stats_use_case.execute().base_version is None


def test_stats_report_the_served_version_during_and_after_a_failed_train(container):
    seen = []
    TinyPromotingPlugin.after_publish_hook = lambda: seen.append(container.base_version()["version"])
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    assert seen == ["v0"]  # pointer already at v1, but the served instance is v0
    assert container.base_version()["version"] == "v1"


def test_stats_keep_the_served_version_when_the_reload_fails(container):
    TinyPromotingPlugin.fail_load = True
    with pytest.raises(BaseReloadError):
        container.train_use_case.execute(TinyTrainRequest(data_path="roto"))
    assert container.stats_use_case.execute().base_version.version == "v0"


def test_failed_pointer_restore_does_not_mask_the_original_error(container, monkeypatch):
    TinyPromotingPlugin.fail_after_publish = True

    def boom(_version):
        raise OSError("s3 caído")

    monkeypatch.setattr(container._base_store, "set_active_version", boom)  # pylint: disable=protected-access
    with pytest.raises(RuntimeError, match="fallo tras publicar"):
        container.train_use_case.execute(TinyTrainRequest(data_path="x"))
