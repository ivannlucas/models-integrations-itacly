"""DTOs and exceptions of the versioned base model."""
from app.application.dto.base_version_dto import RollbackRequest, RollbackResponse, VersionInfo
from app.application.dto.stats_dto import RuntimeStats, StatsResponse
from app.domain.services.exceptions import (
    BaseReloadError,
    BaseVersionNotFoundError,
    NoPreviousBaseVersionError,
)


def test_version_info_defaults_describe_v0():
    info = VersionInfo(version="v0")
    assert info.parent is None and info.mlflow_run_id is None and info.metrics == {}


def test_rollback_dtos():
    assert RollbackRequest().version is None
    resp = RollbackResponse(base_version={"version": "v1", "parent": "v0"},
                            previous_base_version={"version": "v2", "parent": "v1"})
    assert resp.base_version.version == "v1" and resp.previous_base_version.parent == "v1"


def test_stats_response_base_version_is_optional():
    base = dict(model_name="m", version="1", description="d", task_type="t", framework="f",
                inputs=[], outputs=[], metrics={}, runtime_stats=RuntimeStats(total_predictions=0))
    assert StatsResponse(**base).base_version is None
    assert StatsResponse(**base, base_version={"version": "v3"}).base_version.version == "v3"


def test_exception_hierarchy():
    assert issubclass(BaseVersionNotFoundError, LookupError)
    assert not issubclass(BaseVersionNotFoundError, ValueError)  # /train maps ValueError to 400
    assert issubclass(NoPreviousBaseVersionError, RuntimeError)
    assert issubclass(BaseReloadError, RuntimeError)
