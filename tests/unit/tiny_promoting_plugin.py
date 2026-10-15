"""Minimal plugin that promotes its base on train() — exercises container and router wiring."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from app.application.dto.base_version_dto import VersionInfo
from app.application.dto.stats_dto import RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.infrastructure import base_model_publisher
from app.infrastructure.base_model_publisher import persist_retrained_bundle

TINY_FOLDER = "tiny_model"


class TinyTrainRequest(BaseModel):
    data_path: str
    mlflow_run_id: str = ""


class TinyTrainResponse(BaseModel):
    value: str
    base_version: VersionInfo | None = None
    upload_warning: str | None = None


class TinyPredictRequest(BaseModel):
    mode: str = "inline"


class TinyPredictResponse(BaseModel):
    value: str


class TinyPromotingPlugin(ModelPluginPort):
    """Serves the content of value.txt of the active base version ("base" in v0)."""

    BASE_ARTIFACT_FOLDER = TINY_FOLDER
    fail_load = False
    fail_after_publish = False
    after_publish_hook = None  # callable run right after persist, before train() returns

    def __init__(self) -> None:
        self.value: str | None = None

    def load(self) -> None:
        if TinyPromotingPlugin.fail_load:
            raise RuntimeError("bundle roto")
        local = base_model_publisher._store_for(TINY_FOLDER).local_dir / "value.txt"  # pylint: disable=protected-access
        self.value = local.read_text() if local.exists() else "base"

    def is_loaded(self) -> bool:
        return self.value is not None

    def predict_inline(self, *, features: dict, model_key=None, threshold=None, mlflow_run_id: str = ""):
        return TinyPredictResponse(value=self.value)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = ""):
        return TinyPredictResponse(value=self.value)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        return StatsResponse(model_name="tiny", version="1", description="tiny", task_type="t",
                             framework="f", inputs=[], outputs=[], metrics={},
                             runtime_stats=RuntimeStats(total_predictions=0))

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TinyTrainResponse:
        bundle = Path(tempfile.mkdtemp(prefix="tiny_"))
        try:
            (bundle / "value.txt").write_text(data_path)
            result = persist_retrained_bundle(str(bundle), artifact_folder=TINY_FOLDER,
                                              data_path=data_path, metrics={"mae": 1.0},
                                              mlflow_run_id=mlflow_run_id)
        finally:
            shutil.rmtree(bundle, ignore_errors=True)
        if TinyPromotingPlugin.after_publish_hook:
            TinyPromotingPlugin.after_publish_hook()
        if TinyPromotingPlugin.fail_after_publish:
            raise RuntimeError("fallo tras publicar")
        return TinyTrainResponse(value=data_path, base_version=result.base_version,
                                 upload_warning=result.upload_warning)
