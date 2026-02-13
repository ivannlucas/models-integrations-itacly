from typing import Optional
from app.domain.ports.ml_plugin_port import MLPluginPort, PredictResult

class PredictUseCase:
    def __init__(self, runtime_service):
        self.runtime_service = runtime_service

    def execute(self, data_path: str, output_path: str):
        return self.runtime_service.predict(data_path=data_path, output_path=output_path)

    def execute_inline(self, features: dict, model_key: str | None = None):
        return self.runtime_service.predict_inline(features, model_key=model_key)