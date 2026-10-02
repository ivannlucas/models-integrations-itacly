"""The generic predict/train use cases forward `system` only to plugins that declare it (ml40)."""
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.predict_dto import (
    PredictBatchRequest,
    PredictBatchResponse,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.train_dto import TrainRequest

_BATCH_RESPONSE = PredictBatchResponse(
    model_id="ml40", system="aireado", predictions=[], n_runs=0,
    avg_confidence=0.0, model_health="ESTABLE",
)


class _PluginWithSystem:
    def __init__(self):
        self.calls = []

    def predict_batch(self, *, data_path, mlflow_run_id="", system=None):
        self.calls.append({"data_path": data_path, "mlflow_run_id": mlflow_run_id, "system": system})
        return _BATCH_RESPONSE

    def train(self, *, data_path, mlflow_run_id="", system=None):
        self.calls.append({"data_path": data_path, "mlflow_run_id": mlflow_run_id, "system": system})
        return {}


class _PluginWithoutSystem:
    def __init__(self):
        self.calls = []

    def predict_batch(self, *, data_path, mlflow_run_id=""):
        self.calls.append({"data_path": data_path, "mlflow_run_id": mlflow_run_id})
        return _BATCH_RESPONSE

    def train(self, *, data_path, mlflow_run_id=""):
        self.calls.append({"data_path": data_path, "mlflow_run_id": mlflow_run_id})
        return {}


def test_predict_batch_forwards_system_when_declared():
    plugin = _PluginWithSystem()
    PredictModelUseCase(plugin).execute(PredictBatchRequest(data_path="a.csv", system="aireado"))
    assert plugin.calls[0]["system"] == "aireado"


def test_predict_batch_omits_system_when_not_declared():
    plugin = _PluginWithoutSystem()
    PredictModelUseCase(plugin).execute(PredictBatchRequest(data_path="a.csv", system="aireado"))
    assert "system" not in plugin.calls[0]


def test_train_forwards_system_when_declared():
    plugin = _PluginWithSystem()
    TrainModelUseCase(plugin).execute(TrainRequest(data_path="a.csv", system="refrigeracion"))
    assert plugin.calls[0]["system"] == "refrigeracion"


def test_train_omits_system_when_not_declared():
    plugin = _PluginWithoutSystem()
    TrainModelUseCase(plugin).execute(TrainRequest(data_path="a.csv", system="refrigeracion"))
    assert "system" not in plugin.calls[0]
