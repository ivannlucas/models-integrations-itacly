from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List


@dataclass
class TrainResult:
    model_id: str
    metrics: Dict[str, Any]
    artifacts_path: str


@dataclass
class PredictResult:
    model_id: str
    predictions: List[Dict[str, Any]]  
    output_path: Optional[str] = None


@dataclass
class StatsResult:
    model_id: str
    metrics: Dict[str, Any]
    artifacts_path: str


class MLPluginPort(Protocol):
    def train(self, *, data_path: str) -> TrainResult: ...
    def predict(self, *, data_path: str, output_path: Optional[str] = None) -> PredictResult: ...
    def get_stats(self) -> StatsResult: ...