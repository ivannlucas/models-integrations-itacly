from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class InputField(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    type: str
    format: Optional[list[str]] = None
    default: Optional[Any] = None
    description: str


class OutputField(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    type: str
    description: str


class RuntimeStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    total_predictions: int
    avg_latency_ms: Optional[float]


class StatsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    version: str
    description: str
    task_type: str
    framework: str
    inputs: list[InputField]
    outputs: list[OutputField]
    metrics: dict[str, Any]
    runtime_stats: RuntimeStats
