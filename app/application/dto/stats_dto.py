"""Data Transfer Objects for model statistics."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class InputField(BaseModel):
    """Describes a single input feature expected by the model."""

    model_config = ConfigDict(protected_namespaces=())
    name: str
    type: str
    format: Optional[list[str]] = None
    default: Optional[Any] = None
    description: str


class OutputField(BaseModel):
    """Describes a single output field produced by the model."""

    model_config = ConfigDict(protected_namespaces=())
    name: str
    type: str
    description: str


class RuntimeStats(BaseModel):
    """Aggregated runtime counters collected since the model was loaded."""

    model_config = ConfigDict(protected_namespaces=())
    total_predictions: int
    avg_latency_ms: Optional[float] = None


class StatsResponse(BaseModel):
    """Full stats payload returned by the /stats endpoint for any registered model."""

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
