"""Data Transfer Objects (DTOs) for training a model."""
from pydantic import BaseModel


class TrainRequest(BaseModel):
    """Request body for training a model."""
    data_path: str = ""
    mlflow_run_id: str


class TrainResponse(BaseModel):
    """Response body for training a model."""
    detail: str
    metrics: dict = {}
