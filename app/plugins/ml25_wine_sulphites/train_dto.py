"""DTOs for wine sulphite training operations."""
from pydantic import Field

from app.application.dto.plugin_dto import ConfirmedRunId, PluginDTO, TrainRequestBase


class TrainRequest(TrainRequestBase):
    """Request body that specifies the CSV file to use for training."""

    data_path: str = Field(..., description="Path to the CSV training file inside the container")


class TrainResponse(PluginDTO):
    """Response body returned after a successful training run."""

    detail: str = Field(..., description="Human-readable result message")
    mae_quality: float = Field(
        ..., description="Mean absolute error of the quality model on the held-out test split"
    )
    mae_bound_so2: float = Field(
        ...,
        description="Mean absolute error of the bound SO2 model on the held-out test split (mg/L)",
    )
    n_train: int = Field(..., description="Number of samples used for training (80 % split)")
    n_test: int = Field(..., description="Number of samples used for evaluation (20 % split)")
    training_time_s: float = Field(..., description="Wall-clock training time in seconds")
    upload_warning: str | None = Field(
        default=None, description="Set if artifacts were saved locally but S3 upload failed"
    )
    mlflow_run_id: ConfirmedRunId = ""
