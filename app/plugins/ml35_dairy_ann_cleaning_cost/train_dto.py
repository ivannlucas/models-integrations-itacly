"""Train DTOs for ml35 dairy ANN fine-tuning."""
from pydantic import Field

from app.application.dto.plugin_dto import PluginDTO, TrainRequestBase, UploadWarning


class TrainRequest(TrainRequestBase):
    """Fine-tuning request: CSV with the 8 ANN features plus the consumo_agua_l target."""

    data_path: str = Field(
        ..., description="Path to CSV with 8 ANN features + consumo_agua_l target"
    )


class TrainResponse(PluginDTO):
    """Fine-tuning result: training-data metrics and the MLflow upload outcome."""

    detail: str
    mae: float = Field(
        ..., description="Mean absolute error on training data after fine-tuning (L)"
    )
    r2: float = Field(..., description="R² on training data after fine-tuning")
    n_samples: int = Field(..., description="Number of training samples used")
    upload_warning: UploadWarning = None
