"""Pydantic DTOs for the ml41 acoustic anomaly ``/predict`` endpoint."""
from typing import Annotated, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import BatchRequestBase, BatchResponseBase, PluginDTO


class PredictInlineRequest(PluginDTO):
    """Inline request: one base64 WAV plus the machine combination to score it with."""
    mode: Literal["inline"] = "inline"
    audio_base64: str = Field(..., description="Base64-encoded WAV recording of the machine")
    machine: Literal["fan", "pump", "slider", "valve"] = Field(
        ..., description="Machine type being monitored"
    )
    machine_id: Literal["id_00", "id_02", "id_04", "id_06"] = Field(
        ..., description="Physical unit identifier (each unit has its own trained checkpoint)"
    )
    snr: Literal["-6_dB", "0_dB", "6_dB"] = Field(
        ...,
        description=(
            "Background-noise level of the closest MIMII training combination to the real "
            "installation conditions — not a real-time measurement (see manifest known_issues)"
        ),
    )
    threshold: float | None = Field(
        default=None,
        description="Override the trained decision threshold for maha_score (advanced use)",
    )
    mlflow_run_id: str = Field(
        default="", description="MLflow run ID for a user-trained checkpoint"
    )


class PredictInlineResponse(PluginDTO):
    """Inline response: anomaly scores and predicted label for one WAV."""
    model_id: str
    machine: str
    machine_id: str
    snr: str
    mse_score: float = Field(
        ..., description="Reconstruction MSE — stochastic across runs (~5-6%), informational only"
    )
    maha_score: float = Field(
        ..., description="Mahalanobis distance in PCA latent space (deterministic)"
    )
    predicted_label: int = Field(..., description="0 = normal, 1 = anomalous")
    threshold_used: float


class PredictBatchRequest(BatchRequestBase):
    """Batch request: ZIP with manifest.csv plus the referenced WAV files."""
    data_path: str = Field(
        ...,
        description=(
            "Path to a ZIP file containing manifest.csv (columns: filename,machine,"
            "machine_id,snr) plus the referenced WAV files"
        ),
    )
    threshold: float | None = Field(
        default=None,
        description="Optional decision threshold override for every WAV in this batch",
    )


class PredictBatchResponse(BatchResponseBase):
    """Batch response: one prediction dict per WAV in the manifest."""


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
