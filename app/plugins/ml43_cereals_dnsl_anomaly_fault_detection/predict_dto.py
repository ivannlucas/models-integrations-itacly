"""Pydantic DTOs for the ml43-cereals-dnsl-anomaly-fault-detection /predict endpoint."""
from pydantic import Field

from app.application.dto.plugin_dto import BatchResponseBase, PluginDTO


class PredictBatchRequest(PluginDTO):
    """Batch prediction request: CSV with real 180-row sensor windows."""

    mode: str = Field(
        default="batch",
        description="Único modo soportado: 'batch' (CSV con ventanas reales de 180 filas)",
    )
    data_path: str = Field(..., description="Path to CSV file with sensor time-series data")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-trained model")


class PredictBatchResponse(BatchResponseBase):
    """Batch prediction response: one result per window, plus an optional warning."""

    warning: str | None = Field(
        default=None,
        description=(
            "Aviso no bloqueante sobre el resultado (p.ej. CSV sin cycle_id, o sin ventanas "
            "válidas)"
        ),
    )


# ml43-cereals-dnsl-anomaly-fault-detection requires a real 180-row sensor window: there is no
# single-snapshot ("inline") prediction mode. Any request whose body doesn't match
# PredictBatchRequest (e.g. mode="inline") is rejected by FastAPI/Pydantic validation before
# reaching the plugin — see Ml43CerealsDnslAnomalyFaultDetectionPlugin.predict_inline for the
# defensive backstop required by ModelPluginPort's abstract interface.
PredictRequest = PredictBatchRequest
PredictResponse = PredictBatchResponse
