from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: str = Field(default="batch", description="Único modo soportado: 'batch' (CSV con ventanas reales de 180 filas)")
    data_path: str = Field(..., description="Path to CSV file with sensor time-series data")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-trained model")


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None
    warning: str | None = Field(
        default=None,
        description="Aviso no bloqueante sobre el resultado (p.ej. CSV sin cycle_id, o sin ventanas válidas)",
    )


# modelo43-cereales requires a real 180-row sensor window: there is no single-snapshot
# ("inline") prediction mode. Any request whose body doesn't match PredictBatchRequest
# (e.g. mode="inline") is rejected by FastAPI/Pydantic validation before reaching the
# plugin — see Modelo43CerealesPlugin.predict_inline for the defensive backstop required
# by ModelPluginPort's abstract interface.
PredictRequest = PredictBatchRequest
PredictResponse = PredictBatchResponse
