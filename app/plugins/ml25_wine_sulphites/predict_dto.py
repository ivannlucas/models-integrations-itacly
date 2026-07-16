"""DTOs for wine sulphite prediction operations."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ConfigDict


class PredictBatchRequest(BaseModel):
    """DTO for batch prediction request."""
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description="Path to CSV file with wine physicochemical properties inside the container",
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictBatchResponse(BaseModel):
    """DTO for batch prediction response."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None
    mlflow_run_id: str = ""


class PredictInlineRequest(BaseModel):
    """DTO for inline prediction request."""
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    # Physicochemical wine properties
    fixed_acidity: float = Field(..., description="Fixed acidity (g/dm³), typical range 3.8–14.2")
    volatile_acidity: float = Field(
        ..., description="Volatile acidity (g/dm³), typical range 0.08–1.1"
    )
    citric_acid: float = Field(..., description="Citric acid (g/dm³), typical range 0.0–1.66")
    residual_sugar: float = Field(
        ..., description="Residual sugar (g/dm³), typical range 0.6–65.8"
    )
    chlorides: float = Field(..., description="Chlorides (g/dm³), typical range 0.009–0.346")
    density: float = Field(..., description="Density (g/cm³), typical range 0.987–1.039")
    pH: float = Field(..., description="pH, typical range 2.72–3.82")
    sulphates: float = Field(..., description="Sulphates (g/dm³), typical range 0.22–1.08")
    alcohol: float = Field(..., description="Alcohol (% vol.), typical range 8.0–14.2")
    free_sulfur_dioxide: float = Field(
        ..., description="Current free SO2 (mg/L), typical range 2–289"
    )
    total_sulfur_dioxide: float = Field(
        ..., description="Current total SO2 (mg/L), typical range 9–440"
    )
    # Operational simulation parameters
    min_molecular: float = Field(
        default=0.6,
        description="Minimum molecular SO2 required for microbial protection (mg/L)",
    )
    max_total: float = Field(
        default=200.0,
        description="Maximum legal total SO2 (mg/L)",
    )
    delta_max: float = Field(
        default=40.0,
        description="Maximum free SO2 increment to explore in simulation (mg/L)",
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictInlineResponse(BaseModel):
    """DTO for inline prediction response."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    threshold: float | None = None
    prediction: Any = Field(
        ..., description="True if sulphite intervention is recommended, False otherwise"
    )
    confidence: float | None = Field(
        None, description="Predicted sensory quality at the recommended point (0–10)"
    )
    features_used: list[str]
    recommended_free_so2: float = Field(..., description="Recommended free SO2 dose (mg/L)")
    recommended_bound_so2: float = Field(
        ..., description="Estimated bound SO2 at the recommended point (mg/L)"
    )
    recommended_total_so2: float = Field(
        ..., description="Estimated total SO2 at the recommended point (mg/L)"
    )
    recommended_molecular_so2: float = Field(
        ..., description="Molecular SO2 at the recommended point (mg/L)"
    )
    predicted_quality: float = Field(
        ..., description="Predicted sensory quality at the recommended point (0–10 scale)"
    )
    baseline_predicted_quality: float = Field(
        ..., description="Predicted quality at the current SO2 dose"
    )
    recommendation_reason: str = Field(
        ..., description="Textual explanation of the recommendation decision"
    )
    intervention_recommended: bool = Field(
        ..., description="True if the expected improvement exceeds 1× MAE threshold"
    )
    mae_quality: float | None = Field(
        default=None, description="Cross-validated MAE of the quality model (~0.427)"
    )
    mae_bound: float | None = Field(
        default=None, description="Cross-validated MAE of the bound SO2 model (~14.5 mg/L)"
    )
    xai_feature_values: dict[str, float] = Field(
        default_factory=dict,
        description="Wine analysis input values used for this prediction, echoed back for XAI/explainability display",
    )
    simulation_trajectory: list[dict[str, float]] = Field(
        default_factory=list,
        description=(
            "Every candidate free SO2 dose evaluated during optimization, with its estimated "
            "bound/total/molecular SO2 and predicted quality — the full dose-vs-quality path, "
            "not just the recommended point"
        ),
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
