"""Pydantic request/response DTOs for the ml1 vineyard pest/disease ``/predict`` endpoint."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml1_wine_pest_disease_detection.constants import SENSOR_COLS

_SENSORS_DESC = (
    f"Telemetria IIoT de los ultimos 5 dias, 9 campos obligatorios (configuracion de produccion, "
    f"exclude_gases=true): {list(SENSOR_COLS)}. CO2_ppm/VOC_ppb se aceptan pero se ignoran."
)


class PredictInlineRequest(BaseModel):
    """Inline request: one sample — optional image + required sensor telemetry."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    image_base64: str | None = Field(
        default=None,
        description=(
            "Imagen de hoja/racimo en base64 (JPG/PNG), opcional. Sin ella, el sistema responde "
            "en modo de contingencia tabular (solo riesgo, via Random Forest)."
        ),
    )
    sensors: dict[str, float] = Field(..., description=_SENSORS_DESC)
    mlflow_run_id: str = Field(default="", description="MLflow run ID de los RandomForest reentrenados por el usuario")


class PredictBatchRequest(BaseModel):
    """Batch request: a ZIP containing one CSV (Path_Imagen + sensor columns) and the images
    it references (at the ZIP root or under an images/ subfolder — both are searched by
    filename), the same contract as the original --input CSV + images_dir."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Ruta local o s3:// a un ZIP con el CSV de entrada y las imagenes referenciadas")
    mlflow_run_id: str = Field(default="", description="MLflow run ID de los RandomForest reentrenados por el usuario")


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(BaseModel):
    """Inline response: predicted pathology/pest class, confidence and epidemiological risk."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    enfermedad: str = Field(..., description="Una de las 15 clases, o IMAGEN_NO_DISPONIBLE si no se aporto imagen")
    confianza: float = Field(..., description="Probabilidad softmax de la clase ganadora [0, 1]; 0.0 sin imagen")
    riesgo: float = Field(..., description="Indice de riesgo epidemiologico [0, 1]")
    metodo: str = Field(..., description="HIBRIDO_ENSAMBLE | RIESGO_TABULAR_EXPERT")


class BatchPrediction(BaseModel):
    """One row of a predict_batch result."""

    id: int
    nombre_imagen: str
    enfermedad: str
    confianza: float
    riesgo: float
    metodo: str
    error: str | None = None


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction per input CSV row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[BatchPrediction]
    n_rows: int
    n_predictions: int = Field(..., description="Filas sin error")
    output_path: str | None = None


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
