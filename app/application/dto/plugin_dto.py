"""Base DTOs shared by the plugins' predict_dto.py / train_dto.py.

Each plugin keeps its own request/response classes (same names, docstrings and model-specific
fields); these bases only hold the fields every model declares the same way. Pydantic puts
inherited fields first, in the base's order, and a field redeclared in a subclass keeps its
inherited position. So a plugin class may only inherit from a base whose fields are exactly the
first fields it already had, in that order — that keeps the JSON key order and the OpenAPI schema
of every endpoint unchanged. A plugin whose field differs (another description or default)
redeclares it with its own ``Field``.
"""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.base_version_dto import VersionInfo
from app.application.dto.train_dto import MlflowRunId

# ── Field types repeated verbatim across plugins ─────────────────────────────────────────────
# A field that is not among a class's leading fields cannot come from a base without moving it,
# so these carry the shared declaration instead: ``mlflow_run_id: UserModelRunId = ""``. The
# default stays in the class, exactly as it was; the description lives here once.

# /predict: optional run of a user-retrained model (English / Spanish wording).
UserModelRunId = Annotated[str, Field(description="MLflow run ID for user-trained model")]
UserModelRunIdEs = Annotated[
    str, Field(description="MLflow run ID de un modelo reentrenado por el usuario")
]
# /train of the models that promote their base (ml13-24): the run is optional.
PromotingTrainRunId = Annotated[
    str,
    Field(
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en "
            "él (trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        )
    ),
]
# /train response: run that really holds the retrained model.
ConfirmedRunId = Annotated[
    str,
    Field(
        description=(
            "Run de MLflow al que se subieron los artefactos. La plataforma lo toma como "
            "autoritativo frente al run que pre-creó ella: si aquel falló, este es el que "
            "de verdad tiene el modelo entrenado (train-task-manager.ts, confirmedRunId)."
        )
    ),
]
UploadWarning = Annotated[
    str | None, Field(description="Aviso si el modelo reentrenado no se ha guardado en MLflow")
]
PromotingUploadWarning = Annotated[
    str | None,
    Field(
        description=(
            "Aviso si la subida a MLflow falló; el modelo base se publica igualmente (solo un "
            "fallo al publicar la base es un 502)"
        )
    ),
]
PublishedBaseVersion = Annotated[
    VersionInfo | None, Field(description="Versión del modelo base publicada por este reentreno")
]
UnusedThreshold = Annotated[
    float | None,
    Field(
        description=(
            "No usado por este modelo (no hay umbral de decisión); se acepta por compatibilidad "
            "de contrato."
        )
    ),
]
XaiFeatureValues = Annotated[
    dict[str, float] | None,
    Field(description="Valores de features usados — consumidos por el servicio XAI"),
]


class PluginDTO(BaseModel):
    """Root of the plugin DTOs: ``model_*`` field names are allowed (model_id, model_key…)."""

    model_config = ConfigDict(protected_namespaces=())


class BatchRequestBase(PluginDTO):
    """Leading fields of a batch /predict request: ``mode``, ``data_path``, ``mlflow_run_id``."""

    mode: Literal["batch"] = "batch"
    data_path: str
    mlflow_run_id: str = ""


class InlineRequestBase(PluginDTO):
    """Leading fields of an inline /predict request: ``mode``, ``model_key``, ``threshold``,
    ``mlflow_run_id``."""

    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = ""


class BatchResponseBase(PluginDTO):
    """Leading fields of a batch /predict response: ``model_id``, ``predictions``,
    ``output_path``."""

    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class TrainRequestBase(PluginDTO):
    """Leading fields of a /train request: ``data_path`` and a required, non-empty run id."""

    data_path: str
    mlflow_run_id: MlflowRunId
