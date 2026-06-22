"""Pydantic DTOs for the cereal residue-optimizer /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Train request: a CSV with the features plus the target residue column."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="CSV con features + columna Residuo_Disponible_Suelo_t")


class TrainResponse(BaseModel):
    """Train response: fields match the dict returned by plugin.train()."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    r2_test: float
    n_train: int
    n_test: int
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si el artefacto se guardó en local pero falló el upload a S3",
    )
