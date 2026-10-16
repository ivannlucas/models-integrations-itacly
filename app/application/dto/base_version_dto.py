"""DTOs for the versioned base model (versions/<vN>/ + current.json in the artifact store)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VersionInfo(BaseModel):
    """One published base version, as stored in its version.json (v0 = the delivered artifact)."""

    model_config = ConfigDict(protected_namespaces=())
    version: str = Field(..., description="Versión del modelo base: v0 (entregado), v1, v2…")
    parent: str | None = Field(default=None, description="Versión de la que hereda (None en v0)")
    created_at: str | None = Field(default=None, description="Fecha de publicación (ISO 8601, UTC)")
    mlflow_run_id: str | None = Field(default=None, description="Run de MLflow del reentreno, si lo hubo")
    data_path: str | None = Field(default=None, description="Datos con los que se reentrenó")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Métricas de test del reentreno")


class RollbackRequest(BaseModel):
    """Body of POST /rollback. Without version, the base goes back to the current one's parent."""

    version: str | None = Field(default=None, description="Versión a activar (por defecto, la anterior)")


class RollbackResponse(BaseModel):
    """Result of POST /rollback."""

    base_version: VersionInfo
    previous_base_version: VersionInfo
