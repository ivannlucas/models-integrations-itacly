"""Persist a retrained bundle as the model's new base version.

Used by the train() of the models that declare BASE_ARTIFACT_FOLDER (ml13, 14, 15, 16, 17, 18,
19, 21, 23, 24): every successful /train becomes the base served without mlflow_run_id. The bundle is
also uploaded to the caller's MLflow run when one is given — traceability only: if that upload
fails the base is still published and the response carries upload_warning. If publishing fails,
ModelPersistenceError (→ 502) and the base does not change (see ArtifactStore.publish_version).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.domain.services.exceptions import ModelPersistenceError
from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistResult:
    """Outcome of persist_retrained_bundle."""

    base_version: dict
    upload_warning: str | None


def _store_for(artifact_folder: str) -> ArtifactStore:
    """The store a model's base versions are published to (tests swap it for a temp one)."""
    return ArtifactStore(artifact_folder)


def _json_safe(metrics: dict) -> dict[str, Any]:
    """Keep scalar metrics (numpy scalars included); NaN/inf become None so version.json is valid JSON."""
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            safe[key] = None
        elif value is None or isinstance(value, (bool, int, float, str)):
            safe[key] = value
        else:
            dropped.append(str(key))
    if dropped:
        logger.info("Non-scalar metrics not stored in version.json: %s", ", ".join(dropped))
    return safe


def persist_retrained_bundle(
    bundle_dir: str | Path,
    *,
    artifact_folder: str,
    data_path: str,
    metrics: dict,
    mlflow_run_id: str = "",
    upload_to_mlflow: Callable[[], None] | None = None,
) -> PersistResult:
    """Upload *bundle_dir* to the MLflow run (if any) and publish it as the new base version.

    *upload_to_mlflow* is the plugin's own MLflow logging + upload, called only with a run id.
    The bundle must use the base artifact's canonical filenames; files it lacks are inherited
    from the current base version.
    """
    mlflow_run_id = (mlflow_run_id or "").strip()
    upload_warning = None
    if mlflow_run_id and upload_to_mlflow is not None:
        try:
            upload_to_mlflow()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("MLflow upload failed for run %s: %s", mlflow_run_id, exc)
            upload_warning = (
                f"El modelo no se ha podido guardar en el run de MLflow {mlflow_run_id} "
                f"({type(exc).__name__}: {exc}); sí se ha publicado como modelo base."
            )
    try:
        version = _store_for(artifact_folder).publish_version(bundle_dir, {
            "mlflow_run_id": mlflow_run_id or None,
            "data_path": data_path,
            "metrics": _json_safe(metrics),
        })
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Publishing the new base of '%s' failed: %s", artifact_folder, exc)
        raise ModelPersistenceError(
            f"El modelo reentrenado no se ha podido publicar como modelo base: {exc}"
        ) from exc
    return PersistResult(base_version=version, upload_warning=upload_warning)
