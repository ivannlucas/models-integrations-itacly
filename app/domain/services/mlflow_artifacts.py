"""MLflow artifact plumbing shared by every plugin's ``mlflow_utils.py`` and ``plugin.py``.

What each plugin keeps for itself is only what is specific to its model: which files make up
the retrained model and how to write/rebuild them. Everything around that — the temp folder,
the download/upload through ``BaseMLflowTracker``, the "no model in this run" answer, the
cleanup on failure, the ``ModelPersistenceError`` messages and the best-effort MLflow block of
``stats()`` — lives here so it behaves the same for the 36 plugins.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

from app.application.dto.stats_dto import StatsResponse
from app.domain.services.exceptions import ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker

logger = logging.getLogger(__name__)

T = TypeVar("T")

MODEL_ARTIFACT_PATH = "model"
NO_RUN_MESSAGE = "Sin run de MLflow: el modelo reentrenado no se ha guardado."


# ── download (mlflow_utils.download_user_*_from_mlflow) ──────────────────────────────────────


def download_run_artifacts(
    run_id: str,
    loader: Callable[[str], T | None],
    *,
    prefix: str,
    artifact_path: str = MODEL_ARTIFACT_PATH,
) -> tuple[T, str] | None:
    """Download ``artifact_path`` of MLflow run ``run_id`` and build the model with ``loader``.

    ``loader(local_path)`` receives the downloaded folder and returns the loaded model (any
    shape the plugin uses), or ``None`` if the folder does not hold a complete model.

    Returns ``(loaded, temp_dir)``; the caller owns ``temp_dir`` and MUST remove it after use
    (see :func:`removing_dir`). Returns ``None`` — after removing the temp folder — when the run
    has no such artifacts or ``loader`` returns ``None``. If ``loader`` raises, the temp folder
    is removed and the exception propagates unchanged (``require_user_model`` maps it).
    """
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path=artifact_path)
        if not local_path:
            logger.warning(
                "No %s artifacts found in MLflow run_id=%s", artifact_path, run_id
            )
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        loaded = loader(local_path)
        if loaded is None:
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        logger.info("Downloaded user model from MLflow run_id=%s", run_id)
        return loaded, tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


@contextmanager
def removing_dir(path: str | None) -> Iterator[None]:
    """Remove ``path`` (if any) when the block ends, also on error.

    Wraps the use of a user model downloaded with :func:`download_run_artifacts`::

        with removing_dir(user_temp_dir):
            ...  # predict with the user model
    """
    try:
        yield
    finally:
        if path:
            shutil.rmtree(path, ignore_errors=True)


# ── upload (plugin.train) ────────────────────────────────────────────────────────────────────


def require_train_tracker(mlflow_run_id: str) -> BaseMLflowTracker:
    """Return the tracker of the run a retrain is saved to, or fail if there is none.

    A retrained model lives only in its own MLflow run; without one it would be lost, so
    ``train()`` raises ``ModelPersistenceError`` instead of reporting a success.
    """
    if not mlflow_run_id:
        raise ModelPersistenceError(NO_RUN_MESSAGE)
    return BaseMLflowTracker(mlflow_run_id)


def upload_model_to_mlflow(
    tracker: BaseMLflowTracker,
    write: Callable[[str], None],
    *,
    prefix: str,
    artifact_path: str = MODEL_ARTIFACT_PATH,
    subject: str = "El modelo",
) -> None:
    """Write the retrained model into a temp folder with ``write(dir)`` and upload it to MLflow.

    The temp folder is always removed. Any failure while writing or uploading becomes
    ``ModelPersistenceError("<subject> reentrenado no se ha podido guardar en MLflow: …")``
    so the API reports that the retrain was not saved.
    """
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        write(tmp)
        tracker.upload_artifacts(tmp, artifact_path=artifact_path)
    except Exception as exc:
        logger.error("MLflow artifact upload failed: %s", exc)
        raise ModelPersistenceError(
            f"{subject} reentrenado no se ha podido guardar en MLflow: {exc}"
        ) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def upload_dir_to_mlflow(
    tracker: BaseMLflowTracker,
    local_dir: str,
    *,
    artifact_path: str = MODEL_ARTIFACT_PATH,
    subject: str = "El modelo",
) -> None:
    """Upload an already written folder to MLflow, mapping failures to ``ModelPersistenceError``.

    For plugins whose training writes its bundle to a folder they manage themselves; the folder
    is not removed here.
    """
    try:
        tracker.upload_artifacts(local_dir, artifact_path=artifact_path)
    except Exception as exc:
        logger.error("MLflow artifact upload failed: %s", exc)
        raise ModelPersistenceError(
            f"{subject} reentrenado no se ha podido guardar en MLflow: {exc}"
        ) from exc


# ── stats ────────────────────────────────────────────────────────────────────────────────────


def mlflow_run_summary(mlflow_run_id: str) -> dict:
    """Return ``{"params": …, "metrics": …}`` of an MLflow run (errors propagate)."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    return {"params": tracker.get_params(), "metrics": tracker.get_metrics()}


def add_mlflow_run_to_stats(stats: StatsResponse, mlflow_run_id: str) -> StatsResponse:
    """Add the run's params/metrics under ``stats.metrics["mlflow"]``, best-effort.

    Without ``mlflow_run_id`` the stats are returned untouched. If MLflow cannot be read the
    base stats are still returned (a warning is logged): ``/stats`` never fails because of it.
    """
    if not mlflow_run_id:
        return stats
    try:
        stats.metrics["mlflow"] = mlflow_run_summary(mlflow_run_id)
        logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # stats stay usable
        logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
    return stats
