from __future__ import annotations

import functools
import logging
import os
import pickle
from typing import Any, Callable

from app.domain.services.exceptions import UserModelUnavailableError

logger = logging.getLogger(__name__)


# What the loaders raise when the run's artifacts are missing, truncated, corrupt or belong to
# another architecture (checked against joblib, pickle, torch, numpy, json, keras and YOLO):
# FileNotFoundError/OSError, ValueError (incl. JSONDecodeError, keras), EOFError (empty pickle),
# UnpicklingError, RuntimeError (load_state_dict size/key mismatch) and KeyError (bundle lookups).
# Anything else (AttributeError, TypeError, NameError…) is a bug in our loader, not a bad run:
# it must surface as a 500 so it is retried and alerted, not as a client error.
ARTIFACT_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, RuntimeError,
)


def require_user_model(download: Callable[..., Any]) -> Callable[..., Any]:
    """Make a plugin's ``download_user_*_from_mlflow(run_id, ...)`` fail loudly.

    The wrapped helpers return ``None`` when the run has no complete model or MLflow is
    unreachable. Returning that ``None`` let callers fall back silently to the base model,
    so a user asking for their retrained model got the base one's predictions. With this
    decorator a ``None`` for a non-empty ``run_id`` raises ``UserModelUnavailableError``, and
    so does an ``ARTIFACT_LOAD_ERRORS`` raised while loading the run's files. Other exceptions
    propagate unchanged.
    """
    @functools.wraps(download)
    def wrapper(run_id: str, *args: Any, **kwargs: Any) -> Any:
        try:
            result = download(run_id, *args, **kwargs)
        except UserModelUnavailableError:
            raise
        except ARTIFACT_LOAD_ERRORS as exc:
            if not run_id:
                raise
            # A run that exists but holds a partial/foreign model (e.g. a training that failed
            # after the platform created the run) is still "this run has no loadable model"
            # (→ 422), not a 500.
            logger.exception("Could not load the user model of MLflow run '%s'", run_id)
            raise UserModelUnavailableError(
                f"No se ha podido cargar el modelo reentrenado del run de MLflow '{run_id}': "
                f"{type(exc).__name__}: {exc}. No se usa el modelo base en su lugar."
            ) from exc
        if result is None and run_id:
            raise UserModelUnavailableError(
                f"No se ha podido cargar el modelo reentrenado del run de MLflow '{run_id}': "
                "el run no contiene un modelo completo o MLflow no es accesible. No se usa el "
                "modelo base en su lugar."
            )
        return result
    return wrapper


class BaseMLflowTracker:
    """Generic MLflow tracker reusable across model runtimes.

    Covers the common operations every runtime needs:
      - Connect to an existing run via run_id
      - Log parameters and per-step metrics
      - Upload / download artifact directories
      - Fetch run metadata (metrics, params, tags)

    Model-specific logic (e.g. how to instantiate a PyTorch model from
    downloaded weight files) stays in the plugin and is not part of this class.
    """

    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:5000")

    def __init__(self, run_id: str = "") -> None:
        self.run_id = run_id
        self._client = None

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, run_id: str) -> None:
        """Set the run_id and (re)build the MLflow client."""
        self.run_id = run_id
        self._client = self._build_client()

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self, uri: str | None = None):
        import mlflow
        from mlflow.tracking import MlflowClient
        uri = uri or self.TRACKING_URI
        mlflow.set_tracking_uri(uri)
        return MlflowClient(tracking_uri=uri)

    def is_connected(self) -> bool:
        """Return True if a non-empty run_id has been set."""
        return bool(self.run_id)

    # ── logging ───────────────────────────────────────────────────────────────

    def log_params(self, params: dict) -> None:
        """Log a dict of training hyperparams. No-op if run_id is empty."""
        if not self.run_id:
            return
        for key, value in params.items():
            self.client.log_param(self.run_id, key, str(value))

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        """Log a dict of metrics at a given step (epoch). No-op if run_id is empty."""
        if not self.run_id:
            return
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, key, value, step=step)

    def set_tags(self, tags: dict) -> None:
        """Set arbitrary key-value tags on the run. No-op if run_id is empty."""
        if not self.run_id:
            return
        for key, value in tags.items():
            self.client.set_tag(self.run_id, key, str(value))

    # ── artifacts ─────────────────────────────────────────────────────────────

    def upload_artifacts(self, local_dir: str, artifact_path: str = "") -> None:
        """
        Upload every file under *local_dir* to MLflow.
        If *artifact_path* is set, files are stored under that prefix
        (e.g. artifact_path="model" -> "model/state_dict.pth").
        No-op if run_id is empty.
        """
        if not self.run_id:
            return
        local_dir = os.path.normpath(local_dir)
        for root, _, files in os.walk(local_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, local_dir)
                dest = os.path.join(artifact_path, rel).replace("\\", "/") if artifact_path else rel
                self.client.log_artifact(self.run_id, fpath, os.path.dirname(dest) or ".")

    def download_artifacts(self, dest_dir: str, artifact_path: str = "") -> str:
        """
        Download artifacts from MLflow to *dest_dir*/*artifact_path*.

        Returns the local directory where files were placed
        (e.g. ``dest_dir/model/`` when artifact_path="model"),
        or an empty string on failure.
        """
        if not self.run_id:
            return ""
        try:
            self.client.download_artifacts(self.run_id, artifact_path, dst_path=dest_dir)
            local_path = os.path.join(dest_dir, artifact_path) if artifact_path else dest_dir
            return os.path.normpath(local_path)
        except Exception as exc:
            logger.warning("MLflow artifact download failed (run=%s path=%s): %s",
                           self.run_id, artifact_path, exc)
            return ""

    # ── metadata ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return all logged metrics from the run. Returns {} on failure."""
        if not self.run_id:
            return {}
        try:
            run = self.client.get_run(self.run_id)
            return dict(run.data.metrics)
        except Exception as exc:
            logger.warning("MLflow get_metrics failed: %s", exc)
            return {}

    def get_params(self) -> dict:
        """Return all logged params from the run. Returns {} on failure."""
        if not self.run_id:
            return {}
        try:
            run = self.client.get_run(self.run_id)
            return dict(run.data.params)
        except Exception as exc:
            logger.warning("MLflow get_params failed: %s", exc)
            return {}

    def get_tags(self) -> dict:
        """Return all tags from the run. Returns {} on failure."""
        if not self.run_id:
            return {}
        try:
            run = self.client.get_run(self.run_id)
            return dict(run.data.tags)
        except Exception as exc:
            logger.warning("MLflow get_tags failed: %s", exc)
            return {}
