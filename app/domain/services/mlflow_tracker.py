from __future__ import annotations

import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)


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

    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    FALLBACK_URIS = [
        "http://mlflow:5000",
        "http://mlflow.mlflow:5000",
        "http://mlflow.mlflow.svc.cluster.local:5000",
        "https://mlflow:5000",
        "https://mlflow.mlflow:5000",
        "https://mlflow.mlflow.svc.cluster.local:5000",
        "http://mlflow.mlflow.svc.cluster.local",
        "https://mlflow.mlflow.svc.cluster.local",
    ]

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


def _mlflow_try_uris(operation_fn, *args, **kwargs):
    """Try *operation_fn* with each FALLBACK_URI until one succeeds."""
    from mlflow.exceptions import MlflowException

    uris = [BaseMLflowTracker.TRACKING_URI]
    for fb in BaseMLflowTracker.FALLBACK_URIS:
        if fb not in uris:
            uris.append(fb)

    attempts = []
    last_error = None
    for uri in uris:
        try:
            result = operation_fn(uri, *args, **kwargs)
            attempts.append(f"{uri}=OK")
            logger.info("MLflow operation succeeded using uri=%s", uri)
            return result
        except MlflowException as exc:
            logger.warning("MLflow operation failed for uri=%s: %s", uri, exc)
            attempts.append(f"{uri}=403({exc})")
            last_error = exc
        except Exception as exc:
            logger.warning("MLflow operation failed for uri=%s (unexpected): %s", uri, exc)
            attempts.append(f"{uri}=ERR({exc})")
            last_error = exc

    raise RuntimeError(f"MLflow connection failed. Tried: {'; '.join(attempts)}")


def _download_with_uri(uri: str, run_id: str, artifact_path: str, prefix: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        client.download_artifacts(run_id, artifact_path, dst_path=tmp)
        local_path = os.path.join(tmp, artifact_path) if artifact_path else tmp
        return os.path.normpath(local_path), tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def download_mlflow_artifacts(
    run_id: str,
    artifact_path: str = "",
    prefix: str = "mlflow_",
) -> tuple[str, str] | None:
    """Download MLflow *artifact_path* from *run_id* to a temporary directory.

    Returns ``(temp_dir, local_path)`` on success.
    *local_path* is the directory containing the downloaded files
    (e.g. ``temp_dir/model/`` when artifact_path="model").

    On failure, *temp_dir* is cleaned up and ``None`` is returned.
    The caller **must** call ``shutil.rmtree(temp_dir, ignore_errors=True)``
    after the loaded model is no longer needed.
    """
    try:
        local_path, tmp = _mlflow_try_uris(_download_with_uri, run_id, artifact_path, prefix)
        return tmp, local_path
    except Exception as exc:
        logger.error("MLflow download failed on all URIs: %s", exc)
        return None
