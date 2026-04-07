"""Central artifact resolution for all model plugins.

Local layout (current)
----------------------
artifacts/
  <model_name>/
    <filename>

S3 layout (future, activated when S3_BUCKET env-var is set)
------------------------------------------------------------
s3://<S3_BUCKET>/<S3_PREFIX>/<model_name>/<filename>
                              ^^^^^^^^^^^^^^^^^^^^^^^^^
                              default prefix: "artifacts"

When S3_BUCKET is set and a file is missing locally, ArtifactStore downloads
it to the local cache directory before returning the path.  The rest of the
codebase (every model_loader.py) stays unchanged.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# app/infrastructure/artifact_store.py  →  parents[0]=infrastructure, [1]=app, [2]=repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = _REPO_ROOT / "artifacts"


class ArtifactStore:
    """Resolves artifact file paths for a single model.

    Usage::

        _store = ArtifactStore("wine_sulphite")
        path = _store.path("quality_rf.pkl")   # returns a resolved Path
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._local_dir = ARTIFACTS_ROOT / model_name

    def path(self, filename: str) -> Path:
        """Return the local path to *filename*, downloading from S3 if needed.

        Raises FileNotFoundError if the file is absent and S3 is not configured.
        """
        local = self._local_dir / filename
        if not local.exists():
            if os.environ.get("S3_BUCKET"):
                self._download(filename, local)
            else:
                raise FileNotFoundError(
                    f"Artifact not found: {local}\n"
                    f"Either place the file there or set S3_BUCKET to enable "
                    f"automatic download from S3."
                )
        return local

    # ── S3 download (activated when S3_BUCKET is set) ────────────────────────

    def _download(self, filename: str, dest: Path) -> None:
        bucket = os.environ["S3_BUCKET"]
        prefix = os.environ.get("S3_PREFIX", "artifacts")
        key = f"{prefix}/{self._model_name}/{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", bucket, key, dest)
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 artifact download. "
                "Install it with: pip install boto3"
            ) from exc
        boto3.client("s3").download_file(bucket, key, str(dest))
        logger.info("Downloaded %s (%s)", filename, dest)
