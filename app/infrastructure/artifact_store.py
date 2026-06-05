"""Central artifact resolution for all model plugins.

Local layout
----------------------
artifacts/
  <model_name>/
    <filename>

S3 layout (activated when STORAGE_BUCKET env-var is set)
------------------------------------------------------------
s3://<STORAGE_BUCKET>/artifacts/fixed/<model_name>/<filename>

When STORAGE_BUCKET is set and files are missing or stale locally,
ArtifactStore downloads them before returning the path.
"""

import logging
import os
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

# Load .env if present
_env_path = find_dotenv()
if _env_path:
    logger.info("Fetching .env from: %s", _env_path)
    load_dotenv(_env_path)
else:
    logger.warning("No .env file found in this directory or parent directories.")

# app/infrastructure/artifact_store.py → parents[0]=infrastructure, [1]=app, [2]=repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = _REPO_ROOT / "artifacts"


def _build_s3_client():
    """Build and return an S3 client configured from environment variables."""
    import boto3  # type: ignore[import]

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("CUSTOM_S3_ENDPOINT"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_ID"),
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("CUSTOM_REGION"),
    )


def _file_needs_download(local_path: Path, remote_size: int) -> bool:
    """Return True if the local file is missing or its size differs from S3."""
    if not local_path.exists():
        return True
    return local_path.stat().st_size != remote_size


class ArtifactStore:
    """Resolves and downloads artifact files for a single model.

    Usage::

        _store = ArtifactStore("wine_sulphite")
        path = _store.path("quality_rf.pkl")   # returns a resolved Path
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the store for the given model name."""
        self._model_name = model_name
        self._local_dir = ARTIFACTS_ROOT / model_name

    def get_local_dir(self) -> Path:
        """Return the local directory where this model's artifacts are stored."""
        return self._local_dir

    def path(self, filename: str) -> Path:
        """Return the local path to *filename*, downloading from S3 if needed.

        Raises FileNotFoundError if the file is absent and S3 is not configured.
        """
        local = self._local_dir / filename
        if not local.exists():
            if os.environ.get("STORAGE_BUCKET"):
                self._download_all()
            else:
                raise FileNotFoundError(
                    f"Artifact not found: {local}\n"
                    f"Either place the file there or set STORAGE_BUCKET to "
                    f"enable automatic download from S3."
                )
        return local

    def download_all_if_needed(self) -> None:
        """Download every file under the model's S3 prefix, skipping up-to-date files.

        No-op when STORAGE_BUCKET is not set (assumes artifacts are already local).
        """
        if not os.environ.get("STORAGE_BUCKET"):
            logger.debug(
                "STORAGE_BUCKET not set — skipping S3 sync for '%s'", self._model_name
            )
            return
        self._download_all()

    # ── S3 download ───────────────────────────────────────────────────────────

    def _download_all(self) -> None:
        """Download all files for this model from S3, skipping unchanged ones."""
        try:
            from tqdm import tqdm  # type: ignore[import]
        except ImportError:
            tqdm = None

        bucket = os.environ["STORAGE_BUCKET"]
        s3 = _build_s3_client()
        remote_prefix = f"artifacts/fixed/{self._model_name}/"
        self._local_dir.mkdir(parents=True, exist_ok=True)

        # Collect all remote objects under the model prefix
        paginator = s3.get_paginator("list_objects_v2")
        remote_files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=remote_prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/"):  # skip directory markers
                    remote_files.append(obj)

        if not remote_files:
            logger.warning("No files found in S3 for model: %s", self._model_name)
            return

        # Filter to only files that need downloading
        pending = []
        for obj in remote_files:
            relative_path = obj["Key"][len(remote_prefix):]
            local_path = self._local_dir / relative_path
            if _file_needs_download(local_path, obj["Size"]):
                pending.append((obj["Key"], local_path))
            else:
                logger.info("Skipping (unchanged): %s", relative_path)

        if not pending:
            logger.info("All artifact files are up-to-date. No download needed.")
            return

        logger.info(
            "Downloading %d file(s) for model '%s' to %s ...\n%s",
            len(pending),
            self._model_name,
            self._local_dir.resolve(),
            "\n".join(local_path.name for _, local_path in pending),
        )

        iterator = tqdm(pending, desc="Downloading", unit="file") if tqdm else pending
        for remote_key, local_path in iterator:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, remote_key, str(local_path))
            except Exception as exc:
                logger.error("Failed to download %s: %s", remote_key, exc)
                raise  # bubble up so the app doesn't start with missing artifacts

        logger.info("Artifact download complete for model '%s'.", self._model_name)

    def upload_artifact(self, local_path: Path) -> None:
        """Upload a single artifact file to S3 under the model's prefix."""
        bucket = os.getenv("STORAGE_BUCKET")
        if not bucket:
            logger.warning("STORAGE_BUCKET not set — skipping S3 upload of %s", local_path)
            return

        s3 = _build_s3_client()
        remote_key = f"artifacts/fixed/{self._model_name}/{local_path.name}"
        logger.info("Uploading %s → s3://%s/%s", local_path.name, bucket, remote_key)

        config = TransferConfig(
            multipart_threshold=1024 * 1024 * 1024 * 5,  # 5 GB threshold
            max_concurrency=10,
            use_threads=False                             # Simpler, single-threaded execution
        )

        # Pass the config to the upload_file method
        s3.upload_file(str(local_path), bucket, remote_key, Config=config)
        logger.info("Upload complete: s3://%s/%s", bucket, remote_key)
