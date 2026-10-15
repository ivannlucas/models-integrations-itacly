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

Versioned base model (only models whose train() publishes, see base_model_publisher.py)
--------------------------------------------------------------------------------------
s3://<STORAGE_BUCKET>/artifacts/fixed/<model_name>/current.json   {"version": "vN"}
s3://<STORAGE_BUCKET>/artifacts/fixed/<model_name>/versions/<vN>/  full bundle + version.json

Without current.json the base is "v0": the files at the root, exactly as before. A version is
published as a new folder and current.json is written last, so a failed publish never changes
the served model; nothing is ever overwritten or deleted.
"""

import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import find_dotenv, load_dotenv

from app.domain.services.exceptions import BaseVersionNotFoundError

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

BASE_VERSION = "v0"
POINTER_FILENAME = "current.json"
VERSION_INFO_FILENAME = "version.json"
VERSIONS_DIRNAME = "versions"
_VERSION_RE = re.compile(r"^v(\d+)$")
# Top-level entries of the model folder that are versioning metadata, never v0 artifacts.
_RESERVED_TOP_LEVEL = {POINTER_FILENAME, VERSIONS_DIRNAME}

# Base version in use per model folder, shared by every ArtifactStore of that folder in this
# process (ml15 reads its store from two modules; the container and the publisher build their
# own). Keyed by the local base folder so stores with different roots (tests) never mix.
_ACTIVE_VERSIONS: dict[str, str] = {}


def _build_s3_client():
    """Build and return an S3 client configured from environment variables."""
    import boto3  # type: ignore[import]

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("CUSTOM_S3_ENDPOINT"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("CUSTOM_REGION"),
    )


def _file_needs_download(local_path: Path, remote_size: int) -> bool:
    """Return True if the local file is missing or its size differs from S3."""
    if not local_path.exists():
        return True
    return local_path.stat().st_size != remote_size


def _is_directory_marker(obj: dict, objects: list[dict]) -> bool:
    """True for a folder placeholder, never a real artifact file.

    Besides the usual "key/" convention, the platform's S3 proxy lists every folder created by an
    upload as a 0-byte object WITHOUT a trailing slash ("…/vit_tiny_fan"). Downloading that as a
    file makes the next "…/vit_tiny_fan/fan/…" fail with FileExistsError — ml41, whose checkpoints
    live in subfolders, could never load. A 0-byte key that other keys hang from is a folder.
    """
    key = obj["Key"]
    if key.endswith("/"):
        return True
    if obj.get("Size", 0) != 0:
        return False
    prefix = key + "/"
    return any(other["Key"].startswith(prefix) for other in objects)


def download_s3_file(s3_uri: str) -> str:
    """Download an ``s3://bucket/key`` object to a local temp file and return its path."""
    without_prefix = s3_uri[len("s3://"):]
    bucket, _, key = without_prefix.partition("/")
    s3 = _build_s3_client()
    fd, tmp_path = tempfile.mkstemp(suffix=Path(key).suffix)
    os.close(fd)
    logger.info("Downloading s3://%s/%s to %s", bucket, key, tmp_path)
    s3.download_file(bucket, key, tmp_path)
    return tmp_path


@contextmanager
def local_file_path(data_path: str):
    """Yield a local filesystem path for *data_path*, downloading it first if it's an ``s3://`` URI.

    Any downloaded temp file is removed on exit. Local paths are yielded unchanged.
    """
    if not data_path.startswith("s3://"):
        yield data_path
        return
    tmp_path = download_s3_file(data_path)
    try:
        yield tmp_path
    finally:
        os.unlink(tmp_path)


class ArtifactStore:
    """Resolves and downloads artifact files for a single model.

    Usage::

        _store = ArtifactStore("ml25_wine_sulphites")
        path = _store.path("quality_rf.pkl")   # returns a resolved Path

    Paths resolve against the active base version (v0 unless the model publishes versions).
    """

    def __init__(self, model_name: str, root: Path | None = None, use_s3: bool | None = None) -> None:
        """Initialize the store for the given model name.

        root overrides ARTIFACTS_ROOT and use_s3=False ignores STORAGE_BUCKET (both for tests).
        """
        self._model_name = model_name
        self._local_dir = (root or ARTIFACTS_ROOT) / model_name
        self._use_s3 = use_s3

    # ── versions ──────────────────────────────────────────────────────────────

    def _bucket(self) -> str | None:
        if self._use_s3 is False:
            return None
        return os.environ.get("STORAGE_BUCKET") or None

    def _key(self) -> str:
        return str(self._local_dir)

    def active_version(self) -> str:
        """Return the base version this process serves for the model (v0 by default)."""
        return _ACTIVE_VERSIONS.get(self._key(), BASE_VERSION)

    def _set_active(self, version: str) -> None:
        _ACTIVE_VERSIONS[self._key()] = version

    def _version_dir(self, version: str) -> Path:
        if version == BASE_VERSION:
            return self._local_dir
        return self._local_dir / VERSIONS_DIRNAME / version

    def _remote_root(self) -> str:
        return f"artifacts/fixed/{self._model_name}/"

    def _remote_prefix(self, version: str) -> str:
        if version == BASE_VERSION:
            return self._remote_root()
        return f"{self._remote_root()}{VERSIONS_DIRNAME}/{version}/"

    def _read_json(self, relative: str) -> dict | None:
        """Read <model folder>/<relative> (S3 when configured, else local); None if absent."""
        bucket = self._bucket()
        if bucket:
            try:
                obj = _build_s3_client().get_object(Bucket=bucket, Key=self._remote_root() + relative)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                    return None
                raise
            return json.loads(obj["Body"].read().decode("utf-8"))
        local = self._local_dir / relative
        if not local.exists():
            return None
        return json.loads(local.read_text(encoding="utf-8"))

    def _write_pointer(self, version: str) -> None:
        body = json.dumps({"version": version})
        bucket = self._bucket()
        if bucket:
            _build_s3_client().put_object(Bucket=bucket, Key=self._remote_root() + POINTER_FILENAME,
                                          Body=body.encode("utf-8"))
        else:
            self._local_dir.mkdir(parents=True, exist_ok=True)
            (self._local_dir / POINTER_FILENAME).write_text(body, encoding="utf-8")

    @staticmethod
    def _check_version(version: str) -> None:
        """Reject anything that is not v0 or vN (versions end up in local paths and S3 keys)."""
        if version != BASE_VERSION and not (isinstance(version, str) and _VERSION_RE.fullmatch(version)):
            raise BaseVersionNotFoundError(f"La versión '{version}' es inválida: debe tener el formato vN.")

    def refresh(self) -> str:
        """Read current.json and make its version the active one (v0 if there is none)."""
        pointer = self._read_json(POINTER_FILENAME) or {}
        version = pointer.get("version", BASE_VERSION)
        self._check_version(version)
        self._set_active(version)
        logger.info("Model '%s' base version: %s", self._model_name, version)
        return version

    def _max_version(self) -> int:
        numbers = [0]
        bucket = self._bucket()
        if bucket:
            prefix = f"{self._remote_root()}{VERSIONS_DIRNAME}/"
            paginator = _build_s3_client().get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    match = _VERSION_RE.match(obj["Key"][len(prefix):].split("/", 1)[0])
                    if match:
                        numbers.append(int(match.group(1)))
        else:
            versions_dir = self._local_dir / VERSIONS_DIRNAME
            if versions_dir.exists():
                for child in versions_dir.iterdir():
                    match = _VERSION_RE.match(child.name)
                    if child.is_dir() and match:
                        numbers.append(int(match.group(1)))
        return max(numbers)

    def version_info(self, version: str | None = None) -> dict:
        """Return the version.json of *version* (the active one by default).

        Raises BaseVersionNotFoundError if that version was never published.
        """
        version = version or self.active_version()
        self._check_version(version)
        if version == BASE_VERSION:
            return {"version": BASE_VERSION, "parent": None, "created_at": None,
                    "mlflow_run_id": None, "data_path": None, "metrics": {}}
        local = self._version_dir(version) / VERSION_INFO_FILENAME
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))
        data = self._read_json(f"{VERSIONS_DIRNAME}/{version}/{VERSION_INFO_FILENAME}")
        if data is None:
            raise BaseVersionNotFoundError(
                f"La versión '{version}' del modelo base de '{self._model_name}' no existe."
            )
        return data

    def version_exists(self, version: str) -> bool:
        """True for v0 and for every version with a published version.json."""
        try:
            self.version_info(version)
        except BaseVersionNotFoundError:
            return False
        return True

    def set_active_version(self, version: str) -> None:
        """Point current.json at an already published *version* and serve it from now on."""
        self._check_version(version)
        if not self.version_exists(version):
            raise BaseVersionNotFoundError(
                f"La versión '{version}' del modelo base de '{self._model_name}' no existe."
            )
        self._write_pointer(version)
        self._set_active(version)

    def publish_version(self, bundle_dir: str | Path, info: dict) -> dict:
        """Publish *bundle_dir* on top of the active version as a new version and activate it.

        The new folder holds the active version's files with the bundle's files over them, plus
        version.json (parent, mlflow_run_id, data_path, metrics). current.json is written last:
        if anything fails before, the served version does not change.
        """
        parent = self.active_version()
        bucket = self._bucket()
        if bucket:
            self._download_all()   # the parent must be complete locally to inherit its files
        new_version = f"v{self._max_version() + 1}"
        parent_dir = self._version_dir(parent)

        def _skip_metadata(directory: str, names: list[str]) -> list[str]:
            if Path(directory) != parent_dir:
                return []
            return [n for n in names if n in _RESERVED_TOP_LEVEL or n == VERSION_INFO_FILENAME]

        staging = Path(tempfile.mkdtemp(prefix=f"{self._model_name}_{new_version}_"))
        try:
            if parent_dir.exists():
                shutil.copytree(parent_dir, staging, dirs_exist_ok=True, ignore=_skip_metadata)
            shutil.copytree(bundle_dir, staging, dirs_exist_ok=True)
            version_info = {
                "version": new_version,
                "parent": parent,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mlflow_run_id": info.get("mlflow_run_id") or None,
                "data_path": info.get("data_path") or None,
                "metrics": info.get("metrics") or {},
            }
            (staging / VERSION_INFO_FILENAME).write_text(
                json.dumps(version_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if bucket:
                self._upload_dir(staging, self._remote_prefix(new_version))
            target = self._version_dir(new_version)
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(staging, target)   # local cache (the store itself without S3)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        self._write_pointer(new_version)
        self._set_active(new_version)
        logger.info("Model '%s' base version %s published (parent %s).",
                    self._model_name, new_version, parent)
        return version_info

    def _upload_dir(self, local_dir: Path, remote_prefix: str) -> None:
        """Upload every file of local_dir under remote_prefix, version.json last."""
        bucket = self._bucket()
        s3 = _build_s3_client()
        files = sorted(p for p in local_dir.rglob("*") if p.is_file())
        files.sort(key=lambda p: p == local_dir / VERSION_INFO_FILENAME)
        for file_path in files:
            key = remote_prefix + file_path.relative_to(local_dir).as_posix()
            s3.upload_file(str(file_path), bucket, key)

    # ── artifact paths ────────────────────────────────────────────────────────

    @property
    def local_dir(self) -> Path:
        """Return the local directory of the active base version's artifacts."""
        return self._version_dir(self.active_version())

    def path(self, filename: str) -> Path:
        """Return the local path to *filename*, downloading from S3 if needed.

        Raises FileNotFoundError if the file is absent and S3 is not configured.
        """
        local = self.local_dir / filename
        if not local.exists():
            if self._bucket():
                self._download_all()
            else:
                raise FileNotFoundError(
                    f"Artifact not found: {local}\n"
                    f"Either place the file there or set STORAGE_BUCKET to "
                    f"enable automatic download from S3."
                )
        return local

    def download_all_if_needed(self) -> None:
        """Download every file of the active version from S3, skipping up-to-date files.

        Raises EnvironmentError if STORAGE_BUCKET is not set.
        """
        if not self._bucket():
            raise EnvironmentError(
                "STORAGE_BUCKET is not set. Cannot download artifacts from S3."
            )
        self._download_all()

    # ── S3 download ───────────────────────────────────────────────────────────

    def _download_all(self) -> None:
        """Download all files of the active version from S3, skipping unchanged ones."""
        try:
            from tqdm import tqdm  # type: ignore[import]
        except ImportError:
            tqdm = None

        bucket = self._bucket()
        version = self.active_version()
        s3 = _build_s3_client()
        remote_prefix = self._remote_prefix(version)
        local_dir = self._version_dir(version)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Collect all remote objects under the version prefix
        paginator = s3.get_paginator("list_objects_v2")
        objects = [
            obj
            for page in paginator.paginate(Bucket=bucket, Prefix=remote_prefix)
            for obj in page.get("Contents", [])
        ]
        if version == BASE_VERSION:
            # v0 lives at the model root, next to current.json and versions/: those are not its files.
            objects = [
                obj for obj in objects
                if obj["Key"][len(remote_prefix):].split("/", 1)[0] not in _RESERVED_TOP_LEVEL
            ]
        remote_files = [obj for obj in objects if not _is_directory_marker(obj, objects)]

        if not remote_files:
            logger.warning("No files found in S3 for model: %s (version %s)", self._model_name, version)
            return

        # Filter to only files that need downloading
        pending = []
        for obj in remote_files:
            relative_path = obj["Key"][len(remote_prefix):]
            local_path = local_dir / relative_path
            if _file_needs_download(local_path, obj["Size"]):
                pending.append((obj["Key"], local_path))
            else:
                logger.info("Skipping (unchanged): %s", relative_path)

        if not pending:
            logger.info("All artifact files are up-to-date. No download needed.")
            return

        logger.info(
            "Downloading %d file(s) for model '%s' (version %s) to %s ...\n%s",
            len(pending),
            self._model_name,
            version,
            local_dir.resolve(),
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
