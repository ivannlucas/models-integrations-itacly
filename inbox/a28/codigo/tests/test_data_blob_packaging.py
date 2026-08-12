from __future__ import annotations

from pathlib import Path

from scripts.build_data_manifest import DEFAULT_MANIFEST_PATH
from scripts.package_data_blob import create_data_blob
from scripts.verify_data_blob import verify_zip_file
from tests.conftest import ensure_repro_smoke_pipeline


def test_package_data_blob_creates_zip_and_verify_passes(tmp_path: Path) -> None:
    ensure_repro_smoke_pipeline()
    manifest_before = DEFAULT_MANIFEST_PATH.read_bytes()
    result = create_data_blob(output_dir=tmp_path, stamp="20260518")

    zip_path = Path(result["zip_path"])
    manifest_path = Path(result["manifest_path"])
    sha_path = Path(result["sha256_path"])

    assert zip_path.exists()
    assert manifest_path.exists()
    assert sha_path.exists()
    assert result["zip_sha256"]
    assert result["size_bytes"] > 0

    verification = verify_zip_file(zip_path)
    assert verification["valid"], verification
    assert verification["checked_files"] > 0
    assert DEFAULT_MANIFEST_PATH.read_bytes() == manifest_before
