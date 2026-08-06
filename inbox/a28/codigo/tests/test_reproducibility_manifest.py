from __future__ import annotations

from src.reproducibility.manifest import verify_reproducibility_manifest
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_reproducibility_manifest_contains_hashes() -> None:
    ensure_repro_smoke_pipeline()
    manifest_path = REPO_ROOT / "reproducibility_manifest__mixed_context.json"
    verification = verify_reproducibility_manifest(manifest_path)

    assert verification["valid"], verification
    assert verification["checked_files"] > 0
