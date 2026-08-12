from __future__ import annotations

from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_reproduce_mixed_context_smoke_outputs_exist() -> None:
    ensure_repro_smoke_pipeline()
    expected = [
        REPO_ROOT / "reproducibility_manifest__mixed_context.json",
        REPO_ROOT / "dist/cu28_data_blob_20260518.zip",
        REPO_ROOT / "dist/cu28_data_blob_20260518.manifest.json",
        REPO_ROOT / "dist/cu28_data_blob_20260518.sha256",
    ]
    for path in expected:
        assert path.exists(), path
