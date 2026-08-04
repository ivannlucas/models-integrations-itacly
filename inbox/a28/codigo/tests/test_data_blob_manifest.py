from __future__ import annotations

from pathlib import Path

from scripts.build_data_manifest import OFFICIAL_PROCESSED_FILES, SOURCE_SPECS, build_manifest
from tests.conftest import ensure_repro_smoke_pipeline


def test_data_blob_manifest_builds_with_active_source_evidence(tmp_path: Path) -> None:
    ensure_repro_smoke_pipeline()
    manifest_path = tmp_path / "data_blob_manifest.json"
    manifest = build_manifest(output_path=manifest_path, write_source_manifests=True)

    assert manifest_path.exists()
    assert manifest["scope"] == "mixed_context"

    sources = {source["source_id"]: source for source in manifest["sources"]}
    for source_id, spec in SOURCE_SPECS.items():
        source = sources[source_id]
        assert source["official_url"]
        assert source["license_or_terms_url"]
        if spec["evidence_status"] == "active":
            assert source["raw_files"], f"{source_id} should expose raw files for auditability"

    processed_paths = {entry["path"] for entry in manifest["files"]["processed"]}
    assert set(OFFICIAL_PROCESSED_FILES).issubset(processed_paths)

    for entry in manifest["files"]["processed"]:
        assert entry["sha256"]
        assert entry["size_bytes"] > 0


def test_traced_source_keeps_cached_raw_snapshot() -> None:
    ensure_repro_smoke_pipeline()
    manifest = build_manifest(output_path=None, write_source_manifests=True)
    sources = {source["source_id"]: source for source in manifest["sources"]}

    traced_source = sources["MAPA_PRICES_OM"]
    assert traced_source["evidence_status"] == "traced"
    assert traced_source["raw_files"]
    assert traced_source["raw_files"][0]["path"].endswith("s502025rv0.xlsx")
