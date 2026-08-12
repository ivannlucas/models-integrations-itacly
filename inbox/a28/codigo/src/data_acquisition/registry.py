from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_data_manifest import SOURCE_SPECS, build_source_manifest, git_commit, sha256_file
from src.utils import write_json


def source_specs() -> dict[str, dict[str, Any]]:
    return dict(SOURCE_SPECS)


def raw_manifest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "external" / "raw_manifest__mixed_context.json"


def build_raw_manifest(repo_root: Path, *, config_path: str | None = None) -> dict[str, Any]:
    manifests = [build_source_manifest(source_id, write=True) for source_id in SOURCE_SPECS]
    configuration = None
    if config_path:
        config_candidate = Path(config_path)
        if config_candidate.is_absolute():
            resolved = config_candidate
        else:
            resolved = (repo_root / config_candidate).resolve()
        if resolved.exists():
            configuration = {
                "path": resolved.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(resolved),
            }
    payload = {
        "scope": "mixed_context",
        "generated_from": "data_acquisition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "configuration": configuration,
        "sources": manifests,
        "raw_files": [
            raw_file
            for manifest in manifests
            for raw_file in manifest.get("raw_files", [])
        ],
        "generated_outputs": {
            "raw_manifest": "data/raw/external/raw_manifest__mixed_context.json",
            "source_manifests": [
                f"{SOURCE_SPECS[source_id]['raw_dir']}/source_manifest.json"
                for source_id in SOURCE_SPECS
            ],
            "raw_directories": [SOURCE_SPECS[source_id]["raw_dir"] for source_id in SOURCE_SPECS],
        },
    }
    write_json(raw_manifest_path(repo_root), payload)
    return payload
