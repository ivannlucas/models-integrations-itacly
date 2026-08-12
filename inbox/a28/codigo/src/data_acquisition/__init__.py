from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.build_data_manifest import build_source_manifest
from src.data_acquisition.download_ine_cpi import download_ine_cpi
from src.data_acquisition.download_mapa_prices_om import download_mapa_prices_om
from src.data_acquisition.download_mapa_slaughter import download_mapa_slaughter
from src.data_acquisition.registry import build_raw_manifest
from src.utils import ensure_directory


def run_data_acquisition(
    config: dict[str, Any],
    *,
    use_cached_raw: bool = True,
    skip_download: bool = False,
    fail_on_missing_raw: bool = True,
) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    raw_dir = ensure_directory(repo_root / "data" / "raw" / "external")

    restored = {
        "INE_CPI": download_ine_cpi(raw_dir, use_cached_raw=use_cached_raw, skip_download=skip_download),
        "MAPA_SLAUGHTER_MAPA": download_mapa_slaughter(raw_dir, use_cached_raw=use_cached_raw, skip_download=skip_download),
        "MAPA_PRICES_OM": download_mapa_prices_om(raw_dir, use_cached_raw=use_cached_raw, skip_download=skip_download),
    }

    if fail_on_missing_raw:
        missing = [source_id for source_id, paths in restored.items() if source_id != "MAPA_PRICES_OM" and not paths]
        if missing:
            raise FileNotFoundError(f"Missing active raw snapshots for sources: {missing}")

    for source_id in restored:
        build_source_manifest(source_id, write=True)
    raw_manifest = build_raw_manifest(repo_root, config_path=config["project"].get("config_path"))

    return {
        "scope": "mixed_context",
        "raw_manifest_path": str((raw_dir / "raw_manifest__mixed_context.json").resolve()),
        "restored_sources": {key: [str(path) for path in value] for key, value in restored.items()},
        "source_count": len(raw_manifest["sources"]),
    }
