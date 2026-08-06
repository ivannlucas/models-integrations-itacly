from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from scripts.build_data_manifest import SOURCE_SPECS
from src.reproducibility.runtime import ensure_optional_dependency
from src.utils import ensure_directory


def download_ine_cpi(
    raw_dir: Path,
    *,
    use_cached_raw: bool = True,
    skip_download: bool = False,
) -> list[Path]:
    ensure_optional_dependency("requests", repo_root_path=raw_dir.parents[2])
    import requests

    target_dir = ensure_directory(raw_dir / "INE_CPI")
    existing = sorted(path for path in target_dir.glob("*.csv") if path.is_file())
    if existing and use_cached_raw:
        return existing
    if skip_download and existing:
        return existing
    if skip_download:
        return []

    spec = SOURCE_SPECS["INE_CPI"]
    url = spec["download_url_or_endpoint"]
    filename = Path(urlparse(url).path).name or "76128.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path = target_dir / filename
    output_path.write_bytes(response.content)
    return [output_path]
