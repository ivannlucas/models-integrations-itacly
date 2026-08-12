from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

from scripts.build_data_manifest import SOURCE_SPECS
from src.reproducibility.runtime import ensure_optional_dependency
from src.utils import ensure_directory


def download_mapa_prices_om(
    raw_dir: Path,
    *,
    use_cached_raw: bool = True,
    skip_download: bool = False,
) -> list[Path]:
    ensure_optional_dependency("requests", repo_root_path=raw_dir.parents[2])
    ensure_optional_dependency("bs4", repo_root_path=raw_dir.parents[2])
    import requests
    from bs4 import BeautifulSoup

    target_dir = ensure_directory(raw_dir / "MAPA_PRICES_OM")
    existing = sorted(path for path in target_dir.glob("*.xlsx") if path.is_file())
    if existing and use_cached_raw:
        return existing
    if skip_download and existing:
        return existing
    if skip_download:
        return []

    spec = SOURCE_SPECS["MAPA_PRICES_OM"]
    response = requests.get(spec["official_url"], timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for anchor in soup.select("a[href]"):
        href = str(anchor.get("href", "")).strip()
        if ".xlsx" not in href.lower():
            continue
        url = urljoin(spec["official_url"], href)
        filename = Path(urlparse(url).path).name
        if not filename:
            continue
        content = requests.get(url, timeout=60)
        if not content.ok:
            continue
        output_path = target_dir / filename
        output_path.write_bytes(content.content)
        return [output_path]
    return existing
