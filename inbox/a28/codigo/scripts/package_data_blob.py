from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from scripts.build_data_manifest import DEFAULT_MANIFEST_PATH, REPO_ROOT, build_manifest, sha256_file
except ImportError:  # pragma: no cover - script execution fallback
    from build_data_manifest import DEFAULT_MANIFEST_PATH, REPO_ROOT, build_manifest, sha256_file


def _iter_manifest_entries(manifest: dict) -> list[dict]:
    entries: list[dict] = []
    for category_entries in manifest["files"].values():
        entries.extend(category_entries)
    return entries


def _write_sha256_sums(root_dir: Path) -> Path:
    output_path = root_dir / "SHA256SUMS.txt"
    lines = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root_dir).as_posix()}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _reference_date_stamp() -> str | None:
    config_path = REPO_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        reference_date = config.get("project", {}).get("reference_date")
        if not reference_date:
            return None
        return datetime.fromisoformat(str(reference_date)).strftime("%Y%m%d")
    except Exception:
        return None


def create_data_blob(*, output_dir: str | Path = "dist", stamp: str | None = None) -> dict[str, object]:
    output_directory = REPO_ROOT / Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_stamp = stamp or _reference_date_stamp() or datetime.now(timezone.utc).strftime("%Y%m%d")
    workspace_dist = (REPO_ROOT / "dist").resolve()
    manifest_output = DEFAULT_MANIFEST_PATH if output_directory.resolve() == workspace_dist else None

    manifest = build_manifest(
        output_path=manifest_output,
        write_source_manifests=True,
        generation_command=f"python scripts/package_data_blob.py --output-dir {Path(output_dir).as_posix()}",
    )
    zip_name = f"cu28_data_blob_{run_stamp}.zip"
    manifest_name = f"cu28_data_blob_{run_stamp}.manifest.json"
    sha_name = f"cu28_data_blob_{run_stamp}.sha256"

    with TemporaryDirectory() as temp_dir:
        staging_root = Path(temp_dir)
        for entry in _iter_manifest_entries(manifest):
            source_path = REPO_ROOT / entry["path"]
            target_path = staging_root / entry["path"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        staged_manifest_path = staging_root / "data_blob_manifest.json"
        staged_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_sha256_sums(staging_root)

        zip_path = output_directory / zip_name
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(staging_root.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(staging_root).as_posix())

    manifest_path = output_directory / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_sha256 = sha256_file(zip_path)
    sha_path = output_directory / sha_name
    sha_path.write_text(f"{zip_sha256}  {zip_path.name}\n", encoding="utf-8")

    return {
        "zip_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "sha256_path": str(sha_path),
        "zip_sha256": zip_sha256,
        "size_bytes": zip_path.stat().st_size,
        "file_count": len(_iter_manifest_entries(manifest)) + 2,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package the official CU28 mixed_context data blob.")
    parser.add_argument("--output-dir", default="dist", help="Directory where the zip and sidecars will be written.")
    args = parser.parse_args(argv)

    result = create_data_blob(output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
