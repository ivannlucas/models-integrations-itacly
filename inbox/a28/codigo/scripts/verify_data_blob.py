from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

try:
    from scripts.build_data_manifest import REPO_ROOT, sha256_file
except ImportError:  # pragma: no cover - script execution fallback
    from build_data_manifest import REPO_ROOT, sha256_file


def _iter_manifest_entries(manifest: dict) -> list[dict]:
    entries: list[dict] = []
    for category_entries in manifest["files"].values():
        entries.extend(category_entries)
    return entries


def _verify_required_paths(manifest: dict, available_paths: set[str]) -> list[str]:
    missing: list[str] = []
    for category, paths in manifest.get("required_paths", {}).items():
        for path in paths:
            normalized = Path(path).as_posix()
            if category == "splits":
                if not any(candidate.startswith(f"{normalized}/") for candidate in available_paths):
                    missing.append(normalized)
            elif category == "raw":
                if not any(candidate.startswith(f"{normalized}/") for candidate in available_paths):
                    missing.append(normalized)
            elif normalized not in available_paths:
                missing.append(normalized)
    return missing


def verify_manifest_file(manifest_path: str | Path) -> dict[str, object]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root_dir = manifest_file.parent

    checked_files = 0
    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    available_paths: set[str] = set()

    for entry in _iter_manifest_entries(manifest):
        available_paths.add(entry["path"])
        candidate = root_dir / entry["path"]
        if not candidate.exists():
            missing_files.append(entry["path"])
            continue
        if sha256_file(candidate) != entry["sha256"]:
            hash_mismatches.append(entry["path"])
        checked_files += 1

    missing_files.extend(_verify_required_paths(manifest, available_paths))
    return {
        "valid": not missing_files and not hash_mismatches,
        "checked_files": checked_files,
        "missing_files": sorted(set(missing_files)),
        "hash_mismatches": sorted(set(hash_mismatches)),
        "scope": manifest.get("scope"),
        "commit": manifest.get("commit"),
    }


def verify_zip_file(zip_path: str | Path) -> dict[str, object]:
    archive_path = Path(zip_path)
    zip_sha256 = sha256_file(archive_path)
    sidecar_path = archive_path.with_suffix(".sha256")
    sidecar_match = None
    if sidecar_path.exists():
        expected_hash = sidecar_path.read_text(encoding="utf-8").split()[0]
        sidecar_match = expected_hash == zip_sha256

    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    checked_files = 0

    with ZipFile(archive_path) as archive:
        members = {name for name in archive.namelist() if not name.endswith("/")}
        if "data_blob_manifest.json" not in members:
            return {
                "valid": False,
                "checked_files": 0,
                "missing_files": ["data_blob_manifest.json"],
                "hash_mismatches": [],
                "scope": None,
                "commit": None,
                "zip_sha256": zip_sha256,
                "zip_size_bytes": archive_path.stat().st_size,
                "sidecar_match": sidecar_match,
            }

        manifest = json.loads(archive.read("data_blob_manifest.json").decode("utf-8"))
        for entry in _iter_manifest_entries(manifest):
            rel_path = entry["path"]
            if rel_path not in members:
                missing_files.append(rel_path)
                continue
            payload = archive.read(rel_path)
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                hash_mismatches.append(rel_path)
            checked_files += 1

        missing_files.extend(_verify_required_paths(manifest, members))

    return {
        "valid": not missing_files and not hash_mismatches and (sidecar_match is not False),
        "checked_files": checked_files,
        "missing_files": sorted(set(missing_files)),
        "hash_mismatches": sorted(set(hash_mismatches)),
        "scope": manifest.get("scope"),
        "commit": manifest.get("commit"),
        "zip_sha256": zip_sha256,
        "zip_size_bytes": archive_path.stat().st_size,
        "sidecar_match": sidecar_match,
    }


def _print_summary(result: dict[str, object]) -> None:
    print(f"valid={result['valid']}")
    print(f"scope={result.get('scope')}")
    print(f"commit={result.get('commit')}")
    print(f"checked_files={result['checked_files']}")
    print(f"missing_files={len(result['missing_files'])}")
    print(f"hash_mismatches={len(result['hash_mismatches'])}")
    if "zip_sha256" in result:
        print(f"zip_sha256={result['zip_sha256']}")
        print(f"zip_size_bytes={result['zip_size_bytes']}")
        print(f"sidecar_match={result.get('sidecar_match')}")
    for path in result["missing_files"]:
        print(f"missing={path}")
    for path in result["hash_mismatches"]:
        print(f"hash_mismatch={path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify an auditable CU28 mixed_context data blob.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", help="Path to a packaged CU28 data-blob zip.")
    group.add_argument("--manifest", help="Path to a workspace manifest JSON.")
    args = parser.parse_args(argv)

    if args.zip:
        result = verify_zip_file(args.zip)
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        result = verify_manifest_file(manifest_path)

    _print_summary(result)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
