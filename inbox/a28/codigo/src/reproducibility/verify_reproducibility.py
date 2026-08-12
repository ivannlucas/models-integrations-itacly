from __future__ import annotations

import argparse

from .manifest import verify_reproducibility_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the CU28 reproducibility manifest.")
    parser.add_argument("--manifest", required=True, help="Path to reproducibility_manifest__mixed_context.json")
    args = parser.parse_args(argv)

    result = verify_reproducibility_manifest(args.manifest)
    print(f"valid={result['valid']}")
    print(f"scope={result['scope']}")
    print(f"commit={result['commit']}")
    print(f"checked_files={result['checked_files']}")
    print(f"missing_files={len(result['missing_files'])}")
    print(f"hash_mismatches={len(result['hash_mismatches'])}")
    for item in result["missing_files"]:
        print(f"missing={item}")
    for item in result["hash_mismatches"]:
        print(f"hash_mismatch={item}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
