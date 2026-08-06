from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


def test_readme_contains_official_operational_commands() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    expected_commands = [
        "python -m src.main data_acquisition --mixed-context",
        "python -m src.main etl --mixed-context",
        "python -m src.main feature_engineering --mixed-context",
        "python -m src.main make_splits --mixed-context",
        "python -m src.main train --mixed-context",
        "python -m src.main predict --mixed-context",
        "python -m src.main policy_simulation --mixed-context",
        "python -m src.main get_stats --mixed-context",
        "python scripts/run_notebooks.py --scope mixed_context",
        "python scripts/package_data_blob.py --output-dir dist",
        "python scripts/verify_data_blob.py",
    ]
    for command in expected_commands:
        assert command in readme, command


def test_platform_run_is_documented_only_after_secondary_section() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    secondary_section = "## Ejecución batch sobre CSV de cliente"
    platform_command = "python -m src.main platform_run --input data/demo/customer_upload_example.csv --output outputs/demo_run/"

    assert secondary_section in readme
    assert platform_command in readme
    assert readme.index(secondary_section) < readme.index(platform_command)
    assert "## Comando demo" not in readme
