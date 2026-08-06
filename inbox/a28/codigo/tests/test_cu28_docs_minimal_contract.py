from __future__ import annotations

from scripts.audit_cu28_docs_minimal_contract import (
    DOCS_SNAPSHOT,
    DOCS_WHITELIST,
    REPO_ROOT,
    run_audit,
)
from scripts.build_data_manifest import OFFICIAL_DOCS


def test_docs_directory_matches_exact_whitelist() -> None:
    actual = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "docs").rglob("*")
        if path.is_file()
    }
    assert actual == set(DOCS_WHITELIST)
    assert not (REPO_ROOT / "docs" / "audit").exists()


def test_data_blob_declares_the_same_document_snapshot() -> None:
    assert set(OFFICIAL_DOCS) == set(DOCS_SNAPSHOT)


def test_minimal_document_contract_audit_passes() -> None:
    checks, _ = run_audit()
    failed = [f"{check.name}: {check.detail}" for check in checks if not check.passed]
    assert failed == []
