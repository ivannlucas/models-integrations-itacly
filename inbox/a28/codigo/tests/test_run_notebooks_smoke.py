from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path

from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_run_notebooks_smoke_executes_successfully(tmp_path: Path) -> None:
    ensure_repro_smoke_pipeline()
    report_roots = [
        REPO_ROOT / "reports/eda",
        REPO_ROOT / "reports/notebooks",
        REPO_ROOT / "reports/figures/eda",
        REPO_ROOT / "reports/tables/eda",
    ]
    backup_root = tmp_path / "report_snapshot"
    for report_root in report_roots:
        shutil.copytree(
            report_root,
            backup_root / report_root.relative_to(REPO_ROOT),
        )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_notebooks.py",
                "--scope",
                "mixed_context",
                "--smoke",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        for report_root in report_roots:
            shutil.rmtree(report_root)
            shutil.copytree(
                backup_root / report_root.relative_to(REPO_ROOT),
                report_root,
            )

    assert "failed_notebooks" in result.stdout
