from __future__ import annotations

from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_eda_outputs_exist() -> None:
    ensure_repro_smoke_pipeline()
    summary_json = REPO_ROOT / "reports/eda/eda_summary__mixed_context.json"
    summary = read_json(summary_json)

    assert summary["figure_paths"]
    assert summary["table_paths"]
    assert summary["notebook_html_paths"]
    for rel_path in summary["notebook_html_paths"][:3]:
        assert (REPO_ROOT / rel_path).exists()
